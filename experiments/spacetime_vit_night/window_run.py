"""Joint AF/BA training and publication-style multi-slot foreground IoU."""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from night_model import NightViT
from run import objective, save


def window_rows(cache, roles, protocol, ts=4):
    table = pd.read_csv(cache/'manifest.csv', dtype={'event':str})
    windows = []
    for event, rows in table[table.role.isin(roles)].groupby('event'):
        rows = list(rows.sort_values('date').itertuples())
        n = len(rows)
        if n < ts:
            if rows[0].role != 'train':
                raise ValueError(f'Incomplete evaluation sequence: {event}')
            windows.append((event, rows[0].role, [None]*(ts-n)+rows))
            continue
        if protocol == 'dense':
            starts = range(n-ts+1)
        else:
            stride = 1 if protocol == 'train' else ts
            # The release right-aligns its last window, including repeated tails.
            starts = [min(i,n-ts) for i in range(0,n,stride)]
        windows.extend((event, rows[0].role, rows[i:i+ts]) for i in starts)
    return windows


class Windows(Dataset):
    def __init__(self, cache, roles, protocol, day_only=False, ts=4):
        self.day_only = day_only
        self.cache = cache
        self.windows = window_rows(cache, roles, protocol, ts)
        with np.load(cache/'norm.npz') as f:
            self.mu = f['mu'][:,:,None,None]
            self.sd = f['sd'][:,:,None,None]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        event, role, frames = self.windows[idx]
        x = np.zeros((2,len(frames),15,256,256),dtype='f4')
        labels = np.zeros((4,len(frames),256,256),bool)  # AF, BA, known AF, known BA x time
        dates = []
        for t, row in enumerate(frames):
            dates.append(row.date if row is not None else '')
            if row is None:
                continue
            with np.load(self.cache/row.file) as f:
                a = np.stack((f['AF'],f['BA'])).astype('f4')
                labels[:,t] = np.unpackbits(f['labels'],axis=-1)
            scaled = np.clip((a-self.mu)/self.sd,-12,12)
            x[:,t,:13] = np.where(np.isfinite(scaled),scaled,0)
            x[:,t,13] = np.isfinite(a[:,0])
            x[:,t,14] = np.isfinite(a[:,11:13]).any(1)
        if self.day_only:
            # Keep parameter shapes, day features, labels and pixel support fixed.
            x[:,:,11:13] = 0
            x[:,:,14] = 0
        return torch.from_numpy(x), torch.from_numpy(labels), event, role, dates


def loader(args, roles, protocol, epoch=0):
    return DataLoader(Windows(args.cache, roles, protocol, args.day_only, args.ts), batch_size=args.batch,
                      shuffle=protocol=='train', num_workers=args.workers,
                      generator=torch.Generator().manual_seed(4200+epoch))


@torch.no_grad()
def score(net, args, roles, protocol, write_maps=False):
    net.eval()
    records = []
    for x, y, events, batch_roles, dates in loader(args, roles, protocol):
        for task, name in enumerate(('AF','BA')):
            selected = [i for i,role in enumerate(batch_roles) if
                        role=='val' or role=='test_'+name.lower()]
            if not selected:
                continue
            prob = net(x[selected,task].to(args.device),task).sigmoid().cpu().numpy()
            decisions = prob >= .5
            for j,i in enumerate(selected):
                win = [d[i] for d in dates]
                if write_maps:
                    folder = args.run/'maps'/name/events[i]
                    folder.mkdir(parents=True, exist_ok=True)
                    # Repeated dates in different windows retain their own maps.
                    path = folder/f'{win[0]}_{win[-1]}.npz'
                    if protocol=='paper' or not path.exists():
                        np.savez_compressed(path, dates=win,
                                            pred=np.packbits(decisions[j],axis=-1),
                                            known=np.packbits(y[i,task+2].numpy(),axis=-1))
                for t,d in enumerate(win):
                    known = y[i,task+2,t].numpy()
                    actual = y[i,task,t].numpy()
                    pred = decisions[j,t]
                    tp = int((known & actual & pred).sum())
                    fp = int((known & ~actual & pred).sum())
                    fn = int((known & actual & ~pred).sum())
                    union = tp+fp+fn
                    records.append((protocol,name,events[i],win[0],win[-1],t+1,d,
                                    int(known.sum()),tp,fp,fn,tp/union if union else 1.))
    return pd.DataFrame(records,columns='protocol task event start end slot date known tp fp fn iou'.split())


def summarize(frames):
    by_event = frames.groupby(['protocol','task','event'],as_index=False).agg(
        slot_maps=('iou','size'), mean_iou=('iou','mean'), tp=('tp','sum'),
        fp=('fp','sum'),fn=('fn','sum'))
    summary = by_event.groupby(['protocol','task'],as_index=False).agg(
        events=('event','size'),slot_maps=('slot_maps','sum'),event_macro_iou=('mean_iou','mean'),
        tp=('tp','sum'),fp=('fp','sum'),fn=('fn','sum'))
    denom = summary.tp+summary.fp+summary.fn
    summary['pooled_iou'] = (summary.tp/denom).fillna(1.)
    return by_event, summary


def fit(args):
    torch.manual_seed(4200)
    net = NightViT(ts=args.ts).to(args.device)
    opt = torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.01)
    ckpt = args.run/'last.pt'
    history, first, best, age = [], 1, -1., 0
    if ckpt.exists():
        s = torch.load(ckpt,map_location=args.device,weights_only=True)
        if (s.get('day_only',False), s.get('ts',4)) != (args.day_only, args.ts):
            raise ValueError('Checkpoint and requested input modes differ')
        net.load_state_dict(s['model'])
        opt.load_state_dict(s['opt'])
        history, best, age = s['history'], s['best'], s['age']
        first = len(history)+1
    if not args.smoke:
        train_windows = window_rows(args.cache,['train'],'train',args.ts)
        val_windows = window_rows(args.cache,['val'],'paper',args.ts)
        sizes = (len(train_windows), len({w[0] for w in train_windows}),
                 len(val_windows), len({w[0] for w in val_windows}))
        expected = {2:(2254,124,141,13), 4:(2252,124,73,13), 6:(2233,124,51,13)}
        if sizes != expected[args.ts]:
            raise ValueError(f'Unexpected paired train/validation cohort: {sizes}')
    for epoch in range(first,args.epochs+1):
        if age>=args.patience:
            break
        net.train()
        losses = []
        train = loader(args,['train'],'train',epoch)
        for step,(x,y,*_) in enumerate(train,1):
            opt.zero_grad(set_to_none=True)
            for task in (0,1):
                known = y[:,task+2].to(args.device)
                if known.any():
                    logits = net(x[:,task].to(args.device),task)
                    loss = objective(logits,y[:,task].to(args.device),known)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite training loss')
                    (loss*.5).backward()
                    losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step()
            if step==1 or step%100==0:
                print(f'epoch {epoch} batch {step}/{len(train)}',flush=True)
        if not losses:
            raise ValueError('No supervised training pixels')
        _, val = summarize(score(net,args,['val'],'paper'))
        metrics = val.set_index('task').event_macro_iou
        measure = float((metrics.AF+metrics.BA)/2)
        age += 1
        if measure>best:
            best, age = measure, 0
            save(dict(model=net.state_dict(),epoch=epoch,day_only=args.day_only,ts=args.ts),args.run/'best.pt')
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),
                            af_iou=float(metrics.AF),ba_iou=float(metrics.BA),threshold=.5))
        save(dict(model=net.state_dict(),opt=opt.state_dict(),history=history,
                  best=best,age=age,day_only=args.day_only,ts=args.ts),ckpt)
        pd.DataFrame(history).to_csv(args.run/'validation.csv',index=False)
        print(history[-1],flush=True)
    (args.run/'fit_done').touch()


def evaluate(args):
    net = NightViT(ts=args.ts).to(args.device)
    ck = torch.load(args.run/'best.pt',map_location='cpu',weights_only=True)
    if (ck.get('day_only',False), ck.get('ts',4)) != (args.day_only, args.ts):
        raise ValueError('Evaluate using the same inputs as the checkpoint')
    net.load_state_dict(ck['model'])
    paper = score(net,args,['test_af','test_ba'],'paper',True)
    dense = score(net,args,['test_af'],'dense',True)
    frames = pd.concat([paper,dense],ignore_index=True)
    events, totals = summarize(frames)
    if not args.smoke:
        af = events.query("task=='AF'")
        for protocol, slots in [('paper', ((10+args.ts-1)//args.ts)*args.ts),
                                ('dense', (11-args.ts)*args.ts)]:
            group = af[af.protocol==protocol]
            assert len(group)==17 and group.slot_maps.eq(slots).all()
        assert len(events.query("task=='BA' and protocol=='paper'"))==24
        assert frames.loc[frames.task=='AF','known'].eq(256**2).all()
    frames.to_csv(args.run/'window_slots.csv',index=False)
    events.to_csv(args.run/'events.csv',index=False)
    totals['checkpoint_epoch'] = ck['epoch']
    totals['threshold'] = .5
    totals.to_csv(args.run/'summary.tmp',index=False)
    os.replace(args.run/'summary.tmp',args.run/'summary.csv')
    print(totals.to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage',choices=['fit','evaluate'])
    for name in ('run','cache'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--ts',type=int,choices=(2,4,6),default=4)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--patience',type=int,default=12)
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--day-only', action='store_true')
    p.add_argument('--smoke',action='store_true')
    args = p.parse_args()
    if args.device=='cpu':
        torch.set_num_threads(2)
    (fit if args.stage=='fit' else evaluate)(args)
