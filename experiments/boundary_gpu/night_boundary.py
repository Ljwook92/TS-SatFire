"""Compare the frozen historical head and paired refits on common validation frames."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from train import (ASSETS, BoundaryHead, Windows, THRESHOLDS, make_bases,
                   boundary_loss, hist, confusion)
from directions import Directions, corrected_scores
from night_i4 import cohort


class NightWindows(Windows):
    def __init__(self, rows, cache, bases, ts, nights):
        super().__init__(rows,cache,bases,ts)
        self.nights = nights

    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        r = self.rows[idx]
        with np.load(self.nights/r.event/f'{r.last_input}.npz') as z:
            night = z['night']
        return (*result, night)


def head(ts, extended):
    net = BoundaryHead(ts)
    if extended:
        old = net.net[0]
        layer = nn.Conv2d(old.in_channels+8,old.out_channels,3,padding=1)
        with torch.no_grad():
            layer.weight.zero_()
            layer.weight[:,:-9].copy_(old.weight[:,:-1])
            layer.weight[:,-1:].copy_(old.weight[:,-1:])
            layer.bias.copy_(old.bias)
        net.net[0] = layer
    return net


def forward(net, arm, local, night, base, owner, device):
    if arm == 'night':
        local = torch.cat([local,night],dim=1)
    return net(local.to(device),base.to(device),owner.to(device))


def forecasts(net, arm, loader, directions, device):
    net.eval()
    with torch.inference_mode():
        for local,base,owner,past,recent,truth,indices,night in loader:
            scores = forward(net,arm,local,night,base,owner,device).cpu().numpy()
            for j,idx in enumerate(indices.tolist()):
                row = loader.dataset.rows[idx]
                burned = past[j].numpy().ravel()
                raw = scores[j].ravel()
                raw[burned] = -np.inf
                ids, sectors, q, _ = directions.get(row,burned,recent[j].numpy().ravel())
                yield row, corrected_scores(raw,ids,sectors,q,True), truth[j].numpy().ravel()


def fit(arm, ts, training, validation, directions, dest, epochs, device):
    torch.manual_seed(804)
    net = head(ts, arm=='night').to(device)
    opt = torch.optim.AdamW(net.parameters(),lr=.0005)
    history = []
    for epoch in range(1,epochs+1):
        net.train()
        for local,base,owner,_,_,truth,_,night in training:
            score = forward(net,arm,local,night,base,owner,device)
            loss = boundary_loss(score,owner.to(device),truth.to(device))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite boundary loss')
            opt.zero_grad()
            loss.backward()
            opt.step()
        counts = {}
        for row,score,y in forecasts(net,arm,validation,directions,device):
            counts[row.event] = counts.get(row.event,0) + hist(score,y)
        history.append(counts)
        torch.save(net.state_dict(),dest/f'{arm}-{epoch}.pt')
        print(f'TS={ts} {arm} epoch {epoch}/{epochs}',flush=True)
    # Each event is scored with an epoch and threshold chosen on the other events.
    events = sorted(history[0])
    assert len(events)>=2
    choices = {}
    for event in events:
        array = np.stack([sum(h[e] for e in events if e!=event) for h in history])
        tp,fp,fn = array.transpose(1,0,2)
        iou = tp/np.maximum(1,tp+fp+fn)
        epoch,k = np.unravel_index(iou.argmax(),iou.shape)
        choices[event]=(int(epoch+1),float(THRESHOLDS[k]))
    return choices


def run(args,rows,device):
    ts = args.ts
    dest = args.out/f'TS{ts}'
    dest.mkdir()
    rows = rows[rows.ts==ts]
    train = rows[rows.split=='train']
    val = rows[rows.split=='val'].sort_values(['event','target_date'])
    assert len(train) and len(val) and set(train.event).isdisjoint(val.event)
    bases = dest/'base'
    make_bases(rows,args.cache,bases,ts,device)
    def loader(table, shuffle=False):
        ds = NightWindows(table,args.cache,bases,ts,args.out/'nights')
        return DataLoader(ds,batch_size=2,shuffle=shuffle,num_workers=args.workers,
                          generator=torch.Generator().manual_seed(804))
    directions = Directions(ASSETS,args.cache)
    validation = loader(val)
    choices = {arm:fit(arm,ts,loader(train,True),validation,directions,dest,args.epochs,device)
               for arm in ['control','night']}
    ref = args.reference/f'TS{ts}'
    frozen = json.loads((ref/'choices.json').read_text())
    threshold = frozen['models']['local_extent']['threshold']
    (dest/'choices.json').write_text(json.dumps(dict(paired=choices,historical=frozen),indent=2))
    records = []
    for event, group in val.groupby('event'):
        nets = {}
        for arm in ['control','night','historical']:
            net = head(ts,arm=='night').to(device)
            if arm=='historical':
                path,cut=ref/'best.pt',threshold
            else:
                epoch,cut=choices[arm][event]
                path=dest/f'{arm}-{epoch}.pt'
            net.load_state_dict(torch.load(path,map_location=device,weights_only=True))
            net.eval()
            nets[arm]=(net,cut)
        with torch.inference_mode():
            group_loader=loader(group)
            for local,base,owner,past,recent,truth,indices,night in group_loader:
                outputs={arm:forward(net,arm,local,night,base,owner,device).cpu().numpy()
                         for arm,(net,_) in nets.items()}
                for j,idx in enumerate(indices.tolist()):
                    r=group_loader.dataset.rows[idx]
                    y=truth[j].numpy().ravel()
                    burned=past[j].numpy().ravel()
                    ids,sec,q,_=directions.get(r,burned,recent[j].numpy().ravel())
                    flags={}
                    for arm,scores in outputs.items():
                        raw=scores[j].ravel()
                        raw[burned]=-np.inf
                        scores_corr=corrected_scores(raw,ids,sec,q,True)
                        flags[arm]=scores_corr.astype(np.float64)>=nets[arm][1]
                    for comp in ['control','historical']:
                        gained=int((y & flags['night'] & ~flags[comp]).sum())
                        lost=int((y & ~flags['night'] & flags[comp]).sum())
                        b=confusion(flags[comp],y)
                        n=confusion(flags['night'],y)
                        assert gained-lost==b[2]-n[2]
                        record=dict(event=event,target=r.target_date,ts=ts,comparison=comp,
                                    truth_hash=hashlib.sha256(np.packbits(y)).hexdigest(),
                                    negatives=int((~y).sum()),positives=int(y.sum()),recovered=gained,lost=lost)
                        for name,cm in [('baseline',b),('night',n)]:
                            record.update({name+'_'+k:v for k,v in zip(['tp','fp','fn'],cm)})
                        records.append(record)
    directions.domains.close()
    pd.DataFrame(records).to_csv(dest/'paired.csv',index=False)


def summarize(root):
    frames=pd.concat([pd.read_csv(root/f'TS{ts}'/'paired.csv',dtype={'event':str}) for ts in (2,4,6)])
    keys=['event','target','comparison']
    assert not frames.duplicated(keys+['ts']).any()
    grouped=frames.groupby(keys)
    assert grouped.ts.nunique().eq(3).all() and grouped.truth_hash.nunique().eq(1).all()
    assert (frames.positives+frames.negatives==65536).all()
    def metrics(g):
        totals=g.select_dtypes('number').sum()
        row=dict(windows=len(g),events=g.event.nunique())
        for name in ['baseline','night']:
            tp,fp,fn=[int(totals[name+'_'+k]) for k in ('tp','fp','fn')]
            row.update({name+'_iou':tp/max(1,tp+fp+fn),name+'_fn':fn,name+'_fp':fp,
                        name+'_recall':tp/max(1,tp+fn)})
        row.update(delta_iou=row['night_iou']-row['baseline_iou'],
                   delta_fn=row['night_fn']-row['baseline_fn'],recovered=int(totals.recovered),lost=int(totals.lost))
        return row
    for keys,file in [(['ts','comparison'],'summary.csv'),(['ts','comparison','event'],'by_event.csv')]:
        rows=[dict(**dict(zip(keys,group)),**metrics(data)) for group,data in frames.groupby(keys)]
        pd.DataFrame(rows).to_csv(root/file,index=False)
    print((root/'summary.csv').read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--reference',type=Path)
    p.add_argument('--cache',type=Path)
    p.add_argument('--ts',type=int,choices=[2,4,6])
    p.add_argument('--epochs',type=int,default=6)
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    if a.ts is None:
        summarize(a.out)
    else:
        assert torch.cuda.is_available() and a.epochs>=1
        run(a,cohort(),'cuda')
