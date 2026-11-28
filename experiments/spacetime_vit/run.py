"""Train on whole images, select on validation fires, then map official test fires."""
import argparse,json,os
from datetime import date,timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from model import FireViT

TASKS=('AF','BA')
CUT=np.unique(np.r_[np.geomspace(.0001,.1,40),np.linspace(.1,.99,90)])

class Frames(Dataset):
    def __init__(self,cache,roles):
        self.cache=cache
        rows=pd.read_csv(cache/'manifest.csv',dtype={'event':str})
        self.rows=list(rows[rows.role.isin(roles)].itertuples(index=False))
        self.path={(r.event,date.fromisoformat(r.date)):cache/r.file for r in self.rows}
        with np.load(cache/'norm.npz') as p:self.mu=p['mu'];self.sd=p['sd']

    def __len__(self):return len(self.rows)

    def __getitem__(self,i):
        r=self.rows[i]; day=date.fromisoformat(r.date)
        x=np.zeros((2,4,12,256,256),'f4')
        for t in range(4):
            path=self.path.get((r.event,day-timedelta(days=3-t)))
            if path is None:continue
            with np.load(path) as z:
                for j,key in enumerate(TASKS):
                    a=z[key].astype('f4')
                    valid=np.isfinite(a).all(0)
                    x[j,t,:11]=np.where(valid,np.clip((a-self.mu[j,:,None,None])/self.sd[j,:,None,None],-12,12),0)
                    x[j,t,11]=valid
                if t==3:y=np.unpackbits(z['labels'],axis=-1).astype(bool)
        return torch.from_numpy(x),torch.from_numpy(y),r.event,r.date,r.role

def objective(logit,y,known):
    z=logit[known].float();y=y[known].float()
    bce=nn.functional.binary_cross_entropy_with_logits(z,y)
    p=z.sigmoid()
    return bce+1-(2*(p*y).sum()+1)/(p.sum()+y.sum()+1)

def loader(cache,roles,args,train=False,epoch=0):
    return DataLoader(Frames(cache,roles),batch_size=args.batch,shuffle=train,num_workers=args.workers,
                      generator=torch.Generator().manual_seed(4200+epoch),pin_memory=args.device=='cuda')

@torch.no_grad()
def calibrate(net,dl,device):
    net.eval()
    hist=np.zeros((2,2,len(CUT)+1),dtype='int64')
    for x,y,*_ in dl:
        for j in range(2):
            known=y[:,j+2].numpy()
            if not known.any():continue
            prob=net(x[:,j].to(device),j).sigmoid().cpu().numpy()[known]
            truth=y[:,j].numpy()[known]
            bins=np.searchsorted(CUT,prob,side='right')
            for cls in (0,1):
                hist[j,cls]+=np.bincount(bins[truth==cls],minlength=len(CUT)+1)
    cuts=[];ious=[]
    for h in hist:
        if not h[1].sum():raise ValueError('No positive validation pixels for task')
        fp,tp=np.cumsum(h[:,::-1],axis=1)[:,::-1][:,1:]
        iou=tp/(fp+h[1].sum());k=int(np.argmax(iou))
        cuts.append(float(CUT[k]));ious.append(float(iou[k]))
    return cuts,ious

def save(obj,path):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,path)

def fit(args):
    torch.manual_seed(4200);net=FireViT().to(args.device)
    opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.01)
    path=args.run/'last.pt';start=1;best=-1.;age=0
    valid=loader(args.cache,['val'],args)
    history=[]
    if path.exists():
        ck=torch.load(path,map_location=args.device,weights_only=True)
        net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        start=ck['epoch']+1;best=ck['best'];age=ck['age'];history=ck['history']
    for epoch in range(start,args.epochs+1):
        if age>=args.patience:break
        net.train(); loss_sum=0; steps=0
        for x,y,*_ in loader(args.cache,['train'],args,True,epoch):
            opt.zero_grad(set_to_none=True);parts=0
            for j in range(2):
                known=y[:,j+2].to(args.device)
                if not known.any():continue
                z=net(x[:,j].to(args.device),j)
                loss=objective(z,y[:,j].to(args.device),known)
                if not torch.isfinite(loss):raise FloatingPointError('non-finite training loss')
                (loss*.5).backward();loss_sum+=float(loss.detach());parts+=1
            if parts:
                nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();steps+=parts
        if not steps:raise ValueError('No supervised training pixels')
        cuts,ious=calibrate(net,valid,args.device)
        metric=float(np.mean(ious));age+=1
        if metric>best:
            best=metric;age=0
            save(dict(model=net.state_dict(),thresholds=cuts,epoch=epoch),args.run/'best.pt')
        history.append(dict(epoch=epoch,loss=loss_sum/steps,af_iou=ious[0],ba_iou=ious[1],
                            af_cut=cuts[0],ba_cut=cuts[1]))
        save(dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,best=best,age=age,history=history),path)
        pd.DataFrame(history).to_csv(args.run/'validation.csv',index=False)
        print(history[-1],flush=True)
    (args.run/'fit_done').touch()

@torch.no_grad()
def evaluate(args):
    ck=torch.load(args.run/'best.pt',map_location='cpu',weights_only=True)
    net=FireViT().to(args.device);net.load_state_dict(ck['model']);net.eval()
    cuts=ck['thresholds'];records=[]
    for x,y,events,dates,roles in loader(args.cache,['test_af','test_ba'],args):
        p=np.stack([net(x[:,j].to(args.device),j).sigmoid().cpu().numpy() for j in range(2)],axis=1)
        for i,(ev,d,role) in enumerate(zip(events,dates,roles)):
            j=0 if role=='test_af' else 1
            target,mask=y[i,j].numpy(),y[i,j+2].numpy()
            pred=p[i,j]>=cuts[j]
            tp=int((mask&target&pred).sum());fp=int((mask&~target&pred).sum());fn=int((mask&target&~pred).sum())
            import hashlib
            digest=hashlib.sha256(np.packbits(mask).tobytes()+np.packbits(target).tobytes()).hexdigest()
            records.append((ev,d,TASKS[j],int(mask.sum()),digest,tp,fp,fn))
            dest=args.run/'maps'/ev;dest.mkdir(parents=True,exist_ok=True)
            valid=x[i,:,-1,-1].numpy().astype(bool)
            # Store hard masks before probability quantization for exact reproducibility.
            np.savez_compressed(dest/f'{d}.npz',prob=np.where(valid,p[i],-1).astype('f2'),
                                pred=np.packbits(p[i]>=np.array(cuts)[:,None,None],axis=-1))
    cur=pd.DataFrame(records,columns='event date task pixels pixel_truth_hash tp fp fn'.split())
    key=['event','date','task']
    old=pd.read_csv(Path(__file__).parent/'assets/frames11.csv',dtype={'event':str})
    paired=cur.merge(old[key+['pixels','pixel_truth_hash','tp','fp','fn','full_window','label_sel']],on=key,suffixes=('','_tcn'),validate='one_to_one')
    assert len(paired)==len(cur)
    if not args.smoke:assert len(paired)==len(old)
    assert (paired.pixel_truth_hash==paired.pixel_truth_hash_tcn).all() and (paired.pixels==paired.pixels_tcn).all()
    paired.to_csv(args.run/'frames.csv',index=False)
    overlap=set(pd.read_csv(Path(__file__).parent/'assets/overlap.csv').test_event)
    summary=[];event_results=[]
    for task,g in paired.groupby('task'):
        groups={'all_dates':g,'known_dates':g[g.pixels>0],
                'four_observed_dates':g[g.full_window & (g.pixels>0)]}
        if task=='AF':groups['no_development_overlap']=g[~g.event.isin(overlap)]
        if task=='BA':groups['known_background']=g[(g.label_sel==1)&(g.pixels>0)]
        for subset,gg in groups.items():
            if gg.empty:continue
            for model,suffix in [('ViT',''),('previous_TCN','_tcn')]:
                tp,fp,fn=(gg[k+suffix] for k in ['tp','fp','fn'])
                u=tp+fp+fn
                dates_iou=(tp/u).fillna(1)
                for ev,indices in gg.groupby('event').groups.items():
                    event_results.append((task,subset,model,ev,len(indices),int(tp.loc[indices].sum()),
                                          int(fp.loc[indices].sum()),int(fn.loc[indices].sum()),dates_iou.loc[indices].mean()))
                summary.append((task,subset,model,len(gg),gg.event.nunique(),tp.sum(),fp.sum(),fn.sum(),
                                tp.sum()/max(1,u.sum()),(tp/u).fillna(1).groupby(gg.event).mean().mean()))
    df=pd.DataFrame(summary,columns='task cohort model dates events tp fp fn pooled_iou event_macro_iou'.split())
    pd.DataFrame(event_results,columns='task cohort model event days tp fp fn mean_daily_iou'.split()).to_csv(args.run/'events.csv',index=False)
    df.to_csv(args.run/'summary.tmp',index=False);os.replace(args.run/'summary.tmp',args.run/'summary.csv');print(df.to_string(index=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['fit','evaluate'])
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--patience',type=int,default=12);p.add_argument('--batch',type=int,default=1)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();args.run.mkdir(parents=True,exist_ok=True)
    if args.device=='cpu':torch.set_num_threads(2)
    (fit if args.mode=='fit' else evaluate)(args)
