"""Validation-only paired end-to-end Transformer ablation on dated FirePred indices."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from encoder import Encoder
from train import THRESHOLDS, hist, unpack
from vegetation import study_rows


class Inputs(Dataset):
    def __init__(self,rows,cache,indices,arm):
        self.rows=list(rows.itertuples())
        self.cache=cache
        self.indices=indices
        self.arm=arm

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row=self.rows[idx]
        frames=[]
        previous=prev_day=None
        for i in range(row.target_index-row.ts,row.target_index):
            with np.load(self.cache/row.event/f'{i:03d}.npz') as data:
                frame=data['enc']
                day=str(data['date'])
                if i==row.target_index-1:
                    past=unpack(data['past'])
            assert row.first_input<=day<=row.last_input<row.target_date
            vi=np.load(self.indices/row.event/f'{i:03d}.npy')
            difference=np.full_like(vi,np.nan)
            if previous is not None and (pd.Timestamp(day)-prev_day).days==1:
                difference=vi-previous
            valid=np.isfinite(difference)
            extra=np.zeros((4,256,256), dtype=np.float32)
            # Match missingness and model size across all arms.
            extra[2:]=valid
            if self.arm!='levels':
                extra[0]=np.where(valid[0],difference[0],0)
            if self.arm=='both':
                extra[1]=np.where(valid[1],difference[1],0)
            frames.append(np.concatenate((frame.astype('float32'),extra)))
            previous,prev_day=vi,pd.Timestamp(day)
        with np.load(self.cache/row.event/f'{row.target_index:03d}.npz') as data:
            assert str(data['date'])==row.target_date
            y=unpack(data['new'])
        assert y.sum()==row.nan_zero_positives and not (y&past).any()
        return np.stack(frames),past,y


def validate(model,loader,device):
    counts=np.zeros((3,len(THRESHOLDS)),np.int64)
    model.eval()
    with torch.inference_mode():
        for x,past,y in loader:
            logits=model(x.to(device))[1].cpu().numpy()
            logits[past.numpy()]=-np.inf
            counts+=hist(logits.ravel(),y.numpy().ravel())
    tp,fp,fn=counts
    iou=tp/np.maximum(tp+fp+fn,1)
    k=int(iou.argmax())
    return dict(pooled_iou=float(iou[k]),logit_threshold=float(THRESHOLDS[k]),
                tp=int(tp[k]),fp=int(fp[k]),fn=int(fn[k]))


def run(a):
    rows=study_rows()
    assert (a.indices/'complete').exists()
    rows=rows[rows.ts==a.ts].copy()
    assert not rows.duplicated(['event','target_date']).any()
    if a.smoke:
        rows=rows[rows.event.isin(['20702159','20562846']) & (rows.target_index==a.ts)]
        torch.set_num_threads(2)
    device='cpu' if a.smoke else 'cuda'
    if not a.smoke:
        assert torch.cuda.is_available()
        assert all((a.cache/e/'complete').is_file() for e in rows.event.unique())
    train=rows[rows.split=='train'].sort_values(['event','target_index'])
    val=rows[rows.split=='val'].sort_values(['event','target_index'])
    assert len(train) and len(val) and set(train.event).isdisjoint(val.event)
    folder=a.out/(('smoke-' if a.smoke else '')+f'TS{a.ts}-seed{a.seed}')
    folder.mkdir(parents=True,exist_ok=False)
    prior=train.nan_zero_positives.sum()/(len(train)*65536)
    weight=min(50.,(1-prior)/max(prior,1.e-8))
    history=[]
    selected=[]
    for arm in ['levels','ndvi','both']:
        torch.manual_seed(a.seed)
        model=Encoder(channels=51).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=3.e-4,weight_decay=1.e-4)
        training=DataLoader(Inputs(train,a.cache,a.indices,arm),batch_size=a.batch,shuffle=True,
                            generator=torch.Generator().manual_seed(a.seed),num_workers=0)
        validation=DataLoader(Inputs(val,a.cache,a.indices,arm),batch_size=a.batch,num_workers=0)
        best=-1.
        for epoch in range(1,a.epochs+1):
            model.train()
            for x,past,y in training:
                scores=model(x.to(device))[1]
                y=y.to(device).float()
                bce=F.binary_cross_entropy_with_logits(scores,y,pos_weight=scores.new_tensor(weight))
                probs=scores.sigmoid()
                dice=1-(2*(probs*y).sum()+1)/(probs.sum()+y.sum()+1)
                loss=bce+.5*dice
                if not torch.isfinite(loss): raise ValueError('Non-finite training loss')
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
            metrics=validate(model,validation,device)
            row=dict(arm=arm,ts=a.ts,seed=a.seed,epoch=epoch,train_windows=len(train),
                     val_windows=len(val),val_events=val.event.nunique(),**metrics)
            history.append(row)
            print(row,flush=True)
            if metrics['pooled_iou']>best:
                pick=row
                best=metrics['pooled_iou']
                torch.save(model.state_dict(),folder/f'{arm}.pt')
        selected.append(pick)
        pd.DataFrame(history).to_csv(folder/'validation_history.csv',index=False)
        summary=pd.DataFrame(selected)
        summary['delta_vs_levels']=summary.pooled_iou-selected[0]['pooled_iou']
        summary.to_csv(folder/'validation_choices.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['cache','indices','out']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--ts',type=int,choices=[2,4,6],required=True)
    p.add_argument('--seed',type=int,default=804)
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    assert a.epochs>0 and a.batch>0
    run(a)
