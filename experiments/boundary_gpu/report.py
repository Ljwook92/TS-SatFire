"""Full-crop counts and a matched map of the New Mexico east boundary."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


def draw_east(dest, r, masks, y, past, ids, sector):
    east = np.zeros(y.shape, bool)
    east[ids[sector == 2]] = True
    broad = np.zeros(y.shape, bool)
    broad[ids[np.isin(sector,[1,2,3])]] = True
    rows = []
    for region, subset in [('E',east), ('NE_E_SE',broad), ('full_crop',np.ones_like(y))]:
        for name, flag in masks.items():
            tp = int((subset & y & flag).sum())
            fp = int((subset & ~y & flag).sum())
            fn = int((subset & y & ~flag).sum())
            rows.append(dict(ts=r.ts, model=name, region=region, candidates=int(subset.sum()),
                             tp=tp, fp=fp, fn=fn, iou=tp/(tp+fp+fn) if tp+fp+fn else np.nan))
    colors = ['white','#b8bfc5','#d95e36','#3475b2','#24875d']
    cmap = ListedColormap(colors)
    panels = [('Observed',None),('Existing TS2 forecast',masks['archived_ts2']),
              ('Sector allocation only',masks['allocation_only']),('Local boundary model',masks['local_extent'])]
    fig,axes = plt.subplots(2,len(panels),figsize=(14,7.5),squeeze=False)
    for row, view in enumerate([past|y|broad, east]):
        rr,cc = np.where(view.reshape(256,256))
        if not len(rr):
            rr,cc = np.array([0,255]),np.array([0,255])
        margin = 5 if row==0 else 2
        for ax,(label,flag) in zip(axes[row],panels):
            layer = past.astype('uint8')
            layer[y] = 2
            if flag is not None:
                layer[flag] = 3
                layer[flag & y] = 4
            ax.imshow(layer.reshape(256,256), interpolation='nearest', cmap=cmap,vmin=0,vmax=4)
            if east.any():
                ax.contour(east.reshape(256,256),levels=[.5],colors=['#825398'],linewidths=.7)
            ax.set_xlim(max(-.5,cc.min()-margin),min(255.5,cc.max()+margin))
            ax.set_ylim(min(255.5,rr.max()+margin),max(-.5,rr.min()-margin))
            if row==0: ax.set_title(label)
            if row==1 and flag is not None:
                ax.set_xlabel(f'East   TP {(east&y&flag).sum()}   FP {(east&~y&flag).sum()}   FN {(east&y&~flag).sum()}')
        axes[row,0].set_ylabel(['Fire vicinity (pixel row)','East boundary detail (pixel row)'][row])
    fig.suptitle(f'New Mexico, June 1, 2021    |    TS={r.ts}    |    purple outline = input-defined E candidates')
    fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(colors[1:],['Previously recorded','Observed new / missed','False positive','True positive'])],
               loc='lower center',ncol=4,frameon=False)
    fig.tight_layout(rect=(0,.055,1,.95))
    fig.savefig(dest/'new_mexico_east.png',dpi=150)
    plt.close(fig)
    return rows


def summarize(run):
    frames = pd.concat([pd.read_csv(run/f'TS{ts}/frame_metrics.csv',dtype={'event':str}) for ts in (2,4,6)])
    keys = ['event','target']
    reference = frames.query("model=='archived_ts2'")
    assert not frames.duplicated(keys+['ts','model']).any()
    for (ts,model),g in frames.groupby(['ts','model']):
        ref = reference[reference.ts==ts]
        assert set(map(tuple,g[keys].values))==set(map(tuple,ref[keys].values))
    common = set.intersection(*[set(map(tuple,g[keys].values)) for _,g in reference.groupby('ts')])
    consecutive = reference.groupby(keys).consecutive.all()
    gap_free = common & set(consecutive[consecutive].index)
    regions = []
    for cohort, subset in [('all_rolling',frames),('common_dates',frames[frames.set_index(keys).index.isin(common)]),
                           ('common_consecutive',frames[frames.set_index(keys).index.isin(gap_free)])]:
        for (ts,model),g in subset.groupby(['ts','model']):
            tp,fp,fn = map(int,g[['tp','fp','fn']].sum())
            union = g.tp+g.fp+g.fn
            iou = g.assign(iou=g.tp.div(union.where(union>0)).fillna(1))
            active = iou[iou.tp+iou.fn>0]
            regions.append(dict(cohort=cohort, ts=ts, model=model, windows=len(g),events=g.event.nunique(),
                                tp=tp,fp=fp,fn=fn,pooled_iou=tp/(tp+fp+fn) if tp+fp+fn else np.nan,
                                event_macro_iou=iou.groupby('event').iou.mean().mean(),
                                active_event_macro_iou=active.groupby('event').iou.mean().mean()))
    table = pd.DataFrame(regions)
    base = table.query("model=='archived_ts2'").set_index(['cohort','ts']).pooled_iou
    table['delta_vs_existing_ts2'] = table.pooled_iou - base.reindex(pd.MultiIndex.from_frame(table[['cohort','ts']])).to_numpy()
    table['reaches_40pct_pooled'] = table.pooled_iou>=.4
    table.to_csv(run/'summary.csv',index=False)
    print(table.round(4).to_string(index=False))


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    summarize(p.parse_args().run)
