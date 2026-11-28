"""Pair AF slot maps, then report the night-minus-day change on the official test."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

KEY = ['event','start','end','slot','date']


def read_scores(run):
    totals = pd.read_csv(run/'summary.csv').query("protocol=='paper' and task=='AF'")
    if len(totals)!=1 or totals.threshold.item()!=.5:
        raise ValueError(f'Expected one official AF result at threshold 0.5 in {run}')
    return pd.read_csv(run/'window_slots.csv').query("protocol=='paper' and task=='AF'").set_index(KEY).sort_index()


def compare(night, day, smoke=False):
    scores = {'day': read_scores(day), 'night': read_scores(night)}
    a,b = scores.values()
    if a.index.has_duplicates or not a.index.equals(b.index):
        raise ValueError('The held-out event/window/slot/date keys differ or have duplicates')
    if not a.known.eq(256**2).all() or not a.known.equals(b.known):
        raise ValueError('AF evaluation must cover each full 256 x 256 crop in both arms')
    if not (a.tp+a.fn).equals(b.tp+b.fn):
        raise ValueError('Truth counts differ on paired slots')
    counts = a.groupby(level='event').size()
    if not counts.eq(12).all() or (not smoke and len(counts)!=17):
        raise ValueError('The official AF cohort requires 12 scored slots per event')
    records = []
    for model,rows in scores.items():
        for event,g in rows.groupby(level='event'):
            union = g.tp+g.fp+g.fn
            # Both-empty foreground masks get one for that slot.
            macro = g.tp.div(union).fillna(1.).mean()
            records.append((model,event,len(g),macro,g.tp.sum(),g.fp.sum(),g.fn.sum()))
    events = pd.DataFrame(records,columns='model event slots event_iou tp fp fn'.split())
    report = events.groupby('model').agg(events=('event','size'),slots=('slots','sum'),
                     event_macro_iou=('event_iou','mean'),tp=('tp','sum'),
                     fp=('fp','sum'),fn=('fn','sum'))
    denom = report.tp+report.fp+report.fn
    report['pooled_iou'] = (report.tp/denom).fillna(1.)
    for model,run in [('night',night),('day',day)]:
        saved = pd.read_csv(run/'summary.csv').query("protocol=='paper' and task=='AF'").iloc[0]
        for col in ['event_macro_iou','pooled_iou','tp','fp','fn']:
            if not np.isclose(saved[col], report.at[model,col],rtol=0,atol=1e-10):
                raise ValueError(f'{model}: slot aggregation disagrees with saved {col}')
    fields = ['event_macro_iou','pooled_iou','tp','fp','fn']
    delta = (report.loc['night',fields]-report.loc['day',fields]).to_frame().T
    delta.index = ['night_minus_day']
    output = pd.concat([report[fields],delta])
    output[['tp','fp','fn']] = output[['tp','fp','fn']].astype('int64')
    output.to_csv(day/'af_comparison.csv',index_label='model')
    by_event = events.set_index(['event','model']).unstack('model')
    for field in ['event_iou','tp','fp','fn']:
        by_event[field,'night_minus_day'] = by_event[field,'night']-by_event[field,'day']
    by_event.columns = ['_'.join(col) for col in by_event.columns]
    by_event.to_csv(day/'af_comparison_by_event.csv')
    print(f'AF at 0.5: {len(counts)} fires, {len(a)} window slots')
    print(output.to_string())
    print(f'Night minus day: event macro {100*delta.event_macro_iou.item():+.2f} pp, '
          f'pooled {100*delta.pooled_iou.item():+.2f} pp, '
          f'FP {int(delta.fp.item()):+d}, FN {int(delta.fn.item()):+d}')


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('night',type=Path)
    args = p.parse_args()
    compare(args.night,args.night/'day_control')
