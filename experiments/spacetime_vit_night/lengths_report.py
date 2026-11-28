"""Compare temporal lengths on every paper slot and on identical event-dates."""
import sys
from pathlib import Path

import pandas as pd

REFERENCE = {'AF': {2: .727, 4: .703, 6: .667}, 'BA': {2: .789, 4: .771, 6: .768}}
DATE = ['task', 'event', 'date']
SLOT = DATE + ['start', 'end', 'slot']


def support(frame, keys):
    return frame.set_index(keys)[['known', 'positives']].sort_index()


def compare(parent):
    root = parent / 'lengths_2_4_6'
    manifest = pd.read_csv(parent / 'cache/manifest.csv', dtype={'event': str})
    manifest = manifest[manifest.role.isin(['test_af', 'test_ba'])].copy()
    manifest['task'] = manifest.role.str[5:].str.upper()
    if manifest.duplicated(DATE).any():
        raise ValueError('Duplicate event-dates in cache')
    dates = manifest.groupby(['task', 'event']).size()
    outputs, day_support, date_support = [], {}, None
    for ts in (2, 4, 6):
        for arm in ('day', 'night'):
            run = (parent / 'day_control' if arm == 'day' else parent) if ts == 4 else root / f'ts{ts}-{arm}'
            if not (run / 'summary.csv').exists():
                raise SystemExit(f'Awaiting completed evaluation: {run}')
            meta = pd.read_csv(run / 'summary.csv').query("protocol == 'paper'")
            if set(meta.task) != {'AF', 'BA'} or not meta.threshold.eq(.5).all():
                raise ValueError(f'Unexpected evaluation protocol in {run}')
            rows = pd.read_csv(run / 'window_slots.csv', dtype={'event': str}).query("protocol == 'paper'").copy()
            if rows.duplicated(SLOT).any() or not set(rows.slot) == set(range(1, ts+1)):
                raise ValueError(f'Unexpected slots in {run}')
            expected_slots = ((dates + ts - 1) // ts) * ts
            pd.testing.assert_series_equal(rows.groupby(['task', 'event']).size(), expected_slots)
            rows['positives'] = rows.tp + rows.fn
            if arm == 'day':
                day_support[ts] = support(rows, SLOT)
            else:
                pd.testing.assert_frame_equal(day_support[ts], support(rows, SLOT))
            # Prefer the window ending earliest: a repeated date is counted once,
            # with the fewest later observations available in the paper windows.
            unique = rows.sort_values(['end', 'start']).drop_duplicates(DATE)
            keys = pd.MultiIndex.from_frame(manifest[DATE]).sort_values()
            current = support(unique, DATE)
            pd.testing.assert_index_equal(current.index, keys)
            if date_support is None:
                date_support = current
            else:
                pd.testing.assert_frame_equal(date_support, current)
            for cohort, frame in [('paper', rows), ('same_dates', unique)]:
                for task, group in frame.groupby('task'):
                    tp, fp, fn = (int(group[k].sum()) for k in ['tp', 'fp', 'fn'])
                    macro = group.groupby('event').iou.mean().mean()
                    ref = REFERENCE[task][ts] if cohort == 'paper' else None
                    outputs.append(dict(cohort=cohort, ts=ts, inputs=arm, task=task,
                                        events=group.event.nunique(), maps=len(group),
                                        event_macro_iou=macro, pooled_iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
                                        tp=tp, fp=fp, fn=fn, paper_iou=ref,
                                        delta_paper_pp=100*(macro-ref) if ref is not None else None))
    result = pd.DataFrame(outputs).sort_values(['cohort', 'task', 'ts', 'inputs'])
    path = root / 'comparison.csv'
    result.to_csv(path.with_suffix('.tmp'), index=False)
    path.with_suffix('.tmp').replace(path)
    print(result.to_string(index=False, float_format=lambda v: f'{v:.4f}', na_rep=''))
    print(path)


if __name__ == '__main__':
    compare(Path(sys.argv[1]))
