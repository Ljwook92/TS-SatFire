"""Prepare aligned raw index values for the paired Transformer experiment."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ASSETS = Path(__file__).resolve().parent / "assets"


def study_rows():
    rows = pd.read_csv(ASSETS/'windows.csv.gz', dtype={'event':str})
    assert rows.groupby('event').split.nunique().eq(1).all()
    # These two validation events share dated VIIRS crop pixels with the training set.
    return rows[(rows.split == 'train') |
                ((rows.split == 'val') & ~rows.event.isin(['22712904', '24462610']))].copy()


def prepare(raw, dest, rows):
    import rasterio
    from prepare import CROP, aligned

    dest.mkdir(parents=True, exist_ok=False)
    counts = []
    for event, group in rows[rows.ts==2].groupby('event'):
        folder = dest/event
        folder.mkdir()
        sequence = sorted((raw/event/'VIIRS_Day').glob('*.tif'))
        ordered = group.sort_values('target_index')
        assert [p.name[:10] for p in sequence[2:]] == ordered.target_date.tolist()
        previous = previous_day = None
        valid_pairs = np.zeros(2, dtype='int64')
        changes = np.zeros(2, dtype='int64')
        changed_days = np.zeros(2, dtype='int64')
        n_pairs = 0
        for i,path in enumerate(sequence[:-1]):
            day = path.name[:10]
            with rasterio.open(path) as image:
                ref = dict(full=image.transform, crop=image.window_transform(CROP),
                           crs=image.crs, shape=image.shape)
            with rasterio.open(raw/event/'FirePred'/path.name.replace('VIIRS_Day','FirePred')) as ds:
                assert ds.descriptions[:2] == ('NDVI_last','EVI2_last'), (event,day,ds.descriptions[:2])
                vi = aligned(ds,[1,2],ref) / 10000.
            # Keep NaN separate from a true zero. Delivered files do not expose source QA.
            vi[(vi < -1.) | (vi > 1.)] = np.nan
            np.save(folder/f'{i:03d}.npy',vi.astype('float32'))
            if previous is not None and (pd.Timestamp(day)-previous_day).days == 1:
                delta = vi-previous
                valid_pairs += np.isfinite(delta).sum((1,2))
                changed = np.abs(delta)>1e-5
                changes += changed.sum((1,2))
                changed_days += changed.any((1,2))
                n_pairs += 1
            previous, previous_day = vi, pd.Timestamp(day)
        for k,name in enumerate(['NDVI','EVI2']):
            counts.append(dict(event=event,split=group.split.iloc[0],index=name,
                               pairs=n_pairs,changed_pairs=changed_days[k],
                               valid_pixels=valid_pairs[k],changed_pixels=changes[k]))
        print('prepared',event,flush=True)
    pd.DataFrame(counts).to_csv(dest/'index_changes.csv',index=False)
    (dest/'complete').touch()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    prepare(a.raw,a.out,study_rows())
