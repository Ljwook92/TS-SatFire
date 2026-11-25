"""Prepare last-input night I4 for paired boundary-head validation."""
import argparse
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from vegetation import study_rows


def cohort():
    rows = study_rows()
    keys = ['event', 'target_date']
    assert rows.calendar_consecutive.all() and not rows.duplicated(keys + ['ts']).any()
    val = rows[rows.split == 'val']
    common = val.groupby(keys).ts.nunique()
    return rows[(rows.split == 'train') | rows.set_index(keys).index.isin(common[common == 3].index)].copy()


def read_night(raw, event, day, ref):
    import rasterio
    from prepare import aligned
    path = raw/event/'VIIRS_Night'/f'{day}_VIIRS_Night.tif'
    if not path.exists():
        return np.full((256, 256), np.nan, np.float32)
    with rasterio.open(path) as src:
        if src.count not in (2, 5):
            raise ValueError(f'Unexpected nighttime layout: {path} {src.descriptions}')
        grid = aligned(src, [src.count-1], ref)[0]
    grid[grid <= 0] = np.nan
    return grid


def night_channels(current, previous, past, spacing):
    delta = current - previous
    values = [(current - 280.)/25., delta/12.]
    if past.any():
        indices = distance_transform_edt(~past, sampling=spacing, return_distances=False, return_indices=True)
        values += [v[tuple(indices)] for v in values[:2]]
    else:
        values += [np.full_like(current, np.nan)] * 2
    data = np.stack(values)
    valid = np.isfinite(data)
    return np.concatenate([np.where(valid, np.clip(data, -8, 8), 0), valid]).astype(np.float16)


def prepare(rows, raw, cache, dest):
    import rasterio
    from prepare import CROP
    dest.mkdir(parents=True, exist_ok=False)
    coverage = []
    for event, frames in rows.drop_duplicates(['event','target_date']).groupby('event'):
        folder = dest/event
        folder.mkdir()
        xy = np.load(cache/event/'xy.npy')
        spacing = [np.linalg.norm(xy[1,0]-xy[0,0]), np.linalg.norm(xy[0,1]-xy[0,0])]
        last_day = None
        for r in frames.sort_values('target_index').itertuples():
            t = date.fromisoformat(r.target_date)
            day, prev = str(t-timedelta(days=1)), str(t-timedelta(days=2))
            assert day == r.last_input and r.first_input <= prev
            with np.load(cache/event/f'{r.target_index-1:03d}.npz') as z:
                assert str(z['date']) == day
                past = np.unpackbits(z['past']).reshape(256,256).astype(bool)
            with np.load(cache/event/f'{r.target_index-2:03d}.npz') as z:
                assert str(z['date']) == prev
            with rasterio.open(raw/event/'VIIRS_Day'/f'{day}_VIIRS_Day.tif') as src:
                ref = dict(full=src.transform,crop=src.window_transform(CROP),crs=src.crs,shape=src.shape)
            if last_day == prev and last_ref == ref:
                previous = current
            else:
                previous = read_night(raw,event,prev,ref)
            current = read_night(raw,event,day,ref)
            data = night_channels(current,previous,past,spacing)
            np.savez_compressed(folder/f'{day}.npz', night=data)
            coverage.append(dict(event=event,target=r.target_date,last_night=day,previous_night=prev,
                                 valid_level_pixels=int(data[4].sum(dtype=int)),
                                 valid_change_pixels=int(data[5].sum(dtype=int))))
            last_day, last_ref = day, ref
        print('prepared nights',event,flush=True)
    pd.DataFrame(coverage).to_csv(dest/'coverage.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['raw','cache','out']:
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    prepare(cohort(), args.raw, args.cache, args.out)
