"""Prepare per-observation day/night inputs on the existing ViT crop and split."""
import argparse
import hashlib
import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.ndimage import uniform_filter

from prepare import bands, spectrum, hotspots


def read_night(dayfile, window, grid):
    path = dayfile.parent.parent / 'VIIRS_Night' / dayfile.name.replace('VIIRS_Day', 'VIIRS_Night')
    if not path.is_file():
        return np.full((2, 256, 256), np.nan, dtype='f4'), False
    with rasterio.open(path) as ds:
        if ds.count not in (2, 5):
            raise ValueError(f'Unexpected night bands ({ds.count}): {path}')
        return bands(ds, window, grid)[-2:], True


def build_event(raw, cache, row, points, thresholds, official_masks, moments, limit):
    files = sorted((raw / row.event / 'VIIRS_Day').glob('[0-9]*.tif'))
    if limit:
        files = files[:limit]
    if not files:
        if row.role == 'train' and row.event == '24332700':
            return []
        raise FileNotFoundError(f'Missing VIIRS_Day: {row.event}')
    folder = cache / row.event
    folder.mkdir(exist_ok=True)
    with rasterio.open(files[0]) as ds:
        ox, oy = (ds.width//2-128, ds.height//2-128) if row.role == 'test_af' else (128, 128)
        window = Window(ox, oy, 256, 256)
        grid = ds.window_transform(window), ds.crs
        reference = spectrum(bands(ds, window, grid))
    geom = dict(affine=list(grid[0])[:6], crs=grid[1].to_wkt())
    (folder / 'grid.json').write_text(json.dumps(geom))
    if row.role == 'test_af' and len(files) != len(thresholds[row.event][0]):
        raise ValueError(f'AF threshold/date mismatch: {row.event}')
    lookup = hotspots(grid, points[points.event == row.event])
    peak = np.full((13, 256, 256), np.nan, dtype='f4')
    accum_af = np.zeros((256, 256), bool)
    burned = accum_af.copy()
    rows = []
    for k, file in enumerate(files):
        day = date.fromisoformat(file.name[:10])
        with rasterio.open(file) as ds:
            a = bands(ds, window, grid)
        night, found = read_night(file, window, grid)
        i4, i5 = a[3:5]
        finite = np.isfinite(i4)
        den = uniform_filter(finite.astype('f4'), 7)
        local = np.divide(uniform_filter(np.where(finite, i4, 0), 7), den,
                          out=np.full_like(i4, np.nan), where=den > .25)
        x = np.concatenate([np.stack([i4, i5, i4-i5, i4-local]), reference-spectrum(a),
                            np.stack(lookup(day)), night])
        peak = np.fmax(peak, x)
        af = a[6] > 0 if len(a) >= 7 else np.zeros_like(burned)
        perimeter = a[7] if len(a) >= 8 else np.full_like(i4, np.nan)
        accum_af |= af
        burned |= af | (perimeter > 0)
        ba = burned
        known_af = (np.isfinite(i4) | np.isfinite(night).any(0)) & (len(a) >= 7)
        known_ba = np.isfinite(peak[[0,11,12]]).any(0) & (len(a) >= 7)
        if row.role == 'test_af':
            if not found:
                raise FileNotFoundError(f'Official AF test night missing: {file}')
            td, tn = (part[k] for part in thresholds[row.event])
            af = (i4 > td[0]) | (i5 > td[1]) | (night[0] > tn[0]) | (night[1] > tn[1])
            digest = hashlib.sha256(np.packbits(af).tobytes()).hexdigest()
            if digest != official_masks.loc[(row.event,str(day))]:
                raise ValueError(f"Official AF label mismatch: {row.event} {day}")
            known_af = np.ones_like(af)
            known_ba[:] = False
        elif row.role == 'test_ba':
            known_af[:] = False
            if int(row.label_sel) == 1:
                ba, known_ba = accum_af, np.ones_like(burned)
            else:
                ba, known_ba = perimeter > 0, np.isfinite(perimeter) & (perimeter != -1)
        if row.role == 'train':
            for task, data in enumerate((x, peak)):
                sample = data[:, ::8, ::8].astype('f8')
                moments[task,0] += np.isfinite(sample).sum((1,2))
                moments[task,1] += np.nansum(sample, axis=(1,2))
                moments[task,2] += np.nansum(sample**2, axis=(1,2))
        path = folder / f'{day}.npz'
        np.savez(path, AF=x.astype('f2'), BA=peak.astype('f2'),
                 labels=np.packbits(np.stack([af, ba, known_af, known_ba]), axis=-1))
        rows.append((row.event, row.role, str(day), str(path.relative_to(cache)), int(found)))
    return rows


def main():
    p = argparse.ArgumentParser()
    for key in ('raw', 'cache', 'assets', 'firms'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(args.assets/'split.csv', dtype={'event':str})
    assert split.event.is_unique and split.role.value_counts().to_dict() == {
        'train':125, 'val':13, 'test_af':17, 'test_ba':24}
    if args.smoke:
        split = split.groupby('role', sort=False).head(1)
    points = pd.concat([pd.read_csv(p, dtype={'event':str}) for p in
                        (args.firms, args.assets/'af_firms.csv.gz')])
    thresholds = json.loads((args.assets/'af_thresholds.json').read_text())
    official_masks = pd.read_csv(Path(__file__).with_name("official_af_hashes.csv")).set_index(["event","date"]).sha256
    moments = np.zeros((2,3,13))
    rows, audit = [], []
    for row in split.itertuples():
        limit = 4 if args.smoke and row.role != 'test_af' else None
        frames = build_event(args.raw, args.cache, row, points, thresholds, official_masks, moments, limit)
        audit.append((row.event, row.role, len(frames), sum(r[-1] for r in frames)))
        rows.extend(frames)
        print(row.role, row.event, len(frames), flush=True)
    if (moments[:,0] == 0).any():
        raise ValueError('At least one predictor has no training observations')
    mu = moments[:,1]/moments[:,0]
    sd = np.sqrt(np.maximum(moments[:,2]/moments[:,0] - mu**2, 1e-8))
    np.savez(args.cache/'norm.npz', mu=mu.astype('f4'), sd=sd.astype('f4'))
    pd.DataFrame(audit, columns=['event','role','frames','night_files']).to_csv(args.cache/'inventory.csv', index=False)
    manifest = pd.DataFrame(rows, columns=['event','role','date','file','night_file'])
    assert not manifest.duplicated(['event','date']).any()
    manifest.to_csv(args.cache/'manifest.tmp', index=False)
    os.replace(args.cache/'manifest.tmp', args.cache/'manifest.csv')


if __name__ == '__main__':
    main()
