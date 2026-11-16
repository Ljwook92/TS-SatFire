"""Helpers for the two-day NDVI diagnostic notebook; no model training or source edits."""
from collections import defaultdict
import datetime as dt
from pathlib import Path

import numpy as np
import rasterio


def header(path):
    path = Path(path)
    try:
        with rasterio.open(path) as ds:
            return dict(path=str(path), shape=[ds.height, ds.width], count=ds.count,
                        crs=str(ds.crs) if ds.crs else None,
                        transform=list(ds.transform), resolution=list(ds.res),
                        descriptions=list(ds.descriptions), scales=list(ds.scales),
                        offsets=list(ds.offsets), nodata=str(ds.nodata), error=None)
    except Exception as exc:
        return dict(path=str(path), error=str(exc))


def grid_reasons(reference, candidate):
    if reference.get('error') or candidate.get('error'):
        return ['header_read_error']
    reasons = []
    if reference['shape'] != candidate['shape']:
        reasons.append('shape_mismatch')
    if not reference['crs']:
        reasons.append('reference_crs_missing')
    if not candidate['crs']:
        reasons.append('candidate_crs_missing')
    if reference['crs'] and candidate['crs'] and reference['crs'] != candidate['crs']:
        reasons.append('crs_mismatch')
    if not np.allclose(reference['transform'], candidate['transform'], atol=1e-5, rtol=0):
        reasons.append('transform_mismatch')
    return reasons


def discover_fp(root, records):
    from audit_ts_satfire import file_date, identity
    root = Path(root)
    rows = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in {'.tif', '.tiff'}:
            continue
        event, modality = identity(path, root)
        event_id = Path(event).name
        if event_id not in records or modality == 'UNKNOWN':
            continue
        date = file_date(path)
        rows.append(dict(event_id=event_id, event_path=event,
                         split=records[event_id]['split'], modality=modality,
                         date=date.isoformat() if date else None, path=str(path)))
    return rows


def header_inventory(files):
    rows = []
    for i, item in enumerate(files):
        h = header(item['path'])
        h.update({k: item[k] for k in ['event_id', 'event_path', 'modality', 'date', 'split']})
        rows.append(h)
        if (i+1) % 500 == 0:
            print(f'Headers: {i+1}/{len(files)}', flush=True)
    return rows


def alignment_report(headers):
    groups = defaultdict(list)
    for row in headers:
        if row['date']:
            groups[(row['event_id'], row['date'])].append(row)
    result = []
    for (event, date), items in sorted(groups.items()):
        day = [r for r in items if r['modality'] == 'VIIRS_Day']
        if len(day) != 1:
            result.append(dict(event_id=event, date=date, modality='VIIRS_Day',
                               reasons='missing_or_duplicate_day', reference=None, candidate=None))
            continue
        for item in items:
            reasons = grid_reasons(day[0], item)
            result.append(dict(event_id=event, date=date, modality=item['modality'],
                               reasons=';'.join(reasons) or 'aligned',
                               reference=day[0]['path'], candidate=item['path']))
    return result


def date_index(files, event_id):
    selected = [r for r in files if r['event_id'] == event_id]
    if len({r['event_path'] for r in selected}) > 1:
        raise ValueError('Same event ID occurs in multiple directories; resolve before comparison.')
    index = defaultdict(lambda: defaultdict(list))
    for row in selected:
        if row['date']:
            index[row['modality']][dt.date.fromisoformat(row['date'])].append(Path(row['path']))
    return index


def unique_path(index, modality, date):
    paths = index[modality].get(date, [])
    return paths[0] if len(paths) == 1 else None


def band_stats(path, band):
    with rasterio.open(path) as ds:
        if not 1 <= band <= ds.count:
            raise ValueError(f'Band {band} absent in {path}')
        raw = ds.read(band)
        good = np.isfinite(raw)
        values = raw[good]
        return dict(path=str(path), band=band, finite_pixels=int(good.sum()),
                    nan_pixels=int(np.isnan(raw).sum()),
                    zero_pixels=int((good & (raw == 0)).sum()),
                    positive_pixels=int((good & (raw > 0)).sum()),
                    negative_pixels=int((good & (raw < 0)).sum()),
                    min=float(values.min()) if len(values) else None,
                    max=float(values.max()) if len(values) else None,
                    metadata_scale=ds.scales[band-1], metadata_offset=ds.offsets[band-1],
                    nodata=str(ds.nodata), description=ds.descriptions[band-1])


def read_ndvi(path, source='provided', band=1, scale=None, offset=None,
              reflectance_confirmed=False, raw_valid_range=None):
    """Returns array or an explicit failure. Never silently normalizes raw radiances."""
    if path is None:
        return dict(array=None, status='missing_or_duplicate_ndvi_file')
    try:
        with rasterio.open(path) as ds:
            excluded_raw_pixels = 0
            if source == 'provided':
                raw = ds.read(band, masked=True).astype('float64')
                applied_scale = ds.scales[band-1] if scale is None else scale
                applied_offset = ds.offsets[band-1] if offset is None else offset
                values = raw.compressed()
                values = values[np.isfinite(values)]
                if applied_scale == 0.0001 and values.size and np.max(np.abs(values)) <= 1 and np.any(values != 0):
                    return dict(array=None, status='ndvi_may_already_be_scaled',
                                note='Values already fit [-1,1]; confirm units and use scale=1 rather than scaling twice.')
                if raw_valid_range is not None:
                    lo, hi = raw_valid_range
                    excluded = np.isfinite(raw.data) & ((raw.data < lo) | (raw.data > hi))
                    excluded_raw_pixels = int((excluded & ~np.ma.getmaskarray(raw)).sum())
                    raw = np.ma.masked_where(excluded, raw)
                arr = (raw * applied_scale + applied_offset).filled(np.nan)
            elif source == 'computed':
                if not reflectance_confirmed:
                    return dict(array=None, status='reflectance_not_confirmed')
                red = ds.read(1, masked=True).astype('float64')*ds.scales[0]+ds.offsets[0]
                nir = ds.read(2, masked=True).astype('float64')*ds.scales[1]+ds.offsets[1]
                denom = red+nir
                with np.errstate(divide='ignore', invalid='ignore'):
                    arr = np.ma.where(abs(denom) > 1e-12, (nir-red)/denom, np.nan).filled(np.nan)
                applied_scale, applied_offset = None, None
            else:
                raise ValueError(f'Unknown NDVI source: {source}')
            finite = np.isfinite(arr)
            outside = finite & ((arr < -1) | (arr > 1))
            fraction = float(outside.sum()/finite.sum()) if finite.any() else 0.0
            if not finite.any():
                return dict(array=None, status='no_valid_ndvi_pixels')
            if fraction > 0.05:
                return dict(array=None, status='ndvi_scale_or_band_unverified',
                            outside_range_fraction=fraction, applied_scale=applied_scale)
            arr[~finite | outside] = np.nan
            return dict(array=arr, status='available', applied_scale=applied_scale,
                        applied_offset=applied_offset, outside_range_fraction=fraction,
                        excluded_raw_pixels=excluded_raw_pixels,
                        note='Plausible range is a check, not proof of calibration/cloud quality.')
    except Exception as exc:
        return dict(array=None, status='ndvi_read_error', error=str(exc))


def progression(before_path, after_path, band=8, rule=None):
    from fp_audit import load_ba
    if before_path is None or after_path is None:
        return dict(status='unknown_missing_or_duplicate_label_file', new_mask=None)
    reasons = grid_reasons(header(before_path), header(after_path))
    if reasons:
        return dict(status='unknown_grid_alignment', reasons=reasons, new_mask=None)
    if rule is None:
        return dict(status='unknown_label_encoding', new_mask=None)
    try:
        before = load_ba(before_path, band, rule, rasterio)
        after = load_ba(after_path, band, rule, rasterio)
        if not before[2] or not after[2]:
            return dict(status='unknown_missing_annotation', new_mask=None)
        valid = before[1] & after[1]
        if not valid.any():
            return dict(status='unknown_no_common_label_pixels', new_mask=None)
        new = valid & after[0] & ~before[0]
        lost = valid & before[0] & ~after[0]
        coverage = float(valid.mean())
        n = int(new.sum())
        status = 'observed_increment' if n else 'observed_no_increment'
        if coverage < 1:
            status = 'partial_coverage_increment' if n else 'partial_coverage_no_increment'
        return dict(status=status, new_mask=new, valid_mask=valid,
                    before_mask=before[0], after_mask=after[0],
                    new_pixels=n, shrinking_pixels=int(lost.sum()),
                    compared_pixels=int(valid.sum()), label_coverage=coverage)
    except Exception as exc:
        return dict(status='unknown_label_read_or_encoding_error', error=str(exc), new_mask=None)


def progression_series(index, band=8, rule=None):
    rows = []
    for current in sorted(index['VIIRS_Day']):
        previous, target = current-dt.timedelta(days=1), current+dt.timedelta(days=1)
        current_path = unique_path(index, 'VIIRS_Day', current)
        target_path = unique_path(index, 'VIIRS_Day', target)
        result = progression(current_path, target_path, band, rule)
        rows.append(dict(current_date=str(current), target_date=str(target),
                         two_day_history=bool(unique_path(index, 'VIIRS_Day', previous) and current_path),
                         two_day_aux_complete=all(unique_path(index, 'FirePred', d) is not None for d in [previous, current]),
                         status=result['status'], new_pixels=result.get('new_pixels'),
                         compared_pixels=result.get('compared_pixels'),
                         shrinking_pixels=result.get('shrinking_pixels'),
                         label_coverage=result.get('label_coverage')))
    return rows


def create_demo_dataset(base):
    """Explicitly synthetic, tiny fixture for notebook validation only."""
    base = Path(base)
    root, roi = base/'data', base/'roi'
    if root.exists() or roi.exists():
        raise ValueError('Use a fresh demo directory; existing data will not be overwritten.')
    roi.mkdir(parents=True)
    for year in range(2017, 2022):
        text = 'Id,start_date,end_date,label_sel\n'
        if year == 2017:
            text += 'DEMO_FIRE,2021-08-01,2021-08-07,0\n'
        (roi/f'us_fire_{year}_out_new.csv').write_text(text)
    script = base/'dataset_gen_pred.py'
    script.write_text("val_ids = ['DEMO_FIRE']\n")
    yy, xx = np.mgrid[:32, :32]
    for day in range(1, 8):
        radius = [0, 3, 4, 6, 6, 6, 7, 8][day]
        burned = (xx-16)**2+(yy-16)**2 < radius**2
        ndvi = 0.65+0.08*np.sin(xx/6)-0.02*day-0.45*burned
        for modality, count in [('VIIRS_Day', 8), ('FirePred', 19), ('VIIRS_Night', 2)]:
            folder = root/'DEMO_FIRE'/modality
            folder.mkdir(parents=True, exist_ok=True)
            arr = np.ones((count, 32, 32), dtype='float32')
            if modality == 'VIIRS_Day':
                arr[0] = 0.2
                arr[1] = 0.2*(1+ndvi)/(1-ndvi)
                arr[6] = np.where(burned, 8, 0)
                arr[7] = np.where(burned, 20210800+day, np.nan)
                if day == 6:
                    arr[7] = np.nan
            elif modality == 'FirePred':
                arr[0] = ndvi
                if day == 4:
                    arr[0, :2, :] = np.nan
            with rasterio.open(folder/f'202108{day:02d}.tif', 'w', driver='GTiff',
                               width=32, height=32, count=count, dtype='float32',
                               crs='EPSG:32610', transform=rasterio.transform.from_origin(100, 200, 375, 375),
                               nodata=float('nan')) as ds:
                ds.write(arr)
    return root, roi, script
