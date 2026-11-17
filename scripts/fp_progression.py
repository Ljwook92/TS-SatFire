"""Recorded next-day fire expansion, with explicit correction of legacy NaN truthiness.

Reference: PredDatasetProcessor.pred_dataset_generator_seqtoseq in
legacy/satimg_dataset_processor/satimg_dataset_processor.py at 5451f600.
The original uses logical_or on raw floats (NaN is True). This module does
NOT reproduce that bug. NaN in BA is treated as no recorded detection only
under the caller's explicit nan_policy='no_detection' convention. These
masks describe recorded fire, not independently verified burned area.
"""
import datetime as dt
import numpy as np
import rasterio
from satfire_explore import grid_reasons, header, unique_path


def recorded_detection(raw):
    """Finite nonzero -> recorded detection; neither NaN nor inf -> fire."""
    raw = np.asarray(raw)
    return np.isfinite(raw) & (raw != 0)


def accumulate_step(previous, af, ba=None, label_sel=0):
    if label_sel not in (0, 1):
        raise ValueError('label_sel must be 0 (AF+BA) or 1 (AF)')
    detection = recorded_detection(af)
    if label_sel == 0:
        if ba is None or ba.shape != af.shape:
            raise ValueError('AF+BA requires matching AF and BA arrays')
        detection |= recorded_detection(ba)
    if previous.shape != detection.shape:
        raise ValueError('Cumulative grid changed')
    cumulative = previous | detection
    return cumulative, cumulative & ~previous


def build_progression(index, label_sel, start_date, nan_policy):
    """Return bounded daily rows and cumulative native-grid maps.

    Missing dates/bands, all-missing AF, infinity, or grid changes invalidate
    that date AND subsequent cumulative history. No silent bridging or
    re-projection. BA all-NaN is reported, and allowed only as no recorded
    BA detections under the explicit policy (not proof of no physical fire).
    """
    if nan_policy != 'no_detection':
        raise ValueError('Explicit nan_policy="no_detection" required; see module docstring')
    label_sel = int(label_sel)
    if label_sel not in (0, 1):
        raise ValueError('Unsupported label selection')
    end = max(index['VIIRS_Day'])
    dates = [start_date + dt.timedelta(days=i) for i in range((end-start_date).days+1)]
    rows, maps = [], {}
    base_header = None
    cumulative = literal = None
    broken = False
    for date in dates:
        row = dict(date=str(date), label_sel=label_sel, nan_policy=nan_policy,
                   status='available', cumulative_pixels=None, added_pixels=None,
                   literal_legacy_pixels=None, af_nan_pixels=None, ba_nan_pixels=None,
                   ba_all_nan=None)
        path = unique_path(index, 'VIIRS_Day', date)
        try:
            if path is None:
                raise ValueError('missing_or_duplicate_day_file')
            h = header(path)
            if h.get('error'):
                raise ValueError('header_read_error')
            reasons = grid_reasons(base_header or h, h)
            if reasons:
                raise ValueError('grid_'+','.join(reasons))
            with rasterio.open(path) as ds:
                if ds.count < (8 if label_sel == 0 else 7):
                    raise ValueError('missing_label_band')
                af_raw = ds.read(7).astype(float)
                af = ds.read(7, masked=True).astype(float).filled(np.nan)
                ba_raw = ds.read(8).astype(float) if label_sel == 0 else None
                ba = ds.read(8, masked=True).astype(float).filled(np.nan) if label_sel == 0 else None
            row['af_nan_pixels'] = int(np.isnan(af).sum())
            row['ba_nan_pixels'] = int(np.isnan(ba).sum()) if ba is not None else None
            row['ba_all_nan'] = bool(np.isnan(ba).all()) if ba is not None else None
            if not np.isfinite(af).any():
                raise ValueError('all_missing_af')
            if np.isinf(af).any() or (ba is not None and np.isinf(ba).any()):
                raise ValueError('infinite_label_values')
            if broken:
                raise ValueError('incomplete_prior_history')
            if base_header is None:
                base_header = h
                cumulative = np.zeros(af.shape, dtype=bool)
                literal = cumulative.copy()
            cumulative, added = accumulate_step(cumulative, af, ba, label_sel)
            # Diagnostic ONLY: reproduce raw np.logical_or truthiness of the legacy loader.
            literal |= af_raw.astype(bool)
            if ba_raw is not None:
                literal |= ba_raw.astype(bool)
            row.update(cumulative_pixels=int(cumulative.sum()), added_pixels=int(added.sum()),
                       literal_legacy_pixels=int(literal.sum()))
            maps[date] = cumulative.copy()
        except (ValueError, OSError, rasterio.errors.RasterioError) as exc:
            broken = True
            row['status'] = 'unknown_'+str(exc)
        rows.append(row)
    return rows, maps


def compare_days(maps, current):
    """Compare cumulative recorded regions; zero increment is a valid outcome."""
    target = current + dt.timedelta(days=1)
    if current not in maps or target not in maps:
        return dict(status='unknown_incomplete_history', new_mask=None,
                    new_pixels=None, current_mask=None, next_mask=None)
    before, after = maps[current], maps[target]
    new = after & ~before
    return dict(status='recorded_increment' if new.any() else 'no_new_recorded_fire_pixels',
                new_mask=new, new_pixels=int(new.sum()), current_mask=before, next_mask=after)
