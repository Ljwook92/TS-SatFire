#!/usr/bin/env python3
"""Read-only TS-SatFire inventory. Window counts check file presence, NOT label validity."""
import argparse
import csv
import datetime as dt
import importlib
import json
import math
from pathlib import Path
import re
import statistics
from collections import defaultdict

DEFAULT_ROOT = '/home/jlc3q/data/SatFire/ts-satfire'
MODALITIES = {'viirs_day': 'VIIRS_Day', 'viirs_night': 'VIIRS_Night', 'firepred': 'FirePred'}


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def file_date(path):
    matches = re.findall(r'(?<!\d)(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)(?!\d)', path.stem)
    dates = set()
    for y, m, d in matches:
        try:
            dates.add(dt.date(int(y), int(m), int(d)))
        except ValueError:
            pass
    return next(iter(dates)) if len(dates) == 1 else None


def identity(path, root):
    parts = path.relative_to(root).parts
    for i in range(len(parts) - 2, -1, -1):
        if parts[i].lower() in MODALITIES:
            return '/'.join(parts[:i]) or '.', MODALITIES[parts[i].lower()]
    return str(path.parent.relative_to(root)), 'UNKNOWN'


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metadata(path, rasterio):
    with rasterio.open(path) as ds:
        return dict(path=str(path), width=ds.width, height=ds.height,
                    crs=str(ds.crs), resolution=ds.res, transform=list(ds.transform),
                    bounds=list(ds.bounds), count=ds.count, dtypes=ds.dtypes,
                    nodata=ds.nodatavals, descriptions=ds.descriptions,
                    scales=ds.scales, offsets=ds.offsets,
                    tags=ds.tags(), band_tags=[ds.tags(i) for i in ds.indexes])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lengths', type=int, nargs='+', default=[1, 2, 4, 6, 10])
    parser.add_argument('--metadata-limit', type=int, default=30,
                        help='Evenly spaced metadata samples per modality; 0 = all files')
    parser.add_argument('--inventory-only', action='store_true', help='No rasterio required')
    parser.add_argument('--task', choices=['all', 'pred'], default='all')
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument('--roi-dir', type=Path, default=repo / 'legacy' / 'roi')
    parser.add_argument('--split-script', type=Path,
                        default=repo / 'legacy' / 'scripts' / 'dataset_gen_pred.py')
    parser.add_argument('--check-grids', action='store_true',
                        help='FP: read candidate input headers and compare grids')
    parser.add_argument('--check-labels', action='store_true',
                        help='FP: audit pixels under an explicitly selected BA encoding')
    parser.add_argument('--ba-band', type=int, help='1-based BA band, verified against actual files')
    parser.add_argument('--ba-rule', choices=['finite', 'positive'],
                        help='finite: finite presence codes; positive: positive binary/code values')
    args = parser.parse_args()
    if args.check_labels and (args.task != 'pred' or args.ba_band is None or args.ba_band < 1 or not args.ba_rule):
        parser.error('--check-labels requires --task pred, --ba-band and --ba-rule.')
    if args.check_grids and args.task != 'pred':
        parser.error('--check-grids requires --task pred.')
    if args.inventory_only and (args.check_labels or args.check_grids):
        parser.error('Pixel/grid checks cannot be combined with --inventory-only.')
    if any(n < 1 for n in args.lengths) or args.metadata_limit < 0:
        parser.error('Lengths must be positive; metadata-limit must be nonnegative.')
    root, out = args.root.resolve(), args.out.resolve()
    if not root.is_dir():
        parser.error(f'Data directory not found: {root}')
    if out == root or root in out.parents:
        parser.error('Output must be outside the source data directory.')
    rasterio = None
    if not args.inventory_only:
        try:
            rasterio = importlib.import_module('rasterio')
        except ImportError:
            parser.error('Install rasterio, or use --inventory-only.')
    if out.exists() and any(out.iterdir()):
        parser.error('Use a new or empty output directory to preserve earlier reports.')
    out.mkdir(parents=True, exist_ok=True)
    rows, issues = [], []
    groups = defaultdict(lambda: defaultdict(list))
    by_modality = defaultdict(list)
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in {'.tif', '.tiff'}:
            continue
        event, modality = identity(path, root)
        date = file_date(path)
        row = dict(event=event, modality=modality, date=date.isoformat() if date else '',
                   path=str(path.relative_to(root)), bytes=path.stat().st_size)
        rows.append(row)
        by_modality[modality].append(path)
        if not date or modality == 'UNKNOWN':
            issues.append(dict(path=row['path'], issue='unresolved_date_or_modality'))
        if date:
            groups[event][modality].append(date)
    if not rows:
        parser.error('No GeoTIFF files found. Check the data path.')
    if args.task == 'pred':
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from fp_audit import run_fp_audit
        try:
            run_fp_audit(root, out, args, rows, rasterio, write_csv, metadata, json_safe)
        except (ValueError, OSError, SyntaxError) as exc:
            parser.error(str(exc))
        return
    write_csv(out / 'inventory.csv', rows, ['event', 'modality', 'date', 'path', 'bytes'])
    event_rows, window_rows, spans = [], [], []
    for event, modalities in sorted(groups.items()):
        day = set(modalities.get('VIIRS_Day', []))
        aux = set(modalities.get('FirePred', []))
        duplicates = {m: len(ds) - len(set(ds)) for m, ds in modalities.items()}
        if any(duplicates.values()):
            issues.append(dict(event=event, issue='duplicate_dates', details=duplicates))
        if not day:
            continue
        span = (max(day) - min(day)).days + 1
        spans.append(span)
        event_rows.append(dict(event=event, first=min(day), last=max(day),
                               calendar_span_days=span, observed_days=len(day),
                               missing_days=span-len(day), auxiliary_days=len(aux)))
        for length in args.lengths:
            candidates = 0
            for target in day:
                required = [target-dt.timedelta(days=k) for k in range(1, length+1)]
                if all(d in day and d in aux for d in required):
                    candidates += 1
            window_rows.append(dict(event=event, input_days=length,
                                    file_presence_candidates=candidates,
                                    duplicate_dates_present=any(duplicates.values())))
    write_csv(out / 'events.csv', event_rows,
              ['event', 'first', 'last', 'calendar_span_days', 'observed_days', 'missing_days', 'auxiliary_days'])
    write_csv(out / 'windows.csv', window_rows,
              ['event', 'input_days', 'file_presence_candidates', 'duplicate_dates_present'])
    meta = []
    if rasterio:
        for paths in by_modality.values():
            n = min(args.metadata_limit, len(paths)) if args.metadata_limit else len(paths)
            indices = {round(i*(len(paths)-1)/max(n-1, 1)) for i in range(n)}
            for i in sorted(indices):
                try:
                    meta.append(metadata(paths[i], rasterio))
                except Exception as exc:
                    issues.append(dict(path=str(paths[i]), issue='metadata_read_error', error=str(exc)))
    summary = dict(root=str(root), file_count=len(rows), bytes=sum(r['bytes'] for r in rows),
                   event_count_with_dated_day_files=len(spans),
                   mean_observation_span_days=statistics.mean(spans) if spans else None,
                   median_observation_span_days=statistics.median(spans) if spans else None,
                   metadata_samples=len(meta), issue_count=len(issues),
                   warnings=['Observation span is not verified fire duration.',
                             'Window counts are file-presence candidates only: no label validity checks.',
                             'Band meanings are not assumed; inspect metadata and loader before deriving labels.',
                             'Dates are parsed from filenames only; unresolved names require a verified parser.'])
    for name, obj in [('summary', summary), ('metadata', meta), ('issues', issues)]:
        (out / f'{name}.json').write_text(json.dumps(json_safe(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
