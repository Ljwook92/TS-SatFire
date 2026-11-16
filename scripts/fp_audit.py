"""FP-only split and temporal audit. Never execute legacy code or infer label encoding."""
import ast
from collections import Counter, defaultdict
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics


def read_fp_split(roi_dir, split_script):
    source = split_script.read_text(encoding='utf-8')
    validation = None
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == 'val_ids' for t in node.targets):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
                validation = set(value)
                break
    if not validation:
        raise ValueError('Cannot extract literal val_ids; supply the verified FP split script.')
    records, sources = {}, []
    for year in range(2017, 2022):
        path = roi_dir / f'us_fire_{year}_out_new.csv'
        data = path.read_text(encoding='utf-8-sig')
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for row in csv.DictReader(data.splitlines()):
            event = row['Id'].strip()
            if not event:
                raise ValueError(f'Empty event ID in {path}')
            if event in records:
                raise ValueError(f'Duplicate split event ID: {event}')
            start = dt.date.fromisoformat(row['start_date'])
            end = dt.date.fromisoformat(row['end_date'])
            if end < start:
                raise ValueError(f'Invalid ROI date range: {event}')
            split = 'test' if year == 2021 else ('val' if event in validation else 'train')
            if year == 2021 and event in validation:
                raise ValueError(f'Test/validation ID overlap: {event}')
            records[event] = dict(
                event_id=event, split=split, roi_start=start.isoformat(), roi_end=end.isoformat(),
                roi_span_days=(end-start).days+1,
                legacy_label_sel=row.get('label_sel', '1') if year == 2021 else '1')
    return records, dict(
        split_script=str(split_script), split_script_sha256=hashlib.sha256(source.encode()).hexdigest(),
        roi_sources=sources, validation_candidate_ids=sorted(validation),
        validation_ids_absent_from_roi=sorted(validation-set(records)),
        declared_counts=dict(Counter(r['split'] for r in records.values())),
        note='Assignments follow literal val_ids and CSV intersections in the selected legacy script; not paper prose counts.')


def load_ba(path, band, rule, rasterio):
    import numpy as np
    with rasterio.open(path) as ds:
        if band < 1 or band > ds.count:
            raise ValueError(f'BA band {band} absent in {path}; count={ds.count}')
        values = ds.read(band)
        finite = np.isfinite(values)
        # Finite-presence encodings use NaN/nonfinite as unburned. Nodata=0 is ambiguous.
        if rule == 'finite':
            if ds.nodata is not None and np.isfinite(ds.nodata):
                raise ValueError('finite rule conflicts with finite NoData; inspect encoding first')
            burn, valid = finite, np.ones(values.shape, dtype=bool)
        else:
            valid = finite & (ds.read_masks(band) > 0)
            burn = valid & (values > 0)
        stats = dict(path=str(path), band=band, finite_pixels=int(finite.sum()),
                     negative_pixels=int((finite & (values < 0)).sum()),
                     zero_pixels=int((finite & (values == 0)).sum()),
                     positive_pixels=int((finite & (values > 0)).sum()),
                     min_finite=float(values[finite].min()) if finite.any() else None,
                     max_finite=float(values[finite].max()) if finite.any() else None)
        # An all-nonfinite band may be missing annotation; do not call it a negative day.
        return burn, valid, bool(finite.any() and valid.any()), stats


def run_fp_audit(root, out, args, rows, rasterio, write_csv, metadata, json_safe):
    records, provenance = read_fp_split(args.roi_dir, args.split_script)
    issues, excluded, selected = [], [], []
    locations = defaultdict(set)
    for row in rows:
        event_id = Path(row['event']).name
        if event_id in records and row['modality'] != 'UNKNOWN':
            locations[event_id].add(row['event'])
    ambiguous = {e for e, paths in locations.items() if len(paths) > 1}
    for event_id in sorted(ambiguous):
        issues.append(dict(event_id=event_id, issue='ambiguous_event_directories',
                           paths=sorted(locations[event_id])))
    for row in rows:
        event_id = Path(row['event']).name
        if event_id not in records or event_id in ambiguous or row['modality'] == 'UNKNOWN':
            excluded.append(dict(**row, reason='not_fp_split_or_ambiguous_or_unknown_modality'))
            continue
        selected.append(dict(**row, event_id=event_id, split=records[event_id]['split']))
        if not row['date']:
            issues.append(dict(event_id=event_id, path=row['path'], issue='unresolved_date'))
    index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in selected:
        if row['date']:
            index[row['event_id']][row['modality']][dt.date.fromisoformat(row['date'])].append(root/row['path'])
    headers, grid_differences = {}, {}
    def header(path):
        if path not in headers:
            try:
                with rasterio.open(path) as ds:
                    headers[path] = (ds.crs, ds.transform, ds.width, ds.height, ds.count)
            except Exception as exc:
                headers[path] = None
                issues.append(dict(path=str(path), issue='header_read_error', error=str(exc)))
        return headers[path]
    def same_grid(a, b):
        return a is not None and b is not None and a[0] is not None and a[0] == b[0] and a[2:4] == b[2:4] and a[1].almost_equals(b[1])
    def record_grid_difference(event_id, ref_path, path):
        a, b = header(ref_path), header(path)
        if a is None or b is None or same_grid(a, b):
            return
        reasons = []
        if a[2:4] != b[2:4]:
            reasons.append('shape_mismatch')
        if a[0] is None:
            reasons.append('reference_crs_missing')
        if b[0] is None:
            reasons.append('candidate_crs_missing')
        if a[0] is not None and b[0] is not None and a[0] != b[0]:
            reasons.append('crs_mismatch')
        if not a[1].almost_equals(b[1]):
            reasons.append('transform_mismatch')
        grid_differences[(str(ref_path), str(path))] = dict(
            event_id=event_id, reference_path=str(ref_path), candidate_path=str(path),
            reasons=';'.join(reasons), reference_width=a[2], reference_height=a[3],
            candidate_width=b[2], candidate_height=b[3], reference_crs=str(a[0]),
            candidate_crs=str(b[0]), reference_transform=str(tuple(a[1])),
            candidate_transform=str(tuple(b[1])))
    event_rows, windows, labels = [], [], []
    split_spans = defaultdict(list)
    for event_id, record in sorted(records.items()):
        modal = index[event_id]
        day, aux, night = modal['VIIRS_Day'], modal['FirePred'], modal['VIIRS_Night']
        dates = sorted(day)
        span = (dates[-1]-dates[0]).days+1 if dates else None
        roi_start = dt.date.fromisoformat(record['roi_start'])
        roi_end = dt.date.fromisoformat(record['roi_end'])
        if span is not None:
            split_spans[record['split']].append(span)
        duplicate_dates = 0
        for modality, mapping in modal.items():
            for date, paths in mapping.items():
                if len(paths) > 1:
                    duplicate_dates += 1
                    issues.append(dict(event_id=event_id, modality=modality, date=str(date),
                                       issue='duplicate_date', paths=[str(p) for p in paths]))
        event_rows.append(dict(**record, observed_day_count=len(dates),
                               first_observed=str(dates[0]) if dates else '',
                               last_observed=str(dates[-1]) if dates else '',
                               observation_span_days=span,
                               internal_missing_days=span-len(dates) if dates else None,
                               observed_outside_roi_days=sum(d < roi_start or d > roi_end for d in dates),
                               missing_days_inside_roi=record['roi_span_days']-sum(roi_start <= d <= roi_end for d in dates),
                               firepred_day_count=len(aux), duplicate_dates=duplicate_dates,
                               status='present' if dates else ('ambiguous' if event_id in ambiguous else 'missing_day_files')))
        label_cache = {}
        def label(path):
            if path not in label_cache:
                try:
                    label_cache[path] = load_ba(path, args.ba_band, args.ba_rule, rasterio)
                    labels.append(dict(event_id=event_id, **label_cache[path][3]))
                except Exception as exc:
                    label_cache[path] = None
                    issues.append(dict(event_id=event_id, path=str(path), issue='label_read_error', error=str(exc)))
            return label_cache[path]
        for length in sorted(set(args.lengths)):
            for target in dates:
                # Ignore the unavoidable initial history shorter than L calendar days.
                if target-dt.timedelta(days=length) < dates[0]:
                    continue
                past = [target-dt.timedelta(days=k) for k in range(length, 0, -1)]
                row = dict(event_id=event_id, split=record['split'], input_days=length,
                           input_start=str(past[0]), input_end=str(past[-1]), target_date=str(target),
                           status='file_presence_candidate', grid_checked=False, label_checked=False,
                           compared_pixels=None, newly_burned_pixels=None, shrinking_ba_pixels=None,
                           missing_night_days=sum(d not in night for d in past))
                if any(d not in day for d in past):
                    row['status'] = 'missing_input_date'
                elif any(d not in aux for d in past):
                    row['status'] = 'missing_firepred_date'
                elif any(len(day[d]) != 1 for d in past+[target]) or any(len(aux[d]) != 1 for d in past):
                    row['status'] = 'duplicate_required_file'
                elif any(len(night.get(d, [])) > 1 for d in past):
                    row['status'] = 'duplicate_night_file'
                elif args.check_grids or args.check_labels:
                    ref = header(day[target][0])
                    paths = [(day[d][0], 6) for d in past] + [(aux[d][0], 19) for d in past]
                    paths += [(night[d][0], 2) for d in past if d in night]
                    row['grid_checked'] = True
                    if ref is None or any(header(p) is None for p, _ in paths):
                        row['status'] = 'unreadable_required_file'
                    elif any(header(p)[4] < n for p, n in paths):
                        row['status'] = 'insufficient_input_bands'
                    elif not same_grid(ref, ref) or any(not same_grid(ref, header(p)) for p, _ in paths):
                        row['status'] = 'grid_mismatch_or_missing_crs'
                        for p, _ in paths:
                            record_grid_difference(event_id, day[target][0], p)
                    else:
                        row['status'] = 'grid_checked_candidate'
                        if args.check_labels:
                            before, after = label(day[past[-1]][0]), label(day[target][0])
                            row['label_checked'] = True
                            if before is None or after is None:
                                row['status'] = 'label_read_error'
                            elif not before[2] or not after[2]:
                                row['status'] = 'unannotated_label_pair'
                            else:
                                valid = before[1] & after[1]
                                row['compared_pixels'] = int(valid.sum())
                                row['newly_burned_pixels'] = int((valid & after[0] & ~before[0]).sum())
                                row['shrinking_ba_pixels'] = int((valid & before[0] & ~after[0]).sum())
                                row['status'] = 'encoding_checked_candidate' if valid.any() else 'no_common_valid_label_pixels'
                windows.append(row)
        print(f"FP {event_id}: {record['split']}, {len(dates)} observed days", flush=True)
    sample_meta = []
    if rasterio:
        by_modal = defaultdict(list)
        for row in selected:
            by_modal[row['modality']].append(root/row['path'])
        for paths in by_modal.values():
            paths.sort()
            n = min(args.metadata_limit, len(paths)) if args.metadata_limit else len(paths)
            for i in sorted({round(k*(len(paths)-1)/max(n-1, 1)) for k in range(n)}):
                try:
                    sample_meta.append(metadata(paths[i], rasterio))
                except Exception as exc:
                    issues.append(dict(path=str(paths[i]), issue='metadata_read_error', error=str(exc)))
    fields = ['event', 'modality', 'date', 'path', 'bytes']
    write_csv(out/'inventory.csv', selected, fields+['event_id', 'split'])
    write_csv(out/'excluded_files.csv', excluded, fields+['reason'])
    event_fields = ['event_id', 'split', 'roi_start', 'roi_end', 'roi_span_days', 'legacy_label_sel',
                    'observed_day_count', 'first_observed', 'last_observed', 'observation_span_days',
                    'internal_missing_days', 'observed_outside_roi_days', 'missing_days_inside_roi',
                    'firepred_day_count', 'duplicate_dates', 'status']
    write_csv(out/'events.csv', event_rows, event_fields)
    window_fields = ['event_id', 'split', 'input_days', 'input_start', 'input_end', 'target_date',
                     'status', 'grid_checked', 'label_checked', 'compared_pixels',
                     'newly_burned_pixels', 'shrinking_ba_pixels', 'missing_night_days']
    write_csv(out/'windows.csv', windows, window_fields)
    grid_fields = ['event_id', 'reference_path', 'candidate_path', 'reasons',
                   'reference_width', 'reference_height', 'candidate_width', 'candidate_height',
                   'reference_crs', 'candidate_crs', 'reference_transform', 'candidate_transform']
    write_csv(out/'grid_differences.csv', grid_differences.values(), grid_fields)
    grid_reason_counts = Counter(reason for item in grid_differences.values()
                                 for reason in item['reasons'].split(';'))
    if grid_differences:
        issues.append(dict(issue='grid_alignment_failures',
                           unique_file_pairs=len(grid_differences),
                           reason_counts=dict(grid_reason_counts), details='grid_differences.csv'))
    totals = []
    good = {'file_presence_candidate', 'grid_checked_candidate', 'encoding_checked_candidate'}
    for split in ['train', 'val', 'test']:
        for length in sorted(set(args.lengths)):
            subset = [w for w in windows if w['split'] == split and w['input_days'] == length]
            counts = Counter(w['status'] for w in subset)
            totals.append(dict(split=split, input_days=length, assessed_targets=len(subset),
                               passing_candidates=sum(counts[s] for s in good),
                               events_with_candidates=len({w['event_id'] for w in subset if w['status'] in good}),
                               positive_label_windows=(sum(w['status'] == 'encoding_checked_candidate' and w['newly_burned_pixels'] > 0 for w in subset) if args.check_labels else None),
                               status_counts=dict(counts)))
    summary = dict(task='pred', root=str(root), all_scanned_files=len(rows),
                   fp_selected_files=len(selected), excluded_files=len(excluded),
                   fp_selected_bytes=sum(r['bytes'] for r in selected),
                   declared_split_counts=provenance['declared_counts'],
                   observed_split_counts=dict(Counter(e['split'] for e in event_rows if e['observed_day_count'])),
                   observation_span_by_split={s: dict(mean=statistics.mean(v), median=statistics.median(v), count=len(v)) for s, v in split_spans.items()},
                   window_counts=totals, issue_count=len(issues), metadata_samples=len(sample_meta),
                   grid_failure_unique_pairs=len(grid_differences), grid_failure_reasons=dict(grid_reason_counts),
                   label_protocol=dict(checked=args.check_labels, source='provided_BA_consecutive_increment',
                                       band=args.ba_band, rule=args.ba_rule,
                                       note='Explicit BA increment audit, not an exact reproduction of legacy cumulative AF/union labels.'),
                   warnings=['Observation span is not actual fire duration.',
                             'Counts are diagnostics, not training-ready samples or validated ground truth.',
                             'ROI dates are reported but do not silently truncate available files.',
                             'Missing night files are reported, not a hard exclusion, matching optional legacy input.',
                             'Pixel checks use the full scene, without benchmark centre cropping.',
                             'No learned features or future weather are read from target-date inputs.'])
    for name, obj in [('summary', summary), ('fp_splits', provenance), ('metadata', sample_meta),
                       ('issues', issues), ('label_band_stats', labels)]:
        (out/f'{name}.json').write_text(json.dumps(json_safe(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
