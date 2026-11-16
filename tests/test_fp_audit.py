import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'audit_ts_satfire.py'
spec = importlib.util.spec_from_file_location('fp_audit_test_module', SCRIPT.with_name('fp_audit.py'))
fp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp)


class FPAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base/'data'
        self.roi = self.base/'roi'
        self.root.mkdir()
        self.roi.mkdir()
        self.split = self.base/'dataset_gen_pred.py'
        # Reading this file must not execute arbitrary legacy code.
        self.split.write_text("val_ids = ['val2017', 'missing_val']\nraise RuntimeError('do not execute')\n")
        for year in range(2017, 2022):
            with (self.roi/f'us_fire_{year}_out_new.csv').open('w') as f:
                f.write('Id,start_date,end_date,label_sel\n')
                event = f'train{year}' if year < 2021 else 'US_2021_TEST'
                f.write(f'{event},2021-08-01,2021-08-05,0\n')
                if year == 2017:
                    f.write('val2017,2021-08-01,2021-08-05,1\n')

    def run_audit(self, *args):
        out = self.base/'out'
        command = [sys.executable, str(SCRIPT), '--task', 'pred', '--root', str(self.root),
                   '--roi-dir', str(self.roi), '--split-script', str(self.split),
                   '--out', str(out), '--lengths', '1', '2', *args]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        return out

    def touch_days(self, event, days):
        for modality in ['VIIRS_Day', 'FirePred']:
            folder = self.root/event/modality
            folder.mkdir(parents=True)
            for day in days:
                (folder/f'2021-08-{day:02d}.tif').touch()

    def test_split_csv_intersection_without_execution(self):
        records, provenance = fp.read_fp_split(self.roi, self.split)
        self.assertEqual(provenance['declared_counts'], {'train': 4, 'val': 1, 'test': 1})
        self.assertEqual(provenance['validation_ids_absent_from_roi'], ['missing_val'])
        self.assertEqual(records['train2017']['legacy_label_sel'], '1')
        self.assertEqual(records['US_2021_TEST']['legacy_label_sel'], '0')

    def test_fp_filter_calendar_gaps_and_missing_events(self):
        self.touch_days('train2017', [1, 2, 4])
        self.touch_days('unrelated_AF_test', [1, 2, 3, 4])
        out = self.run_audit('--inventory-only')
        summary = json.loads((out/'summary.json').read_text())
        self.assertEqual(summary['fp_selected_files'], 6)
        self.assertEqual(summary['excluded_files'], 8)
        self.assertEqual(summary['observed_split_counts'], {'train': 1})
        totals = {x['input_days']: x for x in summary['window_counts'] if x['split'] == 'train'}
        self.assertEqual(totals[1]['passing_candidates'], 1)
        self.assertEqual(totals[2]['passing_candidates'], 0)
        self.assertIn('missing_input_date', totals[1]['status_counts'])
        with (out/'events.csv').open() as f:
            events = list(csv.DictReader(f))
        self.assertEqual(len(events), 6)
        self.assertEqual(sum(e['status'] == 'missing_day_files' for e in events), 5)

    def test_duplicates_are_excluded_from_fp_candidates(self):
        self.touch_days('train2017', [1, 2])
        (self.root/'train2017/VIIRS_Day/copy_20210801.tif').touch()
        out = self.run_audit('--inventory-only')
        summary = json.loads((out/'summary.json').read_text())
        item = next(x for x in summary['window_counts'] if x['split'] == 'train' and x['input_days'] == 1)
        self.assertEqual(item['passing_candidates'], 0)
        self.assertEqual(item['status_counts']['duplicate_required_file'], 1)

    def require_raster(self):
        try:
            import rasterio
            import numpy
        except ImportError:
            self.skipTest('rasterio and numpy required')
        return rasterio, numpy

    def raster_days(self, offset_aux=False):
        rio, np = self.require_raster()
        for modality, count in [('VIIRS_Day', 8), ('FirePred', 19)]:
            folder = self.root/'train2017'/modality
            folder.mkdir(parents=True)
            for day in [1, 2, 3]:
                arr = np.ones((count, 4, 4), dtype='float32')
                if modality == 'VIIRS_Day':
                    arr[7] = np.nan
                    if day != 3:
                        arr[7, 0, :day] = 100
                transform = rio.transform.from_origin(100+(375 if offset_aux and modality == 'FirePred' else 0), 200, 375, 375)
                with rio.open(folder/f'202108{day:02d}.tif', 'w', driver='GTiff',
                              width=4, height=4, count=count, dtype='float32',
                              nodata=float('nan'), crs='EPSG:32610', transform=transform) as ds:
                    ds.write(arr)

    def test_explicit_ba_increment_and_missing_annotations(self):
        self.raster_days()
        out = self.run_audit('--check-labels', '--ba-band', '8', '--ba-rule', 'finite')
        with (out/'windows.csv').open() as f:
            windows = list(csv.DictReader(f))
        first = next(w for w in windows if w['target_date'] == '2021-08-02')
        self.assertEqual(first['newly_burned_pixels'], '1')
        self.assertEqual(first['status'], 'encoding_checked_candidate')
        self.assertTrue(all(w['status'] == 'unannotated_label_pair' for w in windows if w['target_date'] == '2021-08-03'))

    def test_grid_mismatch_blocks_candidates(self):
        self.raster_days(offset_aux=True)
        out = self.run_audit('--check-grids')
        summary = json.loads((out/'summary.json').read_text())
        item = next(x for x in summary['window_counts'] if x['split'] == 'train' and x['input_days'] == 1)
        self.assertEqual(item['passing_candidates'], 0)
        self.assertEqual(item['status_counts']['grid_mismatch_or_missing_crs'], 2)
        self.assertIn('transform_mismatch', summary['grid_failure_reasons'])
        self.assertNotIn('shape_mismatch', summary['grid_failure_reasons'])
        with (out/'grid_differences.csv').open() as f:
            differences = list(csv.DictReader(f))
        self.assertTrue(differences)
        self.assertTrue(all(d['reference_width'] == d['candidate_width'] for d in differences))

    def test_label_checks_require_explicit_encoding(self):
        result = subprocess.run([sys.executable, str(SCRIPT), '--task', 'pred',
                                 '--out', str(self.base/'invalid'), '--check-labels'], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.base/'invalid').exists())


if __name__ == '__main__':
    unittest.main()
