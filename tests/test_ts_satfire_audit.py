import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'audit_ts_satfire.py'
spec = importlib.util.spec_from_file_location('audit', SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def test_dates(self):
        for name in ['image_2021-08-02.tif', '20210802.tif', 'VIIRS_2021_08_02.tif']:
            self.assertEqual(str(audit.file_date(Path(name))), '2021-08-02')
        for name in ['fire_1234.tif', '20210230.tif', '20210802_20210803.tif']:
            self.assertIsNone(audit.file_date(Path(name)))

    def test_inventory_gaps_duplicates_and_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp)/'data', Path(tmp)/'out'
            for modality in ['VIIRS_Day', 'FirePred']:
                folder = root/'fire_A'/modality
                folder.mkdir(parents=True)
                for day in [1, 2, 3, 5]:
                    (folder/f'2021-08-{day:02d}.tif').touch()
            (root/'fire_A'/'VIIRS_Day'/'duplicate_20210802.tif').touch()
            (root/'fire_A'/'VIIRS_Day'/'unknown.tif').touch()
            subprocess.run([sys.executable, str(SCRIPT), '--root', str(root), '--out',
                            str(out), '--inventory-only'], check=True, capture_output=True)
            summary = json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['file_count'], 10)
            self.assertEqual(summary['mean_observation_span_days'], 5)
            with (out/'windows.csv').open() as f:
                rows = list(csv.DictReader(f))
            counts = {int(r['input_days']): int(r['file_presence_candidates']) for r in rows}
            self.assertEqual(counts[1], 2)
            self.assertEqual(counts[2], 1)
            self.assertEqual(counts[4], 0)
            with (out/'events.csv').open() as f:
                event = next(csv.DictReader(f))
            self.assertEqual(event['missing_days'], '1')
            self.assertTrue(all(r['duplicate_dates_present'] == 'True' for r in rows))
            issues = json.loads((out/'issues.json').read_text())
            self.assertEqual(len(issues), 2)
            rerun = subprocess.run([sys.executable, str(SCRIPT), '--root', str(root),
                                   '--out', str(out), '--inventory-only'], capture_output=True)
            self.assertNotEqual(rerun.returncode, 0)

    def test_metadata_nan(self):
        try:
            import rasterio
            import numpy as np
        except ImportError:
            self.skipTest('rasterio is not installed')
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp)/'data', Path(tmp)/'out'
            folder = root/'fire_A'/'VIIRS_Day'
            folder.mkdir(parents=True)
            with rasterio.open(folder/'20210801.tif', 'w', driver='GTiff', width=8,
                               height=8, count=2, dtype='float32', crs='EPSG:32610',
                               transform=rasterio.transform.from_origin(100, 100, 375, 375),
                               nodata=float('nan')) as ds:
                ds.write(np.ones((2, 8, 8), dtype='float32'))
                ds.set_band_description(1, 'unverified_band')
            subprocess.run([sys.executable, str(SCRIPT), '--root', str(root), '--out',
                            str(out)], check=True, capture_output=True)
            meta = json.loads((out/'metadata.json').read_text())[0]
            self.assertEqual(meta['count'], 2)
            self.assertEqual(meta['resolution'], [375, 375])
            self.assertEqual(meta['nodata'], ['nan', 'nan'])
            self.assertEqual(meta['descriptions'][0], 'unverified_band')


if __name__ == '__main__':
    unittest.main()
