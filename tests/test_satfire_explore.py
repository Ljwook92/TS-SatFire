import datetime as dt
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))


class ExploreTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
            import rasterio
            import satfire_explore as explore
        except ImportError:
            self.skipTest('rasterio/numpy required')
        self.np, self.rio, self.explore = np, rasterio, explore
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root, self.roi, self.split = explore.create_demo_dataset(Path(self.tmp.name)/'demo')
        from fp_audit import read_fp_split
        records, _ = read_fp_split(self.roi, self.split)
        self.files = explore.discover_fp(self.root, records)
        self.index = explore.date_index(self.files, 'DEMO_FIRE')

    def test_all_demo_shapes_and_alignment(self):
        rows = self.explore.header_inventory(self.files)
        self.assertEqual(len(rows), 21)
        self.assertEqual({tuple(r['shape']) for r in rows}, {(32, 32)})
        self.assertTrue(all(r['reasons'] == 'aligned' for r in self.explore.alignment_report(rows)))

    def test_same_shape_does_not_imply_alignment(self):
        a = self.explore.header(self.files[0]['path'])
        b = dict(a, transform=list(a['transform']))
        b['transform'][2] += 375
        self.assertEqual(self.explore.grid_reasons(a, b), ['transform_mismatch'])
        c = dict(a, crs=None)
        self.assertIn('candidate_crs_missing', self.explore.grid_reasons(a, c))
        self.assertNotIn('shape_mismatch', self.explore.grid_reasons(a, c))

    def test_progression_zero_is_distinct_from_missing(self):
        rows = self.explore.progression_series(self.index, 8, 'finite')
        by_date = {r['current_date']: r for r in rows}
        self.assertEqual(by_date['2021-08-03']['status'], 'observed_no_increment')
        self.assertEqual(by_date['2021-08-03']['new_pixels'], 0)
        self.assertEqual(by_date['2021-08-05']['status'], 'unknown_missing_annotation')
        self.assertIsNone(by_date['2021-08-05']['new_pixels'])
        self.assertGreater(by_date['2021-08-02']['new_pixels'], 0)
        self.assertFalse(by_date['2021-08-01']['two_day_history'])

    def test_unconfirmed_label_encoding_is_not_zero(self):
        rows = self.explore.progression_series(self.index)
        self.assertEqual(rows[1]['status'], 'unknown_label_encoding')
        self.assertIsNone(rows[1]['new_pixels'])

    def test_vnp13_raw_units_and_fill_are_restored(self):
        p = self.explore.unique_path(self.index, 'FirePred', dt.date(2021, 8, 2))
        with self.rio.open(p, 'r+') as ds:
            original = ds.read(1)
            raw = original * 10000
            raw[0, 0] = -15000
            ds.write(raw, 1)
        result = self.explore.read_ndvi(p, scale=0.0001, offset=0,
                                        raw_valid_range=(-10000, 10000))
        self.assertEqual(result['status'], 'available')
        self.assertTrue(self.np.isnan(result['array'][0, 0]))
        self.assertEqual(result['excluded_raw_pixels'], 1)
        self.np.testing.assert_allclose(result['array'][1:], original[1:], atol=1e-6)

    def test_already_scaled_values_are_not_scaled_twice(self):
        p = self.explore.unique_path(self.index, 'FirePred', dt.date(2021, 8, 2))
        result = self.explore.read_ndvi(p, scale=0.0001)
        self.assertEqual(result['status'], 'ndvi_may_already_be_scaled')
        self.assertIsNone(result['array'])

    def test_notebook_plot_recovers_stale_reader(self):
        import ast
        import contextlib
        import io
        import json
        from unittest.mock import patch
        try:
            import pandas as pd
            import matplotlib
            matplotlib.use('Agg', force=True)
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest('Notebook plotting dependencies required')
        notebook = SCRIPTS.parent/'notebooks'/'ts_satfire_two_day_ndvi.ipynb'
        cells = json.loads(notebook.read_text())['cells']
        source = next(''.join(c['source']) for c in cells if c['cell_type'] == 'code'
                      and 'def plot_pair' in ''.join(c['source']))
        tree = ast.parse(source)
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        namespace = dict(REPO=SCRIPTS.parent, Path=Path, dt=dt, index=self.index,
                         NDVI_SOURCE='provided', PROVIDED_NDVI_BAND=1,
                         NDVI_SCALE=1.0, NDVI_OFFSET=0.0,
                         REFLECTANCE_CONFIRMED=False, PROVIDED_NDVI_RAW_VALID_RANGE=(-1, 1),
                         BA_BAND=8, BA_RULE='finite', EVENT_ID='DEMO_FIRE', DEMO_MODE=True,
                         REPORT_DIR=Path(self.tmp.name), np=self.np, pd=pd, plt=plt,
                         display=lambda *args: None)
        def obsolete(path, source='provided', band=1, scale=None, offset=None,
                     reflectance_confirmed=False):
            raise AssertionError('Stale NDVI reader must not be used')
        original = self.explore.read_ndvi
        self.explore.read_ndvi = obsolete
        namespace['read_ndvi'] = obsolete
        try:
            exec(compile(tree, str(notebook), 'exec'), namespace)
            with contextlib.redirect_stdout(io.StringIO()), patch.object(plt, 'show'):
                observations, change = namespace['plot_pair'](dt.date(2021, 8, 2))
            self.assertTrue(all(o['status'] == 'available' for o in observations))
            self.assertEqual(change['status'], 'observed_increment')
        finally:
            self.explore.read_ndvi = original
            plt.close('all')

    def test_ndvi_units_and_reflectance_gate(self):
        date = dt.date(2021, 8, 2)
        p = self.explore.unique_path(self.index, 'FirePred', date)
        result = self.explore.read_ndvi(p)
        self.assertEqual(result['status'], 'available')
        self.assertTrue(self.np.isfinite(result['array']).any())
        scaled = self.explore.read_ndvi(p, scale=10000)
        self.assertEqual(scaled['status'], 'ndvi_scale_or_band_unverified')
        self.assertIsNone(scaled['array'])
        day = self.explore.unique_path(self.index, 'VIIRS_Day', date)
        self.assertEqual(self.explore.read_ndvi(day, source='computed')['status'], 'reflectance_not_confirmed')
        computed = self.explore.read_ndvi(day, source='computed', reflectance_confirmed=True)
        self.np.testing.assert_allclose(result['array'], computed['array'], atol=1e-6)


if __name__ == '__main__':
    unittest.main()
