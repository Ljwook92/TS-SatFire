import datetime as dt
from pathlib import Path
import sys
import tempfile
import unittest
from collections import defaultdict
import numpy as np
import rasterio
from rasterio.transform import from_origin
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fp_progression import accumulate_step, build_progression, compare_days

class CumulativeProgressionTests(unittest.TestCase):
    def test_finite_labels_match_legacy_and_nan_does_not_become_fire(self):
        af=np.array([[0.,1.],[0.,0.]])
        ba=np.array([[0.,0.],[2.,0.]])
        before=np.zeros((2,2),bool)
        result,_=accumulate_step(before,af,ba,0)
        np.testing.assert_array_equal(result,np.logical_or(af,ba))
        ba[0,0]=np.nan
        result,_=accumulate_step(before,af,ba,0)
        self.assertFalse(result[0,0])
        self.assertTrue(np.logical_or(af,ba)[0,0])
        af_only,_=accumulate_step(before,af,ba,1)
        self.assertEqual(int(af_only.sum()),1)

    def fixture(self,root,day,af,ba,shift=0):
        path=root/f'{day}.tif'
        a=np.zeros((8,2,2),dtype='float32')
        a[6]=af
        a[7]=ba
        with rasterio.open(path,'w',driver='GTiff',count=8,height=2,width=2,
                           dtype='float32',crs='EPSG:4326',transform=from_origin(100+shift,50,.1,.1)) as ds:
            ds.write(a)
        return path

    def test_next_day_increment_no_increment_and_literal_nan_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); index=defaultdict(lambda:defaultdict(list)); start=dt.date(2021,5,17)
            for k in range(3):
                af=np.zeros((2,2)); af[0,0]=1
                if k>0: af[0,1]=1
                day=start+dt.timedelta(days=k)
                index['VIIRS_Day'][day]=[self.fixture(root,day,af,np.full((2,2),np.nan))]
            rows,maps=build_progression(index,0,start,'no_detection')
            self.assertEqual(rows[0]['literal_legacy_pixels'],4)
            self.assertEqual(rows[0]['cumulative_pixels'],1)
            self.assertTrue(rows[0]['ba_all_nan'])
            self.assertEqual(compare_days(maps,start)['new_pixels'],1)
            self.assertEqual(compare_days(maps,start+dt.timedelta(days=1))['status'],'no_new_recorded_fire_pixels')
            self.assertEqual(compare_days(maps,start+dt.timedelta(days=2))['status'],'unknown_incomplete_history')

    def test_missing_calendar_day_invalidates_later_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); index=defaultdict(lambda:defaultdict(list)); start=dt.date(2021,5,17)
            for k in (0,2):
                day=start+dt.timedelta(days=k)
                index['VIIRS_Day'][day]=[self.fixture(root,day,np.zeros((2,2)),np.zeros((2,2)))]
            rows,maps=build_progression(index,0,start,'no_detection')
            self.assertEqual(len(maps),1)
            self.assertIn('missing_or_duplicate',rows[1]['status'])
            self.assertIn('incomplete_prior_history',rows[2]['status'])

    def test_shifted_grid_and_missing_af_are_unknown(self):
        for kind in ('shift','missing_af'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); index=defaultdict(lambda:defaultdict(list)); start=dt.date(2021,5,17)
                for k in range(2):
                    day=start+dt.timedelta(days=k)
                    af=np.full((2,2),np.nan) if k and kind=='missing_af' else np.zeros((2,2))
                    index['VIIRS_Day'][day]=[self.fixture(root,day,af,np.zeros((2,2)),int(k and kind=='shift'))]
                rows,maps=build_progression(index,0,start,'no_detection')
                self.assertEqual(len(maps),1)
                self.assertTrue(rows[1]['status'].startswith('unknown_'))

if __name__=='__main__':
    unittest.main()
