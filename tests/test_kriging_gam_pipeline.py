import unittest,sys,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'notebooks/kriging_gam'))
import satfire_pipeline as p
from satfire_demo import make_demo

class PipelineContracts(unittest.TestCase):
    def test_split_purge(self):
        g=p.split_windows(list(pd.date_range('2021-01-01',periods=40)))
        for l,r in [('train','val'),('val','test')]:
            self.assertLess(max(w[2] for w in g[l]),min(w[0] for w in g[r]))
    def test_calendar_gap_not_next_day(self):
        days=list(pd.date_range('2021-01-01',periods=40));days.pop(12)
        for ws in p.split_windows(days).values():
            for a,b,c in ws:self.assertEqual((c-a).days,2)
    def test_kriging_observed_immutable_and_no_support(self):
        rr,cc=np.indices((15,15));a=np.sin(rr/5)+cc/10;a[5:7,5:7]=np.nan
        xy=np.column_stack([cc.ravel(),rr.ravel()]).astype(float)
        cfg=dict(min_support=5,max_support=100,neighbors=12,max_distance_km=10)
        filled,var,info=p.fill_kriging(a,xy,cfg,1)
        np.testing.assert_array_equal(filled[np.isfinite(a)],a[np.isfinite(a)])
        self.assertTrue(np.isfinite(filled[5:7,5:7]).all())
        a[:]=np.nan;filled,_,_=p.fill_kriging(a,xy,cfg,1)
        self.assertTrue(np.isnan(filled).all())
    def test_label_unknown_and_legacy_are_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            make_demo(Path(td),'SYNTHETIC')
            cfg=dict(raw_root=td,event_id='SYNTHETIC',crop=None,label_nan_policy='strict_unknown')
            files,days=p.catalog(cfg);_,meta=p.read_tif(files[days[0],'VIIRS_Day'])
            strict=p.labels(files,days,meta,cfg)
            self.assertFalse(strict[days[-1]][1][0,0])
            cfg['label_nan_policy']='legacy_nan_zero';legacy=p.labels(files,days,meta,cfg)
            self.assertTrue(legacy[days[-1]][1][0,0])
            self.assertFalse(legacy[days[-1]][0][0,0])
            cur,_=strict[days[20]];future,_=strict[days[21]]
            self.assertFalse(((future&~cur)&cur).any())
    def test_categorical_blocked(self):
        self.assertEqual(p.SPECS['landcover'][2],'categorical')
    def test_metric_probability_values(self):
        m=p.scores([0,0,1,1],[.1,.2,.8,.9],.5)
        self.assertAlmostEqual(m['brier'],.025);self.assertEqual(m['f1'],1)

if __name__=='__main__':unittest.main()
