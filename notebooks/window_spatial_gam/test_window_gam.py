"""Small synthetic integration + temporal leakage regression checks; no real-data claims."""
import tempfile, unittest, hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import window_gam as w
from satfire_demo import make_demo

class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();root=Path(cls.tmp.name)
        cfg=w.defaults();cfg.update(raw_root=str(root/'raw'),event_id='demo',features=['ndvi','i4','tmax'],
            negative_keep=.5,lambdas=[.01],spatial_lengths_km=[3.],spatial_knots_side=3,n_knots=3,max_iter=2000)
        make_demo(cfg['raw_root'],cfg['event_id']);cls.data=w.prepare(cfg,root/'run')
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_01_dates_and_shared_rows(self):
        data=self.data;sets=[]
        for split,dates in data['targets'].items():
            used=set()
            for d in dates:
                i=data['days'].index(d);used.update(data['days'][i-6:i+1])
                ids,y=w.rows(data,i)
                for ts in [2,4,6]:self.assertEqual(len(w.features(data,i,ts,ids)),len(y))
            sets.append(used)
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        _,target=w.split_dates(pd.date_range('2021-05-17','2021-06-23'),w.defaults())
        self.assertEqual([len(target[s]) for s in ['train','val','test']],[13,3,4])
        self.assertEqual(str(target['test'][0].date()),'2021-06-20')
    def test_02_features_cannot_read_future(self):
        data=self.data;i=data['days'].index(data['targets']['test'][0]);ids,_=w.rows(data,i)
        original=w.cached
        def guarded(data,kind,j):
            self.assertLess(j,i,'Target/future raster entered predictors')
            return original(data,kind,j)
        with patch.object(w,'cached',side_effect=guarded):
            for ts in [2,4,6]:w.features(data,i,ts,ids)
    def test_03_selection_never_reads_test(self):
        data=self.data;first_test=data['days'].index(data['blocks']['test'][0])
        original=w.cached
        def guarded(data,kind,j):
            self.assertLess(j,first_test,'Test raster accessed during selection')
            return original(data,kind,j)
        with patch.object(w,'cached',side_effect=guarded):selected=w.train_select(data)
        self.assertEqual(len(selected),6)
        sample=w.training_rows(data)
        for i,ids,y,weights in sample:
            self.assertTrue(np.all(weights[y==1]==1))
            self.assertTrue(np.all(weights[y==0]==2))
            _,all_y=w.rows(data,i);self.assertEqual(y.sum(),all_y.sum())
        # Evaluate must not call any fitting routine; every predicted pixel ID must match.
        with patch.object(w.AdditiveLogit,'fit',side_effect=AssertionError('Test fit forbidden')):
            metrics=w.evaluate(data,selected)
        self.assertEqual(len(metrics),6);self.assertEqual(metrics.n.nunique(),1)
        self.assertEqual(metrics.positives.nunique(),1)
        self.assertTrue(np.all(np.isfinite(metrics[['ap','brier','log_loss']].values)))
        for date in data['targets']['test']:
            ids_expected=None;y_expected=None
            for key in selected:
                with np.load(data['out']/'predictions'/f'{key}_{date.date()}.npz') as z:
                    if ids_expected is None:ids_expected=z['ids'];y_expected=z['y']
                    np.testing.assert_array_equal(ids_expected,z['ids']);np.testing.assert_array_equal(y_expected,z['y'])
                    self.assertTrue(np.all((z['p']>=0)&(z['p']<=1)))
    def test_04_missing_values_keep_rows(self):
        x=np.array([[0.,np.nan],[1.,1.],[2.,2.],[3.,3.],[4.,4.],[5.,5.]])
        y=np.array([0,0,0,1,1,1]);xy=np.column_stack([np.arange(6),np.zeros(6)])
        model=w.AdditiveLogit(self.data['cfg'],.01).fit(x,xy,y,np.ones(6))
        p=model.predict(np.full_like(x,np.nan),xy)
        self.assertEqual(len(p),len(y));self.assertTrue(np.isfinite(p).all())
    def test_05_input_gaps_fail_closed(self):
        with self.assertRaises(ValueError):w.split_dates(pd.date_range('2021-01-01',periods=40).delete(4),w.defaults())

if __name__=='__main__':unittest.main(verbosity=2)
