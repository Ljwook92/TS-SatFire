import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
import st_kriging as sk
from satfire_demo import make_demo

class MathematicalTests(unittest.TestCase):
    def test_covariance_psd_and_time_decay(self):
        xy=np.array([[0,0],[1,0],[0,0]],float);t=np.array([0,0,1])
        K=sk.covariance(xy,t,xy,t,2.,3.)
        np.testing.assert_allclose(K,K.T)
        self.assertGreater(np.linalg.eigvalsh(K).min(),0)
        self.assertAlmostEqual(K[0,2],np.exp(-1/3))
    def test_laplace_mode_and_true_kriging_system(self):
        cfg=sk.defaults();offset=np.array([-3.,-2.,-1.,0.]);y=np.array([0,1,0,1]);groups=np.zeros(4,int)
        fit=sk.LatentKriging(cfg,5,2).fit(np.array([[0.,0.]]),np.array([1]),offset,groups,y)
        v=1+cfg['kernel_jitter']
        objective=lambda f:np.sum(np.logaddexp(0,offset+f)-y*(offset+f))+f*f/(2*v)
        reference=minimize_scalar(objective,bracket=(-2,2),method='brent')
        self.assertAlmostEqual(fit.f[0],reference.x,places=5)
        p=expit(offset+fit.f[0]);g=np.sum(y-p);z=fit.f[0]+g/fit.W[0]
        mean,var=fit.predict(np.array([[0.,0.]]),2)
        k=np.exp(-.5)
        self.assertAlmostEqual(mean[0],k*z/(v+1/fit.W[0]),places=5)
        self.assertAlmostEqual(var[0],1-k*k/(v+1/fit.W[0]),places=5)
        self.assertTrue(np.all(np.diff(fit.objectives)<=1e-8))
        far,_=fit.predict(np.array([[0.,0.]]),100);self.assertLess(abs(far[0]),1e-10)
        with self.assertRaises(ValueError):fit.predict(np.array([[0.,0.]]),1)
    def test_probability_integration_and_zero_strength(self):
        p=np.array([1e-6,.2,.5,.9]);m=np.zeros(4);v=np.ones(4)
        np.testing.assert_array_equal(sk.probabilities(p,m,v,0),p)
        q=sk.probabilities(p,m,v,1);self.assertAlmostEqual(q[2],.5)
        self.assertTrue(np.all((q>0)&(q<1)))

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();root=Path(cls.tmp.name)
        cfg=sk.defaults();cfg.update(raw_root=str(root/'raw'),event_id='SYNTHETIC',features=['ndvi','i4','tmax'],
            negative_keep=.5,lambdas=[.01],n_knots=3,max_iter=2000,block_pixels=7,
            spatial_ranges_km=[3.],temporal_ranges_days=[2.],history_days=2)
        make_demo(cfg['raw_root'],cfg['event_id']);cls.data=sk.prepare(cfg,root/'run')
        start_test=cls.data['days'].index(cls.data['blocks']['test'][0])
        old_cached=sk.core.cached;old_rows=sk.core.rows
        def guarded_cached(d,k,i):
            if i>=start_test:raise AssertionError('Test raster entered selection')
            return old_cached(d,k,i)
        def guarded_rows(d,i):
            if i>=start_test:raise AssertionError('Test labels entered selection')
            return old_rows(d,i)
        with patch.object(sk.core,'cached',side_effect=guarded_cached),patch.object(sk.core,'rows',side_effect=guarded_rows):
            cls.selected=sk.train_select(cls.data)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_01_disjoint_dates_shared_windows(self):
        _,targets=sk.core.split_dates(pd.date_range('2021-05-17','2021-06-23'),sk.defaults())
        self.assertEqual([len(targets[k]) for k in ['train','val','test']],[13,3,4])
        sets=[]
        for split,dates in self.data['targets'].items():
            used=set()
            for date in dates:
                i=self.data['days'].index(date);used.update(self.data['days'][i-6:i+1])
            sets.append(used)
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    def test_02_forecast_reads_only_past_and_no_in_sample_residuals(self):
        d=self.data;day=d['days'].index(d['targets']['test'][0]);fc=self.selected[2][0]
        old=sk.core.cached
        def guard(data,kind,i):
            self.assertLess(i,day,'Current/future target or raster read during prediction')
            return old(data,kind,i)
        with patch.object(sk.core,'cached',side_effect=guard):r=fc.forecast(day,3.,2.)
        self.assertTrue(np.isfinite(r['mean']).all());self.assertTrue(np.isfinite(r['var']).all())
        support=fc.support(day)
        self.assertGreater(support['days'].min(),d['train_end']);self.assertLess(support['days'].max(),day)
        self.assertEqual(len(support['y']),sum(len(fc.observed(i,day)['y']) for i in range(day-2,day)))
        with self.assertRaises(ValueError):fc.observed(day,day)
        with self.assertRaises(ValueError):fc.observed(d['train_end'],day)
    def test_03_future_perturbation_invariance(self):
        d=self.data;day=d['days'].index(d['targets']['test'][0]);fc=self.selected[4][0]
        before=fc.forecast(day,3.,2.);backups={}
        for kind in ['x','c','k']:
            path=d['cache']/f'{kind}_{day:03d}.npy';backups[path]=path.read_bytes();a=np.load(path)
            a[:]=0 if kind=='x' else ~a
            np.save(path,a)
        try:
            # Recompute forecast baseline rather than merely replaying its cached result.
            (d['out']/'base_predictions'/f'TS4_day{day:03d}.npz').unlink()
            after=fc.forecast(day,3.,2.)
            for key in before:np.testing.assert_allclose(before[key],after[key],rtol=0,atol=0)
        finally:
            for path,raw in backups.items():path.write_bytes(raw)
    def test_04_past_observation_changes_prediction(self):
        d=self.data;day=d['days'].index(d['targets']['test'][0]);fc=self.selected[6][0];support=fc.support(day)
        a=sk.LatentKriging(d['cfg'],3,2).fit(**support).predict(d['centres'],day)[0]
        changed=dict(support);changed['y']=np.zeros_like(support['y'])
        b=sk.LatentKriging(d['cfg'],3,2).fit(**changed).predict(d['centres'],day)[0]
        self.assertGreater(np.max(np.abs(a-b)),1e-4)
    def test_05_full_evaluation_without_GAM_refit(self):
        with patch.object(sk.core.AdditiveLogit,'fit',side_effect=AssertionError('GAM refit at test')):
            m=sk.evaluate(self.data,self.selected)
        self.assertEqual(len(m),6);self.assertEqual(m.n.nunique(),1);self.assertEqual(m.positives.nunique(),1)
        for date in self.data['targets']['test']:
            common=None
            for ts in [2,4,6]:
                with np.load(self.data['out']/'predictions'/f'TS{ts}_{date.date()}.npz') as z:
                    if common is None:common=(z['ids'].copy(),z['y'].copy())
                    np.testing.assert_array_equal(z['ids'],common[0]);np.testing.assert_array_equal(z['y'],common[1])
                    self.assertTrue(np.isfinite(z['gam_stk']).all());self.assertTrue(np.all((z['gam_stk']>=0)&(z['gam_stk']<=1)))
    def test_06_selection_seal(self):
        self.selected[2][1]['strength']+=.123
        try:
            with self.assertRaises(ValueError):sk.evaluate(self.data,self.selected)
        finally:self.selected[2][1]['strength']-=.123
    def test_07_all_positive_samples_preserved(self):
        for i,ids,y,w in sk.core.training_rows(self.data):
            self.assertEqual(y.sum(),sk.core.rows(self.data,i)[1].sum())
            np.testing.assert_array_equal(w,np.where(y==1,1.,2.))

if __name__=='__main__':unittest.main(verbosity=2)
