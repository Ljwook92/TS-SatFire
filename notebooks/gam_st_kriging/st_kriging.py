"""Logistic GAM offset + past-only latent spatiotemporal Gaussian-process kriging.

Exact pixel Bernoulli likelihood conditional on a shared block/day latent effect;
block-centroid covariance approximation; Laplace posterior and Gaussian quadrature.
This is generalized kriging, not Gaussian kriging of raw binary residuals.
"""
from pathlib import Path
from itertools import product
import hashlib, json, platform, warnings
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from scipy.special import expit, logit
from numpy.polynomial.hermite import hermgauss
from sklearn.metrics import log_loss
import gam_core as core


def defaults():
    cfg=core.defaults()
    cfg.update(block_pixels=64, history_days=3, spatial_ranges_km=[5.,20.],
               temporal_ranges_days=[1.,3.], latent_variance=1.,
               correction_strengths=[0.,.5,1.], laplace_max_iter=60,
               laplace_tol=1e-5, kernel_jitter=1e-6, quadrature_nodes=15)
    return cfg


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(cfg,out):
    # Isolate configuration from later notebook edits.
    cfg=json.loads(json.dumps(cfg))
    if cfg['windows'] != [2,4,6]:raise ValueError('This controlled comparison requires windows=[2,4,6]')
    if cfg['history_days']<1 or cfg['block_pixels']<1:raise ValueError('Positive history/block size required')
    if cfg['latent_variance']<=0 or min(cfg['spatial_ranges_km']+cfg['temporal_ranges_days'])<=0:raise ValueError('Positive covariance parameters required')
    data=core.prepare(cfg,out)
    for sub in ['base_predictions','observations','predictions','models']: (data['out']/sub).mkdir()
    h,w=data['meta']['shape'];rr,cc=np.indices((h,w));size=cfg['block_pixels']
    labels=(rr//size)*int(np.ceil(w/size))+(cc//size);labels=labels.ravel()
    count=np.bincount(labels)
    centres=np.column_stack([np.bincount(labels,weights=data['xy'][:,j])/count for j in range(2)])
    data.update(block_ids=labels,centres=centres,train_end=data['days'].index(data['targets']['train'][-1]))
    np.savez(data['out']/'spatial_blocks.npz',pixel_block=labels,centres_km=centres,counts=count)
    core.dump(data['out']/'method.json',dict(
        model='GAM offset + Bernoulli latent ST GP, block-centroid covariance, Laplace generalized kriging',
        likelihood='All eligible observed pixels; no subsampling in kriging likelihood; offsets vary within block',
        kernel='variance * exp(-distance_km/spatial_range - abs(day_difference)/temporal_range)',
        train_end=str(data['days'][data['train_end']].date()),
        target='Next-day newly recorded cumulative AF-or-BA area, not ignition or pure BA growth',
        sequential_protocol='Daily rolling-origin, labels of day d are available only to forecasts for d+1 or later',
        uncertainty='Conditional latent Laplace variance only; not calibrated prediction intervals or GAM uncertainty',
        sources={name:sha(Path(__file__).parent/name) for name in ['st_kriging.py','gam_core.py','satfire_demo.py']}))
    return data


def covariance(x,t,z,u,space,time,variance=1.):
    return variance*np.exp(-cdist(x,z)/space-np.abs(np.asarray(t)[:,None]-np.asarray(u)[None,:])/time)


class LatentKriging:
    """Laplace mode for f~N(0,K), y_i~Bernoulli(sigmoid(offset_i+f[group_i])).

    Newton pseudo observations z=f+(k-sum(p))/W give
    mean_* = K_*o (K_oo + diag(1/W))^-1 z at convergence.
    We use the equivalent stable B=I+sqrt(W)Ksqrt(W) Cholesky solves.
    """
    def __init__(self,cfg,space,time):
        self.cfg=cfg;self.space=space;self.time=time
    def fit(self,xy,days,offset,groups,y):
        xy=np.asarray(xy,float);days=np.asarray(days,int);offset=np.asarray(offset,float)
        groups=np.asarray(groups,int);y=np.asarray(y,float);m=len(xy)
        if m==0 or len(offset)!=len(groups) or len(y)!=len(offset):raise ValueError('Invalid support')
        if not np.isfinite(offset).all() or np.any((y!=0)&(y!=1)):raise ValueError('Invalid observations')
        if groups.min()<0 or groups.max()>=m or np.any(np.bincount(groups,minlength=m)==0):raise ValueError('Every latent node needs observations')
        self.xy=xy;self.days=days
        self.K=covariance(xy,days,xy,days,self.space,self.time,self.cfg['latent_variance'])
        self.K.flat[::m+1]+=self.cfg['kernel_jitter']
        self.Kfactor=cho_factor(self.K,lower=True,check_finite=False)
        counts=np.bincount(groups,weights=y,minlength=m)
        def state(f):
            eta=offset+f[groups];p=expit(eta)
            # Constants -sum(y*offset) cancel in line search; retain for objective audit.
            loss=float(np.sum(np.logaddexp(0,eta)-y*eta))
            grad=counts-np.bincount(groups,weights=p,minlength=m)
            W=np.maximum(np.bincount(groups,weights=p*(1-p),minlength=m),1e-12)
            invf=cho_solve(self.Kfactor,f,check_finite=False)
            return loss+.5*float(f@invf),grad,W,invf
        f=np.zeros(m);objectives=[]
        for iteration in range(self.cfg['laplace_max_iter']):
            obj,g,W,invf=state(f);objectives.append(obj)
            error=float(np.max(np.abs(g-invf)))
            if error<self.cfg['laplace_tol']:break
            sw=np.sqrt(W);B=np.eye(m)+(sw[:,None]*self.K)*sw[None,:]
            bf=cho_factor(B,lower=True,check_finite=False)
            b=W*f+g
            a=b-sw*cho_solve(bf,sw*(self.K@b),check_finite=False)
            proposal=self.K@a;step=1.
            while step>2**-20:
                candidate=f+step*(proposal-f)
                if state(candidate)[0]<=obj+1e-10:break
                step*=.5
            if step<=2**-20:raise RuntimeError('Laplace line search failed')
            f=candidate
        else:
            obj,g,W,invf=state(f);error=float(np.max(np.abs(g-invf)))
            if error>=self.cfg['laplace_tol']:raise RuntimeError(f'Laplace did not converge: gradient={error:g}')
        obj,g,W,invf=state(f)
        self.f=f;self.a=invf;self.W=W;self.sw=np.sqrt(W)
        self.Bfactor=cho_factor(np.eye(m)+(self.sw[:,None]*self.K)*self.sw[None,:],lower=True,check_finite=False)
        self.diagnostics=dict(nodes=m,observations=len(y),positives=int(y.sum()),iterations=iteration+1,
                              gradient_max=float(np.max(np.abs(g-invf))),objective=float(obj),
                              first_support_day=int(days.min()),last_support_day=int(days.max()))
        self.objectives=objectives
        return self
    def predict(self,xy,day):
        if int(day)<=int(self.days.max()):raise ValueError('Forecast target must be AFTER every support observation')
        k=covariance(self.xy,self.days,xy,np.full(len(xy),day),self.space,self.time,self.cfg['latent_variance'])
        mean=k.T@self.a;sk=self.sw[:,None]*k
        variance=self.cfg['latent_variance']-np.sum(sk*cho_solve(self.Bfactor,sk,check_finite=False),axis=0)
        if variance.min() < -1e-7:raise RuntimeError('Invalid predictive variance')
        return mean,np.maximum(variance,0.)


def probabilities(p,mean,var,strength,nodes=15):
    if strength==0:return np.asarray(p).copy()
    x,w=hermgauss(nodes);out=np.zeros_like(p,dtype=float)
    eta=logit(np.clip(p,1e-12,1-1e-12))+strength*mean
    scale=strength*np.sqrt(2*var)
    for a,b in zip(x,w):out+=b*expit(eta+a*scale)
    return out/np.sqrt(np.pi)


class PastOnlyForecaster:
    def __init__(self,data,model,ts):
        self.data=data;self.model=model;self.ts=ts;self.logs=[]
    def baseline(self,day):
        """Compute for PRIOR-known at-risk pixels; no current-day raster or label access."""
        d=self.data
        if day<=d['train_end']:raise ValueError('In-sample residuals forbidden: day must follow GAM training')
        file=d['out']/'base_predictions'/f'TS{self.ts}_day{day:03d}.npz'
        if file.exists():
            with np.load(file) as z:return z['ids'],z['p']
        before=core.cached(d,'c',day-1);known=core.cached(d,'k',day-1)
        ids=np.flatnonzero((~before & known).ravel())
        x=core.features(d,day,self.ts,ids);p=np.empty(len(ids));chunk=d['cfg']['chunk_size']
        for a in range(0,len(ids),chunk):p[a:a+chunk]=self.model.predict(x[a:a+chunk],d['xy'][ids[a:a+chunk]])
        if not np.isfinite(p).all():raise RuntimeError('Nonfinite GAM probability')
        np.savez(file,ids=ids,p=p)
        return ids,p
    def observed(self,day,asof):
        d=self.data
        if day>=asof or day<=d['train_end']:raise ValueError('Residual observation is future/current or in-sample')
        file=d['out']/'observations'/f'TS{self.ts}_day{day:03d}.npz'
        if file.exists():
            with np.load(file) as z:return {k:z[k] for k in ['blocks','offset','y']}
        ids,p=self.baseline(day)
        valid=core.cached(d,'k',day).ravel()[ids]
        y=core.cached(d,'c',day).ravel()[ids][valid].astype('uint8')
        result=dict(blocks=d['block_ids'][ids[valid]],offset=logit(np.clip(p[valid],1e-12,1-1e-12)),y=y)
        np.savez(file,**result)
        return result
    def support(self,day):
        d=self.data;all_xy=[];all_days=[];offset=[];groups=[];y=[];cursor=0
        for past in range(max(d['train_end']+1,day-d['cfg']['history_days']),day):
            r=self.observed(past,asof=day)
            if len(r['y'])==0:continue
            used,inv=np.unique(r['blocks'],return_inverse=True)
            all_xy.append(d['centres'][used]);all_days.append(np.full(len(used),past))
            groups.append(inv+cursor);offset.append(r['offset']);y.append(r['y']);cursor+=len(used)
        if not y:raise ValueError('No post-training historical observations available')
        return dict(xy=np.concatenate(all_xy),days=np.concatenate(all_days),offset=np.concatenate(offset),groups=np.concatenate(groups),y=np.concatenate(y))
    def forecast(self,day,space,time):
        ids,p=self.baseline(day);support=self.support(day)
        fit=LatentKriging(self.data['cfg'],space,time).fit(**support)
        mean,var=fit.predict(self.data['centres'],day)
        self.logs.append(dict(ts=self.ts,target_day=day,target_date=str(self.data['days'][day].date()),
                              spatial_range=space,temporal_range=time,**fit.diagnostics))
        block=self.data['block_ids'][ids]
        return dict(ids=ids,p=p,mean=mean[block],var=var[block])


def scored(data,day,pred):
    """Read target labels only AFTER prediction, for common known-label evaluation mask."""
    ids,y=core.rows(data,day)
    positions=np.searchsorted(pred['ids'],ids)
    if len(ids) and (positions.max()>=len(pred['ids']) or not np.array_equal(pred['ids'][positions],ids)):
        raise AssertionError('Evaluation IDs absent from prior-known forecast mask')
    return y,{k:pred[k][positions] for k in ['ids','p','mean','var']}


def train_select(data):
    import joblib, sklearn, scipy
    out=data['out'];cfg=data['cfg'];sample=core.training_rows(data)
    y=np.concatenate([r[2] for r in sample]);weights=np.concatenate([r[3] for r in sample]);xy=np.concatenate([data['xy'][r[1]] for r in sample])
    np.savez(out/'training_rows.npz',day=np.concatenate([np.full(len(r[1]),r[0]) for r in sample]),pixel=np.concatenate([r[1] for r in sample]),y=y,weights=weights)
    base_tuning=[];krige_tuning=[];selected={};selection=[];logs=[]
    for ts in cfg['windows']:
        x=np.concatenate([core.features(data,r[0],ts,r[1]) for r in sample]);best=None
        for lam in cfg['lambdas']:
            print(f'TS={ts} GAM lambda={lam}',flush=True)
            try:model=core.AdditiveLogit(cfg,lam).fit(x,xy,y,weights)
            except RuntimeError as exc:
                base_tuning.append(dict(ts=ts,lam=lam,status=str(exc),val_log_loss=None));continue
            val=core.predict_split(data,model,ts,'val');vy=np.concatenate([r['y'] for r in val]);vp=np.concatenate([r['p'] for r in val]);loss=float(log_loss(vy,vp,labels=[0,1]))
            base_tuning.append(dict(ts=ts,lam=lam,status='converged',val_log_loss=loss))
            if best is None or loss<best[0]:best=(loss,model,vy,vp)
        if best is None:raise RuntimeError('No converged GAM candidates')
        base_loss,model,vy,vp=best;forecaster=PastOnlyForecaster(data,model,ts)
        base_threshold=core.choose_threshold(vy,vp)
        # alpha=0 is an explicit no-correction candidate, selected on validation only.
        winner=dict(loss=base_loss,space=None,time=None,strength=0.,threshold=base_threshold)
        krige_tuning.append(dict(ts=ts,space=None,time=None,strength=0.,status='GAM fallback',val_log_loss=base_loss))
        for space,time in product(cfg['spatial_ranges_km'],cfg['temporal_ranges_days']):
            print(f'TS={ts} latent ST kriging: spatial={space} km, temporal={time} d',flush=True)
            forecasts=[];labels=[]
            try:
                for date in data['targets']['val']:
                    day=data['days'].index(date);raw=forecaster.forecast(day,space,time)
                    yv,pred=scored(data,day,raw);labels.append(yv);forecasts.append(pred)
            except RuntimeError as exc:
                krige_tuning.append(dict(ts=ts,space=space,time=time,strength=None,status=str(exc),val_log_loss=None));continue
            yv=np.concatenate(labels);np.testing.assert_array_equal(yv,vy)
            for strength in cfg['correction_strengths']:
                if strength==0:continue
                pv=np.concatenate([probabilities(r['p'],r['mean'],r['var'],strength,cfg['quadrature_nodes']) for r in forecasts])
                loss=float(log_loss(yv,pv,labels=[0,1]))
                krige_tuning.append(dict(ts=ts,space=space,time=time,strength=strength,status='converged',val_log_loss=loss))
                if loss<winner['loss']:winner=dict(loss=loss,space=space,time=time,strength=strength,threshold=core.choose_threshold(yv,pv))
        if not any(r.get('status')=='converged' for r in krige_tuning if r['ts']==ts):raise RuntimeError('All kriging candidates failed; do not silently substitute GAM')
        chosen=dict(ts=ts,gam_lambda=model.lam,gam_threshold=base_threshold,gam_val_log_loss=base_loss,
                    kriging_space=winner['space'],kriging_time=winner['time'],strength=winner['strength'],
                    kriging_threshold=winner['threshold'],kriging_val_log_loss=winner['loss'])
        selected[ts]=(forecaster,chosen);selection.append(chosen);logs.extend(forecaster.logs);forecaster.logs=[]
        joblib.dump(model,out/'models'/f'TS{ts}_GAM.joblib')
        pd.DataFrame(base_tuning).to_csv(out/'gam_tuning.csv',index=False);pd.DataFrame(krige_tuning).to_csv(out/'kriging_tuning.csv',index=False)
    pd.DataFrame(logs).to_csv(out/'validation_conditioning_audit.csv',index=False)
    core.dump(out/'selection_before_test.json',selection)
    core.dump(out/'selection_seal.json',dict(selection_sha256=sha(out/'selection_before_test.json'),config_sha256=sha(out/'config.json')))
    core.dump(out/'versions.json',dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__))
    return selected


def evaluate(data,selected):
    out=data['out'];cfg=data['cfg'];seal=json.loads((out/'selection_seal.json').read_text())
    if sha(out/'selection_before_test.json')!=seal['selection_sha256'] or sha(out/'config.json')!=seal['config_sha256']:raise ValueError('Selection/config changed after validation')
    if json.loads((out/'config.json').read_text())!=cfg:raise ValueError('In-memory config changed after validation')
    saved=json.loads((out/'selection_before_test.json').read_text())
    if saved!=[selected[ts][1] for ts in cfg['windows']]:raise ValueError('Selected parameters changed')
    all_metrics=[];daily=[];calibration=[];logs=[]
    manifest=data['manifest'].query("split == 'test'").set_index('target_date')
    for ts,(fc,choice) in selected.items():
        gathered={'GAM':[],'GAM_STK':[]};ys=[]
        for date in data['targets']['test']:
            day=data['days'].index(date);date_s=str(date.date())
            if choice['strength']>0:raw=fc.forecast(day,choice['kriging_space'],choice['kriging_time'])
            else:
                ids,p=fc.baseline(day);raw=dict(ids=ids,p=p,mean=np.zeros(len(p)),var=np.zeros(len(p)))
            y,pred=scored(data,day,raw)
            expected=manifest.loc[date_s]
            assert len(y)==expected['n'] and hashlib.sha256(pred['ids'].tobytes()).hexdigest()==expected['pixel_sha256']
            corrected=probabilities(pred['p'],pred['mean'],pred['var'],choice['strength'],cfg['quadrature_nodes'])
            if not np.isfinite(corrected).all() or np.any((corrected<0)|(corrected>1)):raise RuntimeError('Invalid kriging probability')
            ys.append(y)
            np.savez_compressed(out/'predictions'/f'TS{ts}_{date_s}.npz',ids=pred['ids'],y=y,gam=pred['p'],gam_stk=corrected,
                                latent_mean=pred['mean'],latent_sd=np.sqrt(pred['var']),date=date_s,strength=choice['strength'])
            for kind,p,threshold in [('GAM',pred['p'],choice['gam_threshold']),('GAM_STK',corrected,choice['kriging_threshold'])]:
                gathered[kind].append(p);daily.append(dict(ts=ts,model=kind,date=date_s,**core.metrics(y,p,threshold)))
        yy=np.concatenate(ys)
        for kind in gathered:
            p=np.concatenate(gathered[kind]);threshold=choice['gam_threshold' if kind=='GAM' else 'kriging_threshold']
            all_metrics.append(dict(ts=ts,model=kind,strength=0. if kind=='GAM' else choice['strength'],threshold=threshold,**core.metrics(yy,p,threshold)))
            cuts=np.unique(np.quantile(p,np.linspace(0,1,11)));bins=np.searchsorted(cuts[1:-1],p,side='right')
            for b in np.unique(bins):
                use=bins==b;calibration.append(dict(ts=ts,model=kind,bin=int(b),n=int(use.sum()),mean_probability=float(p[use].mean()),observed_fraction=float(yy[use].mean())))
        logs.extend(fc.logs);fc.logs=[]
    pd.DataFrame(logs).to_csv(out/'test_conditioning_audit.csv',index=False)
    table=pd.DataFrame(all_metrics);table.to_csv(out/'test_metrics.csv',index=False)
    pd.DataFrame(daily).to_csv(out/'test_metrics_by_date.csv',index=False);pd.DataFrame(calibration).to_csv(out/'calibration.csv',index=False)
    differences=[]
    for ts in cfg['windows']:
        a=table.query("ts==@ts and model=='GAM'").iloc[0];b=table.query("ts==@ts and model=='GAM_STK'").iloc[0]
        differences.append(dict(ts=ts,**{f'delta_{m}':float(b[m]-a[m]) for m in ['ap','brier','log_loss','f1','iou']}))
    pd.DataFrame(differences).to_csv(out/'paired_differences.csv',index=False)
    tr=data['manifest'].query("split=='train'");rate=float(tr.positives.sum()/tr.n.sum())
    core.dump(out/'constant_baseline.json',dict(train_prevalence=rate,**core.metrics(yy,np.full(len(yy),rate),.5)))
    return table


def plots(data):
    import matplotlib.pyplot as plt
    out=data['out'];table=pd.read_csv(out/'test_metrics.csv')
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
    for ax,metric in zip(axes,['ap','brier','f1']):
        for kind in ['GAM','GAM_STK']:
            t=table[table.model==kind];ax.plot(t.ts,t[metric],'o-',label=kind)
        ax.set(xlabel='Input days (TS)',ylabel=metric,xticks=[2,4,6])
    axes[0].legend();fig.savefig(out/'comparison.png',dpi=150);plt.show()
    day=str(data['targets']['test'][-1].date());shape=data['meta']['shape']
    fig,axes=plt.subplots(2,4,figsize=(14,8),constrained_layout=True);cmap=plt.get_cmap('magma').copy();cmap.set_bad('lightgray')
    for col,ts in enumerate(data['cfg']['windows']):
        with np.load(out/'predictions'/f'TS{ts}_{day}.npz') as z:
            for row,kind in enumerate(['gam','gam_stk']):
                a=np.full(np.prod(shape),np.nan);a[z['ids']]=z[kind]
                im=axes[row,col].imshow(a.reshape(shape),vmin=0,vmax=1,cmap=cmap)
                axes[row,col].set_title(f"TS={ts} {kind.upper()}"+(f" alpha={float(z['strength']):g}" if row else ''))
            observed=np.full(np.prod(shape),np.nan);observed[z['ids']]=z['y']
    axes[0,3].imshow(observed.reshape(shape),vmin=0,vmax=1,cmap=cmap);axes[0,3].set_title('Observed new record')
    axes[1,3].axis('off');fig.colorbar(im,cax=axes[1,3].inset_axes([.2,.1,.1,.8]),label='Probability')
    for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle(day+': common pixels; gray = excluded');fig.savefig(out/'test_maps.png',dpi=150);plt.show()
