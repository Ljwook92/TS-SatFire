"""Same-event retrospective temporal-window ablation; no test-label spatial correction."""
from pathlib import Path
import hashlib, json, re, warnings, platform
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform
from scipy.ndimage import distance_transform_edt, uniform_filter
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                            log_loss, precision_recall_curve, precision_score,
                            recall_score, f1_score, jaccard_score)

BANDS = {'ndvi':('FirePred',1),'evi2':('FirePred',2),
         'wind_speed':('FirePred',4),'wind_direction':('FirePred',5),
         'tmax':('FirePred',7),'erc':('FirePred',8),
         'specific_humidity':('FirePred',9),'i4':('VIIRS_Day',4)}

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False))

def defaults():
    return dict(raw_root='/home/jlc3q/data/SatFire/ts-satfire',
        event_id='US_2021_NM3323810847220210520', windows=[2,4,6], seed=20261120,
        features=['ndvi','evi2','i4','tmax','erc','specific_humidity','wind_speed','wind_direction'],
        label_nan_policy='legacy_nan_zero',
        label_note='Finite positive AF or BA is recorded fire; NaN is no recorded positive, not verified no-fire.',
        timing_note='Retrospective recorded expansion only; original product publication times not established.',
        split_fractions=[0.50,0.25], negative_keep=0.05,
        lambdas=[1e-4,1e-3,1e-2], spatial_lengths_km=[10.,30.],
        spatial_penalty_ratio=1., spatial_knots_side=5, n_knots=4,
        max_iter=1500, chunk_size=8192)

def split_dates(days, cfg):
    days=pd.DatetimeIndex(days)
    if len(days)<3*(max(cfg['windows'])+1): raise ValueError('Insufficient dates for three purged blocks')
    if not np.all(np.diff(days.values)==np.timedelta64(1,'D')): raise ValueError('Nonconsecutive dates')
    a=int(len(days)*cfg['split_fractions'][0]); b=a+int(len(days)*cfg['split_fractions'][1])
    blocks=dict(train=list(days[:a]),val=list(days[a:b]),test=list(days[b:]))
    targets={k:v[max(cfg['windows']):] for k,v in blocks.items()}
    if not all(targets.values()): raise ValueError('Empty split after maximum-window purge')
    return blocks,targets

def read(path):
    with rasterio.open(path) as ds:
        a=ds.read(masked=True).astype('float32').filled(np.nan)
        a=a*np.asarray(ds.scales,dtype='float32')[:,None,None]+np.asarray(ds.offsets,dtype='float32')[:,None,None]
        a[~np.isfinite(a)]=np.nan
        meta=dict(shape=a.shape[1:],crs=ds.crs,transform=ds.transform)
        return a,meta,ds.descriptions

def prepare(cfg, out):
    """Inventory + date-local caches. Future rasters never enter earlier feature construction."""
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    cache=out/'cache';cache.mkdir()
    root=Path(cfg['raw_root'])/cfg['event_id']; files={}
    for product in ['VIIRS_Day','FirePred']:
        for path in sorted((root/product).glob('*.tif')):
            match=re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})',path.name)
            if not match: raise ValueError(f'No date: {path}')
            key=(pd.Timestamp('-'.join(match.groups())),product)
            if key in files: raise ValueError(f'Duplicate: {key}')
            files[key]=path
    days=sorted({d for d,p in files})
    if not days: raise FileNotFoundError(f'No raw GeoTIFFs: {root}')
    blocks,targets=split_dates(days,cfg)
    reference=None; inventory=[]; file_manifest=[]; feature_names=[]
    for name in cfg['features']:
        if name not in BANDS: raise ValueError(f'Unsupported feature {name}')
        feature_names.extend([name+'_sin',name+'_cos'] if name=='wind_direction' else [name])
    for i,date in enumerate(days):
        arrays={}
        for product in ['VIIRS_Day','FirePred']:
            if (date,product) not in files: raise ValueError(f'Missing {date} {product}; no zero fill')
            path=files[date,product];a,meta,desc=read(path)
            if reference is None:
                reference=meta; cumulative=np.zeros(meta['shape'],bool); known=np.ones(meta['shape'],bool)
            if meta['shape']!=reference['shape'] or meta['crs']!=reference['crs'] or not meta['transform'].almost_equals(reference['transform']):
                raise ValueError(f'Grid mismatch: {path}')
            if len(a)<(8 if product=='VIIRS_Day' else 19): raise ValueError(f'Missing bands: {path}')
            arrays[product]=a
            stat=path.stat();file_manifest.append(dict(date=str(date.date()),product=product,path=str(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
            for j,band in enumerate(a):
                good=band[np.isfinite(band)]
                inventory.append(dict(date=str(date.date()),product=product,band=j+1,description=desc[j],
                    finite_fraction=float(np.isfinite(band).mean()),minimum=float(good.min()) if len(good) else None,
                    maximum=float(good.max()) if len(good) else None))
        values=[]
        for name in cfg['features']:
            product,band=BANDS[name];v=arrays[product][band-1]
            if name=='wind_direction': values.extend([np.sin(np.deg2rad(v)),np.cos(np.deg2rad(v))])
            else: values.append(v)
        np.save(cache/f'x_{i:03d}.npy',np.stack(values).astype('float32'))
        raw=arrays['VIIRS_Day'][[6,7]]
        hit=np.any(np.isfinite(raw)&(raw>0),axis=0)
        if cfg['label_nan_policy']=='legacy_nan_zero': valid=np.ones_like(hit)
        elif cfg['label_nan_policy']=='strict_unknown': valid=np.all(np.isfinite(raw),axis=0)
        else: raise ValueError('Invalid label policy')
        cumulative|=hit;known&=valid
        np.save(cache/f'c_{i:03d}.npy',cumulative)
        np.save(cache/f'k_{i:03d}.npy',known|cumulative)
    if reference['crs'] is None: raise ValueError('Missing CRS')
    rr,cc=np.indices(reference['shape']);tr=reference['transform']
    xx=tr.c+tr.a*(cc+.5)+tr.b*(rr+.5);yy=tr.f+tr.d*(cc+.5)+tr.e*(rr+.5)
    lon,lat=warp_transform(reference['crs'],'EPSG:4326',[float(xx.mean())],[float(yy.mean())])
    local=f'+proj=aeqd +lat_0={lat[0]} +lon_0={lon[0]} +datum=WGS84 +units=m'
    x,y=warp_transform(reference['crs'],local,xx.ravel(),yy.ravel())
    xy=np.column_stack([x,y]).astype('float32')/1000
    np.save(cache/'xy_km.npy',xy)
    manifest=pd.DataFrame(file_manifest);manifest.to_csv(out/'source_files.csv',index=False)
    pd.DataFrame(inventory).to_csv(out/'inventory.csv',index=False)
    dump(out/'config.json',cfg)
    dump(out/'grid.json',dict(shape=reference['shape'],crs=str(reference['crs']),transform=list(tr),local_crs=local))
    data=dict(cfg=cfg,out=out,cache=cache,days=days,blocks=blocks,targets=targets,meta=reference,feature_names=feature_names,xy=xy)
    rows=[]
    for split,dates in targets.items():
        for date in dates:
            idx=days.index(date); ids,y=target_rows(data,idx)
            if len(ids)==0: raise ValueError(f'No eligible pixels {date}')
            np.savez(cache/f'rows_{idx:03d}.npz',ids=ids,y=y)
            rows.append(dict(split=split,target_date=str(date.date()),input_start_TS6=str(days[idx-max(cfg['windows'])].date()),
                last_input=str(days[idx-1].date()),n=len(y),positives=int(y.sum()),
                pixel_sha256=hashlib.sha256(ids.tobytes()).hexdigest()))
    data['manifest']=pd.DataFrame(rows);data['manifest'].to_csv(out/'evaluation_manifest.csv',index=False)
    return data

def cached(data, kind, i):
    return np.load(data['cache']/f'{kind}_{i:03d}.npy',mmap_mode='r')

def target_rows(data,i):
    before=cached(data,'c',i-1);after=cached(data,'c',i)
    eligible=~before & cached(data,'k',i-1) & cached(data,'k',i)
    ids=np.flatnonzero(eligible.ravel())
    return ids,(after & ~before).ravel()[ids].astype('uint8')

def rows(data,i):
    with np.load(data['cache']/f'rows_{i:03d}.npz') as z:return z['ids'],z['y']

def features(data,i,ts,ids):
    """Only [i-ts,i) predictors and cumulative state through i-1. Never read target i."""
    if ts not in data['cfg']['windows'] or i<ts: raise ValueError('Invalid window')
    stack=np.stack([cached(data,'x',j).reshape(len(data['feature_names']),-1)[:,ids].T for j in range(i-ts,i)])
    finite=np.isfinite(stack); count=finite.sum(axis=0)
    mean=np.divide(np.nansum(stack,axis=0),count,out=np.full_like(stack[0],np.nan),where=count>0)
    x=np.concatenate([stack[-1],mean,stack[-1]-stack[0],1-finite.mean(axis=0)],axis=1)
    fire=np.asarray(cached(data,'c',i-1))
    distance=distance_transform_edt(~fire) if fire.any() else np.full(fire.shape,np.hypot(*fire.shape))
    density=uniform_filter(fire.astype('float32'),size=5,mode='constant')
    growth=fire & ~cached(data,'c',i-2)
    gdensity=uniform_filter(growth.astype('float32'),size=5,mode='constant')
    return np.column_stack([x,distance.ravel()[ids],density.ravel()[ids],gdensity.ravel()[ids]]).astype('float32')

def training_rows(data):
    cfg=data['cfg'];result=[]
    p=cfg['negative_keep']
    if not 0<p<=1: raise ValueError('negative_keep must be (0,1]')
    for date in data['targets']['train']:
        i=data['days'].index(date);ids,y=rows(data,i)
        rng=np.random.default_rng(cfg['seed']+i)
        keep=(y==1)|(rng.random(len(y))<p)
        weights=np.where(y[keep]==1,1.,1/p)
        result.append((i,ids[keep],y[keep],weights))
    return result

class AdditiveLogit:
    """Ridge-penalized additive cubic B-splines + optional jointly fitted Gaussian spatial basis.

    Mean weighted Bernoulli NLL + lambda/2*(||beta||^2 + ratio*||u||^2).
    Intercept unpenalized. RBF basis induces finite-rank Gaussian spatial prior,
    not exact Matérn kriging and not an estimated posterior uncertainty model.
    """
    def __init__(self,cfg,lam,length=None):self.cfg=cfg;self.lam=lam;self.length=length
    def fit(self,x,xy,y,w):
        if len(np.unique(y))!=2:raise ValueError('Training requires both classes')
        self.medians=np.zeros(x.shape[1]);self.active=[]
        for j in range(x.shape[1]):
            good=np.isfinite(x[:,j])
            if good.any():
                v=x[good,j];ww=w[good];order=np.argsort(v);v=v[order];ww=ww[order]
                self.medians[j]=v[np.searchsorted(np.cumsum(ww),ww.sum()/2)]
                if v[-1]>v[0]:self.active.append(j)
        if not self.active:raise ValueError('No varying training predictors')
        clean=self.clean(x)
        self.spline=SplineTransformer(n_knots=self.cfg['n_knots'],degree=3,knots='uniform',include_bias=False,extrapolation='constant')
        basis=self.spline.fit_transform(clean)
        self.scaler=StandardScaler().fit(basis,sample_weight=w)
        design=self.scaler.transform(basis)
        if self.length is not None:
            side=self.cfg['spatial_knots_side']
            gx=np.linspace(xy[:,0].min(),xy[:,0].max(),side);gy=np.linspace(xy[:,1].min(),xy[:,1].max(),side)
            self.knots=np.array(np.meshgrid(gx,gy)).reshape(2,-1).T
            r=self.rbf(xy);self.space_scaler=StandardScaler().fit(r,sample_weight=w)
            design=np.column_stack([design,self.space_scaler.transform(r)/np.sqrt(self.cfg['spatial_penalty_ratio'])])
        self.model=LogisticRegression(C=1/(self.lam*w.sum()),solver='lbfgs',max_iter=self.cfg['max_iter'],tol=1e-6)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            self.model.fit(design,y,sample_weight=w)
        if any(issubclass(v.category,ConvergenceWarning) for v in caught):raise RuntimeError('Nonconverged fit; increase max_iter or regularization')
        return self
    def clean(self,x):return np.where(np.isfinite(x),x,self.medians)[:,self.active]
    def rbf(self,xy):return np.exp(-np.sum((xy[:,None,:]-self.knots[None,:,:])**2,axis=2)/(2*self.length**2))
    def predict(self,x,xy):
        design=self.scaler.transform(self.spline.transform(self.clean(x)))
        if self.length is not None:design=np.column_stack([design,self.space_scaler.transform(self.rbf(xy))/np.sqrt(self.cfg['spatial_penalty_ratio'])])
        return self.model.predict_proba(design)[:,1]

def predict_split(data,model,ts,split):
    result=[];chunk=data['cfg']['chunk_size']
    for date in data['targets'][split]:
        i=data['days'].index(date);ids,y=rows(data,i);p=np.empty(len(y))
        # Build one day only; transform to dense spline basis in bounded batches.
        x=features(data,i,ts,ids)
        for a in range(0,len(ids),chunk):
            b=a+chunk;p[a:b]=model.predict(x[a:b],data['xy'][ids[a:b]])
        if not np.isfinite(p).all():raise ValueError('Nonfinite probability')
        result.append(dict(date=str(date.date()),ids=ids,y=y,p=p))
    return result

def metrics(y,p,threshold):
    q=p>=threshold
    return dict(n=len(y),positives=int(y.sum()),prevalence=float(np.mean(y)),
        ap=float(average_precision_score(y,p)) if y.any() else None,
        roc_auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
        brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p,labels=[0,1])),
        precision=float(precision_score(y,q,zero_division=0)),recall=float(recall_score(y,q,zero_division=0)),
        f1=float(f1_score(y,q,zero_division=0)),iou=float(jaccard_score(y,q,zero_division=0)))

def choose_threshold(y,p):
    if len(np.unique(y))!=2:raise ValueError('Validation must contain both classes; do not choose new dates based on labels')
    pr,rc,th=precision_recall_curve(y,p)
    f=2*pr[:-1]*rc[:-1]/np.maximum(pr[:-1]+rc[:-1],1e-15)
    return float(th[np.flatnonzero(f==f.max())[-1]])

def train_select(data):
    """Does not access test target rows, features, or labels for fitting/selection."""
    import joblib, sklearn, scipy
    cfg=data['cfg'];out=data['out'];sample=training_rows(data)
    y=np.concatenate([r[2] for r in sample]);w=np.concatenate([r[3] for r in sample])
    xy=np.concatenate([data['xy'][r[1]] for r in sample])
    dump(out/'training_sample.json',dict(n=len(y),positives=int(y.sum()),weighted_prevalence=float(np.average(y,weights=w)),negative_keep=cfg['negative_keep']))
    np.savez(out/'training_rows.npz',day=np.concatenate([np.full(len(r[1]),r[0]) for r in sample]),
             pixel=np.concatenate([r[1] for r in sample]),y=y,weights=w)
    tuning=[];selected={}; selections=[]
    for ts in cfg['windows']:
        x=np.concatenate([features(data,r[0],ts,r[1]) for r in sample])
        for kind in ['GAM','SpatialGAM']:
            best=None
            for length in ([None] if kind=='GAM' else cfg['spatial_lengths_km']):
                for lam in cfg['lambdas']:
                    print(f'TS={ts} {kind} lambda={lam:g} spatial_length={length}',flush=True)
                    try:
                        model=AdditiveLogit(cfg,lam,length).fit(x,xy,y,w)
                    except RuntimeError as exc:
                        tuning.append(dict(ts=ts,model=kind,lam=lam,length_km=length,status=str(exc),val_log_loss=None));continue
                    pred=predict_split(data,model,ts,'val')
                    vy=np.concatenate([r['y'] for r in pred]);vp=np.concatenate([r['p'] for r in pred])
                    loss=float(log_loss(vy,vp,labels=[0,1]))
                    tuning.append(dict(ts=ts,model=kind,lam=lam,length_km=length,status='converged',val_log_loss=loss))
                    if best is None or loss<best[0]:best=(loss,model,vy,vp)
            if best is None:raise RuntimeError(f'No converged candidate {ts} {kind}')
            loss,model,vy,vp=best;threshold=choose_threshold(vy,vp)
            key=f'TS{ts}_{kind}';selected[key]=(model,threshold,ts)
            selections.append(dict(key=key,ts=ts,model=kind,lam=model.lam,length_km=model.length,threshold=threshold,val_log_loss=loss))
            joblib.dump(model,out/f'{key}.joblib')
            pd.DataFrame(tuning).to_csv(out/'tuning.csv',index=False)
    dump(out/'selection_before_test.json',selections)
    dump(out/'versions.json',dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__,rasterio=rasterio.__version__))
    return selected

def evaluate(data,selected):
    """Evaluate fixed models/thresholds on full common test masks; no refitting."""
    out=data['out'];metrics_rows=[];daily=[];calibration=[]
    model_dir=out/'predictions';model_dir.mkdir(exist_ok=True)
    manifest=data['manifest'].query("split == 'test'").set_index('target_date')
    for key,(model,threshold,ts) in selected.items():
        pred=predict_split(data,model,ts,'test')
        y=np.concatenate([r['y'] for r in pred]);p=np.concatenate([r['p'] for r in pred])
        metrics_rows.append(dict(key=key,ts=ts,model='SpatialGAM' if model.length is not None else 'GAM',threshold=threshold,**metrics(y,p,threshold)))
        for r in pred:
            expected=manifest.loc[r['date']]
            assert len(r['ids'])==expected['n'] and hashlib.sha256(r['ids'].tobytes()).hexdigest()==expected['pixel_sha256']
            np.savez_compressed(model_dir/f"{key}_{r['date']}.npz",**r)
            daily.append(dict(key=key,date=r['date'],**metrics(r['y'],r['p'],threshold)))
        edges=np.unique(np.quantile(p,np.linspace(0,1,11)))
        group=np.searchsorted(edges[1:-1],p,side='right')
        for b in np.unique(group):
            keep=group==b;calibration.append(dict(key=key,bin=int(b),n=int(keep.sum()),mean_prediction=float(p[keep].mean()),observed_fraction=float(y[keep].mean())))
    table=pd.DataFrame(metrics_rows);table.to_csv(out/'test_metrics.csv',index=False)
    pd.DataFrame(daily).to_csv(out/'test_metrics_by_date.csv',index=False)
    pd.DataFrame(calibration).to_csv(out/'calibration.csv',index=False)
    # Population prevalence baseline from ALL training eligible labels (no sampled-prevalence bias).
    train=data['manifest'].query("split == 'train'");rate=float(train.positives.sum()/train.n.sum())
    base=np.full(len(y),rate);dump(out/'constant_baseline.json',dict(train_prevalence=rate,**metrics(y,base,.5)))
    diff=[]
    for ts in data['cfg']['windows']:
        a=table.query("ts == @ts and model == 'GAM'").iloc[0];b=table.query("ts == @ts and model == 'SpatialGAM'").iloc[0]
        diff.append(dict(ts=ts,delta_AP=b.ap-a.ap,delta_Brier=b.brier-a.brier,delta_log_loss=b.log_loss-a.log_loss,delta_F1=b.f1-a.f1))
    pd.DataFrame(diff).to_csv(out/'paired_model_differences.csv',index=False)
    return table

def plots(data):
    import matplotlib.pyplot as plt
    out=data['out'];m=pd.read_csv(out/'test_metrics.csv')
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,metric in zip(axes,['ap','brier','f1']):
        for kind in ['GAM','SpatialGAM']:
            t=m[m.model==kind];ax.plot(t.ts,t[metric],'o-',label=kind)
        ax.set(xlabel='Input days (TS)',ylabel=metric,xticks=data['cfg']['windows'])
    axes[0].legend();fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.show()
    date=str(data['targets']['test'][-1].date());fig,axes=plt.subplots(2,4,figsize=(14,8),constrained_layout=True)
    cmap=plt.get_cmap('magma').copy();cmap.set_bad('#d9d9d9')
    last=None
    for row,kind in enumerate(['GAM','SpatialGAM']):
        for col,ts in enumerate(data['cfg']['windows']):
            with np.load(out/'predictions'/f'TS{ts}_{kind}_{date}.npz') as z:
                a=np.full(np.prod(data['meta']['shape']),np.nan);a[z['ids']]=z['p'];last=(z['ids'].copy(),z['y'].copy())
            im=axes[row,col].imshow(a.reshape(data['meta']['shape']),vmin=0,vmax=1,cmap=cmap);axes[row,col].set_title(f'{kind} TS={ts}')
    a=np.full(np.prod(data['meta']['shape']),np.nan);a[last[0]]=last[1]
    axes[0,3].imshow(a.reshape(data['meta']['shape']),vmin=0,vmax=1,cmap=cmap);axes[0,3].set_title('Observed new record')
    axes[1,3].axis('off');cax=axes[1,3].inset_axes([.2,.1,.1,.8]);fig.colorbar(im,cax=cax,label='Probability (shared 0–1)')
    for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle(f'{date}: common eligible pixels; gray = excluded');fig.savefig(out/'test_maps.png',dpi=160);plt.show()
