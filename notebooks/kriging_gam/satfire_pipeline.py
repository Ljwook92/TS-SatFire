"""Single-event exploratory FP pipeline. No dependency on the legacy repository."""
from pathlib import Path
from datetime import timedelta
import re, json, hashlib, warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform as warp_transform
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt, uniform_filter
from pykrige.ok import OrdinaryKriging
from pygam import LogisticGAM, s
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             log_loss, precision_score, recall_score, f1_score, jaccard_score)
import matplotlib.pyplot as plt

# One-based RAW GeoTIFF band indices, NOT the packed 27-channel model stack.
AUX = ['ndvi','evi2','precip','wind_speed','wind_direction','tmin','tmax','erc',
       'specific_humidity','slope','aspect','elevation','pdsi','landcover',
       'forecast_precip','forecast_wind_speed','forecast_wind_direction',
       'forecast_temperature','forecast_specific_humidity']
SPECS = {n: ('FirePred', i+1, 'continuous') for i,n in enumerate(AUX)}
SPECS.update({n:('VIIRS_Day',i+1,'continuous') for i,n in enumerate(['i1','i2','i3','i4','i5','m11'])})
for n in ['wind_direction','forecast_wind_direction','aspect']:
    SPECS[n] = ('FirePred',AUX.index(n)+1,'circular')
SPECS['landcover'] = ('FirePred',14,'categorical')


def dump(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,default=str,allow_nan=False))


def date_of(path):
    m=re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})',Path(path).name)
    if not m: raise ValueError(f'No calendar date: {path}')
    return pd.Timestamp('-'.join(m.groups()))


def catalog(cfg):
    event=Path(cfg['raw_root'])/cfg['event_id']
    if not event.is_dir(): raise FileNotFoundError(f'Actual event folder missing: {event}; no synthetic fallback.')
    files={}
    for folder in ['VIIRS_Day','FirePred']:
        for p in sorted((event/folder).glob('*.tif')):
            key=(date_of(p),folder)
            if key in files: raise ValueError(f'Duplicate date/product: {key}')
            files[key]=p
    days=sorted(d for d,k in files if k=='VIIRS_Day')
    if len(days)<12: raise ValueError('Need at least 12 daily files for chronological train/validation/test with purge.')
    return files,days


def read_tif(path, crop=None):
    with rasterio.open(path) as ds:
        win=None if crop is None else Window(crop[1],crop[0],crop[3],crop[2])
        if win is not None and (win.col_off+win.width>ds.width or win.row_off+win.height>ds.height):
            raise ValueError(f'Crop outside raster: {path}')
        a=ds.read(window=win,masked=True).astype('float64').filled(np.nan)
        # Apply documented GDAL scale/offset; never guess NDVI scale from value range.
        scales=np.array(ds.scales); offsets=np.array(ds.offsets)
        a=a*scales[:,None,None]+offsets[:,None,None]
        a[~np.isfinite(a)]=np.nan
        tr=ds.transform if win is None else ds.window_transform(win)
        meta={'shape':a.shape[1:],'transform':tr,'crs':ds.crs,
              'descriptions':ds.descriptions,'nodata':ds.nodata,'scales':scales.tolist(),
              'offsets':offsets.tolist(),'count':ds.count}
    return a,meta


def same_grid(a,b):
    return a['shape']==b['shape'] and a['crs']==b['crs'] and a['transform'].almost_equals(b['transform'])


def audit(cfg,out):
    files,days=catalog(cfg); rows=[]; reference=None; issues=[]
    for (day,kind),path in sorted(files.items()):
        a,m=read_tif(path,cfg['crop'])
        if reference is None: reference=m
        aligned=same_grid(reference,m)
        if not aligned: issues.append(f'Grid mismatch {path}')
        needed=8 if kind=='VIIRS_Day' else 19
        if a.shape[0]<needed: issues.append(f'Missing bands {path}')
        for i,v in enumerate(a):
            good=v[np.isfinite(v)]
            rows.append(dict(date=str(day.date()),product=kind,path=str(path),band=i+1,
                description=m['descriptions'][i],height=v.shape[0],width=v.shape[1],
                crs=str(m['crs']),nodata=str(m['nodata']),scale=m['scales'][i],offset=m['offsets'][i],
                valid_fraction=float(np.isfinite(v).mean()),min=float(good.min()) if len(good) else None,
                max=float(good.max()) if len(good) else None,aligned=aligned))
    table=pd.DataFrame(rows);table.to_csv(out/'inventory.csv',index=False)
    dump(out/'audit.json',{'issues':issues,'dates':[str(d.date()) for d in days],
        'missing_FirePred':[str(d.date()) for d in days if (d,'FirePred') not in files],
        'day_gaps':[str(d.date()) for d,e in zip(days,days[1:]) if (e-d).days!=1],
        'note':'Raster nodata is not a cloud QA mask. Zero-filled legacy tensors cannot reveal original missingness.'})
    return files,days,reference,table


def coordinates(meta):
    rr,cc=np.indices(meta['shape']);tr=meta['transform']
    x=tr.c+tr.a*(cc+.5)+tr.b*(rr+.5);y=tr.f+tr.d*(cc+.5)+tr.e*(rr+.5)
    if meta['crs'] is None: raise ValueError('Missing CRS; cannot calculate kriging distances.')
    # Convert all grids to a local azimuthal equidistant metric CRS.
    lon,lat=warp_transform(meta['crs'],'EPSG:4326',[float(x.mean())],[float(y.mean())])
    dst=f'+proj=aeqd +lat_0={lat[0]} +lon_0={lon[0]} +datum=WGS84 +units=m'
    xx,yy=warp_transform(meta['crs'],dst,x.ravel().tolist(),y.ravel().tolist())
    return np.column_stack([xx,yy])/1000.0  # kilometres


def fill_kriging(a,xy,cfg,seed):
    """Snapshot-only ordinary kriging. Unfillable pixels stay NaN; originals stay exact."""
    flat=a.ravel();good=np.flatnonzero(np.isfinite(flat));missing=np.flatnonzero(~np.isfinite(flat))
    result=flat.copy();variance=np.full(flat.shape,np.nan);variance[good]=0
    info={'n_observed':len(good),'n_missing':len(missing),'n_filled':0,'status':'no_gap'}
    if not len(missing):return result.reshape(a.shape),variance.reshape(a.shape),info
    if len(good)<cfg['min_support']:
        info['status']='insufficient_support';return a.copy(),variance.reshape(a.shape),info
    rng=np.random.default_rng(seed)
    support=rng.choice(good,min(len(good),cfg['max_support']),replace=False)
    dist,_=cKDTree(xy[support]).query(xy[missing],k=min(cfg['min_support'],len(support)))
    eligible=missing[np.max(np.atleast_2d(dist),axis=1)<=cfg['max_distance_km']]
    info['n_support']=len(support)
    if len(eligible)==0:
        info['status']='outside_support';return a.copy(),variance.reshape(a.shape),info
    z=flat[support]
    if np.ptp(z)<1e-10:
        # A constant field does not identify a variogram; do not manufacture kriging variance.
        info['status']='constant_field_unfilled';return a.copy(),variance.reshape(a.shape),info
    try:
        ok=OrdinaryKriging(xy[support,0],xy[support,1],z,variogram_model='spherical',
                          nlags=6,verbose=False,enable_plotting=False,pseudo_inv=True)
        for ids in np.array_split(eligible,max(1,int(np.ceil(len(eligible)/2048)))):
            pred,var=ok.execute('points',xy[ids,0],xy[ids,1],backend='loop',
                                n_closest_points=min(cfg['neighbors'],len(support)))
            result[ids]=np.asarray(pred);variance[ids]=np.maximum(np.asarray(var),0)
        info.update(status='kriged',n_filled=int(np.isfinite(result[missing]).sum()),
                    variogram_parameters=ok.variogram_model_parameters.tolist())
    except (ValueError,np.linalg.LinAlgError) as exc:
        info.update(status='failed',error=str(exc))
    return result.reshape(a.shape),variance.reshape(a.shape),info


def mask_blocks(a,seed):
    rng=np.random.default_rng(seed);h,w=a.shape;mask=np.zeros(a.shape,bool)
    for fraction in [.07,.14,.23]:
        size=max(2,int(min(h,w)*fraction));r=rng.integers(0,max(1,h-size));c=rng.integers(0,max(1,w-size))
        mask[r:r+size,c:c+size]=True
    return mask & np.isfinite(a)


def restoration_check(a,xy,cfg,seed):
    hold=mask_blocks(a,seed);masked=a.copy();masked[hold]=np.nan
    filled,var,info=fill_kriging(masked,xy,cfg,seed)
    use=hold & np.isfinite(filled);n=int(use.sum())
    row={'n_hidden':int(hold.sum()),'n_scored':n,'coverage_fraction':n/max(1,int(hold.sum())),'status':info['status']}
    if n:
        e=filled[use]-a[use];sd=np.sqrt(var[use])
        row.update(mae=float(abs(e).mean()),rmse=float(np.sqrt((e*e).mean())),bias=float(e.mean()),
                   interval95_coverage=float((abs(e)<=1.96*sd).mean()),interval95_width=float((3.92*sd).mean()))
        good=np.flatnonzero(np.isfinite(masked.ravel()));q=np.flatnonzero(use.ravel())
        d,ix=cKDTree(xy[good]).query(xy[q],k=min(8,len(good)))
        weights=1/np.maximum(d,1e-9)**2
        idw=(weights*masked.ravel()[good[ix]]).sum(axis=1)/weights.sum(axis=1)
        row['idw_mae_same_pixels']=float(abs(idw-a[use]).mean())
        row['idw_rmse_same_pixels']=float(np.sqrt(((idw-a[use])**2).mean()))
    return row


def labels(files,days,reference,cfg):
    """Three-valued cumulative AF|BA. A known positive persists; unknown never becomes negative."""
    positive=np.zeros(reference['shape'],bool);known=np.ones(reference['shape'],bool);out={}
    last=None
    for day in days:
        if last is not None and (day-last).days!=1: known[:]=False
        a,m=read_tif(files[day,'VIIRS_Day'],cfg['crop']);last=day
        if not same_grid(reference,m) or a.shape[0]<8:
            known[:]=False;out[day]=(positive.copy(),(known|positive).copy());continue
        raw=a[[6,7]]
        if cfg['label_nan_policy']=='legacy_nan_zero':
            valid=np.ones(reference['shape'],bool);hit=np.any(np.nan_to_num(raw,nan=0)>0,axis=0)
        elif cfg['label_nan_policy']=='strict_unknown':
            valid=np.all(np.isfinite(raw),axis=0);hit=np.any(np.isfinite(raw)&(raw>0),axis=0)
        else: raise ValueError('Unknown label policy')
        positive|=hit;known&=valid;out[day]=(positive.copy(),(known|positive).copy())
    return out


def split_windows(days):
    windows=[(a,b,c) for a,b,c in zip(days,days[1:],days[2:]) if (b-a).days==1 and (c-b).days==1]
    n=len(windows);a=int(.55*n);b=int(.78*n)
    # Two windows purged at each boundary: no raster date shared across splits.
    groups={'train':windows[:a], 'val':windows[a+2:b], 'test':windows[b+2:]}
    if min(map(len,groups.values()))<1: raise ValueError('Too few continuous windows after 2-window purge.')
    for l,r in [('train','val'),('val','test')]:
        assert max(x[2] for x in groups[l]) < min(x[0] for x in groups[r])
    return groups


def feature_day(day,files,reference,xy,cfg,out,diagnose=False):
    names=[];raws=[];fills=[];logs=[];checks=[]
    products={}
    for name in cfg['features']:
        product,band,kind=SPECS[name]
        if product not in products:
            p=files.get((day,product))
            if p is None:products[product]=None
            else:
                arr,meta=read_tif(p,cfg['crop']);products[product]=(arr,meta) if same_grid(reference,meta) else None
        loaded=products[product]
        if loaded is None or loaded[0].shape[0]<band:
            a=np.full(reference['shape'],np.nan)
        else:a=loaded[0][band-1].copy()
        # User-verified scale and valid range. No implicit zero->missing or cloud inference.
        rule=cfg['value_rules'].get(name,{})
        a=a*rule.get('scale',1)+rule.get('offset',0)
        for sentinel in rule.get('invalid_values',[]):a[a==sentinel]=np.nan
        if 'range' in rule:
            lo,hi=rule['range'];a[(a<lo)|(a>hi)]=np.nan
        if kind=='categorical':
            raise ValueError('Categorical bands need a separately validated encoding; do not spline/krige landcover IDs.')
        if kind=='circular':
            components=[(name+'_sin',np.sin(np.deg2rad(a))),(name+'_cos',np.cos(np.deg2rad(a)))]
        else:components=[(name,a)]
        for component,z in components:
            filled,var,log=fill_kriging(z,xy,cfg['kriging'],cfg['seed']+day.dayofyear)
            if 'range' in rule and kind=='continuous':
                lo,hi=rule['range'];bad=(filled<lo)|(filled>hi);filled[bad]=np.nan;var[bad]=np.nan
            logs.append(dict(date=str(day.date()),feature=component,**log))
            if diagnose:
                checks.append(dict(date=str(day.date()),feature=component,
                    **restoration_check(z,xy,cfg['kriging'],cfg['seed']+band)))
            names.append(component);raws.append(z);fills.append(filled)
            save_raster(out/f'{day.date()}_{component}_restoration.tif',
                        np.stack([filled,var,(~np.isfinite(z)).astype(float),
                                  ((~np.isfinite(z))&np.isfinite(filled)).astype(float)]),reference,
                        ['value','conditional_kriging_variance','input_missing','filled'])
    return names,np.stack(raws),np.stack(fills),logs,checks


def save_raster(path,a,meta,descriptions):
    if a.ndim==2:a=a[None]
    with rasterio.open(path,'w',driver='GTiff',height=a.shape[1],width=a.shape[2],count=len(a),
          dtype='float32',crs=meta['crs'],transform=meta['transform'],nodata=np.nan,compress='deflate') as ds:
        ds.write(a.astype('float32'))
        for i,name in enumerate(descriptions,1):ds.set_band_description(i,name)


def build_tables(files,days,meta,cfg,out):
    xy=coordinates(meta);states=labels(files,days,meta,cfg);groups=split_windows(days)
    diagnostic_days={g[0][1] for g in groups.values()}
    needed=sorted({d for group in groups.values() for a,b,c in group for d in [a,b]})
    cache={};logs=[];checks=[]
    for day in needed:
        names,raw,filled,log,check=feature_day(day,files,meta,xy,cfg,out,day in diagnostic_days)
        cache[day]=(raw,filled);logs+=log;checks+=check
        print('Features',day.date(),flush=True)
    pd.DataFrame(logs).to_csv(out/'restoration_log.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'restoration_validation.csv',index=False)
    tables={};splits=[];feature_names=[f'{n}_{lag}' for lag in ['tm1','t'] for n in names]+['distance_pixels','fire_fraction_5x5','recent_growth_5x5']
    for split,windows in groups.items():
        chunks=[]
        for a,b,c in windows:
            prev,kprev=states[a];cur,kcur=states[b];future,kfuture=states[c]
            if not cur.any():continue
            # Growth target only where current and future cumulative states are known.
            eligible=(~cur)&kcur&kfuture
            geometry=np.stack([distance_transform_edt(~cur),
                uniform_filter(cur.astype(float),size=5),
                uniform_filter((cur&~prev&kprev).astype(float),size=5)])
            raw=np.concatenate([cache[a][0],cache[b][0],geometry]).reshape(len(feature_names),-1).T
            filled=np.concatenate([cache[a][1],cache[b][1],geometry]).reshape(len(feature_names),-1).T
            ids=np.flatnonzero(eligible.ravel());target=(future&~cur).ravel()[ids].astype(int)
            # Fixed-rate uniform sampling, independent of class; validation/test keep all eligible pixels.
            if split=='train':
                keep=np.random.default_rng(cfg['seed']+b.dayofyear).random(len(ids))<cfg['train_pixel_fraction']
                ids=ids[keep];target=target[keep]
            frame=pd.DataFrame(filled[ids],columns=feature_names)
            for j,n in enumerate(feature_names):frame['raw__'+n]=raw[ids,j]
            frame['y']=target;frame['pixel']=ids;frame['date']=str(b.date());frame['target_date']=str(c.date())
            frame['imputed_fraction']=np.mean(~np.isfinite(raw[ids,:-3]),axis=1)
            chunks.append(frame)
            splits.append(dict(split=split,tm1=str(a.date()),t=str(b.date()),target=str(c.date()),
                               n=len(ids),positives=int(target.sum()),known_fraction=float(eligible.mean())))
        if not chunks:raise ValueError(f'No eligible windows: {split}. Inspect NaN label semantics; never silently set unknown to zero.')
        tables[split]=pd.concat(chunks,ignore_index=True)
        if tables[split].empty: raise ValueError(f'No evaluable pixels in {split}; inspect label validity and sampling.')
    pd.DataFrame(splits).to_csv(out/'split_manifest.csv',index=False)
    dump(out/'feature_names.json',feature_names)
    return tables,feature_names,states


def scores(y,p,threshold):
    y=np.asarray(y);p=np.asarray(p);binary=p>=threshold
    return dict(n=len(y),positives=int(y.sum()),prevalence=float(y.mean()),
        average_precision=float(average_precision_score(y,p)) if y.sum()>0 else None,
        roc_auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
        brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p,labels=[0,1])),
        precision=float(precision_score(y,binary,zero_division=0)),recall=float(recall_score(y,binary,zero_division=0)),
        f1=float(f1_score(y,binary,zero_division=0)),iou=float(jaccard_score(y,binary,zero_division=0)))


def fit_evaluate(tables,names,cfg,out):
    for split in ['train','val']:
        if tables[split].y.nunique()!=2:raise ValueError(f'{split} lacks both classes; use additional training events, not a new test split chosen for better scores.')
    records=[];models={};test=tables['test'];predictions=pd.DataFrame()
    for variant in ['median_gam','kriging_logistic','kriging_gam']:
        cols=[('raw__'+n if variant=='median_gam' else n) for n in names]
        # Explicitly drop training-all-missing or constant predictors; never infer from test values.
        active=[col for col in cols if tables['train'][col].dropna().nunique()>1]
        if not active:raise ValueError('No variable training predictors.')
        imp=SimpleImputer(strategy='median',add_indicator=True)
        X={};X['train']=imp.fit_transform(tables['train'][active])
        for split in ['val','test']:X[split]=imp.transform(tables[split][active])
        scale=StandardScaler().fit(X['train']);X={k:scale.transform(v) for k,v in X.items()}
        y=tables['train'].y.to_numpy();candidates=[]
        for lam in cfg['lambdas']:
            if variant=='kriging_logistic':model=LogisticRegression(C=1/lam,max_iter=1000)
            else:
                terms=s(0,n_splines=cfg['n_splines'])
                for j in range(1,X['train'].shape[1]):terms+=s(j,n_splines=cfg['n_splines'])
                model=LogisticGAM(terms,lam=lam,max_iter=cfg['max_iter'])
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always');model.fit(X['train'],y)
            if hasattr(model,'logs_') and model.logs_.get('diffs',[]) and model.logs_['diffs'][-1]>model.tol:
                raise RuntimeError(f'{variant}, lambda={lam}: GAM did not converge. Review training only.')
            pv=prob(model,X['val']);loss=log_loss(tables['val'].y,pv,labels=[0,1])
            candidates.append((loss,lam,model,[str(w.message) for w in ws]))
        loss,lam,model,ws=min(candidates,key=lambda v:v[0]);pv=prob(model,X['val'])
        grid=np.linspace(.01,.99,99);threshold=float(grid[np.argmax([f1_score(tables['val'].y,pv>=v,zero_division=0) for v in grid])])
        pt=prob(model,X['test']);predictions[variant]=pt
        records.append(dict(model=variant,split='test_pooled',lambda_=lam,threshold=threshold,**scores(test.y,pt,threshold)))
        for date,ix in test.groupby('date').groups.items():
            records.append(dict(model=variant,split='test_day',date=date,threshold=threshold,**scores(test.loc[ix,'y'],pt[ix],threshold)))
        for group,mask in [('observed_inputs',test.imputed_fraction==0),('missing_inputs',test.imputed_fraction>0)]:
            if mask.any():records.append(dict(model=variant,split=group,threshold=threshold,**scores(test.loc[mask,'y'],pt[mask],threshold)))
        dump(out/f'{variant}_selection.json',{'lambda':lam,'threshold':threshold,'active_columns':active,
           'warnings':ws,'val_log_loss':loss,'candidate_losses':[(l,la) for l,la,_,_ in candidates],
           'missing_handling':'train median fallback plus indicators after kriging; no claim all gaps restored'})
        import pickle
        with (out/f'{variant}.pkl').open('wb') as f:pickle.dump((imp,scale,model,active),f)
        models[variant]=(model,threshold)
    # Constant training prevalence: prevalence-aware probability baseline.
    pt=np.full(len(test),tables['train'].y.mean())
    records.append(dict(model='training_prevalence',split='test_pooled',threshold=.5,**scores(test.y,pt,.5)))
    predictions=pd.concat([test[['date','target_date','pixel','y','imputed_fraction']].reset_index(drop=True),predictions],axis=1)
    predictions.to_csv(out/'test_predictions.csv.gz',index=False,compression='gzip')
    pd.DataFrame(records).to_csv(out/'metrics.csv',index=False)
    return predictions,pd.DataFrame(records),models


def prob(model,x):
    if isinstance(model,LogisticRegression):return model.predict_proba(x)[:,1]
    # Batched to bound GAM model-matrix memory on full-grid inference.
    return np.concatenate([model.predict_proba(b) for b in np.array_split(x,max(1,int(np.ceil(len(x)/20000))))])


def plot_results(pred,metrics,states,meta,models,out):
    for day,g in pred.groupby('date'):
        shape=meta['shape'];p=np.full(np.prod(shape),np.nan);y=p.copy();missing=p.copy()
        ids=g.pixel.to_numpy();p[ids]=g.kriging_gam;y[ids]=g.y;missing[ids]=g.imputed_fraction
        p=p.reshape(shape);y=y.reshape(shape);missing=missing.reshape(shape)
        cur,known=states[pd.Timestamp(day)];threshold=models['kriging_gam'][1]
        err=np.full(shape,np.nan);ok=np.isfinite(y)
        err[ok]=((p[ok]>=threshold).astype(int)+2*y[ok]) # 0 TN,1 FP,2 FN,3 TP
        save_raster(out/f'{day}_prediction.tif',np.stack([p,y,err,missing]),meta,
                    ['probability','new_recorded_fire_target','error_0TN_1FP_2FN_3TP','input_missing_fraction'])
        fig,ax=plt.subplots(1,5,figsize=(19,4))
        for a,v,title in zip(ax,[cur.astype(float),p,y,err,missing],
                ['Current cumulative record',f'GAM P(new record): {day}', 'Observed next day','0 TN / 1 FP / 2 FN / 3 TP','Input missing fraction']):
            im=a.imshow(v,vmin=0,vmax=3 if title.startswith('0 TN') else 1);a.set_title(title);a.axis('off')
            cb=fig.colorbar(im,ax=a,shrink=.65)
            if title.startswith('0 TN'): cb.set_ticks([0,1,2,3]);cb.set_ticklabels(['TN','FP','FN','TP'])
        fig.tight_layout();fig.savefig(out/f'{day}_panels.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5));ax.plot([0,1],[0,1],'k--')
    rows=[]
    for model in ['median_gam','kriging_logistic','kriging_gam']:
        # Quantile bins with sample counts saved, not just a smooth calibration line.
        bins=pd.qcut(pred[model],q=10,duplicates='drop')
        t=pred.assign(bin=bins).groupby('bin',observed=True).agg(predicted=(model,'mean'),observed=('y','mean'),n=('y','size'))
        ax.plot(t.predicted,t.observed,'o-',label=model)
        t['model']=model;rows.append(t.reset_index().astype({'bin':str}))
    pd.concat(rows).to_csv(out/'calibration_bins.csv',index=False)
    ax.set(xlabel='Mean predicted probability',ylabel='Observed frequency');ax.legend();fig.tight_layout();fig.savefig(out/'calibration.png',dpi=140);plt.close(fig)
