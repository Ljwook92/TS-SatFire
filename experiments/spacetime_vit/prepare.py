"""Build causal 11-channel AF/BA raster streams and the official event split."""
import argparse
from datetime import date,timedelta
from pathlib import Path
import json,os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.warp import reproject,Resampling
from pyproj import Transformer,CRS
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter

ROOT=Path(__file__).resolve().parent
ASSETS=ROOT/'assets'
TASKS=('AF','BA')

def bands(ds,w,grid):
    if (ds.window_transform(w),ds.crs)==grid:
        return ds.read(window=w,masked=True).astype('f4').filled(np.nan)
    out=np.full((ds.count,256,256),np.nan,'f4')
    for i,arr in enumerate(out,1):
        reproject(rasterio.band(ds,i),arr,src_nodata=ds.nodata if ds.nodata is not None else np.nan,
                  dst_transform=grid[0],dst_crs=grid[1],dst_nodata=np.nan,
                  resampling=Resampling.nearest if i>=7 else Resampling.bilinear)
    return out

def spectrum(a):
    red,nir,sw,i4,i5,m11=a[:6]
    def ratio(a,b):
        return np.divide(a-b,a+b,out=np.full_like(a,np.nan),where=(a>=0)&(b>=0)&(a+b>0))
    return np.stack([ratio(nir,red),ratio(nir,m11),np.where(nir>=0,nir,np.nan),
                     ratio(sw,m11),np.where((sw>=0)&(m11>=0)&(sw+m11>0),10*m11-9.8*sw,np.nan)])

def hotspots(grid,points):
    row,col=np.indices((256,256))
    x,y=grid[0]*(col+.5,row+.5)
    lon,lat=Transformer.from_crs(grid[1],4326,always_xy=True).transform(x,y)
    proj=Transformer.from_crs(4326,CRS.from_proj4(f'+proj=aeqd +lat_0={lat.mean()} +lon_0={lon.mean()} +datum=WGS84 +units=m'),always_xy=True)
    xy=np.column_stack([v.ravel() for v in proj.transform(lon,lat)])/1000
    trees={d:cKDTree(np.column_stack(proj.transform(p.longitude.to_numpy(),p.latitude.to_numpy()))/1000)
           for d,p in points.groupby('acq_date')}
    def lookup(day):
        tree=trees.get(str(day-timedelta(days=1)))
        if tree is None:return np.full((256,256),-50,'f4'),np.zeros((256,256),'f4')
        dist=np.minimum(50,tree.query(xy)[0])
        density=tree.query_ball_point(xy,2,return_length=True)/(4*np.pi)
        return -dist.reshape(256,256).astype('f4'),density.reshape(256,256).astype('f4')
    return lookup

def event_rows(raw,out,r,points,thresholds,limit):
    files=sorted((raw/r.event/'VIIRS_Day').glob('[0-9]*.tif'))
    if limit:files=files[:limit]
    if not files:
        if r.role=='train':return [],None
        raise FileNotFoundError(raw/r.event/'VIIRS_Day')
    dates=[date.fromisoformat(f.name[:10]) for f in files]
    assert len(set(dates))==len(dates)
    with rasterio.open(files[0]) as ds:
        w=Window(ds.width//2-128,ds.height//2-128,256,256) if r.role=='test_af' else Window(128,128,256,256)
        grid=(ds.window_transform(w),ds.crs)
        anchor=spectrum(bands(ds,w,grid))
    lookup=hotspots(grid,points[points.event==r.event])
    accum_af=np.zeros((256,256),bool); burn=accum_af.copy()
    peak=np.full((11,256,256),np.nan,'f4')
    dest=out/r.event
    dest.mkdir(exist_ok=True)
    (dest/'grid.json').write_text(json.dumps({'affine':list(grid[0])[:6],'crs':grid[1].to_wkt()}))
    rows=[]; moments=np.zeros((2,3,11))
    for k,(f,day) in enumerate(zip(files,dates)):
        with rasterio.open(f) as ds:
            a=bands(ds,w,grid)
        af=a[6]>0 if len(a)>6 else np.zeros_like(burn)
        perim=a[7] if len(a)>7 else np.full(burn.shape,np.nan,'f4')
        accum_af|=af
        burn|=af|(perim>0)
        if k==0:continue
        i4,i5=a[3:5]; ok=np.isfinite(i4)
        den=uniform_filter(ok.astype('f4'),7)
        local=np.divide(uniform_filter(np.where(ok,i4,0),7),den,out=np.full_like(i4,np.nan),where=den>.25)
        x=np.concatenate([np.stack([i4,i5,i4-i5,i4-local]),anchor-spectrum(a),np.stack(lookup(day))])
        valid=np.isfinite(x).all(0);x[:,~valid]=np.nan
        peak=np.fmax(peak,x); observed=np.isfinite(peak[0])
        yf=af;yb=burn; kf=valid.copy();kb=observed.copy()
        if r.role=='test_af':
            nightpath=str(f).replace('VIIRS_Day','VIIRS_Night')
            with rasterio.open(nightpath) as ds:
                assert (ds.window_transform(w),ds.crs)==grid
                n=ds.read(window=w)
            tn,nn=thresholds[r.event];td=tn[k];nt=nn[k]
            yf=(i4>td[0])|(i5>td[1])|(n[-2]>nt[0])|(n[-1]>nt[1])
            kb[:]=False
        elif r.role=='test_ba':
            kf[:]=False
            if int(r.label_sel)==1:yb=accum_af
            else:
                yb=perim>0;kb&=np.isfinite(perim)&(perim!=-1)
        elif len(a)<7:
            kf[:]=False
            if len(a)<8:kb[:]=False
        targets=np.stack([yf,yb,kf,kb])
        if r.role=='train':
            for j,arr in enumerate((x,peak)):
                sample=arr[:,::8,::8].astype('f8')
                moments[j,0]+=np.isfinite(sample).sum((1,2))
                moments[j,1]+=np.nansum(sample,(1,2))
                moments[j,2]+=np.nansum(sample**2,(1,2))
        path=dest/f'{day}.npz'
        np.savez(path,AF=x.astype('f2'),BA=peak.astype('f2'),labels=np.packbits(targets,axis=-1))
        rows.append((r.event,str(day),r.role,str(path.relative_to(out))))
    return rows,moments

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--firms',type=Path,default=ROOT.parent/'boundary_gpu/assets/firms.csv.gz')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args(); out=args.cache;out.mkdir(parents=True,exist_ok=True)
    split=pd.read_csv(ASSETS/'split.csv',dtype={'event':str})
    if args.smoke:split=split.groupby('role',sort=False).head(1)
    pts=pd.concat([pd.read_csv(args.firms,dtype={'event':str}),pd.read_csv(ASSETS/'af_firms.csv.gz')])
    thresholds=json.loads((ASSETS/'af_thresholds.json').read_text())
    moments=np.zeros((2,3,11));manifest=[];audit=[]
    for r in split.itertuples():
        rows,m=event_rows(args.raw,out,r,pts,thresholds,7 if args.smoke else None)
        if m is not None:moments+=m
        manifest+=rows;audit.append((r.event,r.role,len(rows)))
        print(r.role,r.event,len(rows),flush=True)
    assert (moments[:,0]>0).all()
    mu=moments[:,1]/moments[:,0]
    sd=np.sqrt(np.maximum(moments[:,2]/moments[:,0]-mu**2,1e-8))
    np.savez(out/'norm.npz',mu=mu.astype('f4'),sd=sd.astype('f4'))
    pd.DataFrame(audit,columns=['event','role','targets']).to_csv(out/'inventory.csv',index=False)
    frame=pd.DataFrame(manifest,columns=['event','date','role','file'])
    if not args.smoke:
        missing=set(split.event)-set(frame.event)
        if missing-{'24332700'}:raise RuntimeError(f'Unexpected missing fires: {sorted(missing)}')
    frame.to_csv(out/'manifest.tmp',index=False);os.replace(out/'manifest.tmp',out/'manifest.csv')
if __name__=='__main__':main()
