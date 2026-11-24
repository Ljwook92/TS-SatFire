"""Build event-day tensors once from the existing raw TS-SatFire directory."""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

HERE=Path(__file__).resolve().parent
ASSETS=HERE/'assets'
CROP=Window(128,128,256,256)
LOCAL=['distance_km','recent_distance_km','i4','i4_change','i4_valid','edge_i4_change','edge_i4_valid',
       'wind_speed','outward_wind','slope','uphill_alignment','rise_from_edge','frp_log',
       'frp_distance','edge_frp_log','edge_frp_distance','has_hotspot','has_edge','ndvi','evi2','new_record']
def aligned(ds,bands,ref,nearest=()):
    result=np.full((len(bands),256,256),np.nan,np.float32)
    for i,b in enumerate(bands):
        if ds.crs==ref['crs'] and ds.transform.almost_equals(ref['full']) and ds.shape==ref['shape']:
            result[i]=ds.read(b,window=CROP,masked=True).astype('float32').filled(np.nan)
        else:
            reproject(ds.read(b,masked=True).astype('float32').filled(np.nan),result[i],
                      src_transform=ds.transform,src_crs=ds.crs,src_nodata=np.nan,
                      dst_transform=ref['crop'],dst_crs=ref['crs'],dst_nodata=np.nan,
                      resampling=Resampling.nearest if b in nearest else Resampling.bilinear)
    return result

def encoder_input(day,night,aux,running,norm):
    x=np.nan_to_num(np.concatenate((day,night,aux)),nan=0,posinf=0,neginf=0)
    idx=[3,4,6,7]
    np.maximum(running,x[idx],out=running)
    x[idx]=running
    cont=[]
    for i,v in enumerate(x):
        if i in [12,18,24]:cont.extend([np.sin(np.deg2rad(v)),np.cos(np.deg2rad(v))])
        elif i!=21:cont.append(v)
    cont=np.clip((np.stack(cont)-norm['mean'][:,None,None])/norm['std'][:,None,None],-5,5)
    lc=x[21].astype(int)
    assert lc.min()>=0 and lc.max()<18
    return np.concatenate((cont,np.eye(18,dtype=np.float32)[lc].transpose(2,0,1))).astype(np.float16)

def geometry(ref):
    rr,cc=np.indices((256,256))
    lon,lat=ref['crop']*(cc+.5,rr+.5)
    assert ref['crs'].to_epsg()==4326
    crs=f'+proj=aeqd +lat_0={lat.mean()} +lon_0={lon.mean()} +datum=WGS84 +units=m'
    tr=Transformer.from_crs('EPSG:4326',crs,always_xy=True)
    return np.stack(tr.transform(lon,lat),axis=-1).astype('float32')/1000,tr

def local_features(xy,tr,h,past,new,aux,i4,prev_i4):
    sampling=(np.linalg.norm(xy[1,0]-xy[0,0]),np.linalg.norm(xy[0,1]-xy[0,0]))
    rr,cc=np.indices(past.shape)
    if past.any():
        distance,(row,col)=distance_transform_edt(~past,sampling=sampling,return_indices=True)
    else:
        distance=np.full(past.shape,30.);row,col=rr,cc
    recent=distance_transform_edt(~new,sampling=sampling) if new.any() else np.full(past.shape,30.)
    dxy=xy-xy[row,col]
    bearing=np.arctan2(dxy[:,:,0],dxy[:,:,1])
    delta=i4-prev_i4
    if h is not None and len(h):
        points=np.column_stack(tr.transform(h.longitude.to_numpy(),h.latitude.to_numpy()))/1000
        hd,hi=cKDTree(points).query(xy.reshape(-1,2))
        hd=hd.reshape(256,256)
        power=np.where(hd<=10,np.log1p(h.frp.to_numpy()[hi].reshape(256,256)),0)
    else:
        hd=np.full(past.shape,30.);power=np.zeros(past.shape)
    feats=[np.log1p(distance),np.log1p(recent),(i4-310)/25,delta/12,np.isfinite(delta),
           delta[row,col]/12,np.isfinite(delta[row,col]),aux[3]/8,
           aux[3]*np.cos(np.deg2rad(aux[4]+180)-bearing)/8,aux[9]/30,
           np.cos(np.deg2rad(aux[10]+180)-bearing)*aux[9]/30,
           (aux[11]-aux[11,row,col])/200,power/4,np.log1p(np.minimum(hd,30)),
           power[row,col]/4,np.log1p(np.minimum(hd[row,col],30)),hd<=10,
           np.full(past.shape,past.any()),aux[0]/10000,aux[1]/10000,new]
    # Each exterior pixel belongs to a nearest recorded perimeter cell and an outward distance band.
    # -1 marks locations lacking a recorded source or outside the count-loss neighborhood.
    ring=np.digitize(distance,[.5,1,2])
    owner=np.where(~past & past.any() & (distance<=4),(row*256+col)*4+ring,-1).astype(np.int32)
    return np.nan_to_num(np.clip(feats,-8,8)).astype(np.float16),owner

def prepare_event(event,group,raw,cache,fires,norm,limit=None):
    dest=cache/event;dest.mkdir(parents=True,exist_ok=True)
    paths=sorted((raw/event/'VIIRS_Day').glob('*.tif'))
    check=group[group.ts==2].sort_values('target_index')
    dates=[p.name[:10] for p in paths]
    if dates[2:]!=check.target_date.tolist() or dates[:-2]!=check.first_input.tolist():
        raise ValueError(f'{event}: file dates differ from rolling manifest under {raw}')
    if (dest/'complete').exists() and limit is None:return
    label_sel=group.label_sel.iloc[0]
    running=np.zeros((4,256,256),np.float32);past=np.zeros((256,256),bool)
    prev_i4=np.full((256,256),np.nan,np.float32)
    previous_date=None
    for i,path in enumerate(paths[:limit]):
        date=dates[i]
        with rasterio.open(path) as ds:
            current=dict(full=ds.transform,crop=ds.window_transform(CROP),crs=ds.crs,shape=ds.shape)
            if i==0:
                ref=current;xy,tr=geometry(ref)
                np.save(dest/'xy.npy',xy)
            assert current==ref
            day=aligned(ds,range(1,7),ref)
            labs=ds.read([7] if label_sel==1 else [7,8],window=CROP)
        hit=(np.isfinite(labs)&(labs!=0)).any(0)
        new=hit&~past;past|=hit
        if i>=2:assert new.sum()==int(check.iloc[i-2].nan_zero_positives), (event,date)
        night_path=raw/event/'VIIRS_Night'/path.name.replace('VIIRS_Day','VIIRS_Night')
        if night_path.exists():
            with rasterio.open(night_path) as ds:
                assert ds.count in (2,5)
                night=aligned(ds,[ds.count-1,ds.count],ref)
        else:night=np.full((2,256,256),np.nan,np.float32)
        with rasterio.open(raw/event/'FirePred'/path.name.replace('VIIRS_Day','FirePred')) as ds:
            aux=aligned(ds,range(1,20),ref,[5,11,14,17])
        gap=previous_date is None or (pd.Timestamp(date)-pd.Timestamp(previous_date)).days!=1
        if gap:prev_i4.fill(np.nan)
        local,owner=local_features(xy,tr,fires.get((event,date)),past,new,aux,day[3],prev_i4)
        enc=encoder_input(day,night,aux,running,norm)
        tmp=dest/f'{i:03d}.tmp.npz'
        np.savez(tmp,enc=enc,local=local,owner=owner,new=np.packbits(new),past=np.packbits(past),date=date)
        tmp.replace(dest/f'{i:03d}.npz')
        prev_i4=day[3].copy();previous_date=date
    if limit is None:(dest/'complete').write_text(str(len(paths)))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--event');ap.add_argument('--limit',type=int)
    args=ap.parse_args()
    w=pd.read_csv(ASSETS/'windows.csv.gz',dtype={'event':str})
    if args.event:w=w[w.event==args.event]
    assert len(w)
    fires=dict(tuple(pd.read_csv(ASSETS/'firms.csv.gz',dtype={'event':str}).groupby(['event','acq_date'])))
    norm=dict(np.load(ASSETS/'normalizer.npz'))
    for e,g in w.groupby('event'):
        prepare_event(e,g,args.raw,args.cache,fires,norm,args.limit)
        print('prepared',e,flush=True)
