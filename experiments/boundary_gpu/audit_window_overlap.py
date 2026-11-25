"""Verify rolling dates, shared geographic support and repeated pixels across splits."""
import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, bounds
from vegetation import ASSETS

CROP = Window(128,128,256,256)


def compare(raw,cache,protocol_labels,out):
    manifest = pd.read_csv(ASSETS/'windows.csv.gz', dtype={'event':str})
    assert manifest.groupby('event').split.nunique().eq(1).all()
    assert not manifest.duplicated(['event','ts','target_index']).any()
    catalog={}
    for event,g in manifest.groupby('event'):
        files=sorted((raw/event/'VIIRS_Day').glob('*.tif'))
        days=[p.name[:10] for p in files]
        assert days==sorted(set(days))
        for r in g.itertuples():
            seq=days[r.target_index-r.ts:r.target_index]
            assert (seq[0],seq[-1],days[r.target_index])==(r.first_input,r.last_input,r.target_date)
            assert len(seq)==r.ts and all(day<r.target_date for day in seq)
        with rasterio.open(files[0]) as ds:
            assert ds.crs.to_epsg()==4326
            bbox=np.array(bounds(CROP,ds.transform))
            trans=ds.window_transform(CROP)
        catalog[event]=(dict(zip(days,range(len(days)))),files,bbox,trans,g.split.iloc[0])
    def truth(event,i):
        if cache is not None:
            with np.load(cache/event/f'{i:03d}.npz') as z:
                return np.unpackbits(z['new']).reshape(256,256).astype(bool)
        with np.load(protocol_labels/event/f'{i:03d}.npz') as z:
            return z['y'].reshape(256,256).astype(bool)
    pairs=[]
    for a,b in combinations(catalog,2):
        da,fa,ba,ta,sa=catalog[a]
        db,fb,bb,tb,sb=catalog[b]
        days=sorted(da.keys() & db.keys())
        area=np.maximum(np.minimum(ba[2:],bb[2:])-np.maximum(ba[:2],bb[:2]),0).prod()
        if not days or not area:
            continue
        frac=area/min(np.prod(ba[2:]-ba[:2]),np.prod(bb[2:]-bb[:2]))
        row=dict(event_a=a,event_b=b,split_a=sa,split_b=sb,days=len(days),fraction=frac)
        if sa!=sb:
            mapping=~tb*ta
            dx,dy=mapping*(0,0)
            assert np.allclose([mapping.a,mapping.b,mapping.d,mapping.e],[1,0,0,1])
            assert np.allclose([dx,dy],np.rint([dx,dy]),atol=1e-5,rtol=0)
            y,x=np.indices((256,256))
            xx,yy=x+round(dx),y+round(dy)
            take=(xx>=0)&(xx<256)&(yy>=0)&(yy<256)
            y,x,yy,xx=[v[take] for v in (y,x,yy,xx)]
            total=positive=equal=targets=0
            for day in days:
                i,j=da[day],db[day]
                with rasterio.open(fa[i]) as f,rasterio.open(fb[j]) as h:
                    va=f.read(1,window=CROP,masked=True).astype('float32').filled(np.nan)[y,x]
                    vb=h.read(1,window=CROP,masked=True).astype('float32').filled(np.nan)[yy,xx]
                equal+=np.count_nonzero((va==vb)|(np.isnan(va)&np.isnan(vb)))
                total+=len(va)
                if i>=2 and j>=2:
                    targets+=1
                    positive+=np.count_nonzero(truth(a,i)[y,x]&truth(b,j)[yy,xx])
            row.update(pixel_days=total,matching_I1_including_matching_nodata=equal,
                       common_ts2_targets=targets,shared_positive_pixel_days=positive)
        pairs.append(row)
    table=pd.DataFrame(pairs)
    table.to_csv(out,index=False)
    print('Verified windows:',len(manifest),'events:',len(catalog))
    print(table[table.split_a!=table.split_b].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path,required=True)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--cache',type=Path)
    g.add_argument('--protocol-labels',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    compare(a.raw,a.cache,a.protocol_labels,a.out)
