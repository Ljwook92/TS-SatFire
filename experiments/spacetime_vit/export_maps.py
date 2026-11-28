"""Write spatially referenced probability and decision rasters for both tasks."""
import argparse,json
from pathlib import Path
import numpy as np
import rasterio
from affine import Affine

def export(cache,run):
    total=0
    for folder in sorted((run/'maps').iterdir()):
        g=json.loads((cache/folder.name/'grid.json').read_text())
        for f in sorted(folder.glob('*.npz')):
            with np.load(f) as z:
                p=z['prob'].astype('f4')
                mask=np.where(p>=0,np.unpackbits(z['pred'],axis=-1),-1)
            dst=f.with_suffix('.tif');tmp=f.with_suffix('.tmp.tif')
            with rasterio.open(tmp,'w',driver='GTiff',width=256,height=256,count=4,dtype='float32',
                               crs=g['crs'],transform=Affine(*g['affine']),nodata=-1,compress='deflate',tiled=True) as ds:
                ds.write(np.concatenate([p,mask]).astype('f4'))
                for i,name in enumerate(['AF_probability','BA_probability','AF_detection','BA_detection'],1):ds.set_band_description(i,name)
            tmp.replace(dst);total+=1
    if not total:raise ValueError('No prediction maps')
    print('Exported GeoTIFFs:',total)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cache',type=Path);p.add_argument('run',type=Path)
    a=p.parse_args();export(a.cache,a.run)
