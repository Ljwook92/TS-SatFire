"""Synthetic fixture only: moving fire, changing predictors, nodata blocks, real GeoTIFF I/O."""
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

def make_demo(root,event):
    rng=np.random.default_rng(77);h=w=28;rr,cc=np.indices((h,w))
    for d,date in enumerate(pd.date_range('2021-05-01',periods=40)):
        fire=(cc<(3+d*.45))&(rr>4)&(rr<24)
        ndvi=.3+.3*cc/w+.04*np.sin(rr/4)+rng.normal(0,.008,(h,w))
        day=np.stack([ndvi]*8);day[3]=290+fire*35+cc*.15;day[6]=fire;day[7]=fire
        aux=np.stack([ndvi.copy() for _ in range(19)]);aux[6]=20+d*.08+rr*.1
        # Deliberate known unknown labels, not silently background.
        day[6:,0,0]=np.nan
        aux[0,12:16,13:17]=np.nan;aux[0,17:21,20:25]=np.nan;day[3,8:11,8:12]=np.nan
        for product,a in [('VIIRS_Day',day),('FirePred',aux)]:
            folder=Path(root)/event/product;folder.mkdir(parents=True,exist_ok=True)
            with rasterio.open(folder/f'{date.date()}_{product}.tif','w',driver='GTiff',height=h,width=w,
                count=len(a),dtype='float32',crs='EPSG:32612',transform=from_origin(700000,3680000,375,375),nodata=np.nan) as ds:
                ds.write(a.astype('float32'))
                for i in range(len(a)):ds.set_band_description(i+1,f'SYNTHETIC_{product}_{i+1}')
