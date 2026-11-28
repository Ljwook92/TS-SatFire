"""Export four AF or BA decisions per evaluated window in original coordinates."""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine


def export(run):
    count = 0
    for path in sorted((run/'maps').glob('*/*/*.npz')):
        task, event = path.parent.parent.name, path.parent.name
        geo = json.loads((run/'cache'/event/'grid.json').read_text())
        with np.load(path) as data:
            pred = np.unpackbits(data['pred'],axis=-1)
            dates = data['dates'].tolist()
        tmp = path.with_suffix('.tmp.tif')
        with rasterio.open(tmp,'w',driver='GTiff',width=256,height=256,count=4,
                           dtype='uint8',crs=geo['crs'],transform=Affine(*geo['affine']),
                           compress='deflate',tiled=True) as ds:
            ds.write(pred)
            for i,day in enumerate(dates,1):
                ds.set_band_description(i,f'{task} {day}')
        tmp.replace(path.with_suffix('.tif'))
        count += 1
    if not count:
        raise RuntimeError('No window maps to export')
    print(f'Exported {count} four-band decision maps')


if __name__ == '__main__':
    export(Path(sys.argv[1]))
