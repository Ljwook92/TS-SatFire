"""Map two verified sets of TS2 pixels on the original 256x256 VIIRS crop."""
import argparse
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Patch, Rectangle
import numpy as np


EVENT = "US_2021_NM3323810847220210520"
DATE = "2021-06-01"
ASSETS = Path(__file__).resolve().parent / "assets"


def binary(bits):
    return np.unpackbits(bits).reshape(256, 256).astype(bool)


def cell_edges(mask):
    """Perimeter along pixel edges, including holes."""
    r, c = np.nonzero(mask)
    padded = np.pad(mask, 1)
    segments = []
    for dr, dc, ends in [(-1,0,(-.5,-.5,.5,-.5)), (1,0,(-.5,.5,.5,.5)),
                         (0,-1,(-.5,-.5,-.5,.5)), (0,1,(.5,-.5,.5,.5))]:
        outward = ~padded[r+1+dr,c+1+dc]
        a,b,d,e = ends
        starts = np.column_stack((c[outward]+a,r[outward]+b))
        stops = np.column_stack((c[outward]+d,r[outward]+e))
        segments.extend(np.stack((starts,stops), axis=1))
    return segments


def render(cache, archive, output, partial):
    cache = cache / EVENT
    with np.load(cache/'014.npz') as z:
        assert str(z['date']) == '2021-05-31'
        past = binary(z['past'])
    with np.load(cache/'015.npz') as z:
        assert str(z['date']) == DATE
        truth = binary(z['new'])
    with ZipFile(ASSETS/'domains.zip') as domains:
        with np.load(BytesIO(domains.read(f'{EVENT}/015.npz'))) as z:
            ids, sectors = z['pixel'].astype(int), z['sector']
    sector = np.full(65536,-1)
    sector[ids] = sectors
    sector = sector.reshape(256,256)
    east = np.isin(sector,[1,2,3]) & ~past
    flank = east & (sector!=2)
    assert truth.sum()==102 and not (truth & past).any()
    assert east.sum()==52 and (east & truth).sum()==12 and (flank & truth).sum()==7
    if partial:
        # Published per-region counts: both models hit all 5 true E and 0 true NE/SE.
        missed = flank & truth
        gained = np.zeros_like(truth)
    else:
        with np.load(archive) as z:
            local, allocation = [binary(z[name]) for name in ('local_extent','allocation_only')]
        assert (local & truth).sum()==26 and (allocation & truth).sum()==5
        assert (local & truth & east).sum()==5 and (allocation & truth & east).sum()==5
        gained = truth & ~east & local & ~allocation
        missed = truth & flank & ~local & ~allocation
        assert gained.sum()==21 and missed.sum()==7, 'Predictions differ from the reported TS2 run'
    colors = dict(past='#dce1e5',edge='#505960',observed='#9daec1',gain='#007f70',
                  miss='#d64b19',candidates='#7452aa')
    from matplotlib.colors import ListedColormap
    grid = np.zeros_like(past,dtype='uint8')
    grid[past],grid[truth],grid[gained],grid[missed] = 1,2,3,4
    cmap = ListedColormap(['#ffffff',colors['past'],colors['observed'],colors['gain'],colors['miss']])
    rr,cc = np.where(east)
    inset = (cc.min()-2.5,cc.max()+2.5,rr.min()-2.5,rr.max()+2.5)
    fig,axs = plt.subplots(1,2,figsize=(12.8,6),gridspec_kw={'width_ratios':[1.95,1]})
    fig.subplots_adjust(left=.07,right=.97,top=.83,bottom=.23,wspace=.15)
    for i,ax in enumerate(axs):
        ax.imshow(grid,origin='upper',cmap=cmap,vmin=0,vmax=4,interpolation='nearest')
        ax.add_collection(LineCollection(cell_edges(past),colors=colors['edge'],linewidths=.85))
        w=.90
        ax.add_collection(PatchCollection([Rectangle((c-w/2,r-w/2),w,w) for r,c in zip(rr,cc)],
                          facecolors='none',edgecolors=colors['candidates'],linewidths=.8))
        if i==0:
            r,c=np.where(past|truth|east)
            ax.set_xlim(c.min()-2.5,c.max()+2.5)
            ax.set_ylim(r.max()+2.5,r.min()-2.5)
            x1,x2,y1,y2=inset
            ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='#403649',linestyle='--',linewidth=.8))
        else:
            x1,x2,y1,y2=inset
            ax.set(xlim=(x1,x2),ylim=(y2,y1))
        ax.set_title(['Fire footprint and new records','Eastern candidates, enlarged'][i],loc='left',fontsize=13)
        ax.set_xlabel('Column in VIIRS crop')
        ax.set_aspect('equal')
    axs[0].set_ylabel('Row in VIIRS crop')
    fig.suptitle('New Mexico  |  June 1, 2021  |  TS = 2',fontsize=17,y=.98)
    entries=[Patch(facecolor=colors['past'],edgecolor=colors['edge'],label='Recorded through May 31 + perimeter'),
             Patch(facecolor='white',edgecolor=colors['candidates'],label='NE / E / SE candidates (52)'),
             Patch(facecolor=colors['miss'],label='Missed by both in NE / SE (7)')]
    if not partial: entries.append(Patch(facecolor=colors['gain'],label='Gained outside candidates (21)'))
    entries.append(Patch(facecolor=colors['observed'],label='Other June 1 new records'))
    fig.legend(handles=entries,loc='lower center',ncol=3,frameon=False,fontsize=10,bbox_to_anchor=(.5,.04))
    if partial: fig.text(.5,.016,'Positions of the 21 additional hits await the saved HPC prediction mask.',ha='center',fontsize=10)
    fig.savefig(output,dpi=200,bbox_inches='tight',pad_inches=.15)
    plt.close(fig)
    print(output)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--partial',action='store_true',help='Plot label-identifiable groups only')
    parser.add_argument('--mask',type=Path,help='One downloaded prediction NPZ instead of a run directory')
    a = parser.parse_args()
    archive = a.mask
    if not a.partial and archive is None:
        if a.run is None: parser.error('Supply --run or --mask')
        archive = a.run/'TS2'/'masks'/EVENT/f'{DATE}.npz'
    render(a.cache,archive,a.output,a.partial)
