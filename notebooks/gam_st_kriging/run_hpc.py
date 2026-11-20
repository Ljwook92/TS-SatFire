"""Optional headless runner; use allocated HPC compute resources, not a login node."""
import argparse,os
from pathlib import Path
from datetime import datetime,timezone
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'4')
import st_kriging as sk
from threadpoolctl import threadpool_limits

def main():
    p=argparse.ArgumentParser();p.add_argument('--demo',action='store_true');p.add_argument('--raw-root');p.add_argument('--output-root',default='runs')
    args=p.parse_args();cfg=sk.defaults()
    cfg['raw_root']=args.raw_root or os.environ.get('TS_SATFIRE_RAW_ROOT',cfg['raw_root'])
    if args.demo:
        from satfire_demo import make_demo
        cfg.update(raw_root='demo_data',event_id='SYNTHETIC_ONLY',features=['ndvi','i4','tmax'],negative_keep=1.,
                   lambdas=[.001,.01],n_knots=3,max_iter=2000,block_pixels=7,history_days=2,spatial_ranges_km=[3.],temporal_ranges_days=[1.,3.])
        make_demo(cfg['raw_root'],cfg['event_id'])
    out=Path(args.output_root)/(('demo_' if args.demo else 'real_')+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    with threadpool_limits(limits=int(os.environ['OMP_NUM_THREADS'])):
        data=sk.prepare(cfg,out);selected=sk.train_select(data);print(sk.evaluate(data,selected).to_string(index=False));sk.plots(data)
    print('Saved:',out.resolve())
if __name__=='__main__':main()
