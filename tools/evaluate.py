"""Reproduce training, held-out classification, tracking and scan replay without ROS."""
import argparse, csv, io, json, math, platform, subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
def run(exe,*args,output=None):
    p=subprocess.run([str(exe),*map(str,args)],check=True,capture_output=True,text=True)
    if output:Path(output).write_text(p.stdout,encoding='utf-8')
    return p.stdout

def scan_at(x,y,radius=.07,n=720):
    ranges=[]
    for i in range(n):
        a=-math.pi+i*2*math.pi/n
        along=x*math.cos(a)+y*math.sin(a);d=radius**2-(x*x+y*y-along*along)
        ranges.append(along-math.sqrt(d) if d>=0 and along>0 else 0)
    return ranges

def demo_training(exe,out):
    # Independent synthetic seed and geometry; labels describe generated objects only.
    rng=np.random.default_rng(400)
    path=out/'demo_segments.txt'
    with path.open('w') as f:
        for i in range(600):
            label=i%2;x=rng.uniform(.8,2.5);y=rng.uniform(-.6,.6)
            points=[]
            if label:
                rs=scan_at(x,y,rng.uniform(.045,.09))
                points=[(r*math.cos(-math.pi+k*2*math.pi/720),r*math.sin(-math.pi+k*2*math.pi/720)) for k,r in enumerate(rs) if r>0]
            else:
                # Walls and broad curved objects; deliberately not a real-world classifier.
                radius=rng.uniform(.2,.6)
                rs=scan_at(x,y,radius)
                points=[(r*math.cos(-math.pi+k*2*math.pi/720),r*math.sin(-math.pi+k*2*math.pi/720)) for k,r in enumerate(rs) if r>0]
            if len(points)>=3:f.write(f'demo_{i} {label} {len(points)} '+' '.join(str(v) for p in points for v in p)+'\n')
    features=out/'demo_features.txt';run(exe,'features',path,output=features)
    model=out/'demo_nb.txt';run(exe,'train',features,model)
    return model

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--exe',type=Path,required=True);parser.add_argument('--output',type=Path,default=ROOT/'results/local')
    args=parser.parse_args();exe=args.exe.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    metrics={'environment':platform.platform(),'data_scope':'Historical sequence holdout and synthetic scenarios; no new hardware evaluation.'}
    train=ROOT/'data/train_features.txt';test=ROOT/'data/test_features.txt'
    if train.exists() and test.exists():
        lines=train.read_text().splitlines()
        fit_lines=[s for s in lines if int(s.split()[0].split('_')[1])<80]
        val_lines=[s for s in lines if int(s.split()[0].split('_')[1])>=90]
        (out/'fit_features.txt').write_text('\n'.join(fit_lines)+'\n')
        (out/'validation_features.txt').write_text('\n'.join(val_lines)+'\n')
        run(exe,'train',out/'fit_features.txt',out/'historical_nb.txt',out/'validation_features.txt')
        rows=list(csv.DictReader(io.StringIO(run(exe,'classify',test,out/'historical_nb.txt',output=out/'classification.csv'))))
        y=np.array([int(r['label']) for r in rows]);pred=np.array([int(r['prediction']) for r in rows])
        tp=int(((y==1)&(pred==1)).sum());fp=int(((y==0)&(pred==1)).sum());fn=int(((y==1)&(pred==0)).sum());tn=int(((y==0)&(pred==0)).sum())
        precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn)
        threshold=float((out/'historical_nb.txt').read_text().splitlines()[1])
        metrics['classification']={'fit_segments':len(fit_lines),'validation_segments':len(val_lines),'test_segments':len(y),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':precision,'recall':recall,'f1':2*precision*recall/max(1e-12,precision+recall),'threshold':threshold}
    model=demo_training(exe,out)
    rng=np.random.default_rng(732)
    truth=[];noisy=[];available=[]
    with (out/'trajectory.txt').open('w') as f:
        for i in range(400):
            t=i*.1;x=1.4+.35*math.sin(.25*t);y=.35*math.sin(.4*t);truth.append([x,y])
            p=np.array([x,y])+rng.normal(0,.1,2);noisy.append(p)
            missing=(110<=i<114) or (230<=i<243)
            ds=[] if missing else [p]
            if i%17==0:ds.append([4.,-3.])
            available.append(not missing)
            f.write(f'{t:.9f} {x} {y} {len(ds)} '+' '.join(str(v) for d in ds for v in d)+'\n')
    rows=list(csv.DictReader(io.StringIO(run(exe,'track',out/'trajectory.txt',output=out/'tracking.csv'))))
    estimate=np.array([[float(r['estimate_x']),float(r['estimate_y'])] for r in rows]);truth=np.array(truth);noisy=np.array(noisy)
    status=np.array([int(r['status']) for r in rows]);valid=np.array(available)&(status==2)
    error=np.linalg.norm(estimate-truth,axis=1);raw=np.linalg.norm(noisy-truth,axis=1);us=np.array([float(r['us']) for r in rows])
    metrics['tracking']={'seed':732,'frames':len(rows),'scored_frames':int(valid.sum()),'raw_rmse_m':float(np.sqrt(np.mean(raw[valid]**2))),'filtered_rmse_m':float(np.sqrt(np.mean(error[valid]**2))),'latency_median_us':float(np.median(us)),'latency_p95_us':float(np.percentile(us,95)),'lost_frames':int((status==4).sum())}
    plotted=estimate.copy();plotted[(status==0)|(status==4)]=np.nan
    plotted_error=error.copy();plotted_error[(status==0)|(status==4)]=np.nan
    fig,ax=plt.subplots(1,2,figsize=(11,4));ax[0].scatter(noisy[valid,0],noisy[valid,1],s=5,alpha=.25,label='Noisy detections');ax[0].plot(truth[:,0],truth[:,1],label='Ground truth');ax[0].plot(plotted[:,0],plotted[:,1],label='Kalman estimate');ax[0].set(xlabel='x (m)',ylabel='y (m)',title='Synthetic target tracking');ax[0].axis('equal');ax[0].legend(fontsize=8)
    ax[1].plot(np.arange(400)*.1,plotted_error,label='Position error');ax[1].axvspan(11,11.4,alpha=.15,color='orange');ax[1].axvspan(23,24.3,alpha=.15,color='red');ax[1].set(xlabel='Time (s)',ylabel='Error (m)',title='Occlusions and reacquisition');ax[1].legend();fig.tight_layout();fig.savefig(out/'tracking.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(2,1,figsize=(10,5),sharex=True);t=np.arange(400)*.1;ax[0].step(t,status,where='post');ax[0].set_yticks(range(5),['Empty','Tentative','Tracking','Coasting','Lost']);ax[1].plot(t,[float(r['v']) for r in rows],label='Linear m/s');ax[1].plot(t,[float(r['w']) for r in rows],label='Angular rad/s');ax[1].set_xlabel('Time (s)');ax[1].legend();fig.suptitle('Control preview only - no robot dynamics');fig.tight_layout();fig.savefig(out/'states.png',dpi=170);plt.close(fig)
    with (out/'scans.txt').open('w') as f:
        for i in range(120):
            x=1.3+.1*math.sin(i*.03);y=.2*math.sin(i*.05);rs=scan_at(x,y)
            if 50<=i<55:rs=[0]*720
            f.write(f'{i*.1:.9f} {-math.pi} {2*math.pi/720} .05 10 720 '+' '.join(map(str,rs))+'\n')
    scanrows=list(csv.DictReader(io.StringIO(run(exe,'scan',out/'scans.txt',model,output=out/'scan_tracking.csv'))))
    scanus=np.array([float(r['us']) for r in scanrows]);detected=sum(int(r['detections'])>0 for r in scanrows)
    metrics['synthetic_scan']={'frames':120,'frames_with_detections':detected,'median_us':float(np.median(scanus)),'p95_us':float(np.percentile(scanus,95)),'model':'synthetic-only demo model'}
    assert detected>=100,'synthetic scan pipeline unexpectedly failed'
    assert metrics['tracking']['filtered_rmse_m']<metrics['tracking']['raw_rmse_m'],'tracking regression'
    fig,ax=plt.subplots(figsize=(7,4));ax.plot([float(r['x']) for r in scanrows],[float(r['y']) for r in scanrows],label='Estimated segment centroid');ax.set(xlabel='x (m)',ylabel='y (m)',title='Ray-cast scan to classification to tracking');ax.legend();fig.tight_layout();fig.savefig(out/'scan_tracking.png',dpi=170);plt.close(fig)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8');print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
