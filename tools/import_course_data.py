"""Convert original course MAT labels to portable segments; original files stay local.
Usage: python tools/import_course_data.py /path/to/1081 --output /path/to/export
Requires scipy. Recompute all nine features with leg_cli features afterward.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.io import loadmat

def main():
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    manifest={'method':'Original segment-index annotations; recompute C++ feature schema V1.', 'sequences':[]}
    seen={}
    for directory,split in [('data','train'),('data0106','test')]:
        p=args.source/directory
        point=loadmat(p/'point.mat');labels=loadmat(p/'seg.mat')
        xs,ys=point['pointx'],point['pointy'];segs=labels['seg'].ravel();legs=labels['leg'].ravel()
        count=[0,0];bad=0;overlap=0
        output=args.output/(split+'_segments.txt')
        with output.open('w',encoding='ascii') as f:
            for frame in range(xs.shape[1]):
                digest=hashlib.sha256(np.stack([xs[:,frame],ys[:,frame]]).tobytes()).hexdigest()
                if digest in seen:overlap+=1;continue
                seen[digest]=split
                leg_ids=set(np.asarray(legs[frame]).ravel().astype(int).tolist())
                for number,indices in enumerate(segs[frame].ravel(),1):
                    idx=np.asarray(indices).ravel().astype(int)-1
                    if len(idx)<3 or np.any(idx<0) or np.any(idx>=len(xs)):bad+=1;continue
                    xy=np.stack([xs[idx,frame],ys[idx,frame]],axis=1)
                    if not np.isfinite(xy).all():bad+=1;continue
                    label=int(number in leg_ids);count[label]+=1
                    f.write(f'{directory}_{frame:04d}_{number:03d} {label} {len(xy)} '+ ' '.join(format(v,'.17g') for v in xy.ravel())+'\n')
        manifest['sequences'].append({'directory':directory,'split':split,'frames':xs.shape[1],'nonleg':count[0],'leg':count[1],'skipped_segments':bad,'duplicate_frames_excluded':overlap,
          'source_sha256':{n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ['point.mat','seg.mat']}})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
