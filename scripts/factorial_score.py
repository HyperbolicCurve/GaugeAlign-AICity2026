#!/usr/bin/env python3
"""Score the unlock factorial: 8 cells = recenter{on,off} x to_rgb{0,1} x remap{on,off} on W022.
Metric: per-class-averaged DetA (cls_comb_det_av) + Person-only DetA (unaffected by remap, isolates RGB/gauge).
GT = work/gauge/gt_22.txt."""
import os, subprocess
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
W=os.environ.get("WORK_DIR", f"{ROOT}/work")+"/factorial"
GT=os.environ.get("WORK_DIR", f"{ROOT}/work")+"/gauge/gt_22.txt"
SCORER=f"{ROOT}/scripts/evaluate_hota.py"
NFR=150

def parse(log, key):
    if not os.path.exists(log): return None
    txt=open(log,encoding="utf-8",errors="ignore").read().splitlines()
    for i,l in enumerate(txt):
        if l.startswith(f"HOTA: prediction-{key}"):
            for j in range(i+1,min(i+5,len(txt))):
                if txt[j].strip().startswith("COMBINED"):
                    v=txt[j].split()[1:]
                    return dict(HOTA=float(v[0]),DetA=float(v[1]),AssA=float(v[2]),LocA=float(v[7]))
    return None

def score(pred):
    log=pred.replace("pred_","hota_").replace(".txt",".log")
    if not os.path.exists(log) or os.path.getmtime(log)<os.path.getmtime(pred):
        with open(log,"w") as fo:
            subprocess.run(["python",SCORER,"--gt",GT,"--pred",pred,"--cores","6"],
                           stdout=fo,stderr=subprocess.STDOUT,cwd=ROOT)
    return parse(log,"cls_comb_det_av"), parse(log,"prediction-per_class") if False else None

print(f"{'recenter':9}{'to_rgb':7}{'remap':7}{'dets/fr':>8}{'clsav_DetA':>11}{'clsav_HOTA':>11}{'AssA':>8}{'LocA':>8}")
print("-"*69)
rows=[]
for rc in ["on","off"]:
    for rgb in ["0","1"]:
        for rm in ["remap","noremap"]:
            pred=f"{W}/pred_{rc}_rgb{rgb}_{rm}.txt"
            if not os.path.exists(pred):
                print(f"{rc:9}{rgb:7}{rm:7}  MISSING"); continue
            n=sum(1 for _ in open(pred)); r,_=score(pred)
            if r is None:
                print(f"{rc:9}{rgb:7}{rm:7}{n/NFR:8.1f}  score-fail"); continue
            print(f"{rc:9}{rgb:7}{rm:7}{n/NFR:8.1f}{r['DetA']:11.2f}{r['HOTA']:11.2f}{r['AssA']:8.2f}{r['LocA']:8.2f}")
            rows.append((rc,rgb,rm,r))
print("-"*69)
# the full-correct cell vs one-factor-ablated deltas
full=next((r for (rc,rgb,rm,r) in rows if rc=="on" and rgb=="0" and rm=="remap"),None)
if full:
    print(f"\nfull-correct (recenter+rgb0+remap) DetA={full['DetA']:.2f} HOTA={full['HOTA']:.2f}")
    for (rc,rgb,rm,r) in rows:
        if (rc,rgb,rm)!=("on","0","remap"):
            print(f"  drop {rc}/{rgb}/{rm}: DetA {r['DetA']-full['DetA']:+.2f}  HOTA {r['HOTA']-full['HOTA']:+.2f}")
