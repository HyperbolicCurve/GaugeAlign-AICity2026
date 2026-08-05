#!/usr/bin/env python3
"""Score the gauge-stress sweep: for each (scene, gauge-offset d) score pred vs
frame-matched GT and tabulate DetA/DetRe/HOTA/AssA/LocA vs gauge error.
d=0 == our auto_center (gauge canonicalization); d=|M| == absolute-coord baseline."""
import os, re, subprocess, sys, glob
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
W=os.environ.get("WORK_DIR", f"{ROOT}/work")+"/gauge"
SCORER=f"{ROOT}/scripts/evaluate_hota.py"
NFRAMES=150  # end900 stride6
MABS={"20":86.0,"21":86.0,"22":104.0}
GRID=[("000p0",0.0),("010p0",10.0),("025p0",25.0),("050p0",50.0)]

def parse_hota(log, key):
    """extract COMBINED metrics under 'HOTA: prediction-<key>' block."""
    txt=open(log,encoding="utf-8",errors="ignore").read().splitlines()
    for i,l in enumerate(txt):
        if l.startswith(f"HOTA: prediction-{key}"):
            for j in range(i+1,min(i+4,len(txt))):
                if txt[j].strip().startswith("COMBINED"):
                    v=txt[j].split()[1:]
                    # HOTA DetA AssA DetRe DetPr AssRe AssPr LocA
                    return dict(HOTA=float(v[0]),DetA=float(v[1]),AssA=float(v[2]),
                                DetRe=float(v[3]),LocA=float(v[7]))
    return None

def score(sid,tag):
    pred=f"{W}/pred_{sid}_d{tag}.txt"; gt=f"{W}/gt_{sid}.txt"
    log=f"{W}/hota_{sid}_d{tag}.log"
    if not os.path.exists(pred): return None,0
    nrows=sum(1 for _ in open(pred))
    if not os.path.exists(log) or os.path.getmtime(log)<os.path.getmtime(pred):
        with open(log,"w") as fo:
            subprocess.run(["python",SCORER,"--gt",gt,"--pred",pred,"--cores","6"],
                           stdout=fo,stderr=subprocess.STDOUT,cwd=ROOT)
    return parse_hota(log,"cls_comb_det_av"), nrows

print(f"{'scene':6} {'d(m)':>6} {'gauge':8} {'dets/fr':>8} {'DetA':>7} {'DetRe':>7} {'AssA':>7} {'HOTA':>7} {'LocA':>7}")
print("-"*70)
agg={}  # d -> list of DetA
for sid in ["20","21","22"]:
    grid=GRID+[(f"{MABS[sid]:05.1f}".replace('.','p'),MABS[sid])]
    for tag,d in grid:
        r,nrows=score(sid,tag)
        gauge = "ours" if d==0 else ("ABS" if abs(d-MABS[sid])<1e-3 else "")
        if r is None:
            print(f"W0{sid} {d:6.1f} {gauge:8} {'--':>8} MISSING"); continue
        dpf=nrows/NFRAMES
        print(f"W0{sid} {d:6.1f} {gauge:8} {dpf:8.1f} {r['DetA']:7.2f} {r['DetRe']:7.2f} {r['AssA']:7.2f} {r['HOTA']:7.2f} {r['LocA']:7.2f}")
        agg.setdefault(round(d if gauge!='ABS' else -1,1),[]).append(r['DetA'])
print("-"*70)
print("mean DetA across scenes by gauge error:")
for d in sorted(k for k in agg if k>=0):
    print(f"  d={d:5.1f} m : DetA {sum(agg[d])/len(agg[d]):6.2f}  (n={len(agg[d])})")
if -1 in agg:
    print(f"  d=|M| (ABSOLUTE, no recenter): DetA {sum(agg[-1])/len(agg[-1]):6.2f}  (n={len(agg[-1])})")
