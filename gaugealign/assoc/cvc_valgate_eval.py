#!/usr/bin/env python3
"""CVC-Assoc val-gate: learned survival-gate head vs scalar-cd0.9 BASE, val present-class 3D-HOTA.

Both arms are run with the IDENTICAL run_infer recipe (full-res, NO half_res,
score_thr 0.05 -> threshold 0.20, --remap); the gate is the ONLY difference -> apples-to-apples.
  base = run_infer --conf_decay 0.9         (scalar floor)
  cvc  = run_infer --cvc_head ... --cvc_dets ... --cvc_calib ...   (our learned per-slot gate)

CVC head is floor-safe by construction (g_init=0.90 bit-exact base; effective decay in [0.81,0.99]).
GATE (coordinator): CVC must TIE-OR-BEAT base on EVERY present class (DetA no-reg) AND not drop
combined HOTA on any scene. A clear combined-HOTA gain -> submission candidate for a later slot.
"""
import os
os.environ.setdefault("AICITY_SIM_MODE", "3diou")
import sys, time
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK = os.environ.get("WORK_DIR", os.path.join(_REPO, "work"))
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import evaluate_hota_present as EH

# Inputs are produced by two run_infer.py passes with identical settings except the
# gate (see docs/EXPERIMENTS.md): the scalar-decay control and the learned head.
GT_T = os.path.join(WORK, "gt_w{s}_f0-900_off.txt")
SCENES = ["020", "021", "022"]
NCAMS = {"020": 16, "021": 16, "022": 4}
THR = 0.20
CLASS_NAMES = {0:"Person",1:"Forklift",2:"NovaCarter",3:"Transporter",4:"FourierGR1T2",5:"AgilityDigit",6:"PalletTruck"}
BASE_T = os.path.join(WORK, "cvcgate_base_w{s}.txt")
CVC_T  = os.path.join(WORK, "cvcgate_cvc_w{s}.txt")

def log(*a): print(*a, flush=True)

def threshold(src, dst, thr):
    n = 0
    with open(src) as f, open(dst, "w") as o:
        for ln in f:
            p = ln.split()
            if len(p) < 12: continue
            if float(p[11]) >= thr:
                o.write(ln); n += 1
    return n

def ev(scene, pred_txt):
    summ = EH.evaluate(GT_T.format(s=scene), pred_txt, cores=4, quiet=True)
    seq = list(summ["per_scene"].keys())[0]
    sc = summ["per_scene"][seq]
    perc = {c: sc["per_class"][CLASS_NAMES[c]] for c in CLASS_NAMES
            if CLASS_NAMES[c] in sc["per_class"] and sc["per_class"][CLASS_NAMES[c]]["present"]}
    return sc["combined"], perc

def score(tag_fmt, scene):
    raw = tag_fmt.format(s=scene)
    if not os.path.exists(raw):
        return None
    thr_txt = raw.replace(".txt", "_thr020.txt")
    n = threshold(raw, thr_txt, THR)
    comb, perc = ev(scene, thr_txt)
    return dict(comb=comb, perc=perc, n=n)

def main():
    t0 = time.time()
    B, C = {}, {}
    for s in SCENES:
        b = score(BASE_T, s); c = score(CVC_T, s)
        if b is None: log(f"!! missing base pred: {BASE_T.format(s=s)}"); return 2
        if c is None: log(f"!! missing cvc pred:  {CVC_T.format(s=s)}"); return 2
        B[s], C[s] = b, c
        log(f"W{s}(c{NCAMS[s]}): base DetA={b['comb']['DetA']:.2f} HOTA={b['comb']['HOTA']:.2f} | "
            f"cvc DetA={c['comb']['DetA']:.2f} HOTA={c['comb']['HOTA']:.2f}  "
            f"(dDetA={c['comb']['DetA']-b['comb']['DetA']:+.2f} dHOTA={c['comb']['HOTA']-b['comb']['HOTA']:+.2f})")

    L = ["="*96]
    L.append("CVC-ASSOC VAL-GATE -- learned survival gate vs scalar-cd0.9 base, present-class 3D-HOTA (thr>=0.20)")
    L.append("="*96)
    L.append(f"  {'scene':11s} {'variant':7s} {'rows':>7s}  {'DetA':>7s} {'AssA':>7s} {'LocA':>7s} {'HOTA':>7s}")
    for s in SCENES:
        for name, R in (("base", B), ("cvc", C)):
            c = R[s]["comb"]
            lbl = 'W'+s+'(c'+str(NCAMS[s])+')' if name == "base" else ''
            L.append(f"  {lbl:11s} {name:7s} {R[s]['n']:7d}  {c['DetA']:7.2f} {c['AssA']:7.2f} {c['LocA']:7.2f} {c['HOTA']:7.2f}")
    def avg(R, m): return float(np.mean([R[s]["comb"][m] for s in SCENES]))
    L.append("")
    for name, R in (("base", B), ("cvc", C)):
        L.append(f"  AVG(3) {name:7s}  " + " ".join(f"{avg(R,m):7.2f}" for m in ["DetA","AssA","LocA","HOTA"]))
    L.append(f"  AVG(3) delta    " + " ".join(f"{avg(C,m)-avg(B,m):+7.2f}" for m in ["DetA","AssA","LocA","HOTA"]))
    L.append("")
    # per-class base->cvc (AssA is where the gate acts; DetA must not regress)
    L.append("Per-class  base->cvc  (DetA | AssA | HOTA):")
    reg_classes = []
    for s in SCENES:
        pb, pc = B[s]["perc"], C[s]["perc"]
        parts = []
        for cid in sorted(set(pb) | set(pc)):
            nm = CLASS_NAMES[cid]
            if cid in pb and cid in pc:
                dh = pc[cid]['HOTA']-pb[cid]['HOTA']
                parts.append(f"{nm} D{pb[cid]['DetA']:.1f}->{pc[cid]['DetA']:.1f} A{pb[cid]['AssA']:.1f}->{pc[cid]['AssA']:.1f} H{dh:+.1f}")
                if pc[cid]['DetA'] < pb[cid]['DetA'] - 0.5: reg_classes.append(f"W{s}:{nm}(DetA)")
                if pc[cid]['HOTA'] < pb[cid]['HOTA'] - 0.5: reg_classes.append(f"W{s}:{nm}(HOTA)")
        L.append(f"  W{s}: " + " | ".join(parts))
    L.append("")
    # verdict
    det_ok  = all(C[s]["comb"]["DetA"] >= B[s]["comb"]["DetA"] - 0.3 for s in SCENES)
    hota_noreg = all(C[s]["comb"]["HOTA"] >= B[s]["comb"]["HOTA"] - 0.3 for s in SCENES)
    beats = (avg(C,"HOTA") > avg(B,"HOTA") + 0.20)
    L.append("="*96)
    L.append(f"per-class regressions (>0.5 drop): {reg_classes if reg_classes else 'NONE'}")
    L.append(f"combined DetA no-reg (>-0.3 all) = {det_ok};  combined HOTA no-reg (>-0.3 all) = {hota_noreg}")
    L.append(f"avg dHOTA = {avg(C,'HOTA')-avg(B,'HOTA'):+.2f}  beats-base(>+0.20) = {beats}")
    if beats and det_ok and not reg_classes:
        L.append(">>> VERDICT: CVC BEATS BASE (clean) -- submission candidate. Build test dets(026/027)+CVC test inference.")
    elif hota_noreg and det_ok and not reg_classes:
        L.append(">>> VERDICT: CVC TIES BASE (floor-safe, no gain) -- architecture validated for paper; hold floor for submit.")
    else:
        L.append(">>> VERDICT: CVC REGRESSES on some class/scene -- do NOT submit; hold floor. Inspect gate/decay range.")
    L.append("="*96)
    out = "\n".join(L)
    print(out)
    with open(f"{WORK}/RESULTS_cvc_valgate.txt", "w") as f:
        f.write(out + "\n")
    log(f"[eval] {time.time()-t0:.1f}s -> {WORK}/RESULTS_cvc_valgate.txt")

if __name__ == "__main__":
    sys.exit(main() or 0)
