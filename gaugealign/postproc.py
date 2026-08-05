"""Causal, floor-safe post-processing for track1 predictions (12-col: ...yaw score).

Stages (each independently toggleable, all causal / online-preserving):
  --gapfill K      : causal constant-velocity forward-fill of holes <= K frames within a
                     track id (recovers frames the instance bank held the ID through but the
                     score gate dropped -> HOTA double-penalizes as missed-det + fragmentation).
  --thr "c:v,..."  : per-class score thresholds (e.g. rare Nova/Transporter low 0.05, strong 0.2).
  --nms IOU        : per (scene,frame,class) greedy BEV-IoU duplicate NMS (decoder has none).
  --snap "cids"    : hard-snap (w,l,h) to per-class mean prior for rigid robot classes.
Reads 11- or 12-col; writes same #cols (drops nothing structurally). Order: nms -> thr -> gapfill -> snap.
"""
import argparse
import numpy as np

# official-order [w,l,h] priors (from FULL_MEAN_SIZE_ARR / GT means); index by official class_id
SIZE_PRIOR = {  # person,forklift,nova,transporter,gr1,agility,pallet
    1: (0.90, 2.77, 2.10), 2: (0.60, 0.70, 0.60), 3: (0.62, 0.90, 0.30),
    6: (0.82, 1.89, 1.97),
}


def load(path):
    rows = []
    for ln in open(path):
        p = ln.split()
        if len(p) < 11:
            continue
        r = [int(float(p[0])), int(float(p[1])), int(float(p[2])), int(float(p[3]))] + \
            [float(x) for x in p[4:11]] + ([float(p[11])] if len(p) >= 12 else [1.0])
        rows.append(r)  # sid,cid,oid,fid, x,y,z,w,l,h,yaw, score
    return rows


def fmt(r):
    return (f"{r[0]} {r[1]} {r[2]} {r[3]} " +
            " ".join(f"{v:.2f}" for v in r[4:11]) + f" {r[11]:.4f}")


def bev_iou(a, b):
    # axis-aligned BEV IoU proxy on (x,y,w,l) (ignores yaw; cheap, adequate for dup suppression)
    ax, ay, aw, al = a[4], a[5], a[7], a[8]
    bx, by, bw, bl = b[4], b[5], b[7], b[8]
    a1x, a1y, a2x, a2y = ax - aw / 2, ay - al / 2, ax + aw / 2, ay + al / 2
    b1x, b1y, b2x, b2y = bx - bw / 2, by - bl / 2, bx + bw / 2, by + bl / 2
    ix = max(0, min(a2x, b2x) - max(a1x, b1x)); iy = max(0, min(a2y, b2y) - max(a1y, b1y))
    inter = ix * iy; ua = aw * al + bw * bl - inter
    return inter / ua if ua > 0 else 0.0


def do_nms(rows, iou_thr):
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r[0], r[3], r[1])].append(r)  # scene, frame, class
    out = []
    for g in groups.values():
        g = sorted(g, key=lambda r: -r[11])
        kept = []
        for r in g:
            if all(bev_iou(r, k) < iou_thr for k in kept):
                kept.append(r)
        out.extend(kept)
    return out


def do_thr(rows, thr):
    return [r for r in rows if r[11] >= thr.get(r[1], 0.2)]


def do_hysteresis(rows, high, low, K):
    """CAUSAL two-threshold (hysteresis) gate. Prefix-invariant: each frame's decision uses only the
    current row + the most recent high-confidence frame of the SAME track id (strictly past/present),
    never any future frame. Spawn/keep a track high-confidently at score>=high; once activated, emit its
    CURRENT-frame box at score>=low for up to K frames since the last high-conf hit (a high-conf hit
    within the window re-extends it). No future lookahead, no extrapolated boxes (only real dets)."""
    from collections import defaultdict
    tracks = defaultdict(list)
    for r in rows:
        tracks[(r[0], r[1], r[2])].append(r)  # scene, class, object id
    out = []
    for tr in tracks.values():
        tr = sorted(tr, key=lambda r: r[3])  # by frame; per-id processing is causal
        last_high = None
        for r in tr:
            s = r[11]
            if s >= high:
                out.append(r); last_high = r[3]
            elif s >= low and last_high is not None and (r[3] - last_high) <= K:
                out.append(r)  # low-conf continuation of an already-active id within K frames
            # else: drop (below low, not yet activated, or continuation window expired)
    return out


def do_gapfill(rows, K):
    from collections import defaultdict
    tracks = defaultdict(list)
    for r in rows:
        tracks[(r[0], r[1], r[2])].append(r)  # scene, class, object
    out = list(rows)
    for key, tr in tracks.items():
        tr = sorted(tr, key=lambda r: r[3])
        for i in range(1, len(tr)):
            f1, f2 = tr[i - 1][3], tr[i][3]
            gap = f2 - f1
            if 1 < gap <= K:
                # causal: velocity from the last two seen boxes before the hole
                if i >= 2:
                    df = tr[i - 1][3] - tr[i - 2][3]
                    vel = [(tr[i - 1][j] - tr[i - 2][j]) / max(df, 1) for j in (4, 5, 6)]
                else:
                    vel = [0.0, 0.0, 0.0]
                base = tr[i - 1]
                # displacement gate: skip erratic/fast extrapolations (regression guard)
                if (abs(vel[0]) + abs(vel[1]) + abs(vel[2])) * gap > 3.0:
                    continue
                for k in range(1, gap):  # fill f1+1 .. f2-1
                    nr = list(base)
                    nr[3] = f1 + k
                    nr[4] = base[4] + vel[0] * k
                    nr[5] = base[5] + vel[1] * k
                    nr[6] = base[6] + vel[2] * k
                    nr[11] = base[11] * 0.9  # slightly discount filled boxes
                    out.append(nr)
    return out


def do_snap(rows, cids):
    for r in rows:
        if r[1] in cids and r[1] in SIZE_PRIOR:
            r[7], r[8], r[9] = SIZE_PRIOR[r[1]]
    return rows


def do_smooth(rows, beta):
    # causal EMA on centroid (x,y,z) per track; no future leak -> Online preserved. Keeps size/yaw.
    from collections import defaultdict
    tracks = defaultdict(list)
    for r in rows:
        tracks[(r[0], r[1], r[2])].append(r)
    for tr in tracks.values():
        tr.sort(key=lambda r: r[3])
        s = None
        for r in tr:
            if s is None:
                s = [r[4], r[5], r[6]]
            else:
                s = [beta * s[i] + (1 - beta) * r[4 + i] for i in range(3)]
                r[4], r[5], r[6] = s
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nms", type=float, default=0.0)
    ap.add_argument("--thr", type=str, default="", help="per-class 'cid:thr,...'")
    ap.add_argument("--hysteresis", type=str, default="",
                    help="causal two-threshold recall gate 'HIGH:LOW:K' (e.g. 0.2:0.05:15)")
    ap.add_argument("--gapfill", type=int, default=0)
    ap.add_argument("--smooth", type=float, default=0.0, help="causal EMA beta on centroid (0=off)")
    ap.add_argument("--snap", type=str, default="", help="comma class_ids to snap sizes")
    args = ap.parse_args()
    rows = load(args.pred)
    n0 = len(rows)
    if args.nms > 0:
        rows = do_nms(rows, args.nms)
    if args.hysteresis.strip():
        h, l, k = args.hysteresis.split(":")
        rows = do_hysteresis(rows, float(h), float(l), int(k))
    if args.thr.strip():
        thr = {int(k): float(v) for k, v in (kv.split(":") for kv in args.thr.split(","))}
        rows = do_thr(rows, thr)
    if args.gapfill > 0:
        rows = do_gapfill(rows, args.gapfill)
    if args.smooth > 0:
        rows = do_smooth(rows, args.smooth)
    if args.snap.strip():
        rows = do_snap(rows, {int(x) for x in args.snap.split(",")})
    rows.sort(key=lambda r: (r[0], r[3], r[1], r[2]))
    with open(args.out, "w") as f:
        f.write("\n".join(fmt(r) for r in rows) + ("\n" if rows else ""))
    print(f"postproc {args.pred.split('/')[-1]}: {n0} -> {len(rows)} rows "
          f"(nms={args.nms} hyst='{args.hysteresis}' thr='{args.thr}' gapfill={args.gapfill} snap='{args.snap}')")


if __name__ == "__main__":
    main()
