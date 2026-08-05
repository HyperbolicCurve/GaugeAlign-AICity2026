"""Coordinate sanity + floor-ROI filter for Track1 predictions.

Two modes:
  --check   : report predicted xy/z ranges vs the scene's calibrated camera bounds
              (catches a world re-centering / shift-back offset bug on real scenes).
  --filter  : drop predictions whose xy falls outside [cam bounds +/- margin] or whose
              z is outside [zmin,zmax] (geometric floor ROI; NVIDIA's Sparse4D paper
              reports ROI+conf filtering lifts real HOTA). Writes a filtered pred file.

Camera bounds come from the extrinsics (camera centers = -R^T t) which ring the floor.
"""
import argparse, json
import numpy as np


def cam_bounds(calib_path):
    cal = json.load(open(calib_path))
    cc = []
    for s in cal["sensors"]:
        if s.get("type") != "camera":
            continue
        ext = np.array(s["extrinsicMatrix"]); R = ext[:3, :3]; t = ext[:3, 3]
        cc.append((-R.T @ t)[:2])
    cc = np.array(cc)
    return cc[:, 0].min(), cc[:, 0].max(), cc[:, 1].min(), cc[:, 1].max(), cc


def load_pred(p):
    rows = []
    for ln in open(p):
        q = ln.split()
        if len(q) >= 11:
            rows.append(q)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--mode", choices=["check", "filter"], default="check")
    ap.add_argument("--out", default=None)
    ap.add_argument("--margin", type=float, default=8.0, help="xy margin beyond camera bounds (m)")
    ap.add_argument("--zmin", type=float, default=-0.5)
    ap.add_argument("--zmax", type=float, default=3.2)
    args = ap.parse_args()

    xmin, xmax, ymin, ymax, cc = cam_bounds(args.calib)
    rows = load_pred(args.pred)
    xy = np.array([[float(r[4]), float(r[5])] for r in rows]) if rows else np.zeros((0, 2))
    z = np.array([float(r[6]) for r in rows]) if rows else np.zeros((0,))

    lo = np.array([xmin - args.margin, ymin - args.margin])
    hi = np.array([xmax + args.margin, ymax + args.margin])
    inxy = np.ones(len(rows), bool) if not rows else ((xy >= lo) & (xy <= hi)).all(1)
    inz = np.ones(len(rows), bool) if not rows else ((z >= args.zmin) & (z <= args.zmax))
    keep = inxy & inz

    print(f"[{args.pred.split('/')[-1]}] rows={len(rows)}")
    print(f"  cam bounds x[{xmin:.1f},{xmax:.1f}] y[{ymin:.1f},{ymax:.1f}]  (+margin {args.margin})")
    if len(rows):
        print(f"  pred     x[{xy[:,0].min():.1f},{xy[:,0].max():.1f}] y[{xy[:,1].min():.1f},{xy[:,1].max():.1f}] "
              f"z[{z.min():.2f},{z.max():.2f}]")
        print(f"  in-xy-bounds: {inxy.mean()*100:.1f}%   in-z[{args.zmin},{args.zmax}]: {inz.mean()*100:.1f}%   "
              f"keep: {keep.sum()}/{len(rows)} ({keep.mean()*100:.1f}%)")
        # offset alarm: if the bulk of preds sit far outside camera bounds -> shift bug
        if inxy.mean() < 0.5:
            print("  *** ALARM: >50% of preds OUTSIDE camera bounds -> likely re-centering/shift-back offset bug ***")
    if args.mode == "filter":
        out = args.out or args.pred.replace(".txt", "_roi.txt")
        with open(out, "w") as f:
            f.write("\n".join(" ".join(r) for r, k in zip(rows, keep) if k) + "\n")
        print(f"  filtered -> {out}  (dropped {len(rows)-int(keep.sum())})")


if __name__ == "__main__":
    main()
