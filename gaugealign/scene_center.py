#!/usr/bin/env python3
"""Compute the canonical frame functional C-bar for a scene (the GaugeAlign shift).

C-bar is the mean camera position, obtained from calibration alone:

    C_i    = -R_i^T t_i            (camera centre of sensor i, in world coordinates)
    C-bar  = (1/N) sum_i C_i       (paper, Sec. 3.3)

`build_pkl.py --auto_center` applies exactly this internally; this script exposes
it so the per-scene shift values can be inspected, tabulated, or pinned in a
config (see configs/scenes.yaml). Only the xy components are used -- the z axis
is already gravity-aligned and shared between the training and target sites.

Optionally reports the diagnostic behind Sec. 3.2 (the anchor--gauge coupling):
the extent covered by the detector's k-means anchor prior versus how far the
scene sits from it, which is what determines whether the anchors land inside the
camera frustums at all.

Usage
-----
    python -m gaugealign.scene_center --scene_dir <.../Warehouse_022>
    python -m gaugealign.scene_center --scene_dir <...> --anchor <_ov_kmeans900_....npy>
    python -m gaugealign.scene_center --dataset_root <.../MTMC_Tracking_2026> --split test
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def camera_centers(calibration_path: str) -> np.ndarray:
    """Return the (N, 3) world-frame camera centres C_i = -R_i^T t_i.

    Only sensors with type == "camera" are used, matching `build_pkl.py`.
    """
    calibration = json.load(open(calibration_path))
    centers = []
    for sensor in calibration["sensors"]:
        if sensor.get("type") != "camera":
            continue
        ext = np.asarray(sensor["extrinsicMatrix"], dtype=np.float64)  # 3x4 world->cam
        R, t = ext[:3, :3], ext[:3, 3]
        centers.append(-R.T @ t)
    if not centers:
        raise ValueError(f"no cameras found in {calibration_path}")
    return np.asarray(centers)


def scene_center(calibration_path: str) -> np.ndarray:
    """Return C-bar, the mean camera position (the canonical frame functional)."""
    return camera_centers(calibration_path).mean(axis=0)


def _report_scene(scene_dir: str, anchor: str | None, verbose: bool):
    calibration_path = os.path.join(scene_dir, "calibration.json")
    centers = camera_centers(calibration_path)
    cbar = centers.mean(axis=0)
    name = os.path.basename(os.path.normpath(scene_dir))

    if verbose:
        print(f"=== {name} ===")
        for i, c in enumerate(centers):
            print(f"  cam {i:2d}: centre = {np.round(c, 2).tolist()}")
        print(f"  camera xy extent: "
              f"x[{centers[:, 0].min():.1f}, {centers[:, 0].max():.1f}]  "
              f"y[{centers[:, 1].min():.1f}, {centers[:, 1].max():.1f}]")
        print(f"  C-bar (xy) = ({cbar[0]:.4f}, {cbar[1]:.4f})   "
              f"|C-bar| = {np.hypot(*cbar[:2]):.2f} m")

        if anchor:
            a = np.load(anchor)
            print(f"  anchor prior ({a.shape[0]} anchors, training gauge): "
                  f"mean xy = ({a[:, 0].mean():.2f}, {a[:, 1].mean():.2f})  "
                  f"extent x[{a[:, 0].min():.1f}, {a[:, 0].max():.1f}] "
                  f"y[{a[:, 1].min():.1f}, {a[:, 1].max():.1f}]")
            print(f"  -> in the absolute frame the scene sits {np.hypot(*cbar[:2]):.1f} m "
                  "from where the anchors live; canonicalization removes exactly this offset.")
        print()
    return name, len(centers), cbar


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene_dir", help="a single scene directory containing calibration.json")
    ap.add_argument("--dataset_root", help="MTMC_Tracking_2026 root (tabulate a whole split)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"],
                    help="split to tabulate with --dataset_root (default: test)")
    ap.add_argument("--anchor", default=None,
                    help="optional k-means anchor .npy, to also report the anchor-prior extent")
    ap.add_argument("--quiet", action="store_true", help="table only, no per-camera detail")
    args = ap.parse_args()

    if not args.scene_dir and not args.dataset_root:
        ap.error("pass --scene_dir or --dataset_root")

    if args.scene_dir:
        scene_dirs = [args.scene_dir]
    else:
        split_dir = os.path.join(args.dataset_root, args.split)
        scene_dirs = sorted(
            os.path.join(split_dir, d) for d in os.listdir(split_dir)
            if os.path.exists(os.path.join(split_dir, d, "calibration.json")))
        if not scene_dirs:
            raise SystemExit(f"no scenes with calibration.json under {split_dir}")

    rows = [_report_scene(d, args.anchor, verbose=not args.quiet) for d in scene_dirs]

    print(f"{'scene':16} {'cams':>5} {'shift_x':>10} {'shift_y':>10} {'|C-bar|':>9}")
    print("-" * 54)
    for name, n_cams, cbar in rows:
        print(f"{name:16} {n_cams:5d} {cbar[0]:10.4f} {cbar[1]:10.4f} {np.hypot(*cbar[:2]):9.2f}")
    print("\nPass these as `build_pkl.py --shift_x <shift_x> --shift_y <shift_y>`, "
          "or just use --auto_center (identical, computed on the fly).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
