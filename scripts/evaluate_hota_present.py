#!/usr/bin/env python3
"""Correct present-class 3D-HOTA aggregation for AI City 2026 Track 1.

Why this exists
---------------
scripts/evaluate_track1_hota.py drives the 2025 ZV TrackEval fork and prints a
class-averaged HOTA (`cls_comb_cls_av`) that averages over ALL classes in the
CLASSES_TO_EVAL list, counting every class that is ABSENT from a scene's GT as 0.
That is NOT the official 2026 metric and heavily penalises scenes that only
contain a subset of the 7 classes (e.g. Warehouse_022 has only Person/NovaCarter/
Transporter in frames 0-1799).

Official 2026 metric implemented here
-------------------------------------
  1. 3D HOTA per (scene, class)                      [from TrackEval, mean over alpha]
  2. per-scene combined HOTA = mean over classes PRESENT in that scene's GT
  3. overall combined HOTA    = per-scene combined HOTA weighted across scenes by
                                object count (default: unique GT track ids / scene)

"Present" = the class has >=1 GT detection in that scene (Count.GT_Dets > 0).

This module wraps the *existing* TrackEval fork and the AICityChallenge2026 dataset
adapter defined in scripts/evaluate_track1_hota.py -- it does not re-implement HOTA.
A single TrackEval pass yields per-(scene,class) HOTA + Count, which we then
re-aggregate; no per-class re-runs needed. It is both a CLI and an importable
library (see `evaluate()`), so sweep_tracking.py can call it directly.

Usage:
  python eval_hota_present.py --gt GT.txt --pred PRED.txt [--classes ...]
                              [--weight gt_ids|gt_dets|equal] [--json OUT.json] [--quiet]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

# --- locate + reuse the ZV TrackEval fork and the AICityChallenge2026 adapter ---
_THIS = Path(__file__).resolve()
_SCRIPTS_DIR = _THIS.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Importing this module inserts TRACKEVAL_ROOT onto sys.path and pulls in trackeval,
# and gives us the 2026 dataset adapter + canonical CLASS_NAMES. We reuse both.
import evaluate_hota as _zv  # noqa: E402

import trackeval  # noqa: E402  (importable thanks to _zv's sys.path insertion)

CLASS_NAMES = _zv.CLASS_NAMES  # 7 canonical class display names, index == class id
DATASET_NAME = "AICityChallenge2026"
TRACKER_NAME = "prediction"

# class display name -> integer class id (index in CLASS_NAMES)
NAME_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


@contextlib.contextmanager
def _maybe_silence(quiet: bool):
    if not quiet:
        yield
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def run_trackeval(gt_path, pred_path, classes=None, cores=1, quiet=True):
    """Run one TrackEval pass and return the per-sequence result tree.

    Returns res: dict[seq][cls_name_lower] -> {'HOTA': {...arrays...}, 'Count': {...}}
    plus the 'COMBINED_SEQ' key (per-class combined-across-scenes + ZV cls averages).
    Sequence keys are the raw scene ids (strings, e.g. '22').
    """
    gt_path = str(gt_path)
    pred_path = str(pred_path)
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"GT file not found: {gt_path}")
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    classes = list(classes) if classes else list(CLASS_NAMES)

    evaluator = trackeval.Evaluator(
        {
            "USE_PARALLEL": cores > 1,
            "NUM_PARALLEL_CORES": max(1, cores),
            "BREAK_ON_ERROR": True,
            "PRINT_RESULTS": False,
            "PRINT_ONLY_COMBINED": False,
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": False,
            "DISPLAY_LESS_PROGRESS": True,
            "OUTPUT_SUMMARY": False,
            "OUTPUT_DETAILED": False,
            "OUTPUT_EMPTY_CLASSES": True,
            "PLOT_CURVES": False,
        }
    )
    dataset = _zv.AICityChallenge2026(
        {
            "GT_FILE_PATH": gt_path,
            "TRACKER_FILE_PATH": pred_path,
            "CLASSES_TO_EVAL": classes,
            "PRINT_CONFIG": False,
        }
    )
    metrics_list = [trackeval.metrics.HOTA()]
    with _maybe_silence(quiet):
        output_res, output_msg = evaluator.evaluate([dataset], metrics_list)

    msg = output_msg[DATASET_NAME][TRACKER_NAME]
    if msg != "Success":
        raise RuntimeError(f"TrackEval failed: {msg}")
    return output_res[DATASET_NAME][TRACKER_NAME]


def _scalar(field_arr):
    """TrackEval reports per-alpha arrays; the headline metric is their mean, x100."""
    return float(np.mean(field_arr)) * 100.0


def _cls_metrics(cls_res):
    """Extract headline metrics + counts for one (scene,class) result block."""
    hota = cls_res["HOTA"]
    count = cls_res["Count"]
    return {
        "HOTA": _scalar(hota["HOTA"]),
        "DetA": _scalar(hota["DetA"]),
        "AssA": _scalar(hota["AssA"]),
        "LocA": _scalar(hota["LocA"]),
        "GT_Dets": int(count["GT_Dets"]),
        "GT_IDs": int(count["GT_IDs"]),
        "Dets": int(count["Dets"]),
        "IDs": int(count["IDs"]),
    }


def summarize(res, classes=None, weight="gt_ids"):
    """Re-aggregate a TrackEval result tree into the present-class 2026 metric.

    weight: how to weight scenes when forming the overall present-class HOTA.
        'gt_ids'  -> number of unique GT objects (track ids) in the scene [default]
        'gt_dets' -> number of GT detections (boxes) in the scene
        'equal'   -> simple unweighted mean over scenes
    """
    classes = list(classes) if classes else list(CLASS_NAMES)
    class_lowers = [c.lower() for c in classes]
    seqs = [k for k in res.keys() if k != "COMBINED_SEQ"]

    metric_keys = ["HOTA", "DetA", "AssA", "LocA"]
    per_scene = {}
    for seq in sorted(seqs, key=lambda s: (len(s), s)):
        per_class = {}
        present = []
        for cname, clow in zip(classes, class_lowers):
            if clow not in res[seq]:
                continue
            m = _cls_metrics(res[seq][clow])
            m["present"] = m["GT_Dets"] > 0
            per_class[cname] = m
            if m["present"]:
                present.append(cname)

        combined = {k: (float(np.mean([per_class[c][k] for c in present])) if present else 0.0)
                    for k in metric_keys}
        object_count = sum(per_class[c]["GT_IDs"] for c in present)
        gt_det_count = sum(per_class[c]["GT_Dets"] for c in present)
        per_scene[seq] = {
            "scene_name": f"Warehouse_{int(seq):03d}" if seq.isdigit() else seq,
            "present_classes": present,
            "combined": combined,
            "object_count": int(object_count),
            "gt_det_count": int(gt_det_count),
            "per_class": per_class,
        }

    # ----- overall: weight per-scene combined metrics across scenes -----
    def _weight_for(seq):
        if weight == "equal":
            return 1.0
        if weight == "gt_dets":
            return float(per_scene[seq]["gt_det_count"])
        return float(per_scene[seq]["object_count"])  # gt_ids (default)

    overall = {}
    scenes_with_present = [s for s in per_scene if per_scene[s]["present_classes"]]
    for k in metric_keys:
        vals = np.array([per_scene[s]["combined"][k] for s in scenes_with_present], dtype=float)
        if len(vals) == 0:
            overall[f"{k}_weighted"] = 0.0
            overall[f"{k}_unweighted"] = 0.0
            continue
        w = np.array([_weight_for(s) for s in scenes_with_present], dtype=float)
        if w.sum() <= 0:
            w = np.ones_like(w)
        overall[f"{k}_weighted"] = float(np.sum(vals * w) / np.sum(w))
        overall[f"{k}_unweighted"] = float(np.mean(vals))
    overall["weight_scheme"] = weight
    overall["num_scenes"] = len(scenes_with_present)

    # ----- per-class overall (TrackEval's across-scene combine, det-averaged) -----
    per_class_overall = {}
    combined_seq = res.get("COMBINED_SEQ", {})
    for cname, clow in zip(classes, class_lowers):
        if clow in combined_seq and "HOTA" in combined_seq[clow]:
            m = _cls_metrics(combined_seq[clow])
            m["present"] = m["GT_Dets"] > 0
            per_class_overall[cname] = m

    # ----- ZV all-class-averaged HOTA (the metric we are correcting), for reference -----
    zv_all_class = None
    if "cls_comb_cls_av" in combined_seq and "HOTA" in combined_seq["cls_comb_cls_av"]:
        zv_all_class = _scalar(combined_seq["cls_comb_cls_av"]["HOTA"]["HOTA"])

    return {
        "per_scene": per_scene,
        "overall": overall,
        "per_class_overall": per_class_overall,
        "zv_all_class_avg_HOTA": zv_all_class,
        "classes": classes,
        "weight": weight,
    }


def evaluate(gt_path, pred_path, classes=None, cores=1, weight="gt_ids", quiet=True):
    """One-call convenience: TrackEval pass -> present-class 2026 summary dict."""
    res = run_trackeval(gt_path, pred_path, classes=classes, cores=cores, quiet=quiet)
    return summarize(res, classes=classes, weight=weight)


# ---- key scalar accessor used by the sweep harness ----
def headline_hota(summary):
    """The single number to optimise: object-weighted present-class HOTA (0-100)."""
    return summary["overall"].get("HOTA_weighted", 0.0)


def _fmt_row(cols, widths):
    return "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))


def print_report(summary):
    widths = [14, 8, 8, 8, 8, 9, 8, 8]
    header = ["class", "HOTA", "DetA", "AssA", "LocA", "GT_Dets", "GT_IDs", "present"]
    print("=" * 78)
    print("Present-class 3D-HOTA report (official 2026 aggregation)")
    print("=" * 78)
    for seq, sc in summary["per_scene"].items():
        print(f"\n[Scene {seq} = {sc['scene_name']}]  objects(GT_IDs)={sc['object_count']}  "
              f"GT_dets={sc['gt_det_count']}  present={sc['present_classes']}")
        print("  " + _fmt_row(header, widths))
        for cname, m in sc["per_class"].items():
            print("  " + _fmt_row(
                [cname, f"{m['HOTA']:.2f}", f"{m['DetA']:.2f}", f"{m['AssA']:.2f}",
                 f"{m['LocA']:.2f}", m["GT_Dets"], m["GT_IDs"], "yes" if m["present"] else "-"],
                widths))
        c = sc["combined"]
        print(f"  --> present-class avg:  HOTA={c['HOTA']:.2f}  DetA={c['DetA']:.2f}  "
              f"AssA={c['AssA']:.2f}  LocA={c['LocA']:.2f}")

    print("\n" + "-" * 78)
    print("Per-class OVERALL (combined across scenes, det-averaged):")
    print("  " + _fmt_row(header, widths))
    for cname, m in summary["per_class_overall"].items():
        print("  " + _fmt_row(
            [cname, f"{m['HOTA']:.2f}", f"{m['DetA']:.2f}", f"{m['AssA']:.2f}",
             f"{m['LocA']:.2f}", m["GT_Dets"], m["GT_IDs"], "yes" if m["present"] else "-"],
            widths))

    ov = summary["overall"]
    print("\n" + "=" * 78)
    print("OVERALL present-class combined HOTA")
    print("=" * 78)
    print(f"  weight scheme                                 : {ov['weight_scheme']} "
          f"({ov['num_scenes']} scene(s) with present classes)")
    print(f"  present-class HOTA (object-weighted scenes)   : {ov['HOTA_weighted']:.2f}")
    print(f"  present-class HOTA (unweighted scene mean)    : {ov['HOTA_unweighted']:.2f}")
    print(f"  present-class DetA / AssA / LocA (weighted)   : "
          f"{ov['DetA_weighted']:.2f} / {ov['AssA_weighted']:.2f} / {ov['LocA_weighted']:.2f}")
    if summary["zv_all_class_avg_HOTA"] is not None:
        print(f"  [ref] ZV all-class-averaged HOTA (WRONG metric): "
              f"{summary['zv_all_class_avg_HOTA']:.2f}")
    print("=" * 78)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="ground-truth txt (11-col submission format)")
    ap.add_argument("--pred", required=True, help="prediction txt (11-col submission format)")
    ap.add_argument("--classes", nargs="+", default=CLASS_NAMES,
                    help="class display names to evaluate (default: all 7)")
    ap.add_argument("--weight", choices=["gt_ids", "gt_dets", "equal"], default="gt_ids",
                    help="scene weighting for the overall present-class HOTA")
    ap.add_argument("--cores", type=int, default=1,
                    help="TrackEval parallel cores (>1 evaluates scenes in parallel)")
    ap.add_argument("--json", default=None, help="optional path to dump the summary as JSON")
    ap.add_argument("--quiet", action="store_true", help="suppress TrackEval's own stdout")
    return ap.parse_args()


def main():
    args = parse_args()
    summary = evaluate(args.gt, args.pred, classes=args.classes,
                       cores=args.cores, weight=args.weight, quiet=True)
    print_report(summary)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[json] wrote summary -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
