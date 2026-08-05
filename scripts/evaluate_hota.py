#!/usr/bin/env python3
"""Evaluate Track 1 txt files with the 2025 ZV TrackEval adapter.

This is a local development helper, NOT the official evaluation server -- all
hidden-test numbers in the paper come from the challenge server. It reuses the
2025 winning team's TrackEval fork and patches only the 2026 class list plus the
zero-based frame-count boundary.

For the official 2026 aggregation (per-scene mean over the classes actually
present, weighted across scenes by object count) use evaluate_hota_present.py,
which wraps this module.

The TrackEval checkout is expected at third_party/TrackEval; run
scripts/setup_trackeval.sh, or point TRACKEVAL_ROOT at an existing checkout.
"""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import freeze_support
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKEVAL_ROOT = Path(
    os.environ.get("TRACKEVAL_ROOT", REPO_ROOT / "third_party" / "TrackEval")
)

CLASS_NAMES = [
    "Person",
    "Forklift",
    "NovaCarter",
    "Transporter",
    "FourierGR1T2",
    "AgilityDigit",
    "PalletTruck",
]

if TRACKEVAL_ROOT.exists():
    sys.path.insert(0, str(TRACKEVAL_ROOT))

import trackeval  # noqa: E402
from trackeval.datasets.aicity_challenge import AICityChallengeSingleFile  # noqa: E402
from trackeval.utils import TrackEvalException  # noqa: E402


class AICityChallenge2026(AICityChallengeSingleFile):
    @staticmethod
    def get_default_dataset_config():
        return {
            "GT_FILE_PATH": None,
            "TRACKER_FILE_PATH": None,
            "CLASSES_TO_EVAL": CLASS_NAMES,
            "PRINT_CONFIG": True,
        }

    def __init__(self, config=None):
        super().__init__(config)
        self.class_name_to_class_id = {
            "person": 0,
            "forklift": 1,
            "novacarter": 2,
            "transporter": 3,
            "fouriergr1t2": 4,
            "agilitydigit": 5,
            "pallettruck": 6,
        }

    def _get_seq_info(self):
        seq_list = set()
        seq_lengths = {}
        with open(self.gt_file, encoding="utf-8") as f:
            for line in f:
                try:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    scene_id = parts[0]
                    frame_id = int(parts[3])
                    seq_list.add(scene_id)
                    seq_lengths[scene_id] = max(seq_lengths.get(scene_id, -1), frame_id + 1)
                except (IndexError, ValueError) as exc:
                    raise TrackEvalException(
                        f"Error parsing GT file line: {line.strip()}. Reason: {exc}"
                    ) from exc
        return sorted(seq_list), seq_lengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", default=CLASS_NAMES)
    parser.add_argument("--cores", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not TRACKEVAL_ROOT.exists():
        raise FileNotFoundError(
            f"Missing TrackEval checkout: {TRACKEVAL_ROOT}\n"
            "Run  bash scripts/setup_trackeval.sh  (or set TRACKEVAL_ROOT=...)")

    freeze_support()
    evaluator = trackeval.Evaluator(
        {
            "USE_PARALLEL": args.cores > 1,
            "NUM_PARALLEL_CORES": args.cores,
            "PRINT_RESULTS": True,
            "PRINT_ONLY_COMBINED": False,
            "DISPLAY_LESS_PROGRESS": True,
        }
    )
    dataset_list = [
        AICityChallenge2026(
            {
                "GT_FILE_PATH": str(args.gt),
                "TRACKER_FILE_PATH": str(args.pred),
                "CLASSES_TO_EVAL": args.classes,
                "PRINT_CONFIG": True,
            }
        )
    ]
    metrics_list = [trackeval.metrics.HOTA()]
    evaluator.evaluate(dataset_list, metrics_list)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
