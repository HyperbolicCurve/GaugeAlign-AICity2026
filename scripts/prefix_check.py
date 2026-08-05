#!/usr/bin/env python3
"""Prefix-invariance test: does the pipeline ever read a future frame?

The paper (Sec. 4.1) claims every submitted configuration is *online*. Whether a
system is online is a property of the whole pipeline, not of the base detector:
post-processing can silently look ahead. So we measure it instead of asserting it.

The test
--------
For a set of cutoff frames T, take the output produced from the FULL sequence and
restrict it to frames <= T. Then re-run the same post-processing on the PREFIX of
the input (rows with frame <= T only). If the pipeline is causal the two must be
identical: no decision about a frame t <= T may depend on anything after T.

    stage(full sequence) restricted to frames<=T   ==   stage(rows with frame<=T)

What is checked
---------------
  submitted    the shipped rank-1 configuration -- the detector output as emitted,
               with no temporal post-processing at all (configs/gaugealign_cd95.yaml).
  hysteresis   the causal two-threshold gate that was evaluated but not shipped.
  gapfill      the gap-interpolation variant (submission `bundlefix`, 56.38) that
               WAS discarded for being non-causal.

`gapfill` is included on purpose and is EXPECTED TO FAIL. A test nothing can fail
proves nothing; gapfill failing is what shows the check has teeth, and it is the
direct evidence for the paper's statement that gap interpolation forfeits the
online status. The script's exit code is 0 only when every stage behaves as
expected -- including gapfill leaking.

Usage
-----
    python scripts/prefix_check.py <pred_with_scores.txt>
    python scripts/prefix_check.py <pred.txt> --stages submitted hysteresis
    python scripts/prefix_check.py <pred.txt> --k 5 --n_cutoffs 400 --verbose

The prediction file must still carry its 12th (score) column; the packaged
track1.txt has it stripped, so use work/pred_test_cd95_<scene>.txt.

On sampling: a leak is only visible when a cutoff lands inside the window the
stage looks ahead over, which for gapfill means strictly inside a hole of <= K
frames. Those are sparse -- 5 cutoffs miss them entirely and gapfill then looks
clean. The default of 200 catches them reliably (16/200 on Warehouse_026). Cost
is linear in cutoffs x rows, so prefer a short scene (026/027, 1800 frames,
~15 s) over a 9000-frame one.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "gaugealign"))
import postproc as P  # noqa: E402

HIGH, LOW = 0.2, 0.05


def _key(r):
    """Identity of an emitted box: (scene, class, object id, frame) plus geometry."""
    return (r[0], r[1], r[2], r[3]) + tuple(round(v, 4) for v in r[4:11])


STAGES = {
    # the shipped configuration applies no temporal post-processing whatsoever
    "submitted":  lambda rows, k: list(rows),
    "hysteresis": lambda rows, k: P.do_hysteresis(rows, HIGH, LOW, k),
    "gapfill":    lambda rows, k: P.do_gapfill(rows, k),
}
EXPECTED_CAUSAL = {"submitted": True, "hysteresis": True, "gapfill": False}


def check(rows, stage_name, k, cutoffs, verbose):
    """Return (is_causal, n_leaks, n_cutoffs) for one stage.

    Sampling matters. A lookahead leak only becomes visible when a cutoff T falls
    inside the window the stage looks ahead over -- for gapfill, strictly inside a
    hole of <= K frames. Such holes are sparse, so a handful of cutoffs can miss
    them entirely and make a non-causal stage look clean. Hence the dense default.
    """
    stage = STAGES[stage_name]
    full = stage(rows, k)
    frames = sorted({r[3] for r in rows})
    if not frames:
        print("    no frames in input")
        return None, 0, 0

    leaks = 0
    for q in cutoffs:
        T = frames[min(int(len(frames) * q), len(frames) - 1)]
        from_full = sorted(_key(r) for r in full if r[3] <= T)
        from_prefix = sorted(_key(r) for r in stage([r for r in rows if r[3] <= T], k))
        same = from_full == from_prefix
        leaks += (not same)
        if verbose or not same:
            print(f"    T={T:>6}  full|<=T {len(from_full):>8} rows   "
                  f"prefix {len(from_prefix):>8} rows   {'OK' if same else 'LEAK'}")
    if not verbose:
        print(f"    {len(cutoffs)} cutoffs tested, {leaks} leak(s)")
    return leaks == 0, leaks, len(cutoffs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pred", help="12-column prediction file (... yaw score)")
    ap.add_argument("--stages", nargs="+", default=["submitted", "hysteresis", "gapfill"],
                    choices=sorted(STAGES))
    ap.add_argument("--k", type=int, default=5, help="window length for hysteresis/gapfill")
    ap.add_argument("--n_cutoffs", type=int, default=200,
                    help="number of evenly spaced cutoffs (default 200; see note below)")
    ap.add_argument("--cutoffs", nargs="+", type=float, default=None,
                    help="explicit cutoff fractions, overriding --n_cutoffs")
    ap.add_argument("--verbose", action="store_true", help="print every cutoff, not just leaks")
    args = ap.parse_args()

    cutoffs = args.cutoffs or [i / (args.n_cutoffs + 1) for i in range(1, args.n_cutoffs + 1)]

    rows = P.load(args.pred)
    n_frames = len({r[3] for r in rows})
    print(f"input: {args.pred}\n       {len(rows)} rows over {n_frames} frames, "
          f"K={args.k}, {len(cutoffs)} cutoffs\n")

    verdicts = {}
    for name in args.stages:
        expect = "causal" if EXPECTED_CAUSAL[name] else "NON-causal, must leak"
        print(f"  [{name}]  (expected: {expect})")
        verdicts[name] = check(rows, name, args.k, cutoffs, args.verbose)
        print()

    print("=" * 66)
    all_as_expected = True
    for name, (causal, leaks, total) in verdicts.items():
        if causal is None:
            continue
        agrees = (causal == EXPECTED_CAUSAL[name])
        all_as_expected &= agrees
        status = "causal" if causal else f"reads future frames ({leaks}/{total})"
        print(f"  {name:12} {status:32} {'as expected' if agrees else 'UNEXPECTED'}")
    print("=" * 66)

    if verdicts.get("submitted", (None,))[0]:
        print("\nThe submitted configuration is prefix-invariant: it is online.")
    if verdicts.get("gapfill", (None,))[0] is False:
        print("The discarded gapfill variant leaks, confirming the test discriminates.")

    return 0 if all_as_expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
