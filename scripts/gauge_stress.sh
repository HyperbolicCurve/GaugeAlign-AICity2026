#!/usr/bin/env bash
# =============================================================================
# Gauge stress curve -- paper Sec. 4.3, Table 3, Fig. 5 (left).
#
# Sweeps ONE variable: the world-frame offset d (metres) along -C-bar_hat.
#   d = 0        the canonical gauge (what GaugeAlign uses)
#   d = |C-bar|  the absolute site frame (what naive transfer gives you)
# Everything else is held fixed: images, weights, anchors, thresholds and the
# exact decoded frames. Ground truth is always scored in the original absolute
# frame, so the metric stays comparable across the sweep.
#
# The shift handed to build_pkl is  C-bar - d * C-bar_hat: the scene is
# canonicalized and then deliberately displaced back by d metres.
#
# Result (paper Table 3): mean DetA 36.16 at d=0 -> 12.25 at d=|C-bar|, and the
# four-camera scene (W022) collapses to exactly 0.00.
#
# Usage:  DATASET_ROOT=... NGC_MODEL_DIR=... bash scripts/gauge_stress.sh
# Then:   python scripts/gauge_score.py
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:?set DATASET_ROOT to the MTMC_Tracking_2026 root}"
NGC_MODEL_DIR="${NGC_MODEL_DIR:?set NGC_MODEL_DIR to the Sparse4D model dir}"
WORK_DIR="${WORK_DIR:-$REPO/work}"
PREP_ENV="${PREP_ENV:-gaugealign-prep}"
INFER_ENV="${INFER_ENV:-gaugealign}"

W="$WORK_DIR/gauge"
CKPT="$NGC_MODEL_DIR/sparse4d_warehouse_v2.2_r50.pth"
ANCHOR="$NGC_MODEL_DIR/_ov_kmeans900_v2.2_r50.npy"
EXPYAML="$NGC_MODEL_DIR/experiment.yaml"
mkdir -p "$W"

END=900; STRIDE=6           # 150 frames spread across the sequence

# scene -> "C-bar_x C-bar_y |C-bar|"  (from calibration; see configs/scenes.yaml).
# |C-bar| is given to 1 dp so the output filename tags line up with gauge_score.py.
declare -A CBAR
CBAR[20]="-62.1733 -59.4065 86.0"
CBAR[21]="-62.1733 -59.4065 86.0"
CBAR[22]="-62.7303 -83.0016 104.0"

run_one () {   # scene_id gpu d [d ...]
  local SID="$1" GPU="$2"; shift 2; local DS=("$@")
  local SCENE="Warehouse_0${SID}"
  local SCENE_DIR="$DATASET_ROOT/val/$SCENE"
  local FRAMES="$W/frames_${SCENE}"
  local CX CY CN UX UY
  read -r CX CY CN <<< "${CBAR[$SID]}"
  UX=$(python3 -c "print($CX/$CN)")      # unit vector along C-bar
  UY=$(python3 -c "print($CY/$CN)")

  for d in "${DS[@]}"; do
    local SX SY TAG PKL PRED GTARG
    SX=$(python3 -c "print(round($CX - $d*$UX, 4))")
    SY=$(python3 -c "print(round($CY - $d*$UY, 4))")
    TAG=$(python3 -c "print(f'{$d:05.1f}'.replace('.','p'))")
    PKL="$W/pkl_${SID}_d${TAG}.pkl"
    PRED="$W/pred_${SID}_d${TAG}.txt"
    # GT is shift-independent -- write it once, from the d=0 build
    GTARG=""
    if [ "$d" = "0" ] || [ "$d" = "0.0" ]; then GTARG="--out_gt $W/gt_${SID}.txt"; fi

    echo "[$(date +%T)] W$SID d=$d shift=($SX,$SY) gpu$GPU"
    conda run --no-capture-output -n "$PREP_ENV" python "$REPO/gaugealign/build_pkl.py" \
      --scene_dir "$SCENE_DIR" --scene_name "$SCENE" --scene_id "$SID" \
      --start 0 --end $END --stride $STRIDE \
      --frames_dir "$FRAMES" --out_pkl "$PKL" --half_res \
      --shift_x "$SX" --shift_y "$SY" $GTARG \
      > "$W/build_${SID}_d${TAG}.log" 2>&1

    CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n "$INFER_ENV" \
      python "$REPO/gaugealign/run_infer.py" \
        --config "$EXPYAML" --ckpt "$CKPT" --anchor "$ANCHOR" \
        --pkl "$PKL" --out "$PRED" --scene_id "$SID" \
        --score_thr 0.2 --dec_thr 0.0 --to_rgb 0 --remap --half_res --num_workers 6 \
        > "$W/infer_${SID}_d${TAG}.log" 2>&1

    echo "[$(date +%T)] W$SID d=$d done, rows=$(wc -l < "$PRED")"
    rm -f "$PKL"
  done
  rm -rf "$FRAMES"
}

GRID=(0 10 25 50)
# GPU 0: W020 (+ its |C-bar|); GPU 1: W021 then W022
( run_one 20 0 "${GRID[@]}" 86.0 ) &
( run_one 21 1 "${GRID[@]}" 86.0 ; run_one 22 1 "${GRID[@]}" 104.0 ) &
wait

echo "[$(date +%T)] gauge sweep complete -- now run:  python scripts/gauge_score.py"
