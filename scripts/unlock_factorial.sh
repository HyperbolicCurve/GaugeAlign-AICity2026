#!/usr/bin/env bash
# =============================================================================
# Unlock factorial -- paper Sec. 4.3, Table 4, Fig. 5 (right).
#
# Crosses the three corrections on Warehouse_022 (4 cameras, 150 frames):
#   canonicalization {on = C-bar shift, off = absolute (0,0)}
#   x  colour order  {to_rgb 0 = BGR as trained, 1 = RGB}
#   x  class remap   {on, off}
# = 8 cells. Canonicalization is a build-time factor (2 pkls); colour order and
# remap are inference flags (4 runs each).
#
# Ground truth is reused from the gauge sweep (work/gauge/gt_22.txt) -- same
# start/end/stride, so the frames line up. Run scripts/gauge_stress.sh first.
#
# Result (paper Table 4): all four un-canonicalized cells emit ZERO detections.
# With canonicalization on, dropping the remap costs 2.83 DetA and the wrong
# colour order 0.76 -- so the gauge term outweighs the largest protocol
# correction by 41.99 / 2.83 ~ 14.8x.
#
# Usage:  DATASET_ROOT=... NGC_MODEL_DIR=... bash scripts/unlock_factorial.sh
# Then:   python scripts/factorial_score.py
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:?set DATASET_ROOT to the MTMC_Tracking_2026 root}"
NGC_MODEL_DIR="${NGC_MODEL_DIR:?set NGC_MODEL_DIR to the Sparse4D model dir}"
WORK_DIR="${WORK_DIR:-$REPO/work}"
PREP_ENV="${PREP_ENV:-gaugealign-prep}"
INFER_ENV="${INFER_ENV:-gaugealign}"
GPU="${GPU:-0}"

W="$WORK_DIR/factorial"
CKPT="$NGC_MODEL_DIR/sparse4d_warehouse_v2.2_r50.pth"
ANCHOR="$NGC_MODEL_DIR/_ov_kmeans900_v2.2_r50.npy"
EXPYAML="$NGC_MODEL_DIR/experiment.yaml"

SCENE=Warehouse_022; SID=22
SCENE_DIR="$DATASET_ROOT/val/$SCENE"
FRAMES="$W/frames_${SCENE}"
END=900; STRIDE=6
# canonicalization ON = C-bar (identical to gauge sweep d=0); OFF = absolute (0,0)
CX=-62.7303; CY=-83.0016
mkdir -p "$W"

build () {   # tag shift_x shift_y
  local TAG="$1" SX="$2" SY="$3"
  conda run --no-capture-output -n "$PREP_ENV" python "$REPO/gaugealign/build_pkl.py" \
    --scene_dir "$SCENE_DIR" --scene_name "$SCENE" --scene_id "$SID" \
    --start 0 --end $END --stride $STRIDE \
    --frames_dir "$FRAMES" --out_pkl "$W/pkl_${TAG}.pkl" --half_res \
    --shift_x "$SX" --shift_y "$SY" \
    > "$W/build_${TAG}.log" 2>&1
  echo "[$(date +%T)] built pkl_${TAG} (shift $SX,$SY)"
}

infer () {   # canon_tag to_rgb remap{0,1}
  local RC="$1" TORGB="$2" REMAP="$3"
  local FLAG="" RTAG="noremap"
  if [ "$REMAP" = "1" ]; then FLAG="--remap"; RTAG="remap"; fi
  local OUT="$W/pred_${RC}_rgb${TORGB}_${RTAG}.txt"
  CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n "$INFER_ENV" \
    python "$REPO/gaugealign/run_infer.py" \
      --config "$EXPYAML" --ckpt "$CKPT" --anchor "$ANCHOR" \
      --pkl "$W/pkl_${RC}.pkl" --out "$OUT" --scene_id "$SID" \
      --score_thr 0.2 --dec_thr 0.0 --to_rgb "$TORGB" $FLAG --half_res --num_workers 6 \
      > "$W/infer_${RC}_rgb${TORGB}_${RTAG}.log" 2>&1
  echo "[$(date +%T)] canon=$RC rgb=$TORGB $RTAG -> $(wc -l < "$OUT") rows"
}

build on  "$CX" "$CY"
build off 0.0 0.0
for RC in on off; do
  for TORGB in 0 1; do
    for REMAP in 1 0; do
      infer "$RC" "$TORGB" "$REMAP"
    done
  done
done

rm -f "$W/pkl_on.pkl" "$W/pkl_off.pkl"
rm -rf "$FRAMES"
echo "[$(date +%T)] factorial complete -- now run:  python scripts/factorial_score.py"
