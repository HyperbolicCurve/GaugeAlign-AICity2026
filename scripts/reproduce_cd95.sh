#!/usr/bin/env bash
# =============================================================================
# GaugeAlign -- reproduce the rank-1 submission end to end.
#
#   AI City Challenge 2026 Track 1 -- Team EVA -- 3D HOTA 56.5447 (rank 1/10)
#   configs/gaugealign_cd95.yaml is the machine-readable form of this recipe.
#
# For each of the 5 hidden-test scenes this runs:
#   1. build_pkl.py --auto_center   canonicalize the gauge (t_i' = t_i + R_i*C-bar)
#                                   and decode frames at 960x540
#   2. run_infer.py  --remap        frozen Sparse4D forward pass; adds C-bar back
#                    --conf_decay   to every box centre, applies the class remap,
#                                   and sets the instance-bank temporal prior
#   3. floor_roi.py  --mode filter  geometric floor-ROI filter
# then concatenates the 5 scenes and packages track1.zip.
#
# Usage:
#   bash scripts/reproduce_cd95.sh
#   DATASET_ROOT=/data/MTMC_Tracking_2026 NGC_MODEL_DIR=/models/sparse4d_r50 \
#     bash scripts/reproduce_cd95.sh
#
# Configure via environment (defaults shown in docs/REPRODUCE.md):
#   DATASET_ROOT   MTMC_Tracking_2026 root (must contain test/Warehouse_0XX/)
#   NGC_MODEL_DIR  dir with the .pth, the anchor .npy and experiment.yaml
#   WORK_DIR       scratch + outputs                     (default: ./work)
#   GPUS           comma-separated GPU ids to fan out on (default: "0,1")
#   PREP_ENV       conda env for data prep               (default: gaugealign-prep)
#   INFER_ENV      conda env for inference               (default: gaugealign)
#   SCENES         scene ids to run                      (default: "23 24 25 26 27")
#   KEEP_FRAMES    1 to keep decoded frames for reuse    (default: 0)
#
# Disk: decoded frames are the bottleneck (~9000 frames x up to 20 cameras).
# They are deleted per scene unless KEEP_FRAMES=1.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:?set DATASET_ROOT to the MTMC_Tracking_2026 root (see docs/DATA.md)}"
NGC_MODEL_DIR="${NGC_MODEL_DIR:?set NGC_MODEL_DIR to the Sparse4D model dir (see docs/INSTALL.md)}"
WORK_DIR="${WORK_DIR:-$REPO/work}"
GPUS="${GPUS:-0,1}"
PREP_ENV="${PREP_ENV:-gaugealign-prep}"
INFER_ENV="${INFER_ENV:-gaugealign}"
SCENES="${SCENES:-23 24 25 26 27}"
KEEP_FRAMES="${KEEP_FRAMES:-0}"
TAG="${TAG:-cd95}"

# ---- the winning hyper-parameters (configs/gaugealign_cd95.yaml) -------------
CONF_DECAY=0.95        # temporal prior adaptation (lambda); native default 0.8
SCORE_THR=0.2          # emission threshold
DEC_THR=0.0            # decoder score_threshold override
TO_RGB=0               # keep BGR channel order, as in training
ROI_MARGIN=20.0        # floor-ROI margin (m) beyond the camera-centre bounds
ROI_ZMIN=-0.5
ROI_ZMAX=3.2
NUM_WORKERS=6

CKPT="$NGC_MODEL_DIR/sparse4d_warehouse_v2.2_r50.pth"
ANCHOR="$NGC_MODEL_DIR/_ov_kmeans900_v2.2_r50.npy"
EXPYAML="$NGC_MODEL_DIR/experiment.yaml"

# frames per scene, from the challenge data
declare -A NFRAMES=([23]=9000 [24]=9000 [25]=9000 [26]=1800 [27]=1800)

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
mkdir -p "$WORK_DIR/logs"

for f in "$CKPT" "$ANCHOR" "$EXPYAML"; do
  [ -f "$f" ] || { echo "ERROR: missing $f -- see docs/INSTALL.md"; exit 1; }
done
[ -d "$REPO/third_party/tao_pytorch_backend" ] || \
  { echo "ERROR: patched TAO backend missing. Run scripts/setup_tao_backend.sh"; exit 1; }

run_scene () {   # scene_id gpu_id
  local SID="$1" GPU="$2"
  local SCENE="Warehouse_0${SID}"
  local SCENE_DIR="$DATASET_ROOT/test/$SCENE"
  local NF="${NFRAMES[$SID]:-9000}"
  local PKL="$WORK_DIR/test_${SID}.pkl"
  local FRAMES="$WORK_DIR/frames_test/$SCENE"
  local PRED="$WORK_DIR/pred_test_${TAG}_${SID}.txt"

  [ -d "$SCENE_DIR" ] || { echo "ERROR: $SCENE_DIR not found"; exit 1; }
  if [ -s "$PRED" ]; then echo "[$SCENE] prediction exists, skipping"; return; fi

  # -- 1. gauge canonicalization + frame decode ------------------------------
  # --auto_center computes C-bar from calibration.json and rewrites every
  # extrinsic translation as t_i' = t_i + R_i * C-bar. The shift is stored in
  # the pkl metadata so run_infer.py can invert it exactly.
  if [ ! -s "$PKL" ] || [ ! -d "$FRAMES" ]; then
    echo "[$(date +%T)] [$SCENE] canonicalize + decode ($NF frames, gpu $GPU)"
    conda run --no-capture-output -n "$PREP_ENV" python "$REPO/gaugealign/build_pkl.py" \
      --scene_dir "$SCENE_DIR" --scene_name "$SCENE" --scene_id "$SID" \
      --start 0 --end "$NF" --stride 1 \
      --frames_dir "$FRAMES" --out_pkl "$PKL" \
      --auto_center --half_res \
      > "$WORK_DIR/logs/build_${SID}.log" 2>&1
  else
    echo "[$(date +%T)] [$SCENE] pkl + frames present, reusing"
  fi

  # -- 2. frozen detector + gauge inverse + protocol + temporal prior ---------
  echo "[$(date +%T)] [$SCENE] inference (lambda=$CONF_DECAY, gpu $GPU)"
  CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n "$INFER_ENV" \
    python "$REPO/gaugealign/run_infer.py" \
      --config "$EXPYAML" --ckpt "$CKPT" --anchor "$ANCHOR" \
      --pkl "$PKL" --out "$PRED" --scene_id "$SID" \
      --score_thr "$SCORE_THR" --dec_thr "$DEC_THR" --to_rgb "$TO_RGB" \
      --remap --half_res --num_workers "$NUM_WORKERS" \
      --conf_decay "$CONF_DECAY" \
      > "$WORK_DIR/logs/infer_${TAG}_${SID}.log" 2>&1

  [ "$KEEP_FRAMES" = "1" ] || rm -rf "$FRAMES"
  echo "[$(date +%T)] [$SCENE] done -> $PRED ($(wc -l < "$PRED") rows)"
}

# ---- fan the scenes out over the available GPUs -----------------------------
echo "=== GaugeAlign / $TAG : inference over scenes [$SCENES] on GPUs [$GPUS] ==="
i=0
for SID in $SCENES; do
  GPU="${GPU_ARR[$(( i % ${#GPU_ARR[@]} ))]}"
  run_scene "$SID" "$GPU" &
  i=$(( i + 1 ))
  # keep at most one job per GPU in flight (decoding is disk-bound)
  if [ $(( i % ${#GPU_ARR[@]} )) -eq 0 ]; then wait; fi
done
wait
echo "=== inference complete ==="

# ---- 3. floor-ROI filter + package ------------------------------------------
FILTERED=()
for SID in $SCENES; do
  SRC="$WORK_DIR/pred_test_${TAG}_${SID}.txt"
  ROI="$WORK_DIR/roi_${TAG}_${SID}.txt"
  echo "[$(date +%T)] [Warehouse_0${SID}] floor-ROI filter (margin ${ROI_MARGIN} m)"
  conda run -n "$PREP_ENV" python "$REPO/gaugealign/floor_roi.py" \
    --calib "$DATASET_ROOT/test/Warehouse_0${SID}/calibration.json" \
    --pred "$SRC" --mode filter --out "$ROI" \
    --margin "$ROI_MARGIN" --zmin "$ROI_ZMIN" --zmax "$ROI_ZMAX" \
    > "$WORK_DIR/logs/roi_${TAG}_${SID}.log" 2>&1
  # keep the 11 official columns (drop the trailing score column)
  awk '{print $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11}' "$ROI" > "${ROI%.txt}_11col.txt"
  FILTERED+=("${ROI%.txt}_11col.txt")
done

cat "${FILTERED[@]}" > "$WORK_DIR/pred_test_${TAG}_ALL.txt"
conda run -n "$PREP_ENV" python "$REPO/gaugealign/make_submission.py" \
  --pred "$WORK_DIR/pred_test_${TAG}_ALL.txt" \
  --out_txt "$WORK_DIR/track1.txt" --out_zip "$WORK_DIR/track1.zip"

cat <<EOF

=============================================================================
 Submission ready: $WORK_DIR/track1.zip

 Expected score on the final full hidden test set (paper Table 1/2):
     HOTA 56.5447 | DetA 55.6444 | AssA 49.3929 | LocA 79.5558   (rank 1/10)

 Verify the online (causal) property:
     python scripts/prefix_check.py $WORK_DIR/pred_test_${TAG}_23.txt 5
=============================================================================
EOF
