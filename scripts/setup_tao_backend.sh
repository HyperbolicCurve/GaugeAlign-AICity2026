#!/usr/bin/env bash
# =============================================================================
# Fetch NVIDIA's TAO PyTorch backend at the exact release we built against and
# apply the GaugeAlign patch.
#
# We do NOT vendor NVIDIA's code. The upstream repo is Apache-2.0; what this
# repository carries is the diff (patches/tao_sparse4d_gaugealign.patch), so the
# provenance of every line stays clear.
#
# The patch touches five files:
#   instance_bank.py       our per-instance survival gate (CVC-Assoc) and the
#                          cross-view consensus hooks; both OFF by default, so
#                          an unmodified run is bit-identical to upstream
#   criterion.py           optional background-query loss term (fine-tuning study)
#   sparse4d_head.py       plumb max_time_interval through to the instance bank
#   backbone_v2/__init__.py  import ResNet lazily so absent optional backbone
#                          dependencies (transformers, open_clip, ...) do not
#                          break a ResNet-50 run
#   ops/setup_ext.py       new: build the deformable-aggregation CUDA extension
#
# NOTE: the rank-1 result uses NONE of the optional paths above -- it runs the
# frozen upstream model. The patch is required only because of the import fix,
# the max_time_interval plumbing and the CUDA-extension build helper.
#
# Usage:  bash scripts/setup_tao_backend.sh [target_dir]
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-$REPO/third_party/tao_pytorch_backend}"
UPSTREAM="https://github.com/NVIDIA/tao_pytorch_backend"
COMMIT="d99ff772e2cff894015d81a96d2cb92119000e19"   # Release/7.0.1
PATCH="$REPO/patches/tao_sparse4d_gaugealign.patch"

if [ -d "$TARGET/.git" ]; then
  echo "==> $TARGET already exists; leaving it alone."
  echo "    Delete it first if you want a clean re-install."
  exit 0
fi

echo "==> cloning $UPSTREAM"
mkdir -p "$(dirname "$TARGET")"
git clone "$UPSTREAM" "$TARGET"

echo "==> checking out Release/7.0.1 ($COMMIT)"
git -C "$TARGET" checkout --quiet "$COMMIT"

echo "==> applying $PATCH"
git -C "$TARGET" apply --check "$PATCH"
git -C "$TARGET" apply "$PATCH"

cat <<EOF

==> TAO backend ready at $TARGET

Next, build the deformable-aggregation CUDA extension inside the inference env:

    conda activate gaugealign
    cd $TARGET/nvidia_tao_pytorch/cv/sparse4d/model/ops
    python setup_ext.py build_ext --inplace

Then verify:

    python -c "import sys; sys.path.insert(0,'$TARGET'); \\
               from nvidia_tao_pytorch.cv.sparse4d.model.sparse4d import Sparse4D; print('ok')"
EOF
