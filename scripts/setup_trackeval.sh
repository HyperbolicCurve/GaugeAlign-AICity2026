#!/usr/bin/env bash
# =============================================================================
# Fetch the TrackEval fork used for LOCAL 3D-HOTA evaluation.
#
# All hidden-test numbers in the paper come from the challenge evaluation
# server. This checkout only powers the local development metric used for the
# validation-scene studies (gauge stress curve, unlock factorial, CVC-Assoc
# val gate). scripts/evaluate_hota.py adapts it to the 2026 class list and
# zero-based frame indexing; scripts/evaluate_hota_present.py adds the official
# present-class aggregation on top.
#
# The fork is the one released by the 2025 Track 1 winning team (ZIOVISION),
# which carries the 3D-IoU AI City dataset adapter that upstream TrackEval lacks.
#
# Usage:  bash scripts/setup_trackeval.sh [target_dir]
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-$REPO/third_party/TrackEval}"
UPSTREAM="https://github.com/ZIOVISION/AIC2025_Track1_ZV"

if [ -d "$TARGET" ]; then
  echo "==> $TARGET already exists; leaving it alone."
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "==> cloning $UPSTREAM (for its TrackEval subdirectory)"
git clone --depth 1 "$UPSTREAM" "$TMP/zv"

[ -d "$TMP/zv/TrackEval" ] || { echo "ERROR: TrackEval/ not found in the clone"; exit 1; }

mkdir -p "$(dirname "$TARGET")"
cp -r "$TMP/zv/TrackEval" "$TARGET"

cat <<EOF

==> TrackEval ready at $TARGET

Verify:
    python scripts/evaluate_hota.py --help

Note this is a development approximation of the official metric, not the
challenge server. See docs/RESULTS.md.
EOF
