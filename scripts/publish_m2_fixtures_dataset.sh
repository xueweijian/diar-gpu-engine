#!/usr/bin/env bash
# Publish the M1 parity fixtures as a versioned Kaggle dataset for the
# Stage 3.4 K6 kernel (q8-vs-q8 four-fixture endorsement).
#
# Creates weijianxue/diar-m2-fixtures with the four v12/v13 fixture dirs
# (manifest.json + probs.f32 each, ~230KB total):
#   v12-short-streaming-r0    short.wav, default streaming geometry
#   v12-mid-offline-full-r0   video2.wav, production --offline (full attn)
#   v13-mid-streaming-r0      video2.wav, default streaming geometry
#   v13-mid-offline-preset-r0 video2.wav, --preset offline (offline geometry)
#
# Case -> our k5_runner mapping (pinned in kaggle/m2_stage3/m2_stage3_k6.py):
#   streaming cases        -> --run (default geometry), compare POSTgate
#   offline-preset case    -> --run --offline, compare POSTgate
#   offline-full case      -> --full-offline, compare PREGATE (no BirthGate
#                             upstream on the DiarModel::diarize_offline path)
#
# Audio lives in the existing datasets diar-smoke-audio / diar-real-audio-5
# (mounted read-only by the kernel); the q8 GGUF is downloaded from HF at
# kernel runtime (same as .nemo in K5-A).
#
# Usage: scripts/publish_m2_fixtures_dataset.sh [--version]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_ID="weijianxue/diar-m2-fixtures"
FIX_ROOT="$REPO_ROOT/parity/fixtures"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

for d in v12-short-streaming-r0 v12-mid-offline-full-r0 \
         v13-mid-streaming-r0 v13-mid-offline-preset-r0; do
  test -f "$FIX_ROOT/$d/manifest.json" || { echo "missing $FIX_ROOT/$d" >&2; exit 1; }
  test -f "$FIX_ROOT/$d/probs.f32" || { echo "missing $FIX_ROOT/$d/probs.f32" >&2; exit 1; }
  mkdir -p "$STAGE/$d"
  cp "$FIX_ROOT/$d/manifest.json" "$FIX_ROOT/$d/probs.f32" "$STAGE/$d/"
done
cat > "$STAGE/dataset-metadata.json" <<JSON
{
  "title": "diar-m2-fixtures",
  "id": "$DATASET_ID",
  "licenses": [{"name": "MIT"}]
}
JSON

if [ "${1:-}" = "--version" ]; then
  kaggle datasets version -p "$STAGE" -m "sync m2 k6 fixtures"
else
  kaggle datasets create -p "$STAGE" --dir-mode zip
fi
