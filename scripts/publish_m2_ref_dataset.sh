#!/usr/bin/env bash
# Publish the Stage 0 .npz references as a versioned Kaggle dataset so the
# Stage 2 kernels can mount them as inputs (kernel code_file uploads cannot
# carry 164MB of references).
#
# Creates weijianxue/diar-m2-ref (public within the account, private listing)
# with m2_ref_short.npz + m2_ref_mid.npz + m2_stage0_verdict.json.
#
# Usage: scripts/publish_m2_ref_dataset.sh [--version]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_ID="weijianxue/diar-m2-ref"
REF_DIR="/var/minis/shared/diar-gpu-engine/m2-ref"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

for f in m2_ref_short.npz m2_ref_mid.npz m2_stage0_verdict.json; do
  test -f "$REF_DIR/$f" || { echo "missing $REF_DIR/$f" >&2; exit 1; }
  cp "$REF_DIR/$f" "$STAGE/$f"
done
cat > "$STAGE/dataset-metadata.json" <<JSON
{
  "title": "diar-m2-ref",
  "id": "$DATASET_ID",
  "licenses": [{"name": "MIT"}]
}
JSON

if [ "${1:-}" = "--version" ]; then
  kaggle datasets version -p "$STAGE" -m "sync m2 refs"
else
  kaggle datasets create -p "$STAGE" --dir-mode zip
fi
