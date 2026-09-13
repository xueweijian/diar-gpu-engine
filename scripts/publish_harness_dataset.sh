#!/usr/bin/env bash
# Publish harness/diar_harness.py as the Kaggle dataset diar-gpu-engine-harness.
#
# Kernels mount it read-only at /kaggle/input/diar-gpu-engine-harness and add
# that directory to sys.path, so no kernel script needs to duplicate the build
# and measurement boilerplate.
#
# Usage: scripts/publish_harness_dataset.sh [--version]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_ID="weijianxue/diar-gpu-engine-harness"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cp "$REPO_ROOT/harness/diar_harness.py" "$STAGE/diar_harness.py"
cat > "$STAGE/dataset-metadata.json" <<JSON
{
  "title": "diar-gpu-engine-harness",
  "id": "$DATASET_ID",
  "licenses": [{"name": "MIT"}]
}
JSON

if [ "${1:-}" = "--version" ]; then
  kaggle datasets version -p "$STAGE" -m "sync diar_harness.py"
else
  kaggle datasets create -p "$STAGE" --dir-mode zip
fi
