#!/usr/bin/env bash
# Publish the four M1 parity fixtures as the Kaggle dataset
# weijianxue/diar-m2-fixtures (input for the Stage 3.4 K6 kernel).
#
# Shipped as ONE fixtures.zip (scripts/pack_fixtures.py): the standard
# kaggle CLI path dies on this network (api.kaggle.com SSL EOF), so the
# upload goes through scripts/kaggle_dataset_via_ip.py (SNI bypass — same
# channel the kernels use). Kaggle may present the mount extracted OR as
# the zip; the K6 runner's find_fixtures_dir() accepts both layouts.
#
# Create-only: the via-IP helper implements CreateDataset; re-publishing a
# new version would need the version endpoint implemented first.
#
# Usage: scripts/publish_m2_fixtures_dataset.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_ID="weijianxue/diar-m2-fixtures"
STAGE="${STAGE_DIR:-/tmp/diar-m2-fixtures-stage}"

rm -rf "$STAGE"
mkdir -p "$STAGE"

python3 "$REPO_ROOT/scripts/pack_fixtures.py" \
  "$REPO_ROOT/parity/fixtures" "$STAGE/fixtures.zip"

printf '{\n  "title": "diar-m2-fixtures",\n  "id": "%s",\n  "licenses": [{"name": "MIT"}]\n}\n' \
  "$DATASET_ID" > "$STAGE/dataset-metadata.json"

python3 "$REPO_ROOT/scripts/kaggle_dataset_via_ip.py" create --folder "$STAGE"
echo "NOTE: poll ready via: python3 scripts/kaggle_dataset_via_ip.py datasets-status --ref $DATASET_ID"
