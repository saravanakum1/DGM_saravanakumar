#!/usr/bin/env bash
# Upload code repo to Kaggle.
#
# Dataset (PTB-XL) is uploaded ONCE separately as sharav95/ecg-ptbxl-data:
#   uv run kaggle datasets create -p dataset --dir-mode zip
# After that, attach sharav95/ecg-ptbxl-data in the Kaggle notebook via Add Data.
# Never re-upload the data — it is cached on Kaggle permanently.
#
# Usage:
#   bash scripts/kaggle_upload.sh "your commit message"           # exclude dataset/ (normal)
#   bash scripts/kaggle_upload.sh "your commit message" --dataset # include dataset/ (first-time only)

set -e

MESSAGE="${1:-update}"
INCLUDE_DATASET=false
if [[ "$2" == "--dataset" ]]; then
    INCLUDE_DATASET=true
fi

STAGING="/tmp/ecg-kaggle-upload"

echo "Preparing staging directory..."
rm -rf "$STAGING"
mkdir -p "$STAGING/artifacts"

# Source files and dirs to include
cp config.toml pyproject.toml "$STAGING/"
cp -r src backend "$STAGING/"

# Small artifacts only (JSON, not weights/plots)
if ls artifacts/*.json 1>/dev/null 2>&1; then
    cp artifacts/*.json "$STAGING/artifacts/"
fi

# Include dataset/ only if --dataset flag passed
if [ "$INCLUDE_DATASET" = true ]; then
    echo "Including dataset/ (this may be large)..."
    cp -r dataset "$STAGING/"
fi

# dataset-metadata.json required by Kaggle CLI
if [ -f dataset-metadata.json ]; then
    cp dataset-metadata.json "$STAGING/"
fi

echo "Staging contents:"
find "$STAGING" -type f | sort

echo ""
echo "Uploading to Kaggle..."
DATASET_ID=$(python3 -c "import json; d=json.load(open('dataset-metadata.json')); print(d['id'])")
echo "Dataset ID: $DATASET_ID"

# Try to fetch dataset metadata — succeeds only if it exists
if uv run kaggle datasets status "$DATASET_ID" &>/dev/null; then
    echo "Dataset exists — creating new version..."
    uv run kaggle datasets version -p "$STAGING" --dir-mode zip -m "$MESSAGE"
else
    echo "Dataset not found — creating for the first time..."
    uv run kaggle datasets create -p "$STAGING" --dir-mode zip
fi

echo ""
echo "Cleaning up..."
rm -rf "$STAGING"

echo "Done."
