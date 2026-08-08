#!/bin/bash

set -e

EXP_NAME=$1

if [ -z "$EXP_NAME" ]; then
    echo "Usage: $0 <EXP_NAME>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "=== Step 1/6: Preparing Isaac assets ==="
python learning/preprocessing/prepare_isaac_assets.py \
    --input-dir assets/fabrica/ \
    --output-dir learning/assets/fabrica/mesh/fabrica

echo "=== Step 2/6: Preparing Isaac pair YAML ==="
python learning/preprocessing/prepare_isaac_pair_yaml.py \
    --log-dir logs/$EXP_NAME \
    --yaml-path learning/assets/fabrica/yaml/fabrica_asset_info/fabrica_pairs.yaml

echo "=== Step 3/6: Preparing Isaac plan info ==="
python learning/preprocessing/prepare_isaac_plan_info_batch.py \
    --log-dir logs/$EXP_NAME \
    --plan-info-dir learning/isaacgymenvs/tasks/fabrica/data/plan_info

echo "=== Step 4/6: Generating URDF files ==="
python learning/preprocessing/generate_urdf.py

echo "=== Step 5/6: Generating Franka URDF from plan ==="
python learning/preprocessing/generate_franka_urdf_from_plan.py \
    --plan-info-dir plan_info \
    --franka-dir fabrica_franka

echo "=== Step 6/6: Generating YAML file ==="
python learning/preprocessing/generate_yaml.py

echo "=== All preprocessing steps completed ==="

