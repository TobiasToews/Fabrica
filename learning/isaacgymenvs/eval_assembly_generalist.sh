#!/bin/bash

FRICTION=1
GPU=${2:-0}

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="$GPU"
export NUM_ENVS=1024

ASSEMBLIES=("beam" "car" "cooling_manifold" "duct" "gamepad" "plumbers_block" "stool_circular")

CHECKPOINT="runs/_correct_replication_generalist_fabrica_and_own_1500_iterations/env_generalist_05-11-48-09/nn/env_generalist.pth"

cd ~/Fabrica
bash ./learning/preprocessing/prepare_isaac.sh exp_generalist
cd ~/Fabrica/learning/isaacgymenvs

for ASSEMBLY in "${ASSEMBLIES[@]}"; do
    echo "=== Evaluating assembly: $ASSEMBLY ==="
    python train.py task=FabricaFixPlugTaskAssemble \
        task.env.assemblies=["$ASSEMBLY"] \
        task.env.numEnvs=$NUM_ENVS \
        max_iterations=1 \
        headless=True \
        test=True \
        task.env.if_eval=True \
        task.env.franka_friction=$FRICTION \
        checkpoint="$CHECKPOINT"
    echo "=== Finished $ASSEMBLY ==="
done
