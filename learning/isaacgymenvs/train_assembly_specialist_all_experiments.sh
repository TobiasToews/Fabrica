#!/bin/bash
# Run assembly-specialist training for all benchmark assemblies, one after another.

GPU=${1:-0}
FRICTION=${2:-1}

export CUDA_VISIBLE_DEVICES="$GPU"
export NUM_ENVS=1024
export MAX_ITER=1500
# test config: NUM_ENVS must be a multiple of 16, since batch_size = NUM_ENVS * horizon_length (32)
# must be divisible by minibatch_size (512) -> minimum valid NUM_ENVS is 512/32 = 16
# export NUM_ENVS=16
# export MAX_ITER=2

ASSEMBLIES=(
    "car"
    "cooling_manifold"
    "stool_circular"
    "beam"
    "gamepad"
    "plumbers_block"
    "duct"
)

for ASSEMBLY in "${ASSEMBLIES[@]}"
do
    EXP_NAME="exp_${ASSEMBLY}"
    echo "=== Starting $ASSEMBLY (experiment: ${EXP_NAME}_${ASSEMBLY}) ==="

    cd ~/Fabrica
    bash ./learning/preprocessing/prepare_isaac.sh "$EXP_NAME"

    cd ~/Fabrica/learning/isaacgymenvs
    python train.py task=FabricaFixPlugTaskAssemble task.env.assemblies=["$ASSEMBLY"] experiment=${EXP_NAME}_${ASSEMBLY} task.env.numEnvs=$NUM_ENVS max_iterations=$MAX_ITER headless=True task.env.franka_friction=$FRICTION

    echo "=== Finished $ASSEMBLY (experiment: ${EXP_NAME}_${ASSEMBLY}) ==="
done
