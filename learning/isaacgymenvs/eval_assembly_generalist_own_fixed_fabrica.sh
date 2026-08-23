#!/bin/bash
# Evaluate every assembly-specialist checkpoint under runs/#replication/, one
# after another. There can be more than one run directory per assembly (e.g.
# a "____replicate_" run alongside a later plain-named run) -- all of them
# get evaluated, matched to their assembly via the DIR_TO_ASSEMBLY mapping
# below so the correct pairs YAML is regenerated (via prepare_isaac.sh)
# before each one.

GPU=${1:-0}
FRICTION=${2:-1}

# Required on WSL2 so PhysX can find libcuda.so and use GPU physics.
# Without this, physics silently falls back to CPU, which then crashes
# outright on SDF tri-mesh colliders (GPU-only feature).
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES="$GPU"
export NUM_ENVS=1024

ASSEMBLIES=(
    "beam"
    "car"
    "cooling_manifold"
    "duct"
    "gamepad"
    "plumbers_block"
    "stool_circular"
    "socket_set"
    "human_normal"
)

# 2000 epoachs policy:
CHECKPOINT="runs/_generalist_own_and_fabrica_fixed/env_generalist_own_fixed_23-17-40-38/nn/env_generalist_own_fixed.pth"

cd ~/Fabrica
bash ./learning/preprocessing/prepare_isaac.sh exp_generalist_own_fabrica
cd ~/Fabrica/learning/isaacgymenvs

for ASSEMBLY in "${ASSEMBLIES[@]}"
do
    echo "=== Evaluating Assembly $ASSEMBLY in Checkpoint $CHECKPOINT ==="
    python train.py task=FabricaFixPlugTaskAssemble \
        task.env.assemblies=["$ASSEMBLY"] \
        task.env.numEnvs=$NUM_ENVS \
        max_iterations=1 \
        headless=True \
        test=True \
        task.env.franka_friction=$FRICTION \
        checkpoint="$CHECKPOINT"
    echo "=== Finished evaluating $ASSEMBLY ==="
done
