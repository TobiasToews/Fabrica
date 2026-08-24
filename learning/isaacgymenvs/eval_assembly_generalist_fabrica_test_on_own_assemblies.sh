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

# ASSEMBLIES=(
#     "beam"
#     "car"
#     "cooling_manifold"
#     "duct"
#     "gamepad"
#     "plumbers_block"
#     "stool_circular"
# )

ASSEMBLIES=(
    "socket_set"
    "human_normal"
)

# Custom assemblies' planning logs don't live under logs/exp_<assembly>/ by
# assembly name (unlike the 7 paper assemblies) -- map explicitly. prepare_isaac.sh
# regenerates fabrica_pairs.yaml from scratch each call (it does not merge), so it
# must be re-run per assembly, immediately before that assembly's eval.
declare -A EXP_NAME_MAP=(
    ["socket_set"]="exp_socket"
    ["human_normal"]="exp_human_normal"
)

# 1900 epochs policy:
#CHECKPOINT="runs/_generalist_fabricas/env_generalist_17-15-51-14/nn/env_generalist.pth"

# 2000 epoachs policy:
CHECKPOINT="runs/_generalist_fabricas/env_generalist_18-18-12-03/nn/env_generalist.pth"

for ASSEMBLY in "${ASSEMBLIES[@]}"
do
    EXP_NAME="${EXP_NAME_MAP[$ASSEMBLY]}"
    echo "=== Preparing Isaac assets for $ASSEMBLY (experiment: $EXP_NAME) ==="
    cd ~/Fabrica
    bash ./learning/preprocessing/prepare_isaac.sh "$EXP_NAME"
    cd ~/Fabrica/learning/isaacgymenvs

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
