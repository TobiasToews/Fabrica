#!/bin/bash
# Evaluate assembly-specialist checkpoints for all benchmark assemblies, one after another.
# Assumes each was trained via train_assembly_specialist_all_experiments.sh (experiment
# names of the form exp_<assembly>_<assembly>).

GPU=${1:-0}
FRICTION=${2:-1}

export CUDA_VISIBLE_DEVICES="$GPU"
export NUM_ENVS=1024

ASSEMBLIES=(
    "car"
    "cooling_manifold"
    "stool_circular"
    "beam"
    "gamepad"
    "plumbers_block"
    "duct"
)

cd ~/Fabrica/learning/isaacgymenvs

for ASSEMBLY in "${ASSEMBLIES[@]}"
do
    EXP_NAME="exp_${ASSEMBLY}_${ASSEMBLY}"
    CHECKPOINT=$(ls runs/${EXP_NAME}_*/nn/${EXP_NAME}.pth 2>/dev/null | sort | tail -n 1)

    if [ -z "$CHECKPOINT" ]; then
        echo "=== Skipping $ASSEMBLY: no checkpoint found matching runs/${EXP_NAME}_*/nn/${EXP_NAME}.pth ==="
        continue
    fi

    echo "=== Evaluating $ASSEMBLY (checkpoint: $CHECKPOINT) ==="
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
