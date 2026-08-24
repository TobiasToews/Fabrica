#!/bin/bash

GPU=0
FRICTION=1

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=GPU
export NUM_ENVS=1024

#Assemblies for 420 Resolution
declare -A DIR_TO_ASSEMBLY=(
    ["exp_gamepad_gamepad_10-19-12-57"]="gamepad"
    ["exp_duct_duct_10-23-50-52"]="duct"
)

cd ~/Fabrica/learning/isaacgymenvs

for DIR in "${!DIR_TO_ASSEMBLY[@]}"
do
    ASSEMBLY="${DIR_TO_ASSEMBLY[$DIR]}"
    PREPARE_EXP_NAME="exp_${ASSEMBLY}"
    FULL_EXP_NAME="exp_${ASSEMBLY}_${ASSEMBLY}"
    CHECKPOINT="runs/replication/${DIR}/nn/${FULL_EXP_NAME}.pth"

    if [ ! -f "$CHECKPOINT" ]; then
        echo "=== Skipping $DIR ($ASSEMBLY): checkpoint not found at $CHECKPOINT ==="
        continue
    fi

    # Regenerate the pairs YAML for this assembly before evaluating it -- it only
    # ever holds whichever assembly was most recently prepared, so without this
    # every assembly but the last-trained one fails with a ConfigKeyError.
    cd ~/Fabrica
    bash ./learning/preprocessing/prepare_isaac.sh "$PREPARE_EXP_NAME"
    cd ~/Fabrica/learning/isaacgymenvs

    echo "=== Evaluating $DIR (assembly: $ASSEMBLY, checkpoint: $CHECKPOINT) ==="
    python train.py task=FabricaFixPlugTaskAssemble \
        task.env.assemblies=["$ASSEMBLY"] \
        task.env.numEnvs=$NUM_ENVS \
        max_iterations=1 \
        headless=True \
        test=True \
        task.env.franka_friction=$FRICTION \
        checkpoint="$CHECKPOINT"
    echo "=== Finished evaluating $DIR ==="
done
