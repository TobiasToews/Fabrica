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

# Every run directory currently under runs/#replication/, mapped to its assembly.
# declare -A DIR_TO_ASSEMBLY=(
#     ["____replicate_exp_car_car_09-14-27-40"]="car"
#     ["exp_car_car_10-09-33-01"]="car"
#     ["____replicate_exp_cooling_manifold_cooling_manifold_09-17-00-35"]="cooling_manifold"
#     ["____replicate_exp_stool_circular_stool_circular_09-19-37-33"]="stool_circular"
#     ["exp_stool_circular_stool_circular_10-14-29-06"]="stool_circular"
#     ["exp_beam_beam_09-21-23-58"]="beam"
#     ["exp_beam_beam_10-12-03-33"]="beam"
#     ["exp_plumbers_block_plumbers_block_09-23-52-01"]="plumbers_block"
#     ["exp_plumbers_block_plumbers_block_10-07-32-18"]="plumbers_block"
# )   

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
