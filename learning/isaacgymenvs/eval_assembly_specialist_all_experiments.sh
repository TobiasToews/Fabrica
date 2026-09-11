#!/bin/bash
# Evaluate every trained assembly-specialist checkpoint under runs/_replication.
#
# For each run directory we look up its assembly, regenerate the Isaac pair YAML
# for that assembly (fabrica_pairs.yaml only ever holds the most recently
# prepared assembly, so without this every checkpoint but the last-prepared one
# fails with a ConfigKeyError), then run train.py in eval mode against the
# checkpoint.

set -u

GPU=${1:-0}
FRICTION=1

# Required on WSL2 so PhysX can find libcuda.so and use GPU physics.
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES="$GPU"
export NUM_ENVS=1024

PROJECT_ROOT="$HOME/Fabrica"
REPLICATION_DIR="learning/isaacgymenvs/runs/_replication"

# Map each run directory (under $REPLICATION_DIR) to its assembly name. The
# assembly name feeds task.env.assemblies and, as exp_<assembly>, the logs/
# directory that prepare_isaac.sh reads.
# declare -A RUN_TO_ASSEMBLY=(
#     ["____replicate_exp_car_car_09-14-27-40"]="car"
#     ["____replicate_exp_cooling_manifold_cooling_manifold_09-17-00-35"]="cooling_manifold"
#     ["____replicate_exp_stool_circular_stool_circular_09-19-37-33"]="stool_circular"
#     ["exp_beam_beam_09-21-23-58"]="beam"
#     ["exp_beam_beam_10-12-03-33"]="beam"
#     ["exp_car_2_14-13-54-31"]="car"
#     ["exp_car_car_10-09-33-01"]="car"
#     ["exp_plumbers_block_plumbers_block_09-23-52-01"]="plumbers_block"
#     ["exp_plumbers_block_plumbers_block_10-07-32-18"]="plumbers_block"
#     ["exp_stool_circular_stool_circular_10-14-29-06"]="stool_circular"
# )
declare -A RUN_TO_ASSEMBLY=(
    ["exp_duct_duct_10-23-50-52"]="duct"
    ["exp_gamepad_gamepad_10-19-12-57"]="gamepad"
   )

for RUN_DIR in "${!RUN_TO_ASSEMBLY[@]}"
do
    ASSEMBLY="${RUN_TO_ASSEMBLY[$RUN_DIR]}"
    PREPARE_EXP_NAME="exp_${ASSEMBLY}"
    RUN_PATH="${PROJECT_ROOT}/${REPLICATION_DIR}/${RUN_DIR}"

    # The "final" checkpoint is the only .pth in nn/ that is not an ep snapshot.
    CHECKPOINT=$(ls "${RUN_PATH}/nn/"*.pth 2>/dev/null | grep -v '/last_' | head -n1)

    if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
        echo "=== Skipping $RUN_DIR ($ASSEMBLY): no checkpoint found under ${RUN_PATH}/nn/ ==="
        continue
    fi

    # Regenerate fabrica_pairs.yaml (and the rest of the Isaac preprocessing)
    # for this assembly before evaluating it.
    bash "${PROJECT_ROOT}/learning/preprocessing/prepare_isaac.sh" "$PREPARE_EXP_NAME"

    cd "${PROJECT_ROOT}/learning/isaacgymenvs"

    echo "=== Evaluating $RUN_DIR (assembly: $ASSEMBLY, checkpoint: $CHECKPOINT) ==="
    python train.py task=FabricaFixPlugTaskAssemble \
        task.env.assemblies=["$ASSEMBLY"] \
        task.env.numEnvs=$NUM_ENVS \
        max_iterations=1 \
        headless=True \
        test=True \
        task.env.if_eval=True \
        task.env.franka_friction=$FRICTION \
        checkpoint="$CHECKPOINT"
    echo "=== Finished evaluating $RUN_DIR ==="
done
