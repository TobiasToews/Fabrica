#!/bin/bash

EXP_NAME=${1-sr_gen}
FRICTION=${2:-1}
GPU=${3:-0}

export CUDA_VISIBLE_DEVICES="$GPU"
export NUM_ENVS=1024

# List of assemblies
ASSEMBLIES=("beam" "car" "cooling_manifold" "duct" "gamepad" "plumbers_block" "stool_circular")

# Log file
LOG_FILE="${EXP_NAME}.log"

# Ensure the log file exists (creates if it doesn't)
touch "$LOG_FILE"

# Loop through each assembly
for ASSEMBLY in "${ASSEMBLIES[@]}"; do
    # Log both to console and file
    echo "Running evaluation for assembly: $ASSEMBLY" | tee -a "$LOG_FILE"

    # Redirect stdout and stderr to the log file
    python train.py task=FabricaFixPlugTaskAssemble \
        task.env.assemblies=["$ASSEMBLY"] \
        task.env.numEnvs=$NUM_ENVS \
        max_iterations=1 \
        headless=True \
        test=True \
        # TODO: reevaluate all assemblies with task.env.if_eval=True!!! YOU MISSED THIS!!
        task.env.if_eval=True
        task.env.franka_friction=$FRICTION \
        checkpoint="runs/${EXP_NAME}_${ASSEMBLY}/nn/${EXP_NAME}_${ASSEMBLY}.pth" \
        >> "$LOG_FILE" 2>&1
done


# commands for video recording 
# python train.py task=FabricaFixPlugTaskAssemble task.env.assemblies=["human_normal"] task.env.numEnvs=1024 max_iterations=1 headless=False test=True task.env.if_eval=True task.env.franka_friction=1 checkpoint="runs/_own_assemblies/exp_human_normal_16-22-20-04/nn/exp_human_normal.pth"
# python train.py task=FabricaFixPlugTaskAssemble task.env.assemblies=["socket_set"] task.env.numEnvs=1024 max_iterations=1 headless=False test=True task.env.if_eval=True task.env.franka_friction=1 checkpoint="runs/_own_assemblies/exp_socket_15-15-21-07/nn/exp_socket.pth"
# python train.py task=FabricaFixPlugTaskAssemble task.env.assemblies=["socket_set_small"] task.env.numEnvs=1024 max_iterations=1 headless=False test=True task.env.if_eval=True task.env.franka_friction=1 checkpoint="runs/_own_assemblies/exp_socket_set_small_socket_set_small_06-14-03-11/nn/exp_socket_set_small_socket_set_small.pth"
# python train.py task=FabricaFixPlugTaskAssemble task.env.assemblies=["cooling_manifold"] task.env.numEnvs=1024 max_iterations=1 headless=False test=True task.env.if_eval=True task.env.franka_friction=1 checkpoint="runs/_replication/____replicate_exp_cooling_manifold_cooling_manifold_09-17-00-35/nn/exp_cooling_manifold_cooling_manifold.pth" 
# python train.py task=FabricaFixPlugTaskAssemble task.env.assemblies=["car"] task.env.numEnvs=1024 max_iterations=1 headless=False test=True task.env.if_eval=True task.env.franka_friction=1 checkpoint="runs/_replication/____replicate_exp_car_car_09-14-27-40/nn/exp_car_car.pth" 
# python train.py task=FabricaFixPlugTaskAssemble task.env.assemblies=["beam"] task.env.numEnvs=1024 max_iterations=1 headless=False test=True task.env.if_eval=True task.env.franka_friction=1 checkpoint="runs/_replication/exp_beam_beam_10-12-03-33/nn/exp_beam_beam.pth"
# python train.py task=FabricaFixPlugTaskAssemble task.env.assemblies=["duct"] task.env.numEnvs=1024 max_iterations=1 headless=False test=True task.env.if_eval=True task.env.franka_friction=1 checkpoint="runs/_replication/exp_duct_duct_10-23-50-52/nn/exp_duct_duct.pth"
# python train.py task=FabricaFixPlugTaskAssemble task.env.assemblies=["plumbers_block"] task.env.numEnvs=1024 max_iterations=1 headless=False test=True task.env.if_eval=True task.env.franka_friction=1 checkpoint="runs/_replication/exp_plumbers_block_plumbers_block_10-07-32-18/nn/exp_plumbers_block_plumbers_block.pth"
# python train.py task=FabricaFixPlugTaskAssemble task.env.assemblies=["stool_circular"] task.env.numEnvs=1024 max_iterations=1 headless=False test=True task.env.if_eval=True task.env.franka_friction=1 checkpoint="runs/_replication/exp_stool_circular_stool_circular_10-14-29-06/nn/exp_stool_circular_stool_circular.pth"
# python train.py task=FabricaFixPlugTaskAssemble task.env.assemblies=["gamepad"] task.env.numEnvs=1024 max_iterations=1 headless=False test=True task.env.if_eval=True task.env.franka_friction=1 checkpoint="runs/_replication/exp_gamepad_gamepad_10-19-12-57/nn/exp_gamepad_gamepad.pth"
