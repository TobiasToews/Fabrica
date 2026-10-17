#!/bin/bash

FRICTION=1
GPU=${2:-0}

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="$GPU"
export NUM_ENVS=1024

ASSEMBLIES=("beam" "car" "cooling_manifold" "duct" "gamepad" "plumbers_block" "stool_circular" "socket_set" "human_normal")

CHECKPOINT="runs/_correct_replication_generalist_fabrica_and_own_1500_iterations/env_generalist_own_fixed_06-11-11-27/nn/env_generalist_own_fixed.pth"

cd ~/Fabrica
bash ./learning/preprocessing/prepare_isaac.sh exp_generalist_own_fixed
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


# command for evaluating the general policy of all socket_set_small__vN 
# python train.py task=FabricaFixPlugTaskAssemble task.env.assemblies=["socket_set_small__v0" "socket_set_small__v1" "socket_set_small__v2" "socket_set_small__v3" "socket_set_small__v4"] task.env.numEnvs=1024 max_iterations=1 headless=True test=True task.env.if_eval=True task.env.franka_friction=1 Checkpoint="runs/exp_socket_set_small_var_16-18-37-15/nn/exp_socket_set_small_var.pth"