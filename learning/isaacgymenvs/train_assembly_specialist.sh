EXP_NAME=$1
ASSEMBLY=$2
GPU=${3:-0}
FRICTION=${4:-1}

export CUDA_VISIBLE_DEVICES="$GPU"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export NUM_ENVS=1024
export MAX_ITER=1500

python train.py task=FabricaFixPlugTaskAssemble task.env.assemblies=["$ASSEMBLY"] experiment=${EXP_NAME}_${ASSEMBLY} task.env.numEnvs=$NUM_ENVS max_iterations=$MAX_ITER headless=True task.env.franka_friction=$FRICTION
