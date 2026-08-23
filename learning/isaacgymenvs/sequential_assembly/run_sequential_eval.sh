#!/bin/bash
# Chain the trained specialist policy for one assembly across its full,
# planner-ordered part sequence. See run_sequential_eval.py for details.
#
# Usage: ./run_sequential_eval.sh <assembly> <checkpoint> [num_envs] [gpu]

ASSEMBLY=${1:?"usage: run_sequential_eval.sh <assembly> <checkpoint> [num_envs] [gpu]"}
CHECKPOINT=${2:?"usage: run_sequential_eval.sh <assembly> <checkpoint> [num_envs] [gpu]"}
NUM_ENVS=${3:-256}
GPU=${4:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/run_sequential_eval.py" \
    --assembly "$ASSEMBLY" \
    --checkpoint "$CHECKPOINT" \
    --num-envs "$NUM_ENVS" \
    --gpu "$GPU"
