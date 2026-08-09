#!/bin/bash
# Launch TensorBoard over the training runs in ./runs/.
#
# Usage:
#   ./view_tensorboard.sh                        # show all runs (compare side by side)
#   ./view_tensorboard.sh RUN_NAME                # show just one run, e.g.:
#                                                  #   ./view_tensorboard.sh FabricaTaskAssemble_08-13-00-22
#   ./view_tensorboard.sh RUN_NAME 6007            # use a different port

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_NAME=$1
PORT=${2:-6006}

if [ -z "$RUN_NAME" ]; then
    LOGDIR="$SCRIPT_DIR/runs"
else
    LOGDIR="$SCRIPT_DIR/runs/$RUN_NAME"
    if [ ! -d "$LOGDIR" ]; then
        echo "Error: run directory not found: $LOGDIR"
        echo "Available runs:"
        ls "$SCRIPT_DIR/runs"
        exit 1
    fi
fi

echo "Serving TensorBoard for: $LOGDIR"
echo "Open http://localhost:$PORT in your browser (WSL2 forwards this automatically)."
tensorboard --logdir "$LOGDIR" --port "$PORT"
