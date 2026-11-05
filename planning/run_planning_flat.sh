#!/bin/bash
# Same as run_planning.sh, but without a fixture: every part lies flat on the board in a stable resting pose
# (planning/run_pickup_gen.py). Prints a 1:1 placement template to logs/$EXP_NAME/$ASSEMBLY/fixture/pickup_template.pdf

EXP_NAME=$1
ASSEMBLY=$2
SETUP=${3:-panda}
ASSEMBLY_DIR=${4:-fabrica}

if [ "$SETUP" == "panda" ]; then
  ARM="panda"
  GRIPPER="panda"
  FT_SENSOR="none"
else
  echo "Error: run_planning_flat.sh only supports SETUP=panda"
  exit 1
fi

export OMP_NUM_THREADS=1
set -e

LOG_DIR=logs/$EXP_NAME/$ASSEMBLY

if [ ! -f $LOG_DIR/grasps.pkl ]; then
  echo "Running precedence and path planning..."
  python planning/run_preced_plan.py --assembly-dir assets/$ASSEMBLY_DIR/$ASSEMBLY --log-dir $LOG_DIR --num-proc 12 --arm $ARM

  echo "Running grasp and arm IK generation..."
  python planning/run_grasp_arm_gen.py --assembly-dir assets/$ASSEMBLY_DIR/$ASSEMBLY --log-dir $LOG_DIR --num-proc 50 --max-n-grasp 100 --arm $ARM --gripper $GRIPPER --ft-sensor $FT_SENSOR
fi

echo "Filtering move grasps that can pick a flat part..."
python planning/run_pickup_gen.py --assembly-dir assets/$ASSEMBLY_DIR/$ASSEMBLY --log-dir $LOG_DIR --filter

echo "Running sequence planning..."
python planning/run_seq_plan.py --assembly-dir assets/$ASSEMBLY_DIR/$ASSEMBLY --log-dir $LOG_DIR --plot

echo "Running sequence optimization (first hold grasp picks a flat part)..."
python planning/run_pickup_gen.py --assembly-dir assets/$ASSEMBLY_DIR/$ASSEMBLY --log-dir $LOG_DIR --seq-opt --plot

echo "Running flat pickup layout..."
python planning/run_pickup_gen.py --assembly-dir assets/$ASSEMBLY_DIR/$ASSEMBLY --log-dir $LOG_DIR

echo "Running complete motion planning..."
python planning/run_motion_plan.py --assembly-dir assets/$ASSEMBLY_DIR/$ASSEMBLY --log-dir $LOG_DIR --optimized

echo "Done."
