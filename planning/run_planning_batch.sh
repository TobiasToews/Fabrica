#!/bin/bash

EXP_NAME=$1
SETUP=${2:-panda}
ASSEMBLY_DIR=${3:-fabrica}

ARM=""
GRIPPER=""
FT_SENSOR=""

if [ "$SETUP" == "panda" ]; then
  ARM="panda"
  GRIPPER="panda"
  FT_SENSOR="none"
elif [ "$SETUP" == "panda-robotiq" ]; then
  ARM="panda"
  GRIPPER="robotiq-140"
  FT_SENSOR="none"
elif [ "$SETUP" == "xarm7" ]; then
  ARM="xarm7"
  GRIPPER="robotiq-140"
  FT_SENSOR="move"
elif [ "$SETUP" == "ur5e" ]; then
  ARM="ur5e"
  GRIPPER="robotiq-85"
  FT_SENSOR="none"
else
  echo "Error: Unsupported SETUP value '$SETUP'. Please use 'panda' or 'xarm7' or 'ur5e'."
  exit 1
fi

export OMP_NUM_THREADS=1

echo "Running precedence and path planning..."
python planning/run_preced_plan_batch.py --assembly-dir assets/$ASSEMBLY_DIR --log-dir logs/$EXP_NAME --num-proc 3 --outer-num-proc 3 --inner-num-proc 1 --arm $ARM

echo "Running grasp and arm IK generation..."
python planning/run_grasp_arm_gen_batch.py --assembly-dir assets/$ASSEMBLY_DIR --log-dir logs/$EXP_NAME --num-proc 4 --inner-num-proc 50 --max-n-grasp 300 --arm $ARM --gripper $GRIPPER --ft-sensor $FT_SENSOR

echo "Running sequence planning..."
python planning/run_seq_plan_batch.py --assembly-dir assets/$ASSEMBLY_DIR --log-dir logs/$EXP_NAME --num-proc 12 --plot

echo "Running sequence optimization..."
python planning/run_seq_opt_batch.py --log-dir logs/$EXP_NAME --num-proc 12 --plot

echo "Running fixture generation..."
python planning/run_fixture_gen_batch.py --assembly-dir assets/$ASSEMBLY_DIR --log-dir logs/$EXP_NAME --optimized --num-proc 12

echo "Running complete motion planning..."
python planning/run_motion_plan_batch.py --assembly-dir assets/$ASSEMBLY_DIR --log-dir logs/$EXP_NAME --optimized --num-proc 12

echo "Done."
