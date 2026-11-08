#!/bin/bash
# Plan an assembly for the real robot without a fixture (parts lie flat on the table), checked end to end.
# Usage: bash planning/run_planning_flat.sh EXP_NAME ASSEMBLY [ASSEMBLY_ROOT]
# See planning/run_planning_flat.py. Outputs in logs/EXP_NAME/ASSEMBLY: motion.pkl, fixture/*_template_a4.pdf,
# fixture/place_poses.json, verify_report.json, workcell.json, planning.log
set -e
python -u planning/run_planning_flat.py --exp-name "$1" --assembly "$2" --assembly-root "${3:-fabrica}"
