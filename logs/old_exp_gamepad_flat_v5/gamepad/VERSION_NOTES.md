# exp_gamepad_flat_v5: gamepad with grasps from 800 surface points (2026-11-10)

What differs from the standard run `bash planning/run_planning_flat.sh <exp_name> gamepad`:

- The grasps are generated with `--n-surface-pt 800` instead of the default 200 (four times as many sampled surface points per
  part). The cap of 100 grasps per part (`--max-n-grasp 100`) is the same as in the pipeline.
- Everything after that is the standard pipeline: `run_planning_flat.sh` finds `grasps.pkl` in this folder and continues with the
  grasp filter. Planner as of 2026-11-10 19:34 (main checkout), seed 0, robot base 3.0 cm below the table.

Commands (checkout root, fabrica env):

```bash
python planning/run_preced_plan.py --assembly-dir assets/fabrica/gamepad --log-dir logs/exp_gamepad_flat_v5/gamepad --num-proc 12 --arm panda
python planning/run_grasp_arm_gen.py --assembly-dir assets/fabrica/gamepad --log-dir logs/exp_gamepad_flat_v5/gamepad \
    --num-proc 50 --max-n-grasp 100 --n-surface-pt 800 --arm panda --gripper panda --ft-sensor none
bash planning/run_planning_flat.sh exp_gamepad_flat_v5 gamepad
```

Why: `logs/exp_gamepad_flat_v4` (200 points) found no sequence. No move grasp of part 0 (deepest 3.4 mm) and part 1 (just under
5.0 mm) held its part `GRASP_MIN_FINGER_OVERLAP` = 5 mm deep in the fingers, although the equal parts 5 and 2 had such grasps
(5.6 mm, 6.9 mm) and those grasps fit on parts 0 and 1 without contact. The generator builds its grasps from random surface points,
so more points should give parts 0 and 1 deeper grasps as well.

## Result (2026-11-10 21:07): no plan yet

- Grasp generation with 800 points took 56 min on 4 cores (200 points: 8 min). Every part now has move grasps that hold it at least
  5 mm deep. After the flat filter: part 0: 19, part 1: 19, part 2: 11, part 3: 25, part 4: 14, part 5: 9 (v4: 0, 0, 7, 35, 2, 4).
- Sequence attempt 1: transport of part 4 with grasp 1153 found no path in 8 of 8 attempts ("no straight move into the goal pose
  from a pose clear of everything"); the grasp was dropped.
- Sequence attempt 2: `run_motion_plan.py` stopped with "Failed to plan path for hold arm in task switch" (the simulated hold arm
  re-gripping after part 5). That failure names no grasp, so `run_planning_flat.py` gave up.
- `grasps_unfiltered.pkl` in this folder holds the 800-point grasps: `bash planning/run_planning_flat.sh exp_gamepad_flat_v5 gamepad`
  continues from the grasp filter without generating them again.
