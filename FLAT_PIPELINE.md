# Fixture-free pipeline for the real robot

Plans an assembly where the parts lie flat on the table (no printed fixture). The single real Panda (the "move" arm)
picks and inserts every part; a human holds the first part at the assembly spot (`real_robot/run.py --single-arm`).
Plan once, train once, run on the robot: a plan is only reported as done when every motion of the real arm passed the checks below.

## Use

```bash
conda activate fabrica
bash planning/run_planning_flat.sh <exp_name> <assembly>          # e.g. exp_human_normal_flat_v3 human_normal
```
Output in `logs/<exp_name>/<assembly>/`:

| File | What |
|---|---|
| `motion.pkl` | what `real_robot/run.py` executes |
| `fixture/pickup_template_a4.pdf`, `fixture/assembly_template_a4.pdf` | 1:1 print-outs (A4, 100 %): where each part lies (several sheets with 1 cm overlap if the layout is larger), where the human holds the first part |
| `fixture/pickup_q.json` | the checked arm pose of every pickup (used by motion planning) |
| `pickup_ik_cache.pkl` | cached pickup arm poses; makes later runs in the same log fast |
| `fixture/place_poses.json` | for `real_robot/place_parts.py` (the robot shows each spot, in reverse pick order) |
| `verify_report.json` | result of the final check |
| `workcell.json` | real-world values the plan was made with; `run.py` refuses plans made for another robot height |
| `planning.log` | full log of all stages |

Then train and run (the script prints these commands with the right paths):
```bash
conda activate isaacgym
bash learning/preprocessing/prepare_isaac.sh <exp_name>
cd learning/isaacgymenvs && python train.py task=FabricaFixPlugTaskAssemble task.env.assemblies=["<assembly>"] task.env.numEnvs=1024 max_iterations=200 headless=True experiment=<exp_name>
python real_robot/place_parts.py --log-dir logs/<exp_name>/<assembly>
python real_robot/run.py --single-arm --residual --fn logs/<exp_name>/<assembly>/motion.pkl --checkpoint-path learning/isaacgymenvs/runs/<run>/nn/<exp_name>.pth
```

## Before planning: the real setup must match

- Robot height: `get_panda_base_z()` in `planning/robot/workcell.py` (robot mounting 3.0 cm below the table, measured 2026-11-06).
  Changing it requires a new plan (new `<exp_name>`).
- Printed parts as in the CAD files (`assets/fabrica/<assembly>/*.obj`, cm).
- First part exactly on the orange area of `assembly_template_a4.pdf`; parts on `pickup_template_a4.pdf` or placed with `place_parts.py`.

## Stages (`planning/run_planning_flat.py`)

1. Precedence and grasps (`run_preced_plan.py`, `run_grasp_arm_gen.py`), only if the log has none, at the current robot height.
2. Grasp filter (`run_pickup_gen.py --filter`): move grasps that can pick their part lying flat (stable resting pose: tips over only
   above `MIN_TIP_ANGLE` 20°, or above `MIN_TIP_ANGLE_FALLBACK` 10° for a part without such a pose, e.g. a small cylinder that rolls
   lying and stands upright; such parts are marked on the template), gripper at most
   `MAX_TILT` from vertical), stay inside the virtual walls, and hold the part at least `GRASP_MIN_FINGER_OVERLAP` deep in the fingers.
3. Sequence planning (`run_seq_plan.py`) and optimization (`run_pickup_gen.py --seq-opt`, the first held part must be pickable lying flat).
4. Pickup layout (`run_pickup_gen.py`): the parts are placed one by one in reverse pick order (so the parts lying around at each
   pickup are exactly those that still lie there during the run), each at the free spot nearest to the pickup area (1.25 cm grid,
   on the board, outside the assembly zone) and the resting pose / rotation where its pickup passes every check. For every pickup
   several IK solutions of the redundant arm are computed and the best conditioned one is kept, including the poses while lifting
   the part 2.5–10 cm (the arm folds further when lifting close to the robot).
5. Motion planning (`run_motion_plan.py --no-viz --single-arm`): every path of the real arm is checked and re-planned with another seed
   if it fails or the planner gives up (`MAX_PLAN_ATTEMPTS`). The hold arm is a human, so the real arm does not avoid a simulated hold arm.
6. If a stage finds no clean plan it names the move grasp that caused it; that grasp is removed and the next best sequence is planned
   (`MAX_SEQUENCE_ATTEMPTS`).
7. Final check (`verify_plan.py`), place poses (`run_place_poses.py`), `workcell.json`.

## Checks of the real arm (`planning/plan_checks.py`, values in `planning/config.py`)

Exact geometry, no planner buffers, every pose of every path in `motion.pkl`:

| Rule | Value |
|---|---|
| arm links to every part; gripper and carried part to lying parts (the part being picked may be touched near its grasp) | ≥ `ROBOT_PART_CLEARANCE` 1.0 cm |
| gripper / carried part vs the assembly | no contact (except the part just released, near its release; insertion paths are fixed by the grasp) |
| carried part above the table outside the assembly zone, once lifted | ≥ `TRANSFER_PART_CLEARANCE` 5.0 cm |
| gripper / arm above the table | ≥ 0.2 cm / ≥ 1.0 cm |
| franka-interface virtual walls | ≥ `VIRTUAL_WALL_SAFETY` 1.0 cm |
| singularity: smallest singular value of the arm Jacobian | ≥ `MIN_SIGMA` 0.10, while inserting ≥ `MIN_SIGMA_INSERTION` 0.15 |
| every joint to its limits (e.g. elbow not folded up) | ≥ `MIN_JOINT_LIMIT_MARGIN` 0.5 rad |
| the same two rules for the poses while lifting a picked part 2.5–10 cm, and for the insertion poses of every grasp (grasp filter) | |
| part reaches into the fingers (tolerance against a gripper that is too high) | ≥ `GRASP_MIN_FINGER_OVERLAP` 0.5 cm |

## Changes to the original Fabrica pipeline

| File | Change |
|---|---|
| `planning/robot/workcell.py` | robot base height `get_panda_base_z()` (original: 0), table height, franka-interface virtual walls |
| `planning/robot/motion_plan_arm.py` | virtual walls and table clearance of carried parts as constraints; straight lift of a carried part to the clearance height; retract steps where the arm would flip are rejected; `VISUALIZE` switch |
| `planning/run_motion_plan.py` | works without `fixture.obj`; every move-arm path checked and re-planned; names the failing grasp; `--no-viz` |
| `planning/run_pickup_gen.py` (new) | replaces fixture generation: flat resting poses, grasp filter, sequence optimization, checked part-by-part layout with cached best-conditioned pickup IK, 1:1 templates |
| `planning/plan_checks.py`, `planning/verify_plan.py` (new) | the checks above, final check |
| `planning/run_planning_flat.py`, `.sh` (new) | the pipeline with automatic fallback |
| `planning/run_place_poses.py`, `real_robot/place_parts.py` (new) | the robot shows where each part goes |
| `real_robot/run.py` | `--place-at-pickup`; refuses plans for another robot height or without passed check |
| `rendering/` | fixture optional, `--speed`, `show_initial_setup.py` (still view of the layout), Blender 4 import fix |

## Limits

- Only the move arm is checked: the hold arm is the human (single-arm mode). The human's hands at the assembly are protected only
  indirectly (no contact with the assembly, 1 cm to all parts): keep them on the side away from the robot.
- Insertion paths come from the grasp and the disassembly plan; they are checked for singularities, joint limits, walls and table,
  not re-planned.
- Transfer paths are sampled (RRT) without knowing the singularity / joint-limit rules; a path that violates them is re-planned
  (up to `MAX_PLAN_ATTEMPTS`), then the grasp is dropped.
- A plan matches reality only as well as the robot height, the printed parts and the placement of the first part do.
