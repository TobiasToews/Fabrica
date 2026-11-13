# Context: what was changed in this fork, and where it stands

State of 2026-11-13 (commit `f38fc54`). Rules for working here: [AGENT.md](AGENT.md).

## Goal

Fabrica plans and learns multi-part assembly with two robot arms: one inserts parts ("move arm"), one holds the
assembly ("hold arm"), and the parts wait in a 3D-printed fixture. This fork brings it to the lab's setup: one
real Franka Panda, a human in place of the hold arm, and parts that lie flat on the table instead of a fixture.
The plans are used to train insertion policies and to record robot data for VLA training.

## History

From the commit log; the phases before November are described from the commit messages only.

| When | What | Commits |
|---|---|---|
| 2026-01-11 | official Fabrica release, the base of this fork | `215a30f` |
| Aug 2026 | installation (WSL, [INSTALL_GUIDE.md](INSTALL_GUIDE.md)), training of all assemblies, own assemblies `socket_set` and `human_normal`, generalist policy, guide for own assemblies | `3abdbb0` to `1589726` |
| Sep to Oct | evaluation and retraining; single-arm mode for the lab's one arm; robot height adjusted; printed parts and a fixture for socket_set_small; variants of sequences and grasps ([ASSEMBLY_VARIANTS.md](ASSEMBLY_VARIANTS.md)), policies for `socket_set_small__v0..v4`; VLM camera shared with LeRobot over ZMQ | `d0a30fb` to `3a70050` |
| 2026-11-05 to 11-08 | flat pipeline: human_normal without fixture, checked planning for the real robot, required franka-interface change, execution fixes, policy for human_normal flat v3 | `6049243` to `dd122d2` |
| 2026-11-10 to 11-13 | flat pipeline on all ten assemblies: new grasp generation, layout, failure loop, places of the assembly, position test; documentation; start of the PEP 8 / docstring restyle | `ebe133a`, `f38fc54` |

`ebe133a` holds the content changes of 2026-11-10 to 11-13; `f38fc54` only changes style and docstrings. The
syntax tree of every Python file in `f38fc54`, without docstrings, equals the one in `ebe133a`.

## What the flat pipeline changes

Details: [FLAT_PIPELINE.md](FLAT_PIPELINE.md); stage-by-stage comparison with the original:
[FLAT_VS_ORIGINAL.md](FLAT_VS_ORIGINAL.md). The core change: in the original, the fixture is built so that each part
sits the way its grasp needs. Without a fixture the part lies in a resting pose of its own, so one grasp has to fit
two poses, the lying one (pickup) and the assembled one (insertion).

| Area | Change | Why |
|---|---|---|
| Robot | one real arm plans and does every pickup and insertion; a human holds the first part (`--single-arm`) | the lab has one Panda |
| Real limits | robot base 3.0 cm below the table (measured 2026-11-06), franka-interface virtual walls, distance to singularities and joint limits | the original planner does not know them; the robot stops when it reaches a virtual wall, and the height moves every pose relative to the table |
| Checks | every pose of every path of the real arm is measured with exact geometry (`plan_checks.py`), then a final check (`verify_plan.py`); `run.py` refuses unchecked plans | a plan is only "done" if the real arm can execute it |
| Planner | strict mode for the real arm: straight moves out of and into every grasp, release and insertion start; plans with the same rules it is checked against, plus a small reserve | so that the planner finds paths that pass the checks, instead of paths that fail them afterwards |
| Resting poses | from the faces of the convex hull of the part moved to the origin (`flat_grasp.get_flat_rots`) | trimesh's routine lost poses depending on the place and could hang (beam at (+5, -30)) |
| Grasps (`--flat`) | contact on a 0.25 mm grid of the finger pads; candidates made at the assembled and at the lying pose and tested on the other one; the flat-pickup test before the cut to 100; rated choice; second sampling round for parts with few grasps | the original kept 100 random grasps; small round and long thin parts often kept none that can pick the lying part |
| Grasps at the assembly | grasps are dropped if the arm cannot move straight to the insertion start or straight away after the release (first-held part: only the release) | motion planning found these one per attempt (human_normal at (0, -7.5): nine grasps in a row) |
| Hold grasps | as in the original, plus those that pick the first part lying flat and those that leave room for each insertion (`FLAT_HOLD_ROOM`) | car: 3 of 441 hold grasps leave room for the body |
| First part | if no hold grasp can pick it lying flat, the move arm places it at the assembly spot as a planned step | owner's decision of 2026-11-12 |
| Layout | parts placed one by one in reverse pick order, each at the nearest spot where the pickup passes every check and the arm can move straight in and lift straight out; tests from the gripper pose alone before the IK search | a layout far from the arm took 27 min, 4 min with the tests |
| Failure loop | a failing stage names a cause: `grasp` (drop the grasp, plan the next sequence), `spot` (keep the grasp, ban the spot), `insertion` (stop); at most 10 attempts | keep a good grasp when only its spot was bad (owner's wish) |
| Places | place of the assembly and of the pickup area are inputs, stored in the log; workspace search and position test | the planner should work wherever both arms reach (owner's goal) |
| Memory | collision structures of earlier planner calls are released (`gc.collect()` in `get_fns` and per grasp) | duct ran out of memory; 8 gamepad plans at once filled the 15.9 GB |
| Real robot | `run.py`: refuses a second franka_interface, plans for another height and unchecked plans; robot state read directly; recorder and comparison tools; `planning/plan_place_part.py` lets the move arm place a part | failures seen on the robot: stale states from a second franka_interface (arm pulled 30 cm up), "Illegal rotation" crash right after a skill switch |
| franka-interface | impedance skill `safety_factor` 0.1 -> 0.5, virtual walls sides ±0.65 m, floor -0.05 m (local branch `fabrica-ics` in `~/tobi/franka-interface`) | with 10 % of the torques the wrist stalled and the arm left the plan |

## Verification

- Tests: `test_plan_rules.py` 39, `test_flat_grasp.py` 42, `test_workspace_search.py` 9, all passing on
  2026-11-13 after the formatting. The docstring edits after that were checked by comparing syntax trees, the tests
  were not run again.
- All ten assemblies at the default place, final code (sixth batch, 2026-11-12): 8 of 10 verified (socket_set_small,
  socket_set, beam, human_normal, car, gamepad, cooling_manifold, duct). plumbers_block and stool_circular have no
  plan: no move grasp can pick the bolts or the legs lying flat.
- Position test (10 places of the assembly x 3 pickup areas): beam 30 of 30, gamepad 30 of 30, human_normal 30 of
  30 verified. Not run on the final code: socket_set_small (30 of 30 twice on earlier code), socket_set, car,
  cooling_manifold, duct.
- Run notes with every number: [assembly_runs/README.md](assembly_runs/README.md) and one file per assembly.

## Decisions of the owner

Confirmed:
- Values of 2026-11-11: 0.2 cm table clearance of a carried part inside the assembly zone, 3 mm finger depth,
  1 cm wrist clearance, zone margin 4 cm (confirmed 2026-11-13).
- `PART_GAP` 2.0 cm between lying parts ("safety buffer for the rotation of the gripper").
- No limit on how far the hand leans at a pickup; the gripper keeps 0.5 cm and the wrist 1.0 cm to the board.
- Grasps are matched from both poses (assembled and lying).
- The first part is never placed by hand; fallback: the move arm places it as a planned step.
- The failure loop keeps a good grasp when only its spot failed.
- The pipeline should work for any place of the assembly and of the parts that both arms reach.
- `real_robot/place_parts.py` (the robot shows the spots) is not a priority.
- Python style: PEP 8 and Google docstrings, explained for a junior.

Open, waiting for the owner:
1. **`geometry.offset_mesh`**: the inflated part meshes the strict planner tests against have spikes for 8 of 62
   parts (e.g. duct part 0 grows to 116 x 84 x 198 cm). Verified plans are not affected (the checks measure real
   distances), but the planner rejects free poses. A repair changes the planner and needs a new verification round.
2. **Three changes that were made without his agreement** (in `ebe133a`): the grasp filter for the moves at the
   assembly, the own resting-pose computation, the memory release in the planner. Keep or revert.
3. **Reach rule of the position test**: middle and four corners of the assembly's footprint reachable by a hand
   pointing straight down; a place where a part has no grasp counts as not buildable.
4. **Bolts and legs** (plumbers_block, stool_circular): a stand, other fingers, looser rules, or leave them out.
5. **Robot height**: a later measurement gave 3.3 cm instead of the 3.0 cm the plans use.
6. **franka-interface**: fork the repository and push the branch `fabrica-ics` (TODO in FLAT_PIPELINE.md).

## Work in progress

- **Restyle** (PEP 8, Google docstrings): all 21 new files are formatted; docstrings are finished only in
  `planning/robot/panda_condition.py` and in our functions of `planning/robot/geometry.py`, `util_arm.py` and
  `workcell.py`. Still to do: the remaining docstrings, then an independent check that every new docstring follows
  from the code and the old documentation, then the tests again.

## Known limits

The full list is in [FLAT_PIPELINE.md, "Limits"](FLAT_PIPELINE.md#limits). The most important ones:
- Only the move arm is checked; the human's hands are protected only indirectly.
- The wrist camera mount and its cable are not in the robot model.
- Insertion paths come from the grasp and the disassembly plan; they are checked, not re-planned.
- Hands often lean 40 to 80 deg at a pickup, because the sequence optimization chooses by the original scores.
- `logs/exp_human_normal_flat_v3` was planned before the gripper rules were tightened and does not pass them any
  more; re-plan before the next run on the robot.
- [REAL_ROBOT_STARTUP.md](REAL_ROBOT_STARTUP.md), section "Robot height", re-plans with the original pipeline
  (`run_planning.sh`); for a plan without fixture use `run_planning_flat.sh`.

## Where things are

| Path | Content |
|---|---|
| `logs/<exp>/<assembly>/` | one plan: `motion.pkl`, templates in `fixture/`, `verify_report.json`, `workcell.json`, `planning.log` |
| `logs/exp_<assembly>_flat_run1..run6` | the batches on all assemblies (untracked) |
| `logs/exp_<assembly>_pos*` | the position tests; report `logs/exp_<assembly>_pos/position_test_<assembly>.json` (untracked) |
| `learning/isaacgymenvs/runs/` | trained policies |
| `printing/` | STL files of the parts for 3D printing (`real_robot/prepare_parts_printing.py`) |
| `recordings/`, `records/` | robot recordings and renderings (untracked) |
| `.claude/worktrees/` | git worktrees `gamepad-flat-planning`, `sequence-studio` |

The top-level files `[62`, `[67` (empty) and `bash` (a saved traceback of `render_motion_plan.py`) are stray files
from October.
