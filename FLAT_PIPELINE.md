# Fixture-free pipeline for the real robot

Plans an assembly where the parts lie flat on the table (no printed fixture). The "move" arm picks and inserts every
part, the "hold" arm holds the assembly. Both arms are planned as robots: each avoids the other, and no step is left to a
human (since 2026-11-13). Today only the move arm is a real Panda; `real_robot/run.py --single-arm` lets a human do the
steps of the hold arm.
Plan once, train once, run on the robot: a plan is only reported as done when every motion of the real arm passed the checks below.

## Use

```bash
conda activate fabrica
bash planning/run_planning_flat.sh <exp_name> <assembly>          # e.g. exp_human_normal_flat_v3 human_normal
python planning/run_workspace_search.py --exp-name <exp_name> --assembly <assembly>   # also searches the place of the assembly (see "Place of the assembly")
```
Output in `logs/<exp_name>/<assembly>/`:

| File | What |
|---|---|
| `motion.pkl` | what `real_robot/run.py` executes |
| `fixture/pickup_template_a4.pdf`, `fixture/assembly_template_a4.pdf` | 1:1 print-outs (A4, 100 %): where each part lies (several sheets with 1 cm overlap if the layout is larger), where the human holds the first part |
| `fixture/pickup_q.json` | the checked arm pose of every pickup of the move arm (used by motion planning) |
| `fixture/pickup_q_hold.json` | the arm pose of the hold arm at the first part (empty if the move arm brings that part) |
| `pickup_ik_cache.pkl` | cached pickup arm poses; makes later runs in the same log fast |
| `fixture/place_poses.json` | for `real_robot/place_parts.py` (the robot shows each spot, in reverse pick order). Missing if one of its moves between the spots collides (`workcell.json`: `place_poses` false): lay the parts out with the template then |
| `verify_report.json` | result of the final check |
| `workcell.json` | real-world values the plan was made with; `run.py` refuses plans made for another robot height |
| `assembly_center.json` | only if the plan is made for another place of the assembly than the default (0, -15): the place; every script that works on the log reads it |
| `pickup_center.json` | only if the parts are laid out around another place than the default pickup area (0, 5): the place |
| `banned_spots.json` | spots where a part must not lie again with a given grasp (motion planning failed there because of the place); written by the loop of `run_planning_flat.py` |
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

1. Precedence and grasps (`run_preced_plan.py`, `run_grasp_arm_gen.py --flat`), only if the log has none, at the current robot
   height. With `--flat` the grasps are made for this pipeline (`planning/flat_grasp.py`, see "Grasps" below): a move grasp has to
   pick its part lying flat and to insert it, and that is tested before the number of grasps is cut to 100.
   `run_planning_flat.py --grasps-only` stops after this stage (it uses all cores; the later stages use one).
   If a part that has to be inserted got no move grasp, planning stops here and says which part
   (`no plan where the assembly stands now: no move grasp works there for part ...`; the line
   `[generate_grasps] part ...: no usable move grasp` gives the numbers): the real arm cannot build the assembly where it stands.
2. Grasp filter (`run_pickup_gen.py --filter`): move grasps that can pick their part lying flat (see below), stay inside the
   virtual walls and away from singularities and joint limits, and hold the part at least `GRASP_MIN_FINGER_OVERLAP` deep in the
   fingers (`flat_grasp.move_grasp_criteria`, the same criteria as in the grasp generation).
   A grasp can pick its part lying flat if, in one of the resting poses of the part, the whole gripper stays `GROUND_CLEARANCE`
   0.5 cm above the board and the arm link behind the hand (link 7; its place follows from the gripper pose alone)
   `WRIST_CLEARANCE` 1.0 cm, the value the plan checks demand of every arm link. There is no limit on how far the hand leans
   (until 2026-11-11: 30° from vertical); a hand that leans too far fails one of the two clearances.
   Resting poses (`flat_grasp.get_flat_rots`): a part can lie on every face of its convex hull over which its centre of
   mass stands; the tipping angle follows from how far the centre of mass is from the edge of what touches the board and
   how high it is. They are computed for the part moved to the origin, so they are the same wherever the assembly stands
   (until 2026-11-12 a routine of trimesh was used on the part where it stands: it lost poses through rounding, differently
   at different places, and for some hulls it never returned). Stable ones tip over only above `MIN_TIP_ANGLE` 20°. Less stable ones (tipping between
   `MIN_TIP_ANGLE_FALLBACK` 10° and 20°; the centre of mass is above the support area, so the part stays where it is put) are
   used for a part without a stable pose (e.g. a small cylinder that rolls lying and stands upright) and, if
   `FLAT_FALLBACK_POSES` is set, for a part none of whose grasps fits a stable pose. Such parts are marked on the template
   ("place carefully") and recorded in `grasps.pkl` (`settings.flat_pickup.fallback_parts`).
3. Sequence planning (`run_seq_plan.py`) and optimization (`run_pickup_gen.py --seq-opt`). The hold grasp of the first step is
   the only hold grasp that takes its part from the board: it must pick the part lying flat. Candidates that cannot are removed.
   Fallback if none can, and also if the hold arm reaches the part on no spot of the layout (stage 4): the hold grasps stay
   as they are, and the move arm brings the first part to the assembly spot as a planned and checked step (pickup with one
   of the part's own move grasps, set down on the table at its assembled pose, release, back to rest); the hold arm then
   takes the part where it stands. The layout marks such a part with `place` in `flat_pickup.json`. Without a move grasp
   that can do this there is no sequence (for the reach case: the hold grasp is dropped, cause `hold grasp`).
4. Pickup layout (`run_pickup_gen.py`): the parts are placed one by one in reverse pick order (so the parts lying around at each
   pickup are exactly those that still lie there during the run), each at the free spot nearest to the pickup area (1.25 cm grid,
   on the board, outside the assembly zone, `PART_GAP` 2.0 cm between the footprints of two parts) and the resting pose /
   rotation where
   - its pickup passes every check, with the gripper opened (as it arrives) and closed on the part (the first pose of the
     transport), and
   - the arm can move straight into the grasp (back along the hand axis until the gripper is clear of every part) and straight
     up with the part to the transport height. Both are tested with the motion planner's own routine (`PickupMoves`), so a
     spot that the planner cannot approach is skipped here instead of failing later (duct: a hand that leans with its back to
     a virtual wall).

   What follows from the gripper pose alone is tested before an arm pose is searched for a spot, because that search takes
   seconds (up to 52 IK runs): the gripper too close to another lying part or the wrist too close to any part
   (`SceneChecker.pickup_blocked`), the wrist frame or the flange behind a virtual wall at the pickup or on the lift above it
   (`get_hand_wall_margin`). Both stay `PICKUP_POSE_SLACK` 0.1 cm below the checked limits, so no spot is dropped that the
   full check would take. (A layout far from the arm took 27 min without these tests, 4 min with them.)

   The first part (the hold arm picks it, before anything else moves) is placed last, at the nearest spot where the hold
   arm has an arm pose for its hold grasp whose arm and opened gripper touch neither the lying parts, the table nor the move
   arm at rest (`run_pickup_gen.hold_pickup_q`; the pose goes to `fixture/pickup_q_hold.json`). Until 2026-11-13 any free
   spot counted and a human picked the part. If none of the 150 nearest spots works, the layout starts again with the
   fallback of stage 3: the move arm brings the part.

   If none of the `MAX_PLACEMENT_TRIES` 150 nearest spot / resting pose combinations fits, the search goes on over the whole
   board (`MAX_PLACEMENT_TRIES_WIDE` 300 further checks, every second grid spot, only where the footprints of the fingers keep
   1 cm to the parts already placed). For every pickup several IK solutions of the redundant arm are computed and the best
   conditioned one is kept, including the poses while lifting the part 2.5–10 cm (the arm folds further when lifting close to
   the robot). Spots listed in `banned_spots.json` are skipped (stage 6).
5. Motion planning (`run_motion_plan.py --no-viz`): every path of the real arm is checked and re-planned with another seed
   if it fails or the planner gives up (`MAX_PLAN_ATTEMPTS`). Only one arm moves at a time, and it avoids the other arm,
   which stands still meanwhile. The paths of the hold arm are planned the same way and checked for their distance to the
   move arm; one that cannot be planned is a failure with cause `hold grasp` (until 2026-11-13, with `--single-arm`, the
   move arm did not avoid the hold arm, a hold path without a plan was stored as a jump, and a first part out of the hold
   arm's reach was left to the human; the option is gone). When the planner finds no path, the message names what stopped
   its straight move, e.g. `no straight move into the goal pose from a pose clear of everything (virtual wall after 0.25 cm)`.
6. If a stage finds no clean plan it says what has to change (`planning_failure.json`, `cause`), and the loop does that, at
   most `MAX_SEQUENCE_ATTEMPTS` 10 times in total:

   | Cause | When | What the loop does |
   |---|---|---|
   | `grasp` | the arm or the gripper do not fit: an insertion path or the way out of a release pose breaks a rule, the insertion start cannot be approached because of the arm or the gripper, no free path in `MAX_PLAN_ATTEMPTS` tries, or the layout finds no spot on the whole board | the move grasp is removed, the next best sequence is planned |
   | `spot` | the place where the part lies: no straight move into its grasp or out with it, or every rejected path is too close to parts lying beside it | the grasp stays; the spot is added to `banned_spots.json` and the layout runs again (same sequence) |
   | `hold grasp` | the hold arm cannot do its part with this hold grasp: no arm pose to pick the first part, no path in `MAX_PLAN_ATTEMPTS` tries, or too close to the move arm; also the layout when the hold arm reaches the first part nowhere and no move grasp can bring it | the hold grasp is removed (`run_planning_flat.ban_hold_grasp`), the next best sequence is planned |
   | `insertion` | the part itself cannot get to where its insertion starts (too close to the table or to the assembly on the straight move there), however it is grasped | planning stops with that message |
7. Final check (`verify_plan.py`), place poses (`run_place_poses.py`), `workcell.json`. The place poses are a help for laying
   the parts out and do not decide whether the plan counts: `place_parts.py` moves in joint space between the spots, which
   can collide where hands lean far. In that case `place_poses.json` is removed (so the moves cannot be executed) and the
   plan is reported as done with that note.

## Grasps (`planning/flat_grasp.py`, `run_grasp_arm_gen.py --flat`, values in `planning/config.py`)

The original generator makes grasps for the assembled pose and keeps 100 at random; whether a grasp can also pick the part from
the table was only tested afterwards, so small round and long thin parts often kept none. With `--flat`:

| Step | What happens |
|---|---|
| contact | Contact of the fingers with the part is measured on a fixed grid of the two gripping pads (`PAD_CELL` 0.25 mm; the pad is the flat inner face of the finger, 1.75 × 1.75 cm, starting 0.15 mm behind the fingertip) instead of 200 random points on the whole finger. A grid point is in contact if the part surface is at most `PAD_CLOSING` 0.4 mm in front of it, counted from where the pad first touches the part. Stored per grasp: `contact_points` (every 1 mm, for the scores of the original pipeline), `contact_depths` (per finger, from the fingertip), `contact_area`. |
| candidates: matching of the two poses | A grasp is the pose of the gripper relative to the part, so one grasp has to serve the lying and the assembled part. Candidates come from both sides: (a) as in the original generator, antipodal point pairs on the part at the assembled pose, 10 hand directions around each pair; (b) made for the lying part (`lying_grasp_candidates`, `FLAT_LYING_PT` 200 pairs shared by the resting poses): pairs whose line is at most 45° from horizontal with the part lying, the hand from as straight above as the pair allows and leaning 20° to either side, the fingers as low as the board allows. Candidates of (a) are then tested on the lying part, candidates of (b) at the assembled pose; all that pass both are rated together. |
| usable | A move grasp is usable if it passes the criteria of the grasp filter (stage 2). The finger overlap is that of the weaker finger. |
| moves at the assembly | Of the usable grasps, those are dropped with which the real arm cannot do the two straight moves at the assembly that every plan contains (`run_motion_plan.assembly_moves_blocked`): with the part, straight into the insertion start pose from a pose clear of everything; after the release, with the gripper opened, straight back along the hand axis until it is clear of everything. Tested with the motion planner's own routine in the smallest scene that is certain then (the parts that have to be assembled before); with more parts around these moves only get harder. For the part that is held first (nothing is assembled before it) only the release is tested: it is not inserted, also where the precedence plan gives it a path (plumbers_block, duct), and its move grasps serve the first-part fallback. Without this test motion planning found out at the end of a whole planning attempt, one grasp per attempt (human_normal at (0, -7.5): nine grasps of the last part in a row, no plan). The output line counts them: `N more dropped (no straight move of the arm to the insertion start or away after the release)`. |
| more candidates | A part with fewer than `FLAT_MIN_GRASPS` 10 usable move grasps gets a second round: `FLAT_RESAMPLE_PT` 600 further antipodal pairs from new surface points (the first round uses 200). Of those only the candidates that can pick the part lying flat are checked for reachability. |
| choice | If more than `--max-n-grasp` 100 are usable, the best rated are kept (`rate_flat_grasp`: grip depth and area 0.4, centre of mass near the line between the fingers 0.25, hand upright 0.2 (1 from straight above, 0 from the side), insertion score of the original pipeline 0.15), taking turns between the approach directions so that alternatives remain when a grasp is dropped later. |
| hold grasps | Chosen at random as in the original pipeline. The part that is held first also gets up to 50 of its best rated hold grasps that can pick it lying flat. And for every part that can be inserted while this part is held, hold grasps whose gripper leaves room for that insertion are added until `FLAT_HOLD_ROOM` 5 are kept (if that many exist): the random 100 can miss them (car: 3 of 441). |

`python planning/test_flat_grasp.py` tests these functions, the straight moves at a pickup spot and at the assembly, the tests
made before an arm pose is searched, the banned spots, the resting poses and the layout of a planned assembly, on toy parts
with known dimensions (43 tests, about 2 min).

## Places of the assembly and of the parts (`planning/run_workspace_search.py`, `planning/robot/workcell.py`)

Both places are inputs of a plan and are stored in its log. `--pickup-center X Y` (of `run_planning_flat.py` and
`run_workspace_search.py`) sets the place around which the parts are laid out; the default is (0, 5). The layout takes the
nearest spots to it where every check passes, so the parts lie where the real arm can pick them, as close to that place as
possible. An existing log can be planned again around another place (the grasps do not depend on it).

By default the assembly stands at (0, -15) cm on the board, between the two arms. For the real arm that is the edge of its
range: 9 cm to the left its flange meets a virtual wall, and for y > -5 only a strip about 10 cm wide is free of wall and
joint-limit problems for a hand that points straight down; from y = -20 downwards the range is 30 to 35 cm wide (measured on a
5 cm grid with the flange 15 cm above the table). A part that is
inserted from the far side can have no room for its approach at the default place (human_normal part 4).

- `python planning/run_planning_flat.py --exp-name <exp> --assembly <assembly> --assembly-center X Y` plans a new log for
  another place. The log stores it (`assembly_center.json`); the stages, `verify_plan.py`, `plan_place_part.py`,
  `run_place_poses.py`, `rendering/render_seq_plan.py` and `learning/preprocessing/prepare_isaac_plan_info.py` read it from the
  log (`use_log_workcell`), so nothing has to be set by hand afterwards. Logs without the file use the default place.
- `python planning/run_workspace_search.py --exp-name <exp> --assembly <assembly>` searches the place:
  1. places on a 5 cm grid (x -15..25, y -40..0) where the assembly zone stays on the board and off the pickup anchor;
  2. every place is rated without planning: for each part a sample of 24 grasps that can pick it lying flat and whose gripper
     is free at the assembly is tested with the move arm at that place (IK, virtual walls, singularity and joint limits at the
     assembled pose and at the start of the insertion, 3 cm of room behind the insertion start); for the part held first, IK of
     the hold arm. The rating is the share of grasps that pass, for the part with the fewest;
  3. the pipeline runs for the best places (`--max-places`, 3) until one gives a verified plan; a place other than the default
     gets the log `logs/<exp>_x<X>_y<Y>/<assembly>`. Ratings and results: `logs/<exp>/workspace_search_<assembly>.json`.
  `--precedence-from <log>` reuses a precedence plan made at the default place, `--rate-only` stops after step 2.
- A plan at another place needs its own templates and its own policy training, like every new plan.
- `python planning/test_workspace_search.py` tests the stored places, the moved precedence plan, the rating and the places of
  the position test (9 tests, 45 s).
- `python planning/run_position_test.py --exp-name <exp> --assembly <assembly> --precedence-from <log>` tests the pipeline at
  many places: 10 places of the assembly with 3 places of the pickup area each (`--n-assembly`, `--n-pickup`, `--seed`), drawn
  at random on the grid of the board holes from the area both arms reach (`reachable`: a hand pointing straight down, the
  flange 15 cm above the table; move arm inside its walls and away from singularities and joint limits, hold arm: an arm pose
  exists). The middle and the four corners of the assembly's footprint and the middle of the pickup area have to be
  reachable (`--min-corners`), the assembly zone on the board, the pickup area (7.5 cm around its middle) clear of the zone;
  places are 7.5 cm apart (`--min-separation`). Grasps are generated once per place of the assembly, then the pipeline runs
  for every pair. A place where a part has no working move grasp is not a place the real arm can build the assembly at,
  whatever the reach test with the empty hand says: it is recorded with the reason, no pair is planned there, and a spare
  place is used instead (`--n-spare`, 4). Report: `logs/<exp>/position_test_<assembly>.json` and a table.

## Checks of the real arm (`planning/plan_checks.py`, values in `planning/config.py`)

Exact geometry, no planner buffers, every pose of every path in `motion.pkl`. Distances are compared with a tolerance of
0.001 cm (`DISTANCE_TOL`), so that a distance equal to its limit up to rounding passes:

| Rule | Value |
|---|---|
| arm links to every part; gripper and carried part to lying parts | ≥ `ROBOT_PART_CLEARANCE` 1.0 cm |
| gripper vs the part it is about to pick | no contact on the whole approach (it touches the part only when it closes) |
| gripper / carried part vs the assembly, including the part just released | no contact (the carried part is not checked while inserting: insertion paths are fixed by the grasp) |
| gripper / carried part passing by the assembly (a path may start or end closer: leaving after a release, arriving to insert) | ≥ `ASSEMBLY_CLEARANCE` 1.0 cm |
| gripper (hand, fingers) vs the robot's own links 0–5 | ≥ `ROBOT_SELF_CLEARANCE` 1.0 cm |
| the arm that moves (links, gripper, carried part) vs the other arm, which stands still meanwhile; checked for the paths of both arms | no contact; ≥ `ARM_ARM_CLEARANCE` 1.0 cm except in one stretch at the start and one at the end of a path (both hands work at the assembly); while inserting only no contact (since 2026-11-13) |
| carried part above the table outside the assembly zone (assembly + `ASSEMBLY_ZONE_MARGIN` 4.0 cm; 3.0 until 2026-11-12), once lifted | ≥ `TRANSFER_PART_CLEARANCE` 3.0 cm |
| carried part above the table inside the assembly zone, until the last stretch of its path, where it comes down to its goal (not while inserting: an insertion may end on the table) | ≥ `ZONE_PART_TABLE_CLEARANCE` 0.2 cm |
| gripper / arm above the table | ≥ 0.2 cm / ≥ 1.0 cm |
| franka-interface virtual walls | ≥ `VIRTUAL_WALL_SAFETY` 1.0 cm |
| singularity: smallest singular value of the arm Jacobian | ≥ `MIN_SIGMA` 0.10, while inserting ≥ `MIN_SIGMA_INSERTION` 0.15 |
| every joint to its limits (e.g. elbow not folded up) | ≥ `MIN_JOINT_LIMIT_MARGIN` 0.5 rad |
| the same two rules for the poses while lifting a picked part 2.5–10 cm, and for the insertion poses of every grasp (grasp filter) | |
| part reaches into the fingers (tolerance against a gripper that is too high); grasps made with `--flat`: the weaker finger, on the pad grid | ≥ `GRASP_MIN_FINGER_OVERLAP` 0.3 cm |

## Changes to the original Fabrica pipeline

| File | Change |
|---|---|
| `planning/robot/workcell.py` | robot base height `get_panda_base_z()` (original: 0), table height, franka-interface virtual walls |
| `planning/robot/motion_plan_arm.py` | virtual walls and table clearance of carried parts as constraints (inside the assembly zone the carried part only keeps off the 0.2 cm table slab); straight lift of a carried part to the clearance height; retract steps where the arm would flip are rejected; reports what stopped a straight move; `VISUALIZE` switch |
| `planning/run_motion_plan.py` | works without `fixture.obj`; every move-arm path checked and re-planned; hold-arm paths re-planned and checked for their distance to the move arm; names the failing grasp and the cause (move grasp, hold grasp, spot or insertion); `--no-viz` |
| `planning/run_grasp_arm_gen.py` | `--flat`: grasps for this pipeline (pad-grid contact, candidates from both poses, second sampling round, flat test before the cut, straight moves at the assembly, rated choice, hold grasps that leave room; no blocking box where the fixture of the original pipeline stands, so the assembly may stand anywhere); without it the original generator is unchanged |
| `planning/run_position_test.py` (new) | runs the pipeline at many places of the assembly and of the pickup area |
| `planning/flat_grasp.py` (new) | resting poses, flat-pickup test, pad-grid contact, criteria of a move grasp, rating and choice |
| `planning/run_pickup_gen.py` (new) | replaces fixture generation: grasp filter, sequence optimization, checked part-by-part layout with cached best-conditioned pickup IK and the straight moves tested at every spot, 1:1 templates |
| `planning/plan_checks.py`, `planning/verify_plan.py` (new) | the checks above, final check |
| `planning/run_planning_flat.py`, `.sh` (new) | the pipeline; its loop drops a grasp, moves a part to another spot or stops, depending on the cause of a failure |
| `planning/run_workspace_search.py` (new) | rates places of the assembly over the board and plans at the best one |
| `planning/run_place_poses.py`, `real_robot/place_parts.py` (new) | the robot shows where each part goes |
| `real_robot/run.py` | `--place-at-pickup`; refuses plans for another robot height or without passed check |
| `rendering/` | fixture optional, `--speed`, `show_initial_setup.py` (still view of the layout), Blender 4 import fix |

## Limits

- The wrist camera mount and its cable are not in the robot model: the gripper is checked as the bare Franka hand.
- `logs/exp_human_normal_flat_v3` was planned before the gripper rules were tightened (2026-11-10) and no longer passes them
  (the gripper passes through the part it just released on five retracts and touches part 2 before grasping it).
  Re-plan it before the next run on the robot (the changed planner produces a passing plan from the same grasps and layout).
- The hold arm's paths are checked only for their distance to the move arm. Its own walls, singularities, joint limits and
  table distance are not checked (the rules of `plan_checks.py` are written for the move arm and its franka-interface
  walls; the walls of a second robot are not known yet); its planner avoids the parts, the table and the move arm with the
  buffers of the original Fabrica planner.
- With `real_robot/run.py --single-arm` a human does the hold arm's steps. The human's hands at the assembly are protected only
  indirectly (no contact with the assembly, 1 cm to all parts): keep them on the side away from the robot.
- Insertion paths come from the grasp and the disassembly plan; they are checked for singularities, joint limits, walls and table,
  not re-planned.
- Open (measured 2026-11-13, not changed): the planner keeps its clearances by testing against inflated part meshes
  (`geometry.offset_mesh`), and for parts with fine detail these have spikes, 8 of the 62 parts of the ten assemblies: e.g.
  cooling_manifold parts 0 and 4 (1.8 x 1.7 x 4.6 cm) become 9.4 x 19.6 x 7.1 cm, duct part 0 gets spikes of 2 m. For
  those parts the planner refuses poses that are free (fewer spots and grasps, attempts that fail without need); the
  checks of a finished plan measure real distances and are not affected. Details: `assembly_runs/README.md`.
- The motion planner of the real arm (`ArmMotionPlanner.strict`) plans with the rules above, plus a small reserve (`PLANNER_*` in
  `planning/config.py`): a path is a straight move out of the release / grasp pose until the clearances hold, a free path that keeps
  them, and a straight move into the next pose. If a straight move is impossible (e.g. a wall or a neighbouring part in the way)
  there is no path (stage 6 decides what changes); a finished path that still fails the check is re-planned (up to
  `MAX_PLAN_ATTEMPTS`).
- `python planning/test_plan_rules.py` tests every rule, the planner and the causes of a failure on a toy scene (no robot, 40
  tests, about 30 s).
- Memory: the planner's setup for a path (`get_fns`) builds collision structures of all still parts, for assemblies with
  large meshes several hundred MB (gamepad: two parts with 52000 faces). It releases those of the call before when it
  builds new ones; a motion plan of gamepad then needs about 1.4 GB. Plans of several assemblies or places can run at the
  same time only as far as the memory of the machine allows (15.9 GB here: 6 plans of gamepad).
- The first part is picked from the board by the hold arm with its hold grasp. If no first hold grasp can pick it lying flat,
  or the hold arm reaches it on no spot of the layout, the move arm brings it to the assembly spot (stages 3 and 4).
  `real_robot/run.py` executes that step like every other path of the move arm; its prompts for the human were not adapted to it.
- Plans made before 2026-11-13 were made without the move arm avoiding the hold arm. Checked again with the new rule, the
  plan of `v0.1_exp_human_normal_flat_run6` fails it: the move arm's hand passes the hold arm's link 5 closer than 1 cm, and
  carried parts pass its hand closer than 1 cm.
- Inside the assembly zone a carried part may come as low as 0.2 cm above the table (`ZONE_PART_TABLE_CLEARANCE`); a part whose
  insertion starts on the table itself (0 cm) still has no approach. The zone is a fixed margin around the assembly (4 cm): a
  part that is inserted low and is clear of its neighbours only further out than that has no approach either (gamepad part 4
  needs 3.3 cm).
- Hold grasps are chosen at random, plus those that leave room for each part inserted while the part is held
  (`FLAT_HOLD_ROOM`). A plan can still depend on a single hold grasp (car: one of 151 hold grasps of the chassis leaves room
  for the body).
- With no limit on how far the hand leans, the sequence optimisation often takes grasps whose hand leans 40 to 80° at the
  pickup; it chooses by the scores of the original pipeline, not by the pickup. `run_place_poses.py` (straight joint-space
  moves between the spots) can then report a collision although the plan is verified (human_normal run 4).
- Known from the runs on all assemblies (2026-11-11, `assembly_runs/README.md`):
  - Hexagonal bars lying on a face (the stool legs, 1.8 cm across) have no grasp: the fingers grip two faces that slope 60°,
    and with 3 mm of finger on the face the lower fingertip would be less than 1 mm above the board.
  - The grasps that can pick the bolts and the block of plumbers_block lying flat collide at the assembled pose.
- A plan matches reality only as well as the robot height, the printed parts and the placement of the first part do.

## Required change in franka-interface

`run.py` streams every transfer path to franka-interface's joint impedance skill. Upstream franka-interface (iamlab-cmu,
commit `f83bc2a`) limits that skill to 10 % of the rated joint torques (`safety_factor = 0.1`: 1.2 Nm on joints 5-7), too
little to follow larger wrist rotations: the wrist stalls, the other joints follow, and the arm sags off the plan (measured
with `real_robot/record_state.py`; this made the leg of human_normal hit the table). On this PC the change is committed in
`~/tobi/franka-interface`, local branch `fabrica-ics` (commit `4e8c68d`):

- `franka-interface/include/franka-interface/skills/impedance_control_skill.h`: `safety_factor = 0.5` (6 Nm on joints 5-7, 43.5 Nm on joints 1-4)
- `franka-interface/include/franka-interface/termination_handler/termination_handler.h`: virtual walls sides ±0.65 m, floor −0.05 m (the planning checks against these)

Rebuild with `cd ~/tobi/franka-interface/build && make -j8 franka_interface` and restart the control stack after changing it.

## Tools for the real robot

- `planning/plan_place_part.py --part 0`: the move arm takes a part from its pickup spot and stands it at the assembly spot
  (e.g. the first part instead of the human; plan it with only this part on the table, then lay out the others). Writes
  `<log-dir>/place_part<id>/motion.pkl`, run with `real_robot/run.py --single-arm --fn ...`.
- `real_robot/record_state.py` (record joints, TCP, torques, streamed joint goals and cartesian goals while run.py runs) and
  `real_robot/compare_recording.py` (compare with the plan).

## Execution fixes in real_robot/

- `run.py` refuses to start when more than one `franka_interface` runs: extra instances share its shared memory and report
  stale robot states (the arm was pulled 30 cm up during insertions). Stop all and start the control stack once.
- `robot_interface.py` reads joints, velocities and the pose directly from the robot state: frankapy's getters also convert
  `O_T_EE_d`, which is invalid right after a skill switch ("Illegal rotation" crash at the start of an insertion).
- `stop_skill()` waits until franka-interface has accepted a goal before cancelling it, and the insertion skill's buffer time
  is 5 s instead of 100 s (a lost cancel blocked run.py after an insertion that was near its goal immediately).

## TODO

- [ ] Fork `iamlab-cmu/franka-interface` to `TobiasToews` (https://github.com/iamlab-cmu/franka-interface/fork) and push the branch
      `fabrica-ics`, so the change is not only on this PC:
      `cd ~/tobi/franka-interface && git remote add fork git@github.com:TobiasToews/franka-interface.git && git push -u fork fabrica-ics`
