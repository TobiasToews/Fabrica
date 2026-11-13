# socket_set_small (3 parts) — `exp_socket_set_small_flat_run1`

See [README.md](README.md) for the setup and for what Fact / Interpretation / Not examined mean.

**Result: no plan. Stopped in sequence optimisation after 1 min** (2026-11-10 21:09:11–21:10:52, exit code 1).

## Facts

- Console log: precedence, grasp generation, grasp filter and sequence planning finished. Sequence optimisation printed
  `0/18 first hold grasps can pick their part lying flat`, then `Failed sequence planning`, then `no sequence left`.
- `stats.json`: precedence 1.9 s, grasp generation 92.9 s, sequence planning success ("tree fully explored (solution found: True)").
- Grasp filter (console log): part 0 move 100 -> 17, part 1 move 100 -> 13, part 2 move 90 -> 16. Hold grasps were not reduced
  (100, 82, 34).
- Precedence graph (`precedence.pkl`): edges 0 -> 1 and 0 -> 2.
- Computed from `grasps.pkl` and `tree.pkl` with the pipeline's own functions:

  | Part | Size (cm) | Hold grasps | of them can pick the part lying flat |
  |---|---|---|---|
  | 0 | 7.2 x 6.0 x 2.0 | 100 | 43 |
  | 1 | 2.4 x 2.4 x 2.4 | 82 | 51 |
  | 2 | 3.0 x 3.0 x 2.4 | 34 | 24 |

  All 18 "first hold" nodes of the grasp tree belong to part 0, and none of these 18 hold grasps is among the 43.
- Code (`planning/run_pickup_gen.py`, `run_flat_seq_opt`): a first-hold node is removed when its hold grasp cannot pick the part
  lying flat (`MAX_TILT` 30 deg, `GROUND_CLEARANCE` 0.5 cm, `MIN_TIP_ANGLE` 20 deg). With all 18 removed no sequence is left.
  `run_planning_flat.py` then stops; it has no fallback for this case (there is no move grasp to ban).

## Interpretation

- The failure is not about the new gripper rules or the motion planner: it happens before any pickup layout or path is planned.
- Flat-pickable hold grasps for part 0 exist (43), but the sequence tree only contains 18 hold grasps of part 0 and those are
  all among the other 57. The hold grasps are not filtered for flat pickup before sequence planning, so the tree is built without
  knowing this requirement.

## Not examined

- Why the tree contains only those 18 hold grasps (how `run_seq_plan.py` selects hold grasps).
- Whether the requirement is needed at all with `--single-arm`, where a human brings the first part to the assembly spot.

## Suggestions (none implemented)

1. Filter the hold grasps of parts that can be held first for flat pickup before sequence planning, like the move grasps
   (small change in `run_pickup_gen.py --filter`).
2. In single-arm mode, do not require a flat-pickable first hold grasp (the human places part 0). Needs a decision, because
   `planning/plan_place_part.py` (robot places the first part) relies on such a grasp.

---

# Run 2 — `exp_socket_set_small_flat_run2` (2026-11-11, new grasp generation)

Setup of the second batch: [README.md](README.md), section "Second batch".

**Result: plan verified** (grasp generation 1 min 44 s, planning 1 min; first sequence attempt, no grasp dropped).

## Facts

- Grasp generation (`run_grasp_arm_gen.py --flat`), from the console log:

  | Part | Size (cm) | Candidates | Reachable move grasps | Usable (pick the part lying flat, all criteria) | Resting pose | Kept move / hold |
  |---|---|---|---|---|---|---|
  | 0 (held first) | 7.2 x 6.0 x 2.0 | 1065 | 467 | 108 | stable | 100 / 150 (50 hold grasps added that pick it lying flat) |
  | 1 | 2.4 x 2.4 x 2.4 | 1200 | 197 | 54 | stable | 54 / 100 |
  | 2 | 3.0 x 3.0 x 2.4 | 1375 | 194 | 44 | stable | 44 / 100 |

  No part needed the second sampling round. The grasp filter kept every generated move grasp (100, 54, 44).
- Sequence optimisation: 5 of 21 first hold grasps in the tree can pick part 0 lying flat (run 1: 0 of 18).
- Layout (`fixture/flat_pickup.json`): part 0 with hold grasp 939 (tilt 28.4 deg), part 2 with move grasp 282 (4.4 deg), part 1
  with move grasp 743 (2.7 deg); all in stable resting poses.
- Motion planning: no path of the real arm was rejected. One path of the simulated hold arm could not be planned and is stored
  as a jump (`[run_motion_plan] no path for the hold arm in task transport: shown as a jump`).
- `verify_plan.py`: PASSED. `workcell.json`: `verified: true`, `banned_move_grasps: []`.
- The same pipeline run in a scratch folder before the batch (own precedence plan) gave the same grasp sets per part and also a
  verified plan, with other grasps chosen by the sequence optimisation (hold 549, move 507 and 720).

## Interpretation

- The cause found in run 1 (no first hold grasp that picks part 0 lying flat) is removed by the hold grasps that the generator now
  adds for the part held first.
- The two runs choose different grasps from the same grasp sets. Not verified why; the precedence plan differs between them (own
  plan vs. the one copied from run 1), and the insertion score that ranks the grasps is computed from it.

## Not examined

- The plan was not executed, rendered or used for policy training.
- Which hold-arm path became a jump and why (it does not affect the real arm; the human does that step).
- Two additional scratch tests forced the cases "no hold grasp picks part 0 flat" and "no grasp at all picks it flat" by deleting
  grasps: layout, motion planning and `verify_plan.py` passed in both, and `plan_place_part.py --part 0` found checked paths for
  the first case. These were made on this assembly only.

## Suggestions

1. None needed for planning. Before a run on the robot: look at the plan in the viewer, and note that the hold grasp of part 0 is
   tilted 28.4 deg (limit 30 deg).

# Run 4 (2026-11-11/12, night): after the user's decisions

Experiment `exp_socket_set_small_flat_run4`, code as described in [README.md](README.md) ("Fourth batch"): no tilt limit, wrist
rule, 1.2 cm between lying parts, straight moves tested in the layout, causes of a failure.

- Grasp generation 1 min 46 s. Usable move grasps: part 0: 280 (100 kept), part 1: 82, part 2: 84. No second sampling round.
- **Plan verified** in the first sequence attempt, planning 1 min (layout 17 s, motion planning 29 s). No path rejected.
- Layout: part 1 at (0.0, 5.0), move grasp 743, hand 3 deg from vertical; part 2 at (-3.75, 3.75), move grasp 568, 17 deg;
  part 0 (held first, picked by the human in single-arm mode) at (0.0, 11.25), hold grasp 780, 70 deg.
- Place poses for `place_parts.py`: all moves checked, virtual wall margin 9.4 cm or more.

Not examined: how a hold arm would pick part 0 with a hand leaning 70 deg (the gripper is 0.55 cm above the board there).

# Run 5 (2026-11-12): fifth batch and position test

Batch, default place (`exp_socket_set_small_flat_run5`): **verified**, first sequence attempt. Grasps 2 min 16 s, layout 3 s,
motion planning 29 s; poses for `place_parts.py` without collision.

- Move grasps usable / kept: part 0: 358 / 100 (71 more dropped for the straight moves at the assembly: part 0 is held
  first, so only the release counts), part 1: 102 / 100, part 2: 101 / 100 (1 dropped).
- Candidates made for the lying part among the usable ones: 75 (part 0), 6 (part 1), 0 (part 2).
- Pickups: part 1 move grasp 743, hand 3 deg from vertical; part 2 move grasp 568, 17 deg; part 0 hold grasp 825, 65 deg.

Position test (`exp_socket_set_small_pos_*`, see README): 30 of 30 pairs of places verified, twice (before and after the
changes of that morning), each plan in the first attempt, 0.7 to 1.4 min. Places of the assembly over the whole board
(middle of the footprint from y = -34.5 to +35.5). This is the easiest assembly; it served as a smoke test.

# Run 6 (2026-11-12, 22:02): sixth batch, final code of the day

Default place (`exp_socket_set_small_flat_run6`): **verified**, first sequence attempt. Grasps 2 min 55 s, layout 3 s,
motion planning 23 s; `place_parts.py` poses without collision. Usable move grasps: 291 (part 0), 120, 96. Pickups: part 1
hand 1 deg, part 2: 5 deg; part 0 hold grasp 1036, 48 deg. The position test was not repeated on this code.
