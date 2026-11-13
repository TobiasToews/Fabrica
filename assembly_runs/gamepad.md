# gamepad (6 parts) — `exp_gamepad_flat_run1`

See [README.md](README.md) for the setup and for what Fact / Interpretation / Not examined mean,
and [grasp_filter_finding.md](grasp_filter_finding.md) for how the table below was computed.

**Result: no plan. Stopped in the first sequence attempt after 6 min** (2026-11-10 21:26:45–21:33:01, exit code 1).

## Facts

- Console log: precedence and grasp generation finished; the grasp filter left at least one part without a move grasp; sequence
  optimisation printed `No complete sequence`, then `no sequence left`. No pickup layout and no path was planned.
- Move grasps per part and how many fail each criterion of the filter (a grasp can fail several):

  | Part | Size (cm) | Resting poses | Move grasps | walls | sigma | joint | overlap (old formula) | flat | pass | pass (old overlap) |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 0 | 1.2 x 1.2 x 2.9 | 2 | 36 | 2 | 15 | 1 | 36 (16) | 28 | 0 | 4 |
  | 1 | 2.3 x 2.3 x 3.0 | 2 | 30 | 1 | 6 | 1 | 30 (0) | 6 | 0 | 20 |
  | 2 | 2.3 x 2.3 x 3.0 | 2 | 79 | 0 | 9 | 2 | 72 (0) | 8 | 7 | 63 |
  | 3 | 13.5 x 7.6 x 2.5 | 2 | 100 | 4 | 26 | 6 | 16 (1) | 53 | 29 | 40 |
  | 4 | 1.4 x 6.0 x 1.2 | 4 | 32 | 0 | 0 | 0 | 21 (13) | 15 | 2 | 6 |
  | 5 | 1.2 x 1.2 x 2.9 | 2 | 52 | 2 | 14 | 2 | 37 (7) | 21 | 4 | 25 |

## Interpretation

- Caused by the corrected overlap formula together with the 0.5 cm limit: parts 0 and 1 lose all move grasps on the overlap criterion (36 of 36 and 30 of 30 fail it). With the old formula every part keeps grasps (4, 20, 63, 40, 6, 25).
- The earlier gamepad runs of other sessions failed later, in motion planning; this run did not get that far, so it says nothing about the new planner on gamepad.

## Not examined

- The gamepad run of the other session (`exp_gamepad_flat_v5`) and its notes were not compared with this run.

## Suggestions (none implemented)

1. Decide on the overlap measure first ([grasp_filter_finding.md](grasp_filter_finding.md)), then repeat this run.

---

# Run 2 — `exp_gamepad_flat_run2` (2026-11-11, new grasp generation)

Setup of the second batch: [README.md](README.md), section "Second batch".

**Result: no plan after 10 sequence attempts (planning 137 min). Every attempt failed at the same step: the transport of part 4
to the start of its insertion.** All parts have usable grasps (grasp generation 3 min 15 s).

## Facts

- Grasp generation (`run_grasp_arm_gen.py --flat`), from the console log:

  | Part | Size (cm) | Candidates | Reachable move grasps | Usable (pick the part lying flat, all criteria) | Resting pose | Kept move / hold |
  |---|---|---|---|---|---|---|
  | 0 | 1.2 x 1.2 x 2.9 | 1205 | 149 | 52 | no stable pose (the one used tips at 18 deg) | 52 / 100 |
  | 1 | 2.3 x 2.3 x 3.0 | 1045 | 110 | 38 | stable | 38 / 100 |
  | 2 | 2.3 x 2.3 x 3.0 | 1305 | 138 | 57 | stable | 57 / 100 |
  | 3 (held first) | 13.5 x 7.6 x 2.5 | 985 | 424 | 152 | stable | 100 / 150 (50 hold grasps added that pick it lying flat) |
  | 4 | 1.4 x 6.0 x 1.2 | 960 | 46 | 11 | stable | 11 / 41 |
  | 5 | 1.2 x 1.2 x 2.9 | 1195 | 178 | 51 | no stable pose (the one used tips at 18 deg) | 51 / 100 |

  No part needed the second sampling round. The grasp filter kept every generated move grasp. (Run 1: parts 0 and 1 had no move
  grasp after the filter.)
- Sequence optimisation: 43 of 88 first hold grasps can pick part 3 lying flat. Sequence of the last attempt (move, hold):
  (2, 3), (1, 3), (4, 3), (5, 3), (0, 3).
- All 10 attempts ended with
  `[run_motion_plan] FAILED for part 4 (move grasp ...): transport path: no path found: no straight move into the goal pose from a pose clear of everything`,
  each after 8 planning attempts with the same message (80 rejections). The grasps blamed and removed were 292, 504, 343, 519,
  448, 354, 479, 453, 463, 403; of the 11 move grasps of part 4 only 466 is left. The layout and the transports of parts 2 and 1
  passed in every attempt.
- Geometry (computed from the assembly and `precedence.pkl`): in the assembly part 4 lies between 0.65 and 1.85 cm above the
  table. It is inserted horizontally: its disassembly direction is +y, the insertion starts 2.53 cm from the final pose.
- Scratch diagnosis of the last layout (`diag_goal_retract.py` in the session scratchpad; it walks from the insertion start
  backwards along the disassembly direction in 0.25 cm steps with the planner's own tests), for grasps 403, 292 and 504:
  - every step keeps the basic rules (no contact, joint limits, singularity, walls) up to 11.5 cm;
  - no step is "clear of everything": the planner's test reports `move part and ground` on every step, i.e. the carried part is
    inside the inflated table box that the planner keeps carried parts out of. Its top is 0.78 cm above the table (measured:
    bounds of `ground_mesh_buffered` in `planning/robot/motion_plan_arm.py`); the part is carried 0.65 cm above the table;
  - at 11.75 cm the carried part touches a lying part and the straight move ends.

## Interpretation

- The failure does not depend on the grasp. The strict planner (2026-11-10) requires a straight move into the insertion start
  that begins at a pose clear of everything. For a part that is inserted horizontally less than 0.78 cm above the table such a
  pose does not exist on that line, whatever the grasp. Banning grasps and re-planning could not help: 10 attempts of about
  14 min each were spent on it.
- The plan checks themselves (`planning/plan_checks.py`) have no rule against a carried part low above the table inside the
  assembly zone; the limit is the planner's table buffer for carried parts, taken over from the original planner.
- The other session's note (`logs/exp_gamepad_flat_v5/gamepad/VERSION_NOTES.md`) reports the same message for part 4.

## Not examined

- Whether a plan exists once part 4 can be approached (the transports of parts 5 and 0 and all later steps were never reached).
- Whether the remaining grasp 466 behaves differently (by the geometry above it cannot).
- Parts 0 and 5 have no stable resting pose; the pose used in the layout tips at 18 deg (marked on the template). How well that
  works on the real table is not known.

## Suggestions (none implemented; 1 and 2 change the motion planner and need a decision)

1. Let the straight move into an insertion start end at a pose that is clear of the parts and the assembly but may be as low as
   the insertion itself, inside the assembly zone (the checks allow a carried part to be low there). The free path would then end
   above that pose and descend straight down before the straight move, or the table buffer for carried parts is not applied
   inside the zone.
2. Stop the sequence loop when a failure does not depend on the grasp (here: "no pose clear of everything on the line"), instead
   of banning one grasp after the other.
3. Without planner changes: none found. Raising the assembly on the table (a base plate under part 3) would lift part 4 above
   the buffer, but changes the real setup.

# Run 4 (2026-11-11/12, night): after the user's decisions

Experiment `exp_gamepad_flat_run4`, code as described in [README.md](README.md) ("Fourth batch"): among other things a carried
part may come down to 0.2 cm above the table inside the assembly zone (the footprint of the assembly plus 3 cm; the planner
uses 2.5 cm).

## What happened

- Grasp generation 3 min 23 s. Usable move grasps: part 0: 70, part 1: 45, part 2: 63, part 3: 254, part 4: 14, part 5: 77.
- Attempt 1: layout complete. Motion planning: the transport of part 4 (move grasp 292) was rejected 8 of 8 times:
  `no straight move into the goal pose from a pose clear of everything (move part and part after 11.75 cm)`.
- Cause recorded: the insertion itself, whatever the grasp. Planning stopped there, after 9 min (run 2: ten sequence attempts,
  137 min, ten grasps removed).

## Why the approach still fails (measured, scratch script `diag_goal_retract.py`)

Part 4 (1.4 x 6.0 x 1.2 cm) is pushed in along -y, 0.65 cm above the table; assembled it spans y = -15.7 to -9.7, and -9.7 is
also the edge of the whole assembly. The insertion starts 2.53 cm further out.

| Pose of part 4 | Its far end | Planner's zone ends at y = -7.2 | Checked zone ends at y = -6.7 |
|---|---|---|---|
| start of the insertion | y = -7.17 | 0.03 cm outside | inside |
| first pose clear of the assembled parts (0.75 cm further out) | y = -6.42 | 0.78 cm outside | 0.28 cm outside |

Outside the zone the planner keeps the carried part above its table buffer (0.78 cm), so no pose on the line counts as clear;
the straight move goes on until the part touches lying part 0 after 11.75 cm. With the table buffer switched off everywhere
(for the measurement only) the pose 0.75 cm from the insertion start is clear of everything.

So the rule built for this part is right in kind but 0.3 to 0.8 cm short: the part is 6 cm long and has to come out of its
slot before it is clear of its neighbours.

## Scratch test: zone margin 4 cm instead of 3 cm

In a copy of the code (session scratchpad, `fix2`) with `ASSEMBLY_ZONE_MARGIN = 4.0` and the grasps of run 4: **plan verified
in the first sequence attempt**, no path rejected (layout 290 s, motion planning 1033 s). Place poses: all moves checked.

Interpretation: a fixed margin fits this part, not every part. A part that has to travel further before it is clear needs a
larger zone again. A rule that follows the part would be: low is allowed inside the zone and along the straight approach of
the part into its own insertion. Not built.

# Run 5 (2026-11-12): fifth batch

Default place (`exp_gamepad_flat_run5`): **verified**, first sequence attempt: the first plan of gamepad in a batch (run 4
stopped at the insertion of part 4; with the zone margin of 4 cm a scratch copy had found a plan). Grasps 5 min 44 s,
layout 3 min 33 s, motion planning 21 min 25 s, 41 min for the plan in all; poses for `place_parts.py` without collision.

- Move grasps usable / kept: part 0: 156 / 100, part 1: 65 / 65, part 2: 71 / 71, part 3 (held first): 359 / 100,
  part 4: 22 / 22, part 5: 162 / 100. Dropped for the straight moves at the assembly: 1, 2, 0, 92 (part 3, release
  only), 0, 5.
- Pickups: part 4 hand 0 deg from vertical, parts 1 and 2: 6 and 7 deg, part 0: 20 deg, part 5: 32 deg; part 3 hold
  grasp 825, 75 deg. Parts 0 and 5 lie in a pose that tips at 18 deg.

Not examined: where the 21 min of motion planning go.

## Position test (`exp_gamepad_pos_*`, table in README)

First start (plans from 16:14): stopped after 10 min. Eight motion plans at once held 14.3 GB of the 15.9 GB of the
machine. Cause, measured: each setup of the planner for a path left its collision structures behind, 218 MB per setup for
gamepad (parts 0 and 5 have 52000 faces each, part 4 has 21000). The planner now releases them before it builds new ones;
a motion plan of gamepad then holds about 1.4 GB.

Second start (plans 16:31 to 19:47, 6 at once): **30 of 30 pairs verified**, every plan in the first sequence attempt, no
grasp and no spot dropped, 29 to 54 min per plan. 10 places of the assembly (x from 0 to +12.5, y from -30 to -7.5; no
further place 5 cm from these has all four corners of the 13.5 x 10.6 cm footprint in the reach of both arms), 3 pickup
areas each (y from -35 to +37.5). Usable move grasps of part 4: 21 at (0, -20), 22 at (0, -15), 12 at (+10, -25).
`place_parts.py` poses without collision for 24 of the 30 plans.

# Run 6 (2026-11-12, 22:02): sixth batch, final code of the day

Default place (`exp_gamepad_flat_run6`): **verified**, first sequence attempt. Grasps 7 min 23 s, layout 2 min 41 s,
motion planning 14 min 32 s (run 5: 21 min 25 s), 30 min for the plan in all; `place_parts.py` poses without collision.
Usable move grasps: 146, 52, 110, 319 (part 3, held first), 22 (part 4), 163. Pickups of the move arm with hands 0 to
20 deg; part 3 hold grasp 775, 72 deg.

To know: the inflated mesh the planner uses for part 0 has a spike (1.2 x 1.2 x 2.9 cm becomes 7.3 x 4.2 x 5.3 cm), see
README, "the inflated part meshes of the planner have spikes".
