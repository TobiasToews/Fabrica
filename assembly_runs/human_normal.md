# human_normal (7 parts) — reference runs before the batch

See [README.md](README.md) for the setup and for what Fact / Interpretation / Not examined mean.

| Run | Result | Duration | Notes |
|---|---|---|---|
| `exp_human_normal_flat_v5`, 1st try | no plan | 39.6 min | 57 rejected paths, all "carried part N closer than 1.0 cm to lying part M" a few cm after the grasp |
| `exp_human_normal_flat_v5`, 2nd try | plan verified | 3.4 min (grasps reused) | after the rule change below; 0 rejected paths |
| `exp_human_normal_flat_v6` | plan verified | 21.0 min (from scratch, machine shared) | 0 rejected paths, 0 banned grasps |

## v5, 1st try: no plan

Facts:
- `planning.log`: 10 sequence attempts, 9 move grasps banned, then `no sequence left`.
- The 57 rejected paths were 32x part 2 vs lying part 6, 16x part 3 vs lying part 6, 9x part 1 vs lying part 3; the pose ranges
  were nearly the same in every attempt (e.g. 60..79).
- The strict planner lifts a grasped part straight up. The rule at that time allowed the carried part to be closer than 1 cm to
  lying parts only within 5 cm of its pickup pose (`CONTACT_RADIUS`). Part 6 lying is 6.0 cm high.

Interpretation:
- The straight lift beside a taller neighbour was still within 1 cm of it beyond the 5 cm radius. Supported by the result of the
  change: the rule now allows closeness until the part is clear for the first time (and forbids contact always), and the same
  inputs then planned without a rejected path (2nd try).

## v6: plan verified

Facts:
- `stats.json`: precedence 91 s, grasp generation 475 s, sequence planning 0.07 s, sequence optimisation 1.1 s, pickup layout
  540 s, motion planning 100 s. Whole pipeline 21.0 min.
- `planning.log`: 0 rejected paths. `workcell.json`: no banned move grasps, verified.
- Plan: 68 steps, 4777 move-arm poses (159 s at 30 fps). Move order 1, 2, 3, 5, 4, 6.
- v5 and v6 chose different move grasps for parts 3 and 1 (v5: 19, 838; v6: 2, 562) although both ran the same command.

Not examined:
- Why the grasp generation differs between two runs of the same command.
- The plan was not run on the robot. It uses the default robot height of 3.0 cm, which the measurements of 2026-11-10 put in
  doubt (fingertips 9–10 mm lower than planned).

---

# Run 2 — `exp_human_normal_flat_run2` (2026-11-11, new grasp generation)

Setup of the second batch: [README.md](README.md), section "Second batch".

**Result: no plan (planning 18 min, 4 sequence attempts). Part 4 had 3 usable move grasps; with each of them the transport to
the start of its insertion failed. This is worse than v6, which has a verified plan.**

## Facts

- Grasp generation (`run_grasp_arm_gen.py --flat`, 4 min 23 s), from the console log:

  | Part | Size (cm) | Candidates | Reachable move grasps | Usable (pick the part lying flat, all criteria) | Resting pose | Kept move / hold |
  |---|---|---|---|---|---|---|
  | 0 (held first) | 11.0 x 8.0 x 4.0 | 195 | 84 | 33 | stable | 33 / 72 |
  | 1 | 2.4 x 2.5 x 8.0 | 1520 | 316 | 15 | stable | 15 / 100 |
  | 2 | 2.4 x 2.5 x 8.0 | 1585 | 369 | 21 | stable | 21 / 100 |
  | 3 | 8.0 x 4.5 x 5.0 | 1195 | 292 | 162 | stable | 100 / 100 |
  | 4 | 4.3 x 3.8 x 3.0 | 1855 | 21 | 3 | stable | 3 / 100 |
  | 5 | 4.3 x 3.8 x 3.0 | 1805 | 66 | 8 | stable | 8 / 13 |
  | 6 | 6.0 x 6.0 x 6.5 | 1825 | 319 | 17 | stable | 17 / 100 |

  No part got the second sampling round: it starts below `FLAT_MIN_GRASPS` = 3 usable grasps, and part 4 has exactly 3.
  For comparison, v6 (old generator and filter, overlap limit 0.5 cm on the old measure) kept 19, 5, 6, 50, 2, 7, 5 move grasps.
- Sequence optimisation: 5 of 16 first hold grasps can pick part 0 lying flat. Sequence (move, hold): (1, 0), (2, 0), (3, 1),
  (5, 3), (4, 0), (6, 0), the same order as in v6.
- Attempts 1 to 3 ended with
  `[run_motion_plan] FAILED for part 4 (move grasp 47 / 137 / 262): transport path: no path found: no straight move into the goal pose from a pose clear of everything`
  (8 planning attempts each, 24 rejections). After the third ban part 4 had no move grasp: `no sequence left`. Layout and the
  transports and insertions of parts 1, 2, 3 and 5 passed in every attempt.
- Scratch diagnosis (`diag_goal_retract.py` in the session scratchpad: from the insertion start backwards along the disassembly
  direction, with the planner's own tests). The insertion of part 4 starts 2.5 cm from its final pose and runs along +x, so the
  straight move comes from -x:

  | Grasp of part 4 | Wall margin at the insertion start | Result of the straight move |
  |---|---|---|
  | run 2, 47 | 1.2 cm | blocked at 0.25 cm: virtual wall |
  | run 2, 137 | 1.5 cm | blocked at 0.50 cm: virtual wall |
  | run 2, 262 | 1.7 cm | blocked at 0.75 cm: virtual wall |
  | v6, 84 (the grasp of the verified plan) | 1.9 cm | clear of everything after 0.75 cm, wall margin there 1.1 cm |

  At the insertion start itself the carried part is too close to assembled part 3 to count as clear, so the arm has to back
  off; the wall in the way is the plane y = -0.65 m of the robot frame (world x = -20 cm), which the flange must keep 10 cm
  from (`get_panda_virtual_walls`), plus `VIRTUAL_WALL_SAFETY` 1.0 cm. The flange stands at world x = -8.1 to -8.8 cm.
- The v6 grasp 84, measured on the pad grid: depth per finger 1.55 cm and 0.25 cm, contact area 2.9 cm^2. Its overlap by the old
  measure (deepest sampled contact point of either finger) was 1.46 cm. With the rule of the new generation (weaker finger at
  least `GRASP_MIN_FINGER_OVERLAP` 0.3 cm) it fails by 0.5 mm. Grasp ids are not comparable between v6 and run 2 (other
  surface samples).

## Interpretation

- Part 4 is inserted from the side away from the robot, right in front of a virtual wall: the approach has between 0.2 and
  0.9 cm of room, depending on how the grasp places the flange. v6 passed with 0.1 cm to spare. Whether a plan exists therefore
  depends on having a grasp of part 4 with the flange a few millimetres further from the wall.
- Two things of the new generation worked against this assembly: the weaker-finger rule removed the grasp v6 used, and with
  exactly 3 usable grasps the second sampling round did not start, so there were no further candidates.

## Scratch experiment after the batch (not a pipeline run in `logs/`)

- Same pipeline in the session scratchpad with the second sampling round starting below 10 usable grasps instead of 3
  (`FLAT_MIN_GRASPS` patched in the process, repository unchanged): part 4 got 1855 + 5125 candidates, 52 reachable, 6 usable
  (47, 137, 262, 2052, 2387, 4219); part 5 got 33 instead of 8. Of the six grasps of part 4, the diagnosis finds a clear pose
  only for 2387 (wall margin 1.8 cm at the insertion start, clear after 0.75 cm with exactly 1.0 cm to the wall).
- Result of that run: **plan verified in the fourth sequence attempt** (grasp generation 5 min 33 s, planning 24 min). Attempts
  1 to 3 failed like run 2 with grasps 47, 2052 and 137 of part 4; attempt 4 used grasp 2387 (pickup tilt 28.5 deg) and passed
  motion planning and `verify_plan.py`. Part 5 is picked with grasp 1989 (23.1 deg), the other parts with the grasps of run 2.

## Run 3 — `exp_human_normal_flat_run3` (same day, after `FLAT_MIN_GRASPS` 3 -> 10 and the two check fixes)

- **Plan verified in the fourth sequence attempt** (grasp generation 5 min 9 s, planning 23 min), the same course as the scratch
  experiment: grasps 47, 2052 and 137 of part 4 were removed after the failure described above (24 rejected paths),
  grasp 2387 passed. `workcell.json`: `verified: true`, `banned_move_grasps: [["4", 47], ["4", 2052], ["4", 137]]`.
- Pickup grasps: part 0 hold 90; parts 1, 2, 3, 6 with move grasps 742, 786, 22, 786 as in run 2; part 5 with 1989 (tilt
  23.1 deg), part 4 with 2387 (28.5 deg).
- This plan uses other grasps and another layout than v3 / v4 / v6: it needs its own templates and its own policy training.

## Search for the place of the assembly — `exp_human_normal_ws` (same day, evening)

Asked for after the batch: search pickup and assembly positions over the whole workspace of the two arms instead of the fixed
place. Tool: `planning/run_workspace_search.py` (description in `FLAT_PIPELINE.md`, "Place of the assembly").

Facts:

- Rating of 61 places on a 5 cm grid (10 min, `logs/exp_human_normal_ws/workspace_search_human_normal.json`): for each part 24
  grasps that can pick it lying flat with a free gripper at the assembly are tested with the arm at the place; the rating is
  the share that passes for the part with the fewest (walls, singularity, joint limits at the insertion poses, 3 cm of room
  behind the insertion start).
  - Default place (0, -15): rating 0.00. Part 4 has no passing grasp; the other parts 0.17 to 0.71.
  - Every place with x <= 0 and every place with y >= -15 rates 0.00. Places with a rating: x +5 to +20, y -20 to -35; the
    best are (+15, -30), (+10, -25), (+15, -25), (+5, -25), (+10, -30), (+15, -35), (+5, -20), all 0.17 (4 of 24 grasps for the
    weakest part).
- Pipeline at the best place, assembly centre (+15, -30), log `logs/exp_human_normal_ws_x+15.0_y-30.0/human_normal`:
  **plan verified in the first sequence attempt, no grasp removed, no path rejected** (11.5 min: grasps 5 min 19 s, layout 227 s,
  motion planning 82 s).
- Usable move grasps per part at that place against the default place (run 3): part 0: 45 (33), 1: 42 (15), 2: 49 (21),
  3: 191 (162), 4: 18 (6, after the second sampling round), 5: 24 (33, after the second round), 6: 41 (17). No part needed the
  second sampling round. Part 5 has no hold grasp there (0 kept).
- Pickup grasps: part 4 with grasp 1431, tilt 5.3 deg (default place: 2387, 28.5 deg); part 5 with 62, 13.9 deg.
- Check of the stored place: `verify_plan.py` called in a fresh shell on the log passes; on a copy of the log without
  `assembly_center.json` it reports violations (the tools then assume the default place). The plan at the default place
  (run 3) still passes.

Interpretation:

- The wall problem of part 4 is a property of the default place, not of the assembly: 15 cm towards the move arm and 15 cm
  further from the robot bases, every part has several times as many usable grasps and the first sequence works.

Not examined:

- Whether the real table reaches that far: at (+15, -30) the assembly stands at x 8.7 to 21.3, y -34 to -26 on the board, and
  its zone (3 cm around it) reaches y = -37, i.e. 57 cm in front of the line of the robot bases (y = +20), and x = +24.3.
- Whether a human can hold the first part comfortably there (the hold arm is a human in single-arm mode).
- The other well rated places were not planned; (+5, -20) is the one nearest to the default place.

## Not examined

- Whether the plan with grasp 2387 would be robust on the real robot: the flange passes 1.0 cm inside the limit that stops the robot.
- Why only 21 grasps of part 4 are reachable at the assembled pose (1855 candidates), against 32 in v6.

## Suggestions (2 is implemented, see Run 3; the others are not)

1. Give the insertion of part 4 room: move the assembly spot a few centimetres towards the robot (`get_assembly_center`), or turn
   the assembly so that no part is inserted from the far side. This removes the cause; it changes templates and existing plans.
2. Start the second sampling round earlier (`FLAT_MIN_GRASPS` 10 instead of 3). Cheap, more alternatives for parts with few
   grasps; here it yields one grasp that fits, with no margin to spare.
3. Decide on the weaker-finger rule: a grasp like v6's 84 holds the part with 2.5 mm of one finger and 15.5 mm of the other. If
   that is acceptable, the limit could apply to the deeper finger with a lower limit for the weaker one.
4. Let the grasp filter test whether the insertion start can be approached with respect to the walls (a few steps back along the
   insertion direction), so that such grasps are sorted out, or preferred, before sequence planning.

# Run 4 (2026-11-11/12, night): after the user's decisions

Experiment `exp_human_normal_flat_run4`, assembly at the default place (0, -15), code as described in [README.md](README.md)
("Fourth batch").

## What happened

- Grasp generation 5 min 27 s. Usable move grasps for parts 0 to 6: 53, 94, 125, 190, 26, 18, 112 (at most 100 kept). Part 4
  got the second sampling round and has 26 (run 3: 6).
- Attempt 1: the transport of part 4 with move grasp 47 was rejected 8 of 8 times:
  `no straight move into the goal pose from a pose clear of everything (virtual wall after 0.25 cm)`. Cause recorded: the move
  grasp (the arm, at the start of the insertion). Grasp 47 removed.
- Attempt 2: **plan verified** (`verify_plan.py` PASSED), part 4 with move grasp 2139. Planning 12 min in total.
- Then `run_place_poses.py` (the poses for `real_robot/place_parts.py`) reported `part 4: to above COLLISION`. The pipeline
  ended with exit code 1 and wrote no `workcell.json`.

Layout of attempt 2 and the hand at each pickup (from vertical):

| Part | Spot | Grasp | Hand leans |
|---|---|---|---|
| 6 | (0.0, 5.0) | move 298 | 61 deg |
| 4 | (0.0, -2.5) | move 2139 | 29 deg |
| 5 | (-6.25, 3.75) | move 581 | 44 deg |
| 3 | (0.0, 12.5) | move 22 | 81 deg |
| 2 | (6.25, 12.5) | move 283 | 60 deg |
| 1 | (-7.5, 12.5) | move 247 | 66 deg |
| 0 (held first) | (8.75, 1.25) | hold 130 | 56 deg |

## The collision of the place poses (measured, scratch)

`place_parts.py` moves the arm in joint space from above one spot to above the next, and `run_place_poses.py` checks these
moves at 50 interpolated poses. From above part 6 to above part 4 one joint changes by 3.87 rad; 18 of the 50 poses are
rejected (12: gripper in the table, 6: gripper in part 6, which already lies there). The other moves pass (largest joint
change 0.87 to 2.97 rad).

Interpretation: the plan itself is checked and passes. What fails is the helper that lets the robot show the spots: with
hands that lean 29 to 81 deg, the arm configurations above two neighbouring spots can be far apart in joint space, and a
straight joint-space move between them swings through the table.

## Not examined

- Whether the same grasps would be chosen with a preference for upright hands (the sequence optimisation does not look at the
  pickup at all).
- Run 3 (verified, hands within 30 deg) and the plan at (+15, -30) from the workspace search remain valid plans; they were not
  re-planned.

## Suggestions (none implemented)

1. Decide what a colliding place pose should mean. Now: the pipeline reports an error and writes no `workcell.json`, although
   the plan is verified; `place_parts.py` itself only prints a warning for such a file and would execute the move.
2. Let `place_parts.py` go through the rest pose between two spots when the direct move collides (both sides, planning and
   robot script, must change together).
3. Prefer upright hands at the pickup where a part has such grasps (e.g. in the sequence optimisation, after its own scores).

# Run 5 (2026-11-12): position test and fifth batch

## Position test (`exp_human_normal_pos_*`, details and tables in README)

First run (02:43 to 05:24, 8 places of the assembly): stopped after 4 verified pairs and 3 without a plan. It showed three
problems of the pipeline, each measured and then changed:

1. At the assembly place (0, -7.5) part 6, inserted last, got one move grasp after the other that could not move straight
   back after the release (joint limit after 0.75 to 3 cm). Each cost a whole planning attempt; after 10 attempts there
   was no plan, at all three pickup areas. Of the 100 move grasps of part 6 there, 21 cannot leave, and the nine chosen in
   a row were among them. Now such grasps are dropped when the grasps are generated.
2. With the pickup area 30 cm and more up the board a plan took 40 to 47 min instead of 6.5: the layout searched arm
   poses at spots where the wrist or the flange is behind a virtual wall. Now tested before the search (27 min -> 4 min for
   one layout).
3. At two places part 4 has no working grasp at all (virtual wall). Such a place is now reported right after the grasp
   stage, with the part and the reason.

Second run on the changed code (05:45 to 07:56): **30 of 30 pairs verified**, every plan in the first sequence attempt, 2.6
to 9.4 min per plan. 10 places of the assembly (x from -2.5 to +12.5, y from -32.5 to -7.5) with 3 pickup areas each (y from
-35 to +37.5); an eleventh place, (-2.5, -2.5), is not buildable (no move grasp for part 4: 289 of 295 feasible ones behind
a virtual wall) and was replaced. The place (0, -7.5) of the first run: three plans in 3 to 4 min.

Part 4 is the part with the fewest move grasps: 5 at (-2.5, -27.5), 13 at (+5, -27.5), 26 to 69 at the other places.
`place_parts.py` poses without collision for 10 of the 30 plans.

## Fifth batch, default place (`exp_human_normal_flat_run5`)

**Verified** in the second sequence attempt. Attempt 1: the approach to part 1 with move grasp 247 passes closer than 1 cm
to the assembled part 0 (in all 8 planning attempts); grasp 247 dropped. Attempt 2: plan found
(the transport of part 4 needed a second planning attempt: 1 pose 1 cm too close to part 3). Grasps 7 min 52 s, layout
2 min 44 s, motion planning 3 min 21 s. No poses for `place_parts.py`.

- Move grasps usable / kept: part 0 (held first): 72 / 72, part 1: 158 / 100, part 2: 226 / 100, part 3: 251 / 100,
  part 4: 34 / 34, part 5: 42 / 42, part 6: 169 / 100. Dropped for the straight moves at the assembly: 22, 2, 1, 21,
  16, 0, 7.
- Pickups: part 4: hand 9 deg from vertical, part 5: 20 deg, part 6: 5 deg; part 1: 53 deg, part 2: 60 deg, part 3:
  81 deg; part 0 hold grasp 130, 56 deg.

Still true: three of the six pickups of the move arm are made with the hand leaning 53 to 81 deg (suggestion 3 above).

# Run 6 (2026-11-12, evening): final code of the day

## Position test, third run (19:47 to 22:02; table in README)

**30 of 30 pairs verified** on the code with the new resting poses and the planner that releases its memory, at the same
places as the second run. 22 plans from the first sequence attempt, 5 from the second, one each from the third, fourth and
fifth (second run: all 30 from the first). 14 move grasps were dropped on the way: part 1 five times because the layout
found no spot, parts 4, 1, 2 seven times because the arm touches a part on the straight move to the insertion start, part
6 twice for a singularity on its insertion path. 3.0 to 33.3 min per plan. The place (-2.5, -2.5) is again not buildable
(no move grasp for part 4).

Not examined: why more attempts were needed than in the second run. Two things differ: other grasps are generated (the
resting poses changed, with them the candidates made for the lying part and the random numbers after them), and the
planner's inflated meshes of parts 1 and 2 have spikes (2.4 x 2.5 x 8.0 cm becomes 4.8 x 4.9 x 15.1 cm; README), which
was the case in the second run as well.

## Sixth batch, default place (`exp_human_normal_flat_run6`)

**Verified** in the first sequence attempt (run 5: second). Grasps 7 min 34 s, layout 38 s, motion planning 1 min 46 s;
`place_parts.py` poses without collision (run 5: none). Usable move grasps: 80 (part 0), 142, 189, 241, **13** (part 4),
40, 194. Pickups: parts 2, 3, 5 with hands 14 to 20 deg, parts 4 and 6: 36 and 39 deg, part 1: 58 deg; part 0 hold grasp
135, 72 deg. Fewer hands lean far than in run 5 (53, 60 and 81 deg there).
