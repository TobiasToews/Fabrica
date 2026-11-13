# cooling_manifold (7 parts) — `exp_cooling_manifold_flat_run1`

See [README.md](README.md) for the setup and for what Fact / Interpretation / Not examined mean,
and [grasp_filter_finding.md](grasp_filter_finding.md) for how the table below was computed.

**Result: no plan. Stopped in the first sequence attempt after 5 min** (2026-11-10 21:36:15–21:41:52, exit code 1).

## Facts

- Console log: precedence and grasp generation finished; the grasp filter left at least one part without a move grasp; sequence
  optimisation printed `No complete sequence`, then `no sequence left`. No pickup layout and no path was planned.
- Move grasps per part and how many fail each criterion of the filter (a grasp can fail several):

  | Part | Size (cm) | Resting poses | Move grasps | walls | sigma | joint | overlap (old formula) | flat | pass | pass (old overlap) |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 0 | 1.8 x 1.7 x 4.6 | 3 | 27 | 2 | 9 | 2 | 15 (10) | 27 | 0 | 0 |
  | 1 | 12.0 x 12.0 x 3.5 | 2 | 22 | 1 | 4 | 0 | 16 (1) | 10 | 0 | 8 |
  | 2 | 2.0 x 2.0 x 3.5 | 1 | 54 | 12 | 31 | 3 | 22 (4) | 54 | 0 | 0 |
  | 3 | 2.0 x 2.0 x 3.5 | 1 | 100 | 0 | 43 | 4 | 39 (10) | 100 | 0 | 0 |
  | 4 | 1.8 x 1.7 x 4.6 | 3 | 29 | 2 | 9 | 0 | 18 (14) | 28 | 0 | 0 |
  | 5 | 2.0 x 2.0 x 3.5 | 1 | 100 | 0 | 5 | 0 | 40 (6) | 100 | 0 | 0 |
  | 6 | 2.0 x 2.0 x 3.5 | 1 | 100 | 3 | 0 | 2 | 45 (11) | 100 | 0 | 0 |

## Interpretation

- Independent of the overlap formula: five of seven parts (0, 2, 3, 5, 6) have no flat-pickable move grasp at all, and part 4 has one of 29. No part keeps a grasp with the new formula; with the old one only part 1 keeps 8.
- The assembly cannot be planned with the flat pipeline as it is.

## Not examined

- Why the small parts (2.0 x 2.0 x 3.5 cm, 1 resting pose) and parts 0 and 4 (1.8 x 1.7 x 4.6 cm) have no flat-pickable grasp.

## Suggestions (none implemented)

1. Same as plumbers_block and car: move grasps for the lying pose of the parts are needed before anything else.

---

# Run 2 — `exp_cooling_manifold_flat_run2` (2026-11-11, new grasp generation)

Setup of the second batch: [README.md](README.md), section "Second batch".

**Result: no plan after 10 sequence attempts (planning 81 min). All parts have usable grasps now; every attempt failed at a
transport path that the exact check rejected next to the pickup spot. Two of the three reasons found are defects of the checks,
the third is a real violation caused by the tight layout (see Interpretation).**

## Facts

- Grasp generation (`run_grasp_arm_gen.py --flat`, 6 min 20 s), from the console log:

  | Part | Size (cm) | Candidates (round 1 + round 2) | Reachable move grasps | Usable | Resting pose the grasps fit | Kept move / hold |
  |---|---|---|---|---|---|---|
  | 0 | 1.8 x 1.7 x 4.7 | 970 + 2580 | 109 | 26 | less stable only | 26 / 100 |
  | 1 (held first) | 12.0 x 12.0 x 3.5 | 120 | 39 | 10 | stable | 10 / 34 |
  | 2 | 2.0 x 2.0 x 3.5 | 1195 + 3225 | 141 | 66 | less stable only | 66 / 100 |
  | 3 | 2.0 x 2.0 x 3.5 | 1190 + 3050 | 275 | 92 | less stable only | 92 / 100 |
  | 4 | 1.8 x 1.7 x 4.7 | 965 | 47 | 4 | stable | 4 / 37 |
  | 5 | 2.0 x 2.0 x 3.5 | 1195 + 3215 | 490 | 362 | less stable only | 100 / 100 |
  | 6 | 2.0 x 2.0 x 3.5 | 1195 + 3130 | 393 | 307 | less stable only | 100 / 100 |

  Parts 0, 2, 3, 5 and 6 have no usable grasp for a stable resting pose (tipping above 20 deg). They got the second sampling
  round and then the less stable resting poses (tipping between 10 and 20 deg; `FLAT_FALLBACK_POSES`); `grasps.pkl` records them
  as `fallback_parts`. In the layouts these parts lie in poses that tip at 14 to 17 deg.
- Sequence optimisation: 11 of 17 first hold grasps can pick part 1 lying flat.
- The 10 attempts and the grasp that was blamed and removed (from the console log; each after 8 rejected planning attempts):

  | Attempt | Part, grasp | Reason given by the check of the transport path |
  |---|---|---|
  | 1 | 2, 3597 | gripper left finger closer than 1.0 cm to lying part 5 (poses 0..8); carried part 2 closer than 1.0 cm to lying part 5 (poses 34..41, 6x) |
  | 2, 3, 4 | 2, 1007 / 3592 / 3992 | carried part 2 closer than 1.0 cm to lying part 5 (poses 34..41, 6x) |
  | 5 | 4, 783 | carried part 4 closer than 1.0 cm to lying part 0 (poses 9..36, 28x) |
  | 6 | 6, 327 | carried part 6 closer than 1.0 cm to lying part 5 (poses 34..41, 6x) |
  | 7 | 4, 786 | carried part 4 closer than 1.0 cm to lying part 3 (poses 6..38, 33x) |
  | 8 | 2, 3857 | gripper right finger closer than 1.0 cm to lying part 3 (poses 0..5) |
  | 9 | 3, 1577 | gripper left finger closer than 1.0 cm to lying part 6 (poses 0..5); carried part 3 closer than 1.0 cm to lying part 6 (poses 34..41, 6x) |
  | 10 | 4, 773 | carried part 4 closer than 1.0 cm to lying part 6 (poses 5..40, 36x) |

- Attempt 1 reproduced in a scratch copy (same grasps, filter, sequence optimisation and layout: the layout lines are identical
  to the batch). Part 5 lies at (0.0, 5.0), part 2 at (2.5, 5.0).
  - At the pickup pose of part 2 the gripper is 1.06 cm from part 5 with the fingers opened (what the layout checked) and
    0.81 cm with the fingers closed on the part (the first pose of the transport).
  - Re-planned transport of part 2 (`diag_transport.py` in the session scratchpad; same two violations as in the batch): the
    part is lifted straight up. Its distance to part 5 is 0.56 cm on the table, grows from a lift height of 1.9 cm, and is
    1.00 cm from 2.7 cm to 3.5 cm of lift height; in that stretch the check flags poses 34, 35, 37, 38, 40 and 41 as closer than
    1.0 cm and accepts poses 33, 36, 39 and 42, all of which print as 1.00 cm. Above 3.5 cm the distance grows (1.12 cm at 4.0 cm).
- Attempt 10, from the files of the batch (last layout): re-planned transport of part 4 with grasp 773 (same violation, poses
  5..40). The part is lifted straight up. Its distance to lying part 6 is 1.17 cm on the table, falls to 0.81 cm between 0.8 and
  2.8 cm of lift height and is back at 1.0 cm at 3.4 cm. The gripper keeps 1.00 cm or more to part 6 on all these poses.

## Interpretation

- Gripper violations at poses 0..N (attempts 1, 8, 9): the layout tests the pickup pose with the opened gripper only. A finger
  that closes moves towards the part and can thereby come closer to a neighbour that lies diagonally beside it. Measured for
  attempt 1 (1.06 cm open, 0.81 cm closed). This is a gap in the layout check: it accepts spots whose first transport pose
  breaks the 1 cm rule.
- "Carried part closer than 1.0 cm ... poses 34..41, 6x" (attempts 1, 2, 3, 4, 6, 9): the lifted part stands exactly 1.00 cm
  beside its neighbour (the layout grid is 1.25 cm, these parts are round with whole-millimetre sizes). The rule lets a carried
  part be close until it is 1.0 cm clear for the first time; rounding of the distance (0.9999 / 1.0001) makes it "clear" on one
  pose and "closer" on the next. Measured for attempt 1. This is a defect of the check, not a real approach.
- Attempt 10 (part 4, 36 poses; attempts 5 and 7 show the same pattern in the log but were not re-planned): a real violation
  of the 1 cm rule. Lifted straight up, part 4 comes closer to its neighbour than it was on the table, because the two shapes are
  not straight-sided. The layout only requires 0.5 cm between the footprints of two lying parts (`PART_GAP`) and does not test
  the lift of the carried part, and the planner has no choice but to lift straight up. So the spot decides, not the grasp, yet
  the grasp is blamed and removed.
- The two defects are in `planning/plan_checks.py` / `planning/run_pickup_gen.py` as they were before the grasp work; they did
  not show on human_normal.

## Fixes made after the batch

- `plan_checks.py`: distances are compared with a tolerance of 0.001 cm (`DISTANCE_TOL`), and `check_pickup` also tests the
  gripper closed on the part; the layout and `plan_place_part.py` use it. Two new tests in `planning/test_plan_rules.py`
  (`carried_part_at_the_limit_up_to_rounding`, `pickup_with_the_gripper_closed`) fail without the fixes and pass with them.
- Scratch run with the two fixes, before they went into the repository (session scratchpad, a copy of the planning code with
  the fixes, the grasps of run 2): **plan verified in the first sequence attempt** (planning 17 min, no rejected path, no grasp
  removed). Same grasps as attempt 1 of run 2; the layout put part 2 at (3.8, 5.0) after 41 spots instead of (2.5, 5.0) after
  3, and part 4 at (-5.0, 3.8) after 57 spots instead of (-3.8, 8.7) after 74.
- The real cause (part lifted next to a neighbour 0.5 cm away) is not fixed; it did not occur in this layout.

## Run 3 — `exp_cooling_manifold_flat_run3` (same day, with the fixes in the repository and `FLAT_MIN_GRASPS` 3 -> 10)

- **Plan verified, first attempt** (grasp generation 6 min 37 s, planning 17 min, no path rejected, no grasp removed).
- Part 4 got the second sampling round (4 usable grasps is below 10): 965 + 2625 candidates, 135 reachable, 7 usable. The
  other parts as in run 2.
- Pickup grasps as in the scratch run: part 1 hold 60; move grasps 1577, 3597, 783, 1598, 327, 1608 for parts 3, 2, 4, 0, 6, 5.
  Parts 0, 2, 3, 5 and 6 lie in poses that tip at 14 to 17 deg (marked on the template).

## Not examined

- Attempts 5 and 7 (part 4 against lying parts 0 and 3) were not re-planned; they are assumed to be like attempt 10.
- How reliably parts lying in poses that tip at 14 to 17 deg stay put on the real table, and whether the gripper pushes them
  over when it closes. Five of seven parts depend on such poses.

## Suggestions

1. Make the gap between lying parts at least the clearance a carried part must keep: `PART_GAP` 1.2 cm instead of 0.5 cm
   (`ROBOT_PART_CLEARANCE` 1.0 cm plus a reserve). The footprints are convex hulls, so a part lifted straight up then cannot come
   within 1 cm of a neighbour, and the rule "close until clear for the first time" is no longer needed for such layouts. The
   layouts and templates of all assemblies become larger. Not implemented.
2. Decide whether the less stable resting poses are acceptable (`FLAT_FALLBACK_POSES` in `planning/config.py`). Without them
   this assembly has no grasps for five parts, as in run 1.
3. When a transport fails right at the pickup spot, try another spot for that part instead of removing the grasp.

# Run 4 (2026-11-11/12, night): after the user's decisions

Experiment `exp_cooling_manifold_flat_run4`, code as described in [README.md](README.md) ("Fourth batch"): among other things
1.2 cm between lying parts (suggestion 1 above) and the straight moves tested at every spot of the layout.

- Grasp generation 9 min 53 s. Usable move grasps: part 1: 22; part 4: 14 and part 0: **2** (both after the second sampling
  round); parts 2, 3, 5, 6: 56, 205, 588, 434, all in a less stable resting pose (no grasp fits a stable one).
- **Plan verified** in the first sequence attempt, planning 24 min (layout 618 s, motion planning 419 s). No path rejected, no
  grasp removed, no spot banned.
- Layout: part 5 at (0.0, 5.0) after 1 spot; part 3 at (-3.75, 5.0) after 19; part 4 at (0.0, 10.0) after 43; part 0 at
  (5.0, 8.75) after 101; part 6 at (0.0, 1.25) after 5; part 2 at (3.75, 2.5) after 8; part 1 (held first) at (-11.25, 10.0).
- Hand at the pickup, from vertical: parts 2, 3, 5, 6: 2 to 6 deg (resting poses that tip over at 17 deg); part 4: 33 deg;
  part 0: 70 deg (move grasp 3366); part 1 (hold grasp 0): 67 deg.
- Place poses for `place_parts.py`: all moves checked.

Not examined: why part 0 has only 2 usable move grasps of 180 feasible ones; a plan that depends on two grasps has no reserve
if one of them fails on the robot.

# Run 5 (2026-11-12): fifth batch

Default place (`exp_cooling_manifold_flat_run5`): **verified**, first sequence attempt. Grasps 11 min 48 s, layout 2 min,
motion planning 9 min 43 s. No poses for `place_parts.py` (its joint-space moves collide).

- Move grasps usable / kept: part 0: **1 / 1**, part 1 (held first): 24 / 24, part 2: 242 / 100, part 3: 319 / 100,
  part 4: 20 / 20, part 5: 712 / 100, part 6: 474 / 100. Dropped for the straight moves at the assembly: 16 (part 1),
  36 (part 5), 73 (part 6), 0 for the others.
- Part 0 has one usable move grasp now (run 4: two), grasp 801, and the plan uses it with the hand leaning 71 deg. Of 225
  feasible move grasps 224 fail one of the other criteria; which one was not counted for this part (the count is only
  printed when no grasp is left).
- Parts 2, 3, 5, 6 lie in the less stable pose (tips at 17 deg).
- Pickups: parts 2 to 6 with hands 1 to 19 deg; part 0: 71 deg; part 1 hold grasp 0, 67 deg.

Not examined: why part 0 has so few usable grasps; the position test (12 places of the assembly were drawn; with one grasp
for part 0 at the default place, places where it has none are likely).

# Run 6 (2026-11-12, 22:02): sixth batch, final code of the day

Default place (`exp_cooling_manifold_flat_run6`): **verified** in the fourth sequence attempt. Grasps 14 min 17 s, layout
51 s, motion planning 9 min 2 s; `place_parts.py` poses without collision (run 5: none).

- Usable move grasps: 46 (part 0; run 5: 1), 13 (part 1, held first), 188, 221, 12 (part 4), 653, 488. For part 0 no
  grasp fits a stable resting pose in this run, so the less stable ones are admitted (it tips at 14 deg) and the plan picks
  it with the hand 4 deg from vertical (run 5: 71 deg).
- Pickups of the move arm: hands 4 to 11 deg, part 4: 29 deg; part 1 hold grasp 15, 67 deg.
- Attempts 1 to 3: part 6 (the first part the move arm picks) lay at (5, 5), then (5, 3.75), then (5, 6.25); each time
  motion planning found no straight move into its grasp (`never clear of: arm and part after 20.00 cm`) and the spot was
  dropped; at the fourth spot the plan was found. The grasp (3129) stayed, as intended for this cause.

Why the layout took spots the planner then refused, repeated in a scratch copy: the two use different inflated meshes of
part 4, which lies 9.5 cm from part 6. The planner inflates the part where it lies, and for this part in this pose the
inflated mesh is 84.5 x 7.1 x 427.1 cm (the part: 1.8 x 1.7 x 4.6 cm), so the planner calls the arm too close to part 4
all along the 20 cm of the move back. The layout inflates the part as stored and turns the result; there it is
9.4 x 19.6 x 7.1 cm, also far too large, but not in the way of this move. Parts 0 and 4 (the same shape, 20476 faces)
and part 1 are affected. Measured, not changed: see README, "the inflated part meshes of the planner have spikes".
Interpretation, not measured: the same spikes can cost this assembly grasps (the test of the straight moves at the
assembly uses the planner's meshes) and spots in every run.
