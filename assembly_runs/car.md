# car (6 parts) — `exp_car_flat_run1`

See [README.md](README.md) for the setup and for what Fact / Interpretation / Not examined mean,
and [grasp_filter_finding.md](grasp_filter_finding.md) for how the table below was computed.

**Result: no plan. Stopped in the first sequence attempt after 5 min** (2026-11-10 21:20:49–21:26:45, exit code 1).

## Facts

- Console log: precedence and grasp generation finished; the grasp filter left at least one part without a move grasp; sequence
  optimisation printed `No complete sequence`, then `no sequence left`. No pickup layout and no path was planned.
- Move grasps per part and how many fail each criterion of the filter (a grasp can fail several):

  | Part | Size (cm) | Resting poses | Move grasps | walls | sigma | joint | overlap (old formula) | flat | pass | pass (old overlap) |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 0 | 3.5 x 3.5 x 3.5 | 1 | 8 | 6 | 3 | 0 | 7 (4) | 8 | 0 | 0 |
  | 1 | 17.1 x 5.0 x 6.2 | 3 | 100 | 12 | 32 | 0 | 3 (0) | 42 | 42 | 42 |
  | 2 | 15.8 x 4.5 x 6.0 | 3 | 100 | 5 | 35 | 2 | 5 (0) | 40 | 40 | 41 |
  | 3 | 3.5 x 3.5 x 3.5 | 1 | 49 | 0 | 30 | 8 | 37 (6) | 49 | 0 | 0 |
  | 4 | 3.5 x 3.5 x 3.5 | 1 | 21 | 3 | 16 | 1 | 20 (5) | 21 | 0 | 0 |
  | 5 | 3.5 x 3.5 x 3.5 | 1 | 10 | 0 | 6 | 0 | 10 (3) | 10 | 0 | 0 |

## Interpretation

- Independent of the overlap formula: parts 0, 3, 4 and 5 (3.5 x 3.5 x 3.5 cm, 1 resting pose each; by size presumably the wheels) have no flat-pickable move grasp (all fail the flat criterion). Parts 1 and 2 keep 42 and 40 grasps.
- The assembly cannot be planned with the flat pipeline as it is.

## Not examined

- What the four parts are and how they rest (the size suggests the wheels; not checked in the mesh).
- Why none of their grasps is flat-pickable.

## Suggestions (none implemented)

1. Same as plumbers_block: generate move grasps for the single resting pose of these parts, or pick them from a small stand instead of flat from the table.

---

# Run 2 — `exp_car_flat_run2` (2026-11-11, new grasp generation)

Setup of the second batch: [README.md](README.md), section "Second batch".

**Result: no plan. Parts 0, 3, 4 and 5 have no usable move grasp** (grasp generation 3 min 22 s; the pipeline stopped at the
first sequence optimisation with `no sequence left`).

## Facts

- Grasp generation (`run_grasp_arm_gen.py --flat`), from the console log. "Reachable" means: the arm reaches the grasp at the
  assembled pose without collision. A grasp can fail several criteria.

  | Part | Size (cm) | Candidates (round 1 + round 2) | Reachable move grasps | Usable | Of the reachable ones fail | Kept move / hold |
  |---|---|---|---|---|---|---|
  | 0 | 3.5 x 3.5 x 3.5 | 1110 + 2955 | 7 | 0 | walls 1, singularity / joint limits 3, overlap 0, flat pickup 7 | 0 / 8 |
  | 1 | 17.1 x 5.0 x 6.2 | 1505 | 196 | 71 | | 71 / 100 |
  | 2 (held first) | 15.8 x 4.5 x 6.0 | 950 | 446 | 153 | | 100 / 150 |
  | 3 | 3.5 x 3.5 x 3.5 | 1110 + 2895 | 72 | 0 | walls 0, singularity / joint limits 62, overlap 1, flat pickup 21 | 0 / 54 |
  | 4 | 3.5 x 3.5 x 3.5 | 1110 + 3090 | 26 | 0 | walls 3, singularity / joint limits 23, overlap 5, flat pickup 19 | 0 / 100 |
  | 5 | 3.5 x 3.5 x 3.5 | 1110 + 2950 | 11 | 0 | walls 0, singularity / joint limits 6, overlap 0, flat pickup 11 | 0 / 10 |

- The four parts (by size presumably the wheels) have 1 stable resting pose and 2 with the less stable ones; the counts above
  are for both poses.
- Parts 3 and 4: 51 of 72 and 7 of 26 reachable grasps can pick the part lying flat, but 62 and 23 fail the singularity /
  joint-limit rule for insertion poses (`MIN_SIGMA_INSERTION` 0.15, `MIN_JOINT_LIMIT_MARGIN` 0.5 rad); no grasp passes all criteria.
- Parts 0 and 5: 7 and 11 reachable grasps, none can pick the part lying flat.
- Grasp filter: `NO MOVE GRASP left for part(s) 0, 3, 4, 5`. Sequence planning: `tree fully explored (solution found: False)`.

## Interpretation

- For these four parts the grasps are scarce at the assembled pose already (7 to 72 reachable of about 4000 candidates), as in
  run 1 (8, 49, 21, 10 move grasps). The second sampling round only checks candidates that can pick the part lying flat; it found
  none that is reachable and passes the other criteria.
- For parts 3 and 4 the arm pose during the insertion, not the lying pose, removes most grasps.

## Measurement after the batch (parts 0 and 3; all grasp candidates before any reachability test)

Scratch scripts `diag_no_grasp.py` and `diag_reach.py` (session scratchpad): the candidates of the generator for 200 antipodal
pairs, with pad contact; for those that can pick the part lying flat with 0.3 cm finger depth, what the gripper hits at the
assembled pose.

- Parts 0 and 3 are assembled 4.2 to 7.7 cm above the table, beside parts 1 and 2. One stable resting pose (tips at 56 deg),
  one less stable (11 deg).
- 1080 candidates each; 229 (part 0) and 234 (part 3) can pick the part lying flat with 0.3 cm depth.
- At the assembled pose all of them come from the side (hand 60 to 91 deg from straight down). What the gripper hits there:

  | | part 0 (229) | part 3 (234) |
  |---|---|---|
  | the table, with the hand | 76 | 74 |
  | parts 1 and 2 (assembled before) and the table | 68 | 75 |
  | parts 1 and 2 without the table | 29 | 28 |
  | other combinations | 25 | 22 |
  | no gripper collision (the arm decides) | 31 | 35 |

- In the batch, 7 (part 0) and 72 (part 3) grasps were reachable in the end; none passed all criteria (part 3: 62 fail the
  singularity / joint-limit rule).

Interpretation: as for plumbers_block, the part is oriented differently in the assembly than when it lies, so the grasps
that take it from the table approach it from the side when it is assembled, low above the table and next to the car body. Most
of them put the 20 cm wide hand into the table or the body; the few that remain fail the arm rules of the insertion.

## Not examined

- Which rule fails for parts 3 and 4 (singularity or joint limit, and at which pose of the insertion path).

## Suggestions (none implemented)

1. Look at the insertion poses of parts 3 and 4 first: if the assembly spot or the insertion direction puts the arm near a
   singularity, another place or orientation of the assembly on the table may help more than more grasps.
2. Parts 0 and 5: as in run 1, grasps for the single resting pose of the wheels are missing; a stand that presents the wheel
   upright would make the reachable grasps usable.

# Run 4 (2026-11-11/12, night): after the user's decisions

Experiment `exp_car_flat_run4`, code as described in [README.md](README.md) ("Fourth batch"): among other things the tilt
limit of the hand at the pickup is gone.

## What happened

- Grasp generation 4 min 21 s. For the first time every part has usable move grasps: part 1 (body): 94, part 2 (chassis): 266
  (100 kept), the wheels: part 0: 12, part 3: 36, part 4: 9, part 5: 17, all four after the second sampling round and in the
  less stable resting pose of the wheel (no grasp fits the stable one).
- Sequence planning (`run_seq_plan.py`, unchanged original code): `tree fully explored (solution found: False)`. No sequence.

## Why there is no sequence (measured, scratch script `diag_pairs.py`)

The assembly order is fixed by the parts: chassis (2), body (1) onto it, then the four wheels onto the body. For the step
"insert the body while the chassis is held" the log has **no** pair of a move grasp and a hold grasp:

| | run 1 (original generator) | run 4 |
|---|---|---|
| move grasps of the body | 100 | 94 |
| hold grasps of the chassis | 100 | 150 |
| hold grasps whose gripper is in the way of the body | 99 | 150 |
| pairs for this step | 92 (all with the one free hold grasp) | 0 |

The body (17.1 x 5.0 x 6.2 cm) covers the chassis (15.8 x 4.5 x 6.0 cm) from above and overlaps its upper centimetre on all
sides. Of all 441 feasible hold grasps of the chassis (scratch script `diag_hold_room.py`, before the cut to 100), 3 leave
room for the body: from the side at one end of the chassis, low. Hold grasps are chosen at random (as in the original
pipeline); run 1 happened to keep one of them, run 4 none.

In runs 1 and 2 this was hidden behind the wheels, which had no usable grasp.

# Run 4b (2026-11-12): hold grasps that leave room

Change (my decision, not asked for; switch `FLAT_HOLD_ROOM` in `planning/config.py`, 0 = off): after the random choice of the
hold grasps, for every part that can be inserted while this part is held, hold grasps whose gripper leaves room for that
insertion are added if the choice has fewer than 5 of them (`flat_grasp.holds_with_room`, test `hold_grasps_that_leave_room`).

Experiment `exp_car_flat_run4b`, same setup:

- Grasp generation: the chassis keeps 151 hold grasps, 1 of them added to leave room (hold grasp 875; it is the only one of
  the 151 that leaves room for the body). The step "body while the chassis is held" now has 94 pairs, all with hold grasp 875.
- **Plan verified** in the first sequence attempt, 33 min in total (layout 1236 s, motion planning 241 s). No path rejected, no
  grasp removed. Place poses: all moves checked.
- Layout: wheel 3 at (0.0, 5.0) after 1 spot; wheel 4 at (0.0, 10.0) after 9; wheels 0 and 5 found no spot among the 152
  nearest ones and were placed at (6.25, 11.25) and (-6.25, 11.25) by the search over the whole board (163 and 182 spots
  tried); body at (2.5, -1.25) after 8; chassis (held first) at (-10.0, 1.25).
- Hand at the pickup, from vertical: body 6 deg; wheels 50, 57, 66 and 72 deg; chassis (hold grasp 875) 72 deg.

Facts to keep in mind for the real robot:

- The whole plan depends on one hold grasp of the chassis (875). If the human holds the chassis differently, the body may not
  fit past the hand.
- The wheels lie in a resting pose that tips over at 11 deg (the limit for a "less stable" pose is 10 deg) and are picked by a
  hand that leans 50 to 72 deg.

Not examined: why the stable resting pose of the wheels has no grasp; how the wheels behave in the 11 deg pose when a finger
touches them first.

# Run 5 (2026-11-12): fifth batch

Default place (`exp_car_flat_run5`): **verified**, first sequence attempt (first plan of car in a batch; run 4 had no
sequence, run 4b was a single run). Grasps 7 min 37 s, layout 2 min 48 s, motion planning 5 min 47 s; poses for
`place_parts.py` without collision.

- Move grasps usable / kept: part 0: 169 / 100, part 1: 157 / 100, part 2 (held first): 390 / 100, part 3: 35 / 35,
  part 4: 9 / 9, part 5: 198 / 100. Dropped for the straight moves at the assembly: 9, 12, 147 (part 2, release only),
  0, 0, 2.
- Parts 0, 3, 4, 5 lie in the less stable pose (tips at 11 deg): no grasp fits a stable one.
- Pickups: part 1 hand 0 deg; parts 0 and 5: 42 and 39 deg; parts 3 and 4: 60 and 56 deg; part 2 hold grasp 875, 72 deg
  (the same hold grasp as in run 4b).

Not examined: the position test for car (7 places of the assembly were drawn, 5 cm apart; not run yet when this was written).

# Run 6 (2026-11-12, 22:02): sixth batch, final code of the day

Default place (`exp_car_flat_run6`): **verified**, first sequence attempt. Grasps 9 min 17 s, layout 7 min 16 s (run 5:
2 min 48 s), motion planning 3 min 56 s. No poses for `place_parts.py` (colliding move). Usable move grasps: 148, 197,
353 (part 2, held first), 46, **6** (part 4), 175. Parts 0, 3, 4, 5 lie in the pose that tips at 11 deg. Pickups: part 1
hand 14 deg; parts 0, 3, 4, 5: 54, 60, 54 and 49 deg; part 2 hold grasp 817, 73 deg.

Not run: the position test for car.
