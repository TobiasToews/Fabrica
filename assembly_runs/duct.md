# duct (8 parts) — `exp_duct_flat_run1`

See [README.md](README.md) for the setup and for what Fact / Interpretation / Not examined mean,
and [grasp_filter_finding.md](grasp_filter_finding.md) for how the table below was computed.

**Result: no plan. Stopped in the first sequence attempt after 124 min** (2026-11-10 21:41:52–23:46:15, exit code 1).

## Facts

- Console log: precedence and grasp generation finished; the grasp filter left at least one part without a move grasp; sequence
  optimisation printed `No complete sequence`, then `no sequence left`. No pickup layout and no path was planned.
- `stats.json`: precedence planning 6602 s (110 min), grasp generation 843 s; sequence planning: "tree fully explored (solution found: False)". Precedence planning took 89 % of the 124 min of this run.
- Move grasps per part and how many fail each criterion of the filter (a grasp can fail several):

  | Part | Size (cm) | Resting poses | Move grasps | walls | sigma | joint | overlap (old formula) | flat | pass | pass (old overlap) |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 0 | 10.9 x 12.0 x 14.4 | 5 | 14 | 0 | 10 | 2 | 8 (0) | 4 | 2 | 2 |
  | 1 | 10.6 x 12.0 x 7.1 | 1 | 65 | 3 | 27 | 0 | 44 (0) | 40 | 6 | 12 |
  | 2 | 3.2 x 2.3 x 4.0 | 1 | 100 | 3 | 49 | 0 | 29 (3) | 100 | 0 | 0 |
  | 3 | 3.2 x 2.3 x 4.0 | 1 | 90 | 0 | 49 | 0 | 29 (4) | 90 | 0 | 0 |
  | 4 | 3.2 x 2.3 x 4.0 | 1 | 41 | 0 | 29 | 9 | 14 (1) | 41 | 0 | 0 |
  | 5 | 3.2 x 2.3 x 4.0 | 1 | 82 | 0 | 46 | 10 | 28 (5) | 82 | 0 | 0 |
  | 6 | 3.2 x 2.3 x 4.0 | 1 | 47 | 0 | 21 | 1 | 13 (2) | 47 | 0 | 0 |
  | 7 | 3.2 x 2.3 x 4.0 | 1 | 50 | 6 | 28 | 0 | 14 (5) | 50 | 0 | 0 |

## Interpretation

- Independent of the overlap formula: parts 2 to 7 (six parts of 3.2 x 2.3 x 4.0 cm, 1 resting pose each) have no flat-pickable move grasp; every one of their 41 to 100 grasps fails the flat criterion.
- Parts 0 and 1 keep 2 and 6 grasps. Their overlap numbers show the same effect of the corrected formula as in gamepad (0 fail the old formula, 8 and 44 the new one), but that does not decide the result here.
- The assembly cannot be planned with the flat pipeline as it is.

## Not examined

- Why precedence planning takes 110 min for this assembly (stool_circular with 9 parts took 58 s). Whether other jobs loaded the machine during this run was not recorded.
- Why the six small parts have a single resting pose and no grasp for it.

## Suggestions (none implemented)

1. Move grasps for the lying pose of parts 2 to 7 are needed before anything else (generate grasps for the resting pose, or pick these parts from a stand).
2. Look at the precedence planning time separately; it is unrelated to the grasp filter.

---

# Run 2 — `exp_duct_flat_run2` (2026-11-11, new grasp generation)

Setup of the second batch: [README.md](README.md), section "Second batch".

**Result: no plan after 10 sequence attempts (planning 228 min). All parts have usable grasps now; every attempt failed in the
pickup layout: no spot was found for one of the small parts.** No path was planned.

## Facts

- Grasp generation (`run_grasp_arm_gen.py --flat`, 9 min 38 s), from the console log:

  | Part | Size (cm) | Candidates (round 1 + round 2) | Reachable move grasps | Usable | Resting pose the grasps fit | Kept move / hold |
  |---|---|---|---|---|---|---|
  | 0 (no part has to be there before it) | 10.9 x 12.0 x 14.4 | 1080 + 3290 | 23 | 5 | stable | 5 / 150 |
  | 1 | 10.6 x 12.0 x 7.1 | 1020 | 57 | 12 | stable | 12 / 32 |
  | 2 | 3.2 x 2.3 x 4.0 | 1475 + 3995 | 309 | 78 | less stable only | 78 / 100 |
  | 3 | 3.2 x 2.3 x 4.0 | 1485 + 3845 | 240 | 54 | less stable only | 54 / 94 |
  | 4 | 3.2 x 2.3 x 4.0 | 1475 + 3865 | 123 | 30 | less stable only | 30 / 100 |
  | 5 | 3.2 x 2.3 x 4.0 | 1485 + 3915 | 195 | 47 | less stable only | 47 / 100 |
  | 6 | 3.2 x 2.3 x 4.0 | 1475 + 4005 | 236 | 71 | less stable only | 71 / 70 |
  | 7 | 3.2 x 2.3 x 4.0 | 1475 + 3950 | 346 | 153 | less stable only | 100 / 100 |

  Parts 2 to 7 have no usable grasp for a stable resting pose; `grasps.pkl` records them as `fallback_parts` (resting poses that
  tip between 10 and 20 deg). In run 1 these six parts had no move grasp after the filter.
- Sequence planning 79 s. Sequence optimisation: 19 of 48 first hold grasps can pick the first part lying flat; it took 945 s
  in an attempt looked at during the run and 769 s in the last one (`stats.json`, overwritten by every attempt), more than half
  of the 23 min per attempt.
- All 10 attempts ended in `run_pickup_gen.py` with
  `no spot within 152 (or 153) tries where the pickup passes the checks`, for part 6 (twice), part 3 (twice), part 4 (five times)
  and part 7 (once). The reasons counted are always the same kind: a finger of the gripper closer than 1.0 cm to one of the
  lying parts already placed, e.g. attempt 1: left finger to part 3 (79x), right finger to part 4 (51x), left finger to part 5 (45x).
  Each time the grasp of that part was removed (10 grasps in total).
- Example, attempt 9: part 7 placed at (0.0, 5.0) after 1 spot, part 2 at (-3.8, 5.0) after 51, part 3 at (0.0, 8.7) after 41,
  part 5 at (3.8, 5.0) after 36, part 6 at (0.0, 1.3) after 25; part 4 found no spot within 152.
- From the code (`run_pickup_gen.py`): the spots are tried on a 1.25 cm grid in the order of their distance from the pickup
  area; a spot counts as tried when the footprint is free (0.5 cm to the other footprints); the search stops after
  `MAX_PLACEMENT_TRIES` = 150 tried spots.

## Interpretation

- The six small parts are picked one after the other and all lie in the same small area. Each one placed makes the next one
  harder (1, 51, 41, 36, 25 spots, then more than 150): the fingers need 1 cm to every neighbour, and the 150 nearest free spots
  are all next to the parts already placed. The search gives up before it reaches spots further out. Not verified by a run with
  more spots.
- The failure belongs to the layout, not to the grasp that is blamed: ten grasps of three parts gave the same result.

## Not examined

- Whether spots exist further away from the pickup area (the board is 62 x 92 cm), and how many tries are needed to reach them.
- Anything after the layout: no path of this assembly has been planned yet.
- How reliably the small parts stay in poses that tip at 10 to 20 deg.

## Suggestions (none implemented)

1. Let the layout go on to spots further out when the nearest ones fail (more tries, or a coarser first pass over the board),
   and keep more distance between lying parts (`PART_GAP`, see cooling_manifold).
2. Do not remove a grasp when the layout finds no spot for a part whatever its grasp; stop with that message, or widen the search.
3. The sequence optimisation takes 13 to 16 min for this assembly and is repeated in every attempt; its result could be reused
   when only the layout failed.

# Run 3 (2026-11-11, evening): wider layout search

Experiment `exp_duct_flat_run3`: precedence plan and unfiltered grasps of run 2, the code as after the batch (pickup checked
with the closed gripper, tolerance on the distances) plus the layout search over the whole board. Started 21:28, stopped by me
at 22:10 in the second sequence attempt, because the code was being changed for the next round (the run would have continued
on a mix of old and new code).

## What happened

- Sequence planning 79 s, sequence optimisation 929 s (19 of 48 first hold grasps can pick the first part lying flat), layout
  892 s.
- Layout, attempt 1: all eight parts placed, for the first time. Part 7 at (0.0, 5.0) after 1 spot, part 2 at (-3.8, 5.0)
  after 51, part 4 at (0.0, 8.7) after 41, part 5 at (3.8, 5.0) after 36, part 3 at (0.0, 1.2) after 25. Part 6 found no spot
  among the 153 nearest ones and was placed at (6.3, 1.3) on the first spot of the search over the whole board (154 spots
  tried). Part 1 at (-7.5, 12.5) after 5, part 0 (hold arm) at (6.5, 13.3).
- Motion planning, attempt 1 (first motion planning of this assembly): the approach to the pickup of part 2 (move grasp 1603,
  hand 21.7 deg from vertical) was rejected 8 of 8 times with `no straight move into the goal pose from a pose clear of
  everything`. Grasp 1603 of part 2 was removed, attempt 2 started.

## Why the approach to part 2 failed (measured, scratch script `diag_switch_goal.py`)

The scene at that moment rebuilt from the files of the layout: parts 0, 1, 6, 3, 5, 4 assembled, parts 2 and 7 still lying
(1.60 cm apart). Walking back from the pickup pose along the hand axis, as the planner does:

| moved | margin to the virtual wall | gripper to part 2 | gripper to part 7 | step allowed |
|---|---|---|---|---|
| 0.00 cm | 1.1 cm | 0.39 cm | 2.33 cm | yes |
| 0.25 cm | 1.0 cm (below the limit of 1.0) | 0.39 cm | 2.42 cm | **no: virtual wall** |

The hand axis points to (-0.369, 0, 0.929): the hand leans with its back towards -x, where the virtual wall of the move arm
stops the flange at x = -9. The part lies at x = -4.9 to -2.6. At the pickup pose itself the wall rule passes (1.1 cm), so the
layout accepted the spot; the first step back breaks it. The gripper would have to move back until it is 1.2 cm from part 2.

This is not a matter of how close the parts lie (part 7 is 2.3 cm from the gripper). Earlier on 2026-11-11 I told the user it
was; that statement was a guess and is wrong.

The same grasp at a spot further from the wall, or with the hand turned by 180 deg about the vertical, has the hand axis
pointing away from the wall. The layout did not test the approach, only the pose.

## What was changed because of it

- The layout tests the two straight moves at every spot with the planner's own routine before it takes the spot
  (`PickupMoves` in `planning/run_pickup_gen.py`): into the grasp along the hand axis, and straight up with the part. A spot
  where one of them is missing is skipped like a spot that fails a check.
- The planner reports what stopped a straight move (`last_block`), so the log names the cause, e.g.
  `no straight move into the goal pose from a pose clear of everything (virtual wall after 0.25 cm)`.
- A failure of motion planning carries its cause; for the cause "spot" the grasp is kept and the part is laid out elsewhere.

Result with these changes: run 4 below.

# Run 4 (2026-11-11/12, night): after the user's decisions

Experiment `exp_duct_flat_run4`, code as described in [README.md](README.md) ("Fourth batch").

- Grasp generation 12 min 42 s. Usable move grasps: part 1: 33; part 0: 10 (after the second sampling round); parts 2 to 7:
  203, 153, 42, 100, 94, 186, all in a less stable resting pose (no grasp fits a stable one), as in run 2.
- **Plan verified** in the first sequence attempt, the first plan of this assembly. Planning 57 min: sequence planning 160 s,
  sequence optimisation 1311 s, layout 663 s, motion planning 738 s. No path rejected, no grasp removed, no spot banned.
- Layout: part 7 at (0.0, 5.0) after 1 spot; part 3 at (0.0, 8.75) after 21; part 2 found no spot among the 152 nearest ones
  and was placed at (-6.25, 8.75) by the search over the whole board (175 spots tried); part 4 at (3.75, 5.0) after 16; part 6
  at (0.0, 1.25) after 9; part 5 at (-6.25, 5.0) after 58; part 1 at (7.5, 12.5) after 3; part 0 (held first) at (11.25, 0.0).
- Hand at the pickup, from vertical: parts 7, 4, 6, 3, 5: 13, 12, 21, 27, 29 deg; part 2: 53 deg; part 1: 36 deg; part 0
  (hold grasp 45): 66 deg. Parts 2 to 7 lie in a resting pose that tips over at 16 deg.
- Place poses for `place_parts.py`: all moves checked.

Compared with run 3: there the layout put part 2 at (-3.8, 5.0), where its grasp had no straight approach (virtual wall).
With the approach tested at every spot, 174 spot / resting pose combinations were rejected for part 2 before one fitted.

Not examined: how reliably the six small parts stay in the pose that tips at 16 deg; the sequence optimisation still takes
22 min for this assembly.

# Run 5 (2026-11-12): fifth batch

Default place (`exp_duct_flat_run5`): **verified**, first sequence attempt. Grasps 17 min 1 s (run 4: 12 min 42 s), layout
1 min 41 s, motion planning 12 min 29 s, 39 min for the plan in all. No poses for `place_parts.py`.

Two mistakes of mine showed up on this assembly before it got there (README, "Position test"):

- Part 0 is held first, and the precedence plan gives it a path (a slide of 12.8 cm along the table). My new test for the
  moves at the assembly took such a path for an insertion. On plumbers_block, whose first part also has one, it dropped
  every move grasp of that part (687), and the pipeline then reports "no move grasp" for it. Corrected before duct was
  reached in the batch, so this was not observed on duct itself.
- The same test ran out of memory here: duct has two parts with 28000 and 38000 faces (parts 0 and 1), and the collision
  structures of each tested grasp were not released (60 MB per grasp). The kernel killed a worker at 10:28, the grasp
  generation then waited forever; a second start filled 15 GB of memory and 21 GB of swap. Corrected (release after each
  grasp, one shared planner, fewer processes): third start without trouble, never less than 10.5 GB free.

Result of the third start:

- Move grasps usable / kept: part 0: 26 / 26, part 1: 59 / 59, part 2: 216 / 100, part 3: 146 / 100, part 4: 103 / 100,
  part 5: 194 / 100, part 6: 69 / 69, part 7: 169 / 100. Dropped for the straight moves at the assembly: 4 (part 0,
  release only), 1 (part 1), 6 (part 2).
- Parts 2 to 7 lie in the less stable pose (tips at 16 deg).
- Pickups: parts 1 and 3 to 7 with hands 20 to 32 deg, part 2: 47 deg; part 0 hold grasp 296, 74 deg.

Not examined: the position test for duct (each plan takes about 40 min, 30 pairs would take about 5 hours).

# Run 6 (2026-11-12, 22:02): sixth batch, final code of the day

Default place (`exp_duct_flat_run6`): **verified**, first sequence attempt. Grasps 18 min 29 s, layout 1 min 41 s, motion
planning 12 min 19 s, 34 min for the plan in all; `place_parts.py` poses without collision (run 5: none). Usable move
grasps: 51 (part 0), 62, 222, 153, 92, 119, 106, 189. Parts 2 to 7 lie in the pose that tips at 16 deg. Pickups of the
move arm: hands 16 to 28 deg, part 2: 45 deg; part 0 hold grasp 375, 65 deg.

To know: the inflated mesh the planner uses for part 0, the large part that is held first, has spikes up to 196 cm long
(10.9 x 12.0 x 14.4 cm becomes 116 x 84 x 198 cm), see README, "the inflated part meshes of the planner have spikes".
The plan was found all the same; what the spikes cost here was not measured.

Not run: the position test for duct.
