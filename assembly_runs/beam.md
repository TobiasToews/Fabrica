# beam (5 parts) — `exp_beam_flat_run1`

See [README.md](README.md) for the setup and for what Fact / Interpretation / Not examined mean,
and [grasp_filter_finding.md](grasp_filter_finding.md) for how the table below was computed.

**Result: no plan. Stopped in the first sequence attempt after 5 min** (2026-11-10 21:10:52–21:16:34, exit code 1).

## Facts

- Console log: precedence and grasp generation finished; the grasp filter left at least one part without a move grasp; sequence
  optimisation printed `No complete sequence`, then `no sequence left`. No pickup layout and no path was planned.
- Move grasps per part and how many fail each criterion of the filter (a grasp can fail several):

  | Part | Size (cm) | Resting poses | Move grasps | walls | sigma | joint | overlap (old formula) | flat | pass | pass (old overlap) |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 0 | 1.3 x 7.6 x 1.5 | 3 | 100 | 0 | 4 | 6 | 19 (8) | 50 | 30 | 39 |
  | 1 | 1.3 x 7.6 x 1.5 | 2 | 100 | 0 | 37 | 2 | 37 (5) | 54 | 16 | 24 |
  | 2 | 1.2 x 1.2 x 7.6 | 4 | 100 | 4 | 31 | 4 | 18 (6) | 97 | 0 | 1 |
  | 3 | 1.2 x 1.2 x 7.6 | 4 | 100 | 3 | 26 | 4 | 15 (6) | 96 | 0 | 1 |
  | 6 | 15.2 x 1.8 x 1.3 | 4 | 100 | 3 | 25 | 6 | 55 (16) | 38 | 4 | 32 |

## Interpretation

- Parts 2 and 3 (sticks of 1.2 x 1.2 x 7.6 cm) are the blockers: 97 and 96 of their 100 move grasps cannot pick the part lying flat. The overlap formula only decides about the last remaining grasp of each (pass 1 with the old formula, 0 with the new one).
- With one grasp per part the pipeline would have had no alternative as soon as that grasp failed a later stage, so the old formula would not have made this assembly robust either.

## Not examined

- Why almost no grasp of the two sticks is flat-pickable (which of tilt, ground clearance or resting pose excludes them).

## Suggestions (none implemented)

1. Find out which flat-pickup condition excludes the grasps of parts 2 and 3; for long thin parts lying flat, grasps across the short side from above should exist. If the grasp generation does not produce them, generate grasps for the lying pose directly.
2. See [grasp_filter_finding.md](grasp_filter_finding.md) for the overlap measure.

---

# Run 2 — `exp_beam_flat_run2` (2026-11-11, new grasp generation)

Setup of the second batch: [README.md](README.md), section "Second batch".

**Result: plan verified** (grasp generation 4 min 2 s, planning 9 min; first sequence attempt, no grasp dropped).

## Facts

- Grasp generation (`run_grasp_arm_gen.py --flat`), from the console log:

  | Part | Size (cm) | Candidates | Reachable move grasps | Usable (pick the part lying flat, all criteria) | Resting pose | Kept move / hold |
  |---|---|---|---|---|---|---|
  | 0 | 1.3 x 7.6 x 1.5 | 1595 | 182 | 54 | stable | 54 / 100 |
  | 1 | 1.3 x 7.6 x 1.5 | 1450 | 178 | 34 | stable | 34 / 100 |
  | 2 | 1.2 x 1.2 x 7.6 | 1080 | 352 | 10 | stable | 10 / 100 |
  | 3 | 1.2 x 1.2 x 7.6 | 1125 | 484 | 8 | stable | 8 / 100 |
  | 6 (held first) | 15.2 x 1.8 x 1.3 | 830 | 389 | 94 | stable | 94 / 150 (50 hold grasps added that pick it lying flat) |

  No part needed the second sampling round (it starts below 3 usable grasps). The grasp filter kept every generated move grasp.
- Sequence optimisation: 4 of 11 first hold grasps can pick part 6 lying flat.
- Layout: part 6 with hold grasp 666 (tilt 30.0 deg), part 2 with move grasp 546 (5.3 deg), part 3 with 127 (11.9 deg), part 0
  with 58 (26.4 deg), part 1 with 119 (29.7 deg); all in stable resting poses.
- Motion planning: no path of the real arm was rejected; one hold-arm path is stored as a jump. `verify_plan.py`: PASSED.
  `workcell.json`: `verified: true`, `banned_move_grasps: []`.
- Stage times (`stats.json`): layout 276 s, motion planning 179 s.

## Interpretation

- The two sticks (parts 2, 3) had 0 move grasps after the filter in run 1 because the 100 grasps were cut at random before the flat
  test: 97 and 96 of them could not pick the stick lying flat. Testing all reachable grasps (352 and 484) finds 10 and 8 usable ones.
  This confirms the measurement made before the change (13 flat-pickable grasps among 339 reachable ones for part 2).

## Run 3 — `exp_beam_flat_run3` (same day, after `FLAT_MIN_GRASPS` 3 -> 10 and the two check fixes)

- **Plan verified, first attempt** (grasp generation 4 min 59 s, planning 9 min, no grasp removed, one hold-arm jump).
- Part 3 got the second sampling round (8 usable grasps is below 10): 1125 + 3245 candidates, 578 reachable, 46 usable. The
  other parts as in run 2.
- Pickup grasps: part 6 hold 666 (30.0 deg), part 2 move 546 (5.3 deg), part 3 move 2876 (12.3 deg), part 0 move 39
  (15.9 deg), part 1 move 119 (29.7 deg).

## Not examined

- The plans were not executed, rendered or used for policy training.
- Run 2: three of the five pickup grasps are tilted 26–30 deg (limit 30 deg); run 3: two. How well such tilted pickups work on
  the real robot is not known.

## Suggestions

1. None needed for planning. If tilted pickups turn out badly on the robot, lower `MAX_TILT` or raise the weight of the upright
   term in the rating (`FLAT_RATING_WEIGHTS`); both reduce the number of usable grasps of the sticks.

# Run 4 (2026-11-11/12, night): after the user's decisions

Experiment `exp_beam_flat_run4`, code as described in [README.md](README.md) ("Fourth batch").

- Grasp generation 4 min 19 s. Usable move grasps: part 0: 77, part 1: 55, part 2: 13, part 3: 23, part 6: 146 (100 kept). No
  second sampling round (run 3: part 3 needed it).
- **Plan verified** in the first sequence attempt, planning 11 min (layout 354 s, motion planning 184 s). 14 of 15 first hold
  grasps can pick part 6 lying flat.
- Layout: part 1 at (0.0, 5.0) after 1 spot, move grasp 193, hand 41 deg from vertical; part 0 at (-5.0, 5.0) after 49 spots,
  grasp 383, 48 deg; part 3 at (6.25, 5.0) after 73 spots, grasp 127, 12 deg; part 2 at (-2.5, 10.0) after 15 spots, grasp 546,
  5 deg; part 6 (held first) at (0.0, -1.25), hold grasp 516, 51 deg.

Interpretation: with the tilt limit gone, the sequence optimisation takes grasps whose hand leans 41 and 48 deg at the pickup
(parts 1 and 0). It chooses by the scores of the original pipeline (hold changes, torque of the insertion forces), not by how
upright the hand is at the pickup.

# Run 5 (2026-11-12): fifth batch

Default place (`exp_beam_flat_run5`): **verified**, first sequence attempt. Grasps 8 min 37 s, layout 1 min 15 s, motion
planning 5 min 46 s; poses for `place_parts.py` without collision.

- Move grasps usable / kept: part 0: 153 / 100, part 1: 141 / 100, part 2: 57 / 57, part 3: 66 / 66, part 6 (held
  first): 188 / 100. Dropped for the straight moves at the assembly: 3 (part 3), 13 (part 6, release only).
- Candidates made for the lying part among the kept move grasps: 50 of 100, 56 of 100, 27 of 57, 25 of 66, 35 of 100: for
  this assembly a third to a half of the grasps that are kept come from that side of the matching.
- Pickups: parts 1 and 3 with the hand straight down, parts 0 and 2 leaning 20 deg; part 6 hold grasp 901, 20 deg.

## Position test (`exp_beam_pos_*`, tables in README)

First start (11:31, during the batch): stopped. At the third place of the assembly, (+5, -30), the grasp generation did
not finish: for parts 2 and 6 the computation of the resting poses (a routine of trimesh, called by `get_flat_rots`) never
returned. The same routine also lost resting poses of beam parts depending on the place: part 0 has 4 poses (the part
tips at 46.6, 46.6, 37.6 and 35.8 deg); the routine gave 3 at the default place and between 0 and 4 at twelve other
places; for part 1 it gave 2 at the default place, without the two most stable ones. The batch run above was made with
that routine, so parts 0 and 1 were planned with fewer resting poses than they have.

The resting poses are now computed from the faces of the convex hull of the part moved to the origin (README). All five
parts have 4 resting poses at every place.

Second start on that code (13:07 to 15:02): **30 of 30 pairs verified**, every plan in the first sequence attempt, no
grasp and no spot dropped, 5.5 to 11.6 min per plan. 10 places of the assembly (x from 0 to +12.5, y from -30 to -7.5; no
further place 5 cm from these has all four corners of the 15.2 x 7.6 cm footprint in the reach of both arms), 3 pickup
areas each (y from -37.5 to +37.5). Usable move grasps per part, examples: at (0, -20): 121, 97, 59, 55 (parts 0 to 3)
and 219 (part 6); at (+12.5, -20): 60, 26, 72, 70 and 290. `place_parts.py` poses without collision for 22 of 30 plans.

# Run 6 (2026-11-12, 22:02): sixth batch, final code of the day

Default place (`exp_beam_flat_run6`): **verified**, first sequence attempt. Grasps 6 min 58 s, layout 54 s, motion planning
3 min 15 s (run 5: 5 min 46 s); `place_parts.py` poses without collision. Usable move grasps: 152, 94, 59, 65 (parts 0 to
3) and 178 (part 6). Pickups of the move arm with hands 0 to 16 deg; part 6 hold grasp 646, 52 deg.
