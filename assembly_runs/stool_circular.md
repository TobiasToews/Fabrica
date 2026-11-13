# stool_circular (9 parts) — `exp_stool_circular_flat_run1`

See [README.md](README.md) for the setup and for what Fact / Interpretation / Not examined mean,
and [grasp_filter_finding.md](grasp_filter_finding.md) for how the table below was computed.

**Result: no plan. Stopped in the first sequence attempt after 16 min** (2026-11-10/11 23:46:15–00:03:06, exit code 1).

## Facts

- Console log: precedence and grasp generation finished; the grasp filter left at least one part without a move grasp; sequence
  optimisation printed `No complete sequence`, then `no sequence left`. No pickup layout and no path was planned.
- `stats.json`: precedence planning 58 s, grasp generation 938 s; sequence planning: "tree fully explored (solution found: False)".
- Move grasps per part and how many fail each criterion of the filter (a grasp can fail several):

  | Part | Size (cm) | Resting poses | Move grasps | walls | sigma | joint | overlap (old formula) | flat | pass | pass (old overlap) |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 0 | 1.8 x 1.8 x 3.0 | 6 | 100 | 3 | 31 | 2 | 18 (3) | 41 | 33 | 43 |
  | 1 | 2.1 x 1.8 x 9.5 | 6 | 100 | 3 | 23 | 2 | 10 (3) | 100 | 0 | 0 |
  | 2 | 7.5 x 7.5 x 1.6 | 2 | 100 | 7 | 23 | 5 | 28 (0) | 51 | 18 | 40 |
  | 3 | 2.1 x 1.8 x 9.5 | 6 | 100 | 3 | 28 | 6 | 12 (4) | 100 | 0 | 0 |
  | 4 | 1.8 x 1.8 x 3.0 | 6 | 100 | 2 | 34 | 3 | 26 (9) | 43 | 25 | 32 |
  | 5 | 2.1 x 1.8 x 9.5 | 6 | 100 | 1 | 35 | 7 | 15 (6) | 100 | 0 | 0 |
  | 6 | 1.8 x 1.8 x 3.0 | 6 | 100 | 1 | 30 | 2 | 18 (0) | 41 | 41 | 46 |
  | 7 | 2.1 x 1.8 x 9.5 | 6 | 100 | 5 | 31 | 2 | 10 (1) | 100 | 0 | 0 |
  | 8 | 1.8 x 1.8 x 3.0 | 6 | 100 | 5 | 24 | 2 | 24 (0) | 46 | 34 | 48 |

## Interpretation

- Independent of the overlap formula: parts 1, 3, 5 and 7 (2.1 x 1.8 x 9.5 cm; by size presumably the legs) have no flat-pickable move grasp, although each has 6 resting poses and 100 grasps; all 100 fail the flat criterion.
- The other five parts keep 18 to 41 grasps (25 to 48 with the old overlap formula).
- Long thin parts are the common pattern with beam (parts 2 and 3, 1.2 x 1.2 x 7.6 cm, 97 and 96 of 100 not flat-pickable).

## Not examined

- Which condition of the flat pickup excludes all grasps of the long parts (tilt above 30 deg, ground clearance, or the resting poses found).

## Suggestions (none implemented)

1. Same question as for beam: find out which flat-pickup condition excludes grasps of long thin parts. If the generated grasps all hold the part along its length in the assembled (upright) pose, the lying part needs its own grasps across the short side from above.

---

# Run 2 — `exp_stool_circular_flat_run2` (2026-11-11, new grasp generation)

Setup of the second batch: [README.md](README.md), section "Second batch".

**Result: no plan. The four legs (parts 1, 3, 5, 7) have no usable move grasp** (grasp generation 12 min 38 s; the pipeline
stopped at the first sequence optimisation with `no sequence left`).

## Facts

- Grasp generation (`run_grasp_arm_gen.py --flat`), from the console log. "Reachable" means: the arm reaches the grasp at the
  assembled pose without collision. A grasp can fail several criteria.

  | Part | Size (cm) | Candidates (round 1 + round 2) | Reachable move grasps | Usable | Of the reachable ones fail | Kept move / hold |
  |---|---|---|---|---|---|---|
  | 0 | 1.8 x 1.8 x 3.0 | 1435 | 520 | 230 | | 100 / 100 |
  | 1 | 2.1 x 1.8 x 9.5 | 1430 + 4200 | 482 | 0 | walls 16, singularity / joint limits 123, overlap 48, flat pickup 481 | 0 / 100 |
  | 2 (held first) | 7.5 x 7.5 x 1.6 | 1000 | 404 | 159 | | 100 / 150 |
  | 3 | 2.1 x 1.8 x 9.5 | 1430 + 4175 | 518 | 0 | walls 15, singularity / joint limits 134, overlap 49, flat pickup 518 | 0 / 100 |
  | 4 | 1.8 x 1.8 x 3.0 | 1050 | 423 | 155 | | 100 / 100 |
  | 5 | 2.1 x 1.8 x 9.5 | 1430 + 4200 | 465 | 0 | walls 15, singularity / joint limits 157, overlap 48, flat pickup 464 | 0 / 100 |
  | 6 | 1.8 x 1.8 x 3.0 | 1050 | 455 | 174 | | 100 / 100 |
  | 7 | 2.1 x 1.8 x 9.5 | 1430 + 4175 | 490 | 0 | walls 9, singularity / joint limits 126, overlap 47, flat pickup 490 | 0 / 100 |
  | 8 | 1.8 x 1.8 x 3.0 | 1455 | 569 | 259 | | 100 / 100 |

- The legs have 6 stable resting poses each and no further less stable ones. Of 465 to 518 reachable grasps per leg, 0 or 1 can
  pick the leg lying flat, and that one fails another criterion. The second sampling round (about 4200 further candidates per
  leg) added no usable grasp.
- The five other parts have 155 to 259 usable grasps each.
- Grasp filter: `NO MOVE GRASP left for part(s) 1, 3, 5, 7`. Sequence planning: `tree fully explored (solution found: False)`.

## Interpretation

- The question left open in run 1 is answered for the generated grasps: the flat-pickup test, not the random cut to 100 grasps,
  excludes the legs. Testing every reachable grasp and sampling three times as many candidates changes nothing.
- Earlier scratch measurements of this session (before the change, not repeated in this run) describe the legs as hexagonal
  bars, like parts 1 and 4 of plumbers_block: lying on a face they have no vertical pair of faces, face grasps are tilted 30 deg
  and come closer than 0.5 cm to the board, grasps across two edges are not stable. That fits the numbers above.

## Measurement after the batch (leg, part 1; all grasp candidates before any reachability test)

Scratch script `diag_no_grasp.py` (session scratchpad): the candidates of the generator for 200 antipodal pairs, with pad contact.

- The leg is a hexagonal bar: six side faces 60 deg apart, 1.8 cm across the flats, 2.08 cm across the corners, 9.5 cm long.
  Its six resting poses are the six faces (tips at 30.0 deg, 1.8 cm high).
- 1395 candidates; in all of them the fingers close across the bar (1345 across two flats, 1.8 cm; 50 on a 1.4 cm feature).
- Smallest tilt of the hand over the six resting poses: 14 candidates at 30 deg or less, 290 between 30 and 35 deg, 548
  between 35 and 60 deg, 543 above 60 deg.
- Of the 14 within `MAX_TILT`: the gripper is between -0.21 and 0.70 cm above the board (median 0.04 cm); 4 keep
  `GROUND_CLEARANCE` 0.5 cm; none of these 4 holds the leg 0.3 cm deep with both fingers.

Interpretation: with the bar lying on a face, the two faces a gripper can close on are inclined 60 deg, so the hand has to lean
at least 30 deg, which is exactly the limit. The generator turns the hand about the closing direction in 10 steps (36 deg
apart) from a random start, so most candidates miss the one direction with 30 deg and land between 30 and 35 deg; the few
that hit it bring a fingertip too close to the board.

## Not examined

- Whether tilted face grasps (hand 30 to 35 deg, fingertip close to the board) would hold a leg on the real robot.

## Suggestions (none implemented)

1. Same decision as for the bars of plumbers_block: either allow tilted face grasps for such bars (try `MAX_TILT` 35 deg and
   `GROUND_CLEARANCE` 0.3 cm on one leg first and look at the grasps), or present the legs in a V-block or stand.
2. The second sampling round costs time here without a chance of success (12 min 38 s for the grasp generation, the longest of the
   batch). It could be skipped for a part whose first-round grasps all fail the flat test.

# Run 4 (2026-11-11/12, night): after the user's decisions

Experiment `exp_stool_circular_flat_run4`, code as described in [README.md](README.md) ("Fourth batch"): among other things the
tilt limit of the hand at the pickup is gone.

- Grasp generation 12 min 46 s. Parts 0, 2, 4, 6, 8: 227 to 354 usable move grasps. The legs (parts 1, 3, 5, 7): **0 usable**,
  as before. Of their feasible move grasps the flat pickup fails for 481 of 484 (part 1), 518 of 518 (part 3), 464 of 465
  (part 5), 490 of 492 (part 7).
- No sequence (`no sequence left` after the grasp filter), planning under 1 min.

Measured before the run (scratch script `diag_tilt_lifted.py`, leg 1, 1395 candidates with pad contact): only 4 candidates
keep the gripper 0.5 cm above the board in some resting pose, none of them with 3 mm of finger on the part; with the gripper
allowed down to 0.1 cm above the board it is 13, again none with 3 mm. The tilt limit was not the cause.

Interpretation: the leg is a hexagonal bar, 1.8 cm across the flats, and lies on one of its six faces. The fingers have to
grip two opposite faces, which then slope 60 deg; the lower finger lies along a face that ends at the board. With 3 mm of
finger on that face its outer edge would be below 1 mm above the board.

Not examined: grasps that hold the bar on its two end faces (9.5 cm apart, wider than the gripper opens: 8 cm); a stand that
presents the legs upright.

# Run 5 (2026-11-12): fifth batch

Default place (`exp_stool_circular_flat_run5`): **no plan**, as before: no move grasp for parts 1, 3, 5, 7 (the legs). The
pipeline stops right after the grasp stage and names them. Grasps 20 min 2 s.

- Legs: 690, 685, 695 and 609 feasible move grasps; the flat pickup fails for 683, 676, 692 and 601 of them, and the few
  that pass it fail another criterion (walls, singularity / joint limits or finger overlap).
- The other five parts have 356 to 552 usable move grasps each (100 kept).

Nothing was changed for the legs since run 4.

# Run 6 (2026-11-12, 22:02): sixth batch, final code of the day

Default place (`exp_stool_circular_flat_run6`): **no plan**, as in every run: no move grasp for parts 1, 3, 5, 7 (the
legs). Flat pickup fails for 655 of 667, 630 of 638, 599 of 602 and 561 of 565 feasible move grasps. Grasps 28 min 29 s.
