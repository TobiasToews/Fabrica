# plumbers_block (5 parts) — `exp_plumbers_block_flat_run1`

See [README.md](README.md) for the setup and for what Fact / Interpretation / Not examined mean,
and [grasp_filter_finding.md](grasp_filter_finding.md) for how the table below was computed.

**Result: no plan. Stopped in the first sequence attempt after 4 min** (2026-11-10 21:16:34–21:20:49, exit code 1).

## Facts

- Console log: precedence and grasp generation finished; the grasp filter left at least one part without a move grasp; sequence
  optimisation printed `No complete sequence`, then `no sequence left`. No pickup layout and no path was planned.
- Move grasps per part and how many fail each criterion of the filter (a grasp can fail several):

  | Part | Size (cm) | Resting poses | Move grasps | walls | sigma | joint | overlap (old formula) | flat | pass | pass (old overlap) |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 0 | 4.3 x 8.0 x 5.0 | 4 | 3 | 0 | 2 | 0 | 0 (0) | 3 | 0 | 0 |
  | 1 | 2.9 x 2.5 x 7.0 | 1 | 100 | 0 | 36 | 2 | 49 (7) | 100 | 0 | 0 |
  | 2 | 16.0 x 4.0 x 6.5 | 4 | 100 | 2 | 34 | 3 | 6 (0) | 43 | 33 | 37 |
  | 3 | 10.0 x 4.0 x 2.6 | 3 | 100 | 7 | 42 | 2 | 13 (3) | 55 | 28 | 34 |
  | 4 | 2.9 x 2.5 x 7.0 | 1 | 100 | 4 | 28 | 0 | 29 (2) | 100 | 0 | 0 |

## Interpretation

- Independent of the overlap formula: parts 1 and 4 (2.9 x 2.5 x 7.0 cm, 1 resting pose each) have no flat-pickable move grasp (100 of 100 fail), and part 0 has only 3 move grasps at all, none flat-pickable.
- The assembly cannot be planned with the flat pipeline as it is, whatever the gripper rules or the motion planner do.

## Not examined

- Why parts 1 and 4 have a single resting pose and no grasp for it.
- Why the grasp generation finds only 3 move grasps for part 0 (4.3 x 8.0 x 5.0 cm).

## Suggestions (none implemented)

1. For parts with a single resting pose, generate move grasps for that lying pose (today grasps are generated for the assembled pose and then tested for flat pickup).
2. Alternatively allow such a part to be handed over or placed by the human, like the first part.

---

# Run 2 — `exp_plumbers_block_flat_run2` (2026-11-11, new grasp generation)

Setup of the second batch: [README.md](README.md), section "Second batch".

**Result: no plan. Parts 0, 1 and 4 have no usable move grasp** (grasp generation 3 min 50 s; the pipeline stopped at the first
sequence optimisation with `no sequence left`).

## Facts

- Grasp generation (`run_grasp_arm_gen.py --flat`), from the console log. "Reachable" means: the arm reaches the grasp at the
  assembled pose without collision. The criteria are those of the grasp filter; a grasp can fail several.

  | Part | Size (cm) | Candidates (round 1 + round 2) | Reachable move grasps | Usable | Of the reachable ones fail | Kept move / hold |
  |---|---|---|---|---|---|---|
  | 0 | 4.3 x 8.0 x 5.0 | 1160 + 3495 | 4 | 0 | walls 0, singularity / joint limits 2, overlap 3, flat pickup 4 | 0 / 17 |
  | 1 | 2.9 x 2.5 x 7.0 | 1600 + 4715 | 238 | 0 | walls 17, singularity / joint limits 68, overlap 25, flat pickup 238 | 0 / 100 |
  | 2 (no part has to be there before it) | 16.0 x 4.0 x 6.5 | 760 | 201 | 91 | | 91 / 142 |
  | 3 | 10.0 x 4.0 x 2.6 | 1710 | 200 | 55 | | 55 / 100 |
  | 4 | 2.9 x 2.5 x 7.0 | 1590 + 4750 | 253 | 0 | walls 9, singularity / joint limits 72, overlap 26, flat pickup 253 | 0 / 100 |

- Parts 0, 1 and 4 got the second sampling round (600 further antipodal pairs; of its candidates only those that can pick the part
  lying flat are checked for reachability). It added no usable grasp.
- Resting poses of parts 1 and 4: 1 stable pose, 7 with the less stable ones (tipping between 10 and 20 deg). No reachable grasp
  can pick them in any of the 7. Part 0: 4 resting poses, all stable.
- Grasp filter: `NO MOVE GRASP left for part(s) 0, 1, 4`. Sequence planning: `tree fully explored (solution found: False)`.

## Interpretation

- More candidates do not help here: for parts 1 and 4 not one of 238 and 253 reachable grasps can pick the part lying flat, and
  the flat test of the second round found none either among 4715 and 4750 further candidates that is also reachable.
- Part 0 is a different case: at its assembled pose only 4 of 4655 candidates are reachable at all, so the assembled pose (not
  the lying pose) leaves almost no grasp. The same was seen in run 1 (3 move grasps).
- Earlier scratch measurements of this session (before the change, not repeated in this run) describe parts 1 and 4 as hexagonal
  bars: lying on a face they offer no vertical pair of faces; grasps on two faces are tilted 30 deg and bring the gripper closer
  than 0.5 cm to the board, grasps across two edges are not stable.

## Measurement after the batch (parts 1 and 0; all grasp candidates before any reachability test)

Scratch scripts `diag_no_grasp.py` and `diag_reach.py` (session scratchpad): the candidates of the generator for 200 antipodal
pairs, with pad contact; for those that can pick the part lying flat with 0.3 cm finger depth, what the gripper hits at the
assembled pose.

- Part 1 (the same shape as part 4): hexagonal section, 2.5 cm across the flats, 7.0 cm long; in the assembly it stands upright
  from 3.5 to 10.5 cm above the table, on parts 3 and 2.
  - Resting poses: one stable pose, standing on an end (tips at 27.2 deg) and upside down: the direction that points up in the
    assembly points down. Six less stable poses, lying on a face (tip at 14.0 deg, support margin 0.31 cm).
  - 1565 candidates. None can pick the part in the stable pose. 176 can pick it lying on a face, 120 of them with 0.3 cm depth.
  - Those 120 all come from the side at the assembled pose (hand 60 to 91 deg from straight down). All 120 collide there with
    part 2 (assembled before part 1); 59 of them also put the hand into the table.
- Part 0 (4.3 x 8.0 x 5.0 cm, assembled 2.7 to 7.7 cm above the table, on part 2):
  - 1250 candidates; 518 can pick it lying flat in a stable pose, 511 with 0.3 cm depth.
  - 250 of them examined at the assembled pose: all come from the side and all collide with something: 241 with part 2
    (assembled before), 216 with the table. (In the batch, 4 of 4655 candidates of part 0 were reachable as move grasps.)

Interpretation: these parts are oriented differently in the assembly than when they lie. A grasp that takes the lying part from
above approaches the assembled part from the side, 3 to 10 cm above the table, where the 20 cm wide hand hits the table or the
parts already there. The grasps that are reachable at the assembled pose come from above there, and would have to come from
below for the only stable resting pose of parts 1 and 4.

## Not examined

- Whether a stand that presents parts 1 and 4 upright, the way they are assembled, makes the reachable grasps usable.
- Whether a larger `MAX_TILT` or a smaller `GROUND_CLEARANCE` would admit any grasp that is also reachable at the assembled pose.

## Suggestions (none implemented)

1. Parts 1 and 4: decide whether tilted face grasps are acceptable (test `MAX_TILT` 35 deg and `GROUND_CLEARANCE` 0.3 cm on these
   two parts first), or pick these bars from a small V-block or stand instead of flat from the table.
2. Part 0: find out what blocks its grasps at the assembled pose before changing anything else; with 4 reachable grasps the
   planner has no alternatives even if one of them were usable.

# Run 4 (2026-11-11/12, night): after the user's decisions

Experiment `exp_plumbers_block_flat_run4`, code as described in [README.md](README.md) ("Fourth batch"): among other things the
tilt limit of the hand at the pickup is gone.

- Grasp generation 3 min 50 s. Part 2: 127 usable move grasps, part 3: 68, part 0: **5** (run 2: 0; 17 feasible move grasps
  after the second round). Parts 1 and 4 (the bolts): **0 usable**; of their 238 and 253 feasible move grasps every one fails
  the flat pickup (1 stable resting pose, 7 with the less stable ones).
- No sequence (`no sequence left`), planning under 1 min.

The shape of parts 1 and 4, measured on the mesh (the user doubted my earlier wording "the only stable pose is standing upside
down"): a bolt, 7 cm long. A round shaft of 0.95 cm diameter over the lower 4.9 cm, a hexagonal head (2.5 cm across the flats,
2.9 cm across the corners) over the upper 2.1 cm; the centre of mass is 5.06 cm above the lower end, inside the head.

| Resting pose | Tips over at | Support |
|---|---|---|
| standing on the head, shaft up (1 pose) | 27.2 deg | hexagon of the head, margin 1.00 cm |
| lying on one of the six faces of the head, shaft horizontal (6 poses) | 14.0 deg | 1.44 x 2.00 cm of the face, the centre of mass 0.31 cm inside its edge |

All seven are offered to the grasps (the six lying poses as "less stable"). No feasible move grasp fits one of them. Measured
on the 200 move grasps of the first sampling round that are reachable at the assembled pose (scratch script
`diag_flat_fail.py`, part 1):

- 174 hold the bolt with the hand from above, along its axis; 26 with the hand from the side, across the axis.
- For every one of the 200, the resting pose in which it comes closest to passing is a lying one, and there the gripper is too
  low (for 121 also the arm link behind the hand). The five that come closest are side grasps; with the bolt lying the hand
  leans 35 to 46 deg and the gripper would be 0.04, 0.00, -0.14, -0.22 and -0.25 cm above the board (required: 0.5 cm).

So the limit for the bolts is the distance of the fingertip to the board, not the tilt of the hand.

The other way round (sample measured before the run, `diag_tilt_lifted.py`): of 185 candidates that can pick the bolt lying
with 3 mm of finger, 60 were tested at the assembled pose: 0 reachable (30 collide with other parts, 20 with the table, 10
with the bolt itself).

Not examined: which of the 5 usable grasps of part 0 would survive sequence planning (the bolts stop it before).

# Run 5 (2026-11-12): fifth batch

Default place (`exp_plumbers_block_flat_run5`): **no plan**, as before: no move grasp for parts 1 and 4 (the bolts). The
pipeline now stops right after the grasp stage and says so
(`no plan where the assembly stands now: no move grasp works there for part 1, 4`). Grasps 5 min 27 s.

- Part 1: 296 feasible move grasps, none can pick the bolt lying flat (flat pickup fails for 296 of 296; 1 stable pose, 7
  with the less stable ones). Part 4: 284 feasible, 284 fail the flat pickup.
- Part 0: 2 usable move grasps of 12 feasible (run 4: 5). Part 3: 169 usable, 100 kept.
- Part 2 is held first: 178 usable move grasps, 100 kept (they only serve the first-part fallback).

A mistake of mine was found on this assembly: the precedence plan gives part 2 a path although it is the first part (a
slide of 3.5 cm along the table), and my new test for the straight moves at the assembly treated it as an insertion. The
first generation of this run dropped all 687 otherwise usable move grasps of part 2
(`no straight move to the insertion start (move part and ground after 0.25 cm)`) and listed part 2 among the parts without
a grasp. Corrected (the first part is the part with nothing assembled before it; only its release is tested), grasps made
again: the numbers above.

Not examined: the bolts (unchanged since run 4: the fingertips cannot get low enough beside a bolt that lies on the board).

# Run 6 (2026-11-12, 22:02): sixth batch, final code of the day

Default place (`exp_plumbers_block_flat_run6`): **no plan**, as in every run: no move grasp for parts 1 and 4 (the bolts).
Flat pickup fails for 294 of 295 feasible move grasps of part 1 and for 301 of 301 of part 4. Grasps 6 min 32 s.
