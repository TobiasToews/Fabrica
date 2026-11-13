# socket_set (6 parts) — `exp_socket_set_flat_run1`

See [README.md](README.md) for the setup and for what Fact / Interpretation / Not examined mean,
and [grasp_filter_finding.md](grasp_filter_finding.md) for how the table below was computed.

**Result: no plan. Stopped in the first sequence attempt after 3 min** (2026-11-10 21:33:01–21:36:15, exit code 1).

## Facts

- Console log: precedence and grasp generation finished; the grasp filter left at least one part without a move grasp; sequence
  optimisation printed `No complete sequence`, then `no sequence left`. No pickup layout and no path was planned.
- Move grasps per part and how many fail each criterion of the filter (a grasp can fail several):

  | Part | Size (cm) | Resting poses | Move grasps | walls | sigma | joint | overlap (old formula) | flat | pass | pass (old overlap) |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 0 | 14.0 x 6.0 x 2.0 | 2 | 100 | 0 | 26 | 6 | 19 (9) | 40 | 26 | 32 |
  | 1 | 1.4 x 1.4 x 2.4 | 2 | 62 | 0 | 8 | 0 | 38 (3) | 12 | 20 | 43 |
  | 2 | 1.7 x 1.7 x 2.4 | 2 | 54 | 0 | 8 | 4 | 17 (4) | 8 | 31 | 38 |
  | 3 | 2.2 x 2.2 x 2.4 | 2 | 100 | 0 | 22 | 3 | 94 (14) | 17 | 6 | 57 |
  | 4 | 2.4 x 2.4 x 2.4 | 2 | 100 | 0 | 35 | 5 | 98 (77) | 31 | 0 | 7 |
  | 5 | 3.0 x 3.0 x 2.4 | 2 | 100 | 0 | 42 | 10 | 81 (53) | 31 | 11 | 20 |

## Interpretation

- Caused by the corrected overlap formula together with the 0.5 cm limit: part 4 (2.4 x 2.4 x 2.4 cm) loses all move grasps, 98 of 100 on the overlap criterion (median overlap 0.05 cm). With the old formula it keeps 7.
- Part 4 is weak with either formula (77 of 100 fail the old one), so few alternatives would remain in later stages.

## Not examined

- Why the contact points of the grasps of parts 4 and 5 are so close to the fingertip (median 0.05 and 0.16 cm).

## Suggestions (none implemented)

1. Decide on the overlap measure first ([grasp_filter_finding.md](grasp_filter_finding.md)), then repeat this run.

---

# Run 2 — `exp_socket_set_flat_run2` (2026-11-11, new grasp generation)

Setup of the second batch: [README.md](README.md), section "Second batch".

**Result: plan verified** (grasp generation 3 min 22 s, planning 4 min; first sequence attempt, no grasp dropped).

## Facts

- Grasp generation (`run_grasp_arm_gen.py --flat`), from the console log:

  | Part | Size (cm) | Candidates | Reachable move grasps | Usable (pick the part lying flat, all criteria) | Resting pose | Kept move / hold |
  |---|---|---|---|---|---|---|
  | 0 (held first) | 14.0 x 6.0 x 2.0 | 715 | 324 | 84 | stable | 84 / 150 (50 hold grasps added that pick it lying flat) |
  | 1 | 1.4 x 1.4 x 2.4 | 1315 | 197 | 80 | stable | 80 / 100 |
  | 2 | 1.7 x 1.7 x 2.4 | 1295 | 218 | 85 | stable | 85 / 100 |
  | 3 | 2.2 x 2.2 x 2.4 | 1355 | 174 | 75 | stable | 75 / 100 |
  | 4 | 2.4 x 2.4 x 2.4 | 1310 | 204 | 67 | stable | 67 / 100 |
  | 5 | 3.0 x 3.0 x 2.4 | 1275 | 189 | 39 | stable | 39 / 100 |

  No part needed the second sampling round. The grasp filter kept every generated move grasp.
- Sequence optimisation: 16 of 64 first hold grasps can pick part 0 lying flat.
- Layout: part 0 with hold grasp 584 (tilt 19.7 deg); move grasps 527, 704, 352, 567, 507 for parts 5, 1, 2, 4, 3 (tilt 2.5–11.8
  deg); all in stable resting poses.
- Motion planning: no path of the real arm was rejected, no hold-arm jump. `verify_plan.py`: PASSED. `workcell.json`:
  `verified: true`, `banned_move_grasps: []`.

## Interpretation

- In run 1 part 4 lost all move grasps to the finger-overlap criterion (0.5 cm, measured on a few random contact points). With the
  overlap measured on the pad grid and the limit at 0.3 cm, 67 of its 204 reachable grasps are usable.

## Not examined

- The plan was not executed, rendered or used for policy training.
- How deep the chosen grasps actually hold the small sockets (the depth per finger is stored with every grasp as `contact_depths`).

## Suggestions

1. None needed for planning.

# Run 4 (2026-11-11/12, night): after the user's decisions

Experiment `exp_socket_set_flat_run4`, code as described in [README.md](README.md) ("Fourth batch").

- Grasp generation 3 min 31 s. Usable move grasps: part 0: 180, part 1: 96, part 2: 121, part 3: 107, part 4: 113, part 5: 74
  (at most 100 kept). No second sampling round.
- **Plan verified** in the first sequence attempt, planning 3 min (sequence optimisation 35 s, layout 68 s, motion planning 49 s).
- Layout: part 3 at (0.0, 5.0), hand 12 deg from vertical; part 4 at (-3.75, 5.0), 12 deg; part 5 at (3.75, 3.75), 3 deg;
  part 2 at (0.0, 8.75), 5 deg; part 1 at (-1.25, 1.25), 11 deg; part 0 (held first) at (0.0, 15.0), hold grasp 610, 71 deg.

# Run 5 (2026-11-12): fifth batch

Default place (`exp_socket_set_flat_run5`): **verified**, first sequence attempt. Grasps 4 min 37 s, layout 26 s, motion
planning 1 min 22 s; poses for `place_parts.py` without collision.

- Move grasps usable / kept: part 0: 335 / 100, part 1: 103 / 100, part 2: 119 / 100, part 3: 143 / 100, part 4:
  155 / 100, part 5: 101 / 100. Dropped for the straight moves at the assembly: 42 (part 0, release only), 7, 7, 1, 2, 0.
- Pickups: hands 0 to 12 deg from vertical for the five parts of the move arm (grasps 704, 352, 1745, 567, 527); part 0
  hold grasp 546, 68 deg.

Not run: the position test (the easiest assembly besides socket_set_small).

# Run 6 (2026-11-12, 22:02): sixth batch, final code of the day

Default place (`exp_socket_set_flat_run6`): **verified**, first sequence attempt. Grasps 4 min 43 s, layout 19 s, motion
planning 51 s; `place_parts.py` poses without collision. Usable move grasps: 295 (part 0), 116, 130, 98, 93, 80. Pickups
of the move arm with hands 0 to 15 deg; part 0 hold grasp 630, 58 deg.

To know: the inflated mesh the planner uses for part 1 has spikes (1.4 x 1.4 x 2.4 cm becomes 7.8 x 7.8 x 5.2 cm), see
README, "the inflated part meshes of the planner have spikes".
