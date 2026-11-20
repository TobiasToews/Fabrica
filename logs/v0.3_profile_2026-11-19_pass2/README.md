# Planning time profile, planner v0.3, pass 2 (2026-11-19)

Repetition of pass 1 ([../v0.3_profile_2026-11-18/README.md](../v0.3_profile_2026-11-18/README.md))
under the same conditions. Generated tables: [report.md](report.md).

## Setup

Identical to pass 1: planner v0.3 (commit `3d20bb6`, no uncommitted changes in
`planning/`), `FABRICA_PANDA_BASE_Z=-3.3`, default places (assembly at
(0, -15) cm, parts around (0, 5) cm), default pipeline settings, same machine
and software, one run at a time, same order (plumbers_block, beam,
human_normal, gamepad, duct, cooling_manifold, car, socket_set,
socket_set_small, stool_circular), all in one call.

- Time: 2026-11-19 11:38 to 16:12 (4.6 h); pass 1 ran in the evening and at
  night.
- Logs of the plans: `logs/v0.3_exp_<assembly>_flat_prof2/<assembly>/`, raw
  profiling data in `profile/`.
- Command:

```bash
export FABRICA_PANDA_BASE_Z=-3.3
python -u profiling/profile_planning.py --exp-suffix flat_prof2 \
    --assemblies plumbers_block beam human_normal gamepad duct cooling_manifold car socket_set socket_set_small stool_circular \
    --summary-dir logs/v0.3_profile_2026-11-19_pass2
```

Evaluation change before this report (applies to all passes, pass 1 was
evaluated again with it): samples taken after the end of a run are dropped,
and the summed CPU seconds are taken as their running maximum (the true value
never decreases; a dip is an undercount of a process that has exited and
whose time has not reached its parent yet). The first evaluation of pass 2
showed impossible "other load" values (2910 and 153 threads for duct and
socket_set) because of the last sample after the end; the stage CPU sums were
not affected.

## Results

| assembly | plan | attempts | path re-plans | total [min] | grasp generation [min] | loop [min] | final check [min] | CPU [h] | mean cores | peak PSS [GiB] |
|---|---|---|---|---|---|---|---|---|---|---|
| plumbers_block | no | - | - | 6.4 | 6.0 | - | - | 1.17 | 11.0 | 3.3 |
| beam | yes | 1 | 0 | 16.0 | 6.6 | 6.1 | 2.8 | 1.72 | 6.5 | 2.7 |
| human_normal | yes | 9 | 64 | 30.5 | 7.1 | 21.5 | 0.7 | 2.13 | 4.2 | 8.2 |
| gamepad | yes | 1 | 0 | 51.3 | 7.2 | 27.3 | 15.8 | 1.90 | 2.2 | 2.5 |
| duct | yes | 2 | 8 | 58.3 | 17.5 | 26.5 | 10.0 | 4.14 | 4.3 | 11.8 |
| cooling_manifold | no | 8 | 56 | 34.2 | 13.9 | 19.7 | - | 3.10 | 5.4 | 3.6 |
| car | yes | 1 | 0 | 21.2 | 9.1 | 8.2 | 2.7 | 1.89 | 5.4 | 5.6 |
| socket_set | no | 17 | 136 | 22.0 | 4.5 | 17.2 | - | 1.22 | 3.3 | 1.5 |
| socket_set_small | yes | 3 | 16 | 4.9 | 2.7 | 1.8 | 0.2 | 0.59 | 7.3 | 1.3 |
| stool_circular | no | - | - | 28.6 | 27.6 | - | - | 4.99 | 10.5 | 3.1 |

loop = sequence optimization + layout + motion planning, all attempts added.

**6 of 10 assemblies get a verified plan, the same six as in pass 1.**

### Checks of the measurement (fact)

- CPU seconds of the stages add up to the total of `/usr/bin/time` (largest
  difference 0.2 s), stage durations to the wall time (largest difference
  0.1 s).
- No other planning or training process at any start; other programs used on
  average at most 0.65 of the 16 threads during a stage.

### Runs without a plan (fact)

- **plumbers_block, stool_circular:** as in pass 1, no move grasp for the
  bolts (parts 1, 4) and the legs (parts 1, 3, 5, 7).
- **socket_set:** again the 10 changes of a move grasp or spot used up, again
  mostly the straight move into a pickup blocked by the hold arm (`dual-arm
  after 0.25 cm`), 15 of 17 failures.
- **cooling_manifold:** this time other failures (part 6: the straight move
  into its pickup never gets clear of arm and part within 20 cm, 4x; part 4:
  singularity; part 6: joint limit; one hold arm path), and in attempt 8 the
  same crash as in pass 1 while drawing the PDF template
  (`shapely.errors.GEOSException: TopologyException: Ring edge missing`). The
  crash happened in both passes.

## Comparison with pass 1

| assembly | attempts p1 / p2 | total p1 / p2 [min] | difference | grasp generation p1 / p2 [min] | loop p1 / p2 [min] | final check p1 / p2 [min] |
|---|---|---|---|---|---|---|
| plumbers_block | - / - | 6.5 / 6.4 | -1 % | 6.03 / 5.99 | - | - |
| beam | 1 / 1 | 16.1 / 16.0 | -0 % | 6.74 / 6.64 | 6.1 / 6.1 | 2.76 / 2.77 |
| human_normal | 9 / 9 | 31.2 / 30.5 | -2 % | 7.21 / 7.15 | 22.2 / 21.5 | 0.68 / 0.71 |
| gamepad | 6 / 1 | 63.2 / 51.3 | -19 % | 7.12 / 7.23 | 40.6 / 27.3 | 14.75 / 15.85 |
| duct | 2 / 2 | 65.0 / 58.3 | -10 % | 17.52 / 17.50 | 33.0 / 26.5 | 9.63 / 10.01 |
| cooling_manifold | 7 / 8 | 74.9 / 34.2 | -54 % | 13.84 / 13.93 | 60.5 / 19.7 | - |
| car | 1 / 1 | 21.9 / 21.2 | -3 % | 8.92 / 9.12 | 9.2 / 8.2 | 2.70 / 2.74 |
| socket_set | 17 / 17 | 21.2 / 22.0 | +3 % | 4.49 / 4.50 | 16.5 / 17.2 | - |
| socket_set_small | 1 / 3 | 4.3 / 4.9 | +14 % | 2.69 / 2.72 | 1.2 / 1.8 | 0.25 / 0.24 |
| stool_circular | - / - | 28.8 / 28.6 | -1 % | 27.84 / 27.60 | - | - |

### Observations

Fact:

- **The pipeline is not deterministic.** With the same code, inputs and
  settings, the grasp generation reports the same number of candidates per
  part but different numbers of feasible or kept grasps for 5 of 10
  assemblies (e.g. duct part 0: 119 vs 104 feasible move grasps; human_normal
  part 3: 32 vs 31 kept). gamepad has identical grasp summaries in both passes
  and still needed 6 attempts in pass 1 and 1 in pass 2, so later stages vary
  too. Which random numbers are not seeded was not examined.
- **Where the course of the loop is the same** (same number of attempts:
  plumbers_block, beam, human_normal, car, socket_set, stool_circular), the
  total times of the two passes differ by 0 to 3 %.
- **The grasp generation time is reproducible** within 2.2 % (car), the
  final check within 7.5 % (gamepad, whose plans differ).

Interpretation:

- The spread of the planning time comes mostly from the course of the loop
  (how many attempts until a plan passes), not from measurement noise or
  machine load. A report should give the time per assembly together with the
  number of attempts, and treat the loop as a random process of the planner.
- With this spread, mean and standard deviation over three passes describe
  the loop-heavy assemblies (gamepad, duct, cooling_manifold) only roughly.

## Files

As in pass 1: `report.md`, `results.json`, `summary.csv`, `summary.md`,
`environment.json` here; raw data in
`logs/v0.3_exp_<assembly>_flat_prof2/<assembly>/profile/`. Evaluate again:

```bash
python profiling/profile_planning.py --summary-dir logs/v0.3_profile_2026-11-19_pass2 --resummarize
```
