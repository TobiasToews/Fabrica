# Planning time profile, planner v0.3, pass 3 (2026-11-19) and all three passes

Third repetition, under the same conditions as pass 1
([../v0.3_profile_2026-11-18/README.md](../v0.3_profile_2026-11-18/README.md),
method described there) and pass 2
([../v0.3_profile_2026-11-19_pass2/README.md](../v0.3_profile_2026-11-19_pass2/README.md)).
Generated tables of this pass: [report.md](report.md). The statistics over all
three passes are in [three_passes.csv](three_passes.csv) and below.

## Setup

Identical to passes 1 and 2: planner v0.3 (commit `3d20bb6`, no uncommitted
changes in `planning/`), `FABRICA_PANDA_BASE_Z=-3.3`, default places (assembly
at (0, -15) cm, parts around (0, 5) cm), default pipeline settings, same
machine and software, one run at a time, same order, one call.

- Time: 2026-11-19 16:12 to 21:36 (5.4 h).
- Logs of the plans: `logs/v0.3_exp_<assembly>_flat_prof3/<assembly>/`, raw
  profiling data in `profile/`.
- The profiler process of this pass still had the first evaluation loaded;
  the results were evaluated again with the corrected evaluation
  (`--resummarize`, see pass 2) before this report.

```bash
export FABRICA_PANDA_BASE_Z=-3.3
python -u profiling/profile_planning.py --exp-suffix flat_prof3 \
    --assemblies plumbers_block beam human_normal gamepad duct cooling_manifold car socket_set socket_set_small stool_circular \
    --summary-dir logs/v0.3_profile_2026-11-19_pass3
python profiling/profile_planning.py --summary-dir logs/v0.3_profile_2026-11-19_pass3 --resummarize
```

## Results of pass 3

| assembly | plan | attempts | path re-plans | total [min] | grasp generation [min] | loop [min] | final check [min] | CPU [h] | mean cores | peak PSS [GiB] |
|---|---|---|---|---|---|---|---|---|---|---|
| plumbers_block | no | - | - | 6.5 | 6.1 | - | - | 1.18 | 10.8 | 3.3 |
| beam | yes | 1 | 0 | 17.1 | 6.8 | 7.5 | 2.3 | 1.76 | 6.2 | 2.8 |
| human_normal | yes | 4 | 17 | 20.9 | 7.3 | 11.8 | 0.7 | 2.00 | 5.7 | 8.2 |
| gamepad | yes | 4 | 25 | 64.2 | 7.2 | 40.8 | 15.4 | 2.11 | 2.0 | 2.4 |
| duct | yes | 1 | 0 | 56.7 | 17.5 | 25.2 | 9.2 | 4.12 | 4.4 | 12.3 |
| cooling_manifold | no | 13 | 106 | 78.5 | 13.9 | 64.1 | - | 3.82 | 2.9 | 3.7 |
| car | yes | 1 | 0 | 21.0 | 9.1 | 7.9 | 2.9 | 1.90 | 5.4 | 5.6 |
| socket_set | no | 18 | 144 | 24.4 | 4.5 | 19.7 | - | 1.26 | 3.1 | 1.5 |
| socket_set_small | yes | 3 | 16 | 4.8 | 2.7 | 1.7 | 0.2 | 0.59 | 7.4 | 1.3 |
| stool_circular | no | - | - | 28.7 | 27.7 | - | - | 5.00 | 10.5 | 3.1 |

**6 of 10 verified, the same six as in passes 1 and 2.**

Checks (fact): stage CPU seconds add up to `/usr/bin/time` within 0.23 s,
stage durations to the wall time within 0.6 s; no other planning or training
process at any start; other programs used at most 0.67 of 16 threads.

Runs without a plan (fact):

- **plumbers_block, stool_circular:** as before, no move grasp for the bolts
  and the legs.
- **socket_set:** again the 10 changes used up; 13 of 18 failures are the
  straight move into a pickup blocked by the hold arm (`dual-arm after
  0.25 cm`), 2 the move arm touching the hold arm, 3 no path for the hold arm.
- **cooling_manifold:** this time no crash; the loop used up its 10 changes.
  Failures: the straight move into the pickup of part 3 or 6 never gets clear
  of arm and part within 20 cm (7x), part 2 blocked by the hold arm (4x), one
  singularity, one hold arm path. The template crash of passes 1 and 2 did not
  occur because no layout reached that point.

## All three passes

n = 3 per assembly; mean ± sample standard deviation (n - 1 in the
denominator); CV = coefficient of variation (SD / mean). Times in minutes.

| assembly | plan (of 3) | attempts p1 / p2 / p3 | total, mean ± SD | min - max | CV | grasp generation | loop | final check | CPU [h] | peak PSS max [GiB] |
|---|---|---|---|---|---|---|---|---|---|---|
| beam | 3 | 1 / 1 / 1 | 16.4 ± 0.6 | 16.0 - 17.1 | 4 % | 6.73 ± 0.08 | 6.5 ± 0.8 | 2.61 ± 0.27 | 1.74 ± 0.02 | 2.9 |
| human_normal | 3 | 9 / 9 / 4 | 27.5 ± 5.8 | 20.9 - 31.2 | 21 % | 7.20 ± 0.06 | 18.5 ± 5.8 | 0.69 ± 0.02 | 2.10 ± 0.09 | 8.2 |
| gamepad | 3 | 6 / 1 / 4 | 59.6 ± 7.2 | 51.3 - 64.2 | 12 % | 7.18 ± 0.05 | 36.2 ± 7.7 | 15.32 ± 0.55 | 2.03 ± 0.12 | 2.5 |
| duct | 3 | 2 / 2 / 1 | 60.0 ± 4.4 | 56.7 - 65.0 | 7 % | 17.51 ± 0.01 | 28.2 ± 4.2 | 9.62 ± 0.40 | 4.17 ± 0.07 | 12.3 |
| car | 3 | 1 / 1 / 1 | 21.4 ± 0.5 | 21.0 - 21.9 | 2 % | 9.05 ± 0.11 | 8.4 ± 0.7 | 2.78 ± 0.11 | 1.89 ± 0.00 | 5.6 |
| socket_set_small | 3 | 1 / 3 / 3 | 4.6 ± 0.3 | 4.3 - 4.9 | 7 % | 2.71 ± 0.02 | 1.5 ± 0.3 | 0.24 ± 0.01 | 0.59 ± 0.01 | 1.3 |
| cooling_manifold | 0 | 7 / 8 / 13 | 62.6 ± 24.6 | 34.2 - 78.5 | 39 % | 13.88 ± 0.05 | 48.1 ± 24.7 | - | 3.55 ± 0.39 | 3.7 |
| socket_set | 0 | 17 / 17 / 18 | 22.5 ± 1.7 | 21.2 - 24.4 | 7 % | 4.49 ± 0.01 | 17.8 ± 1.7 | - | 1.23 ± 0.02 | 1.5 |
| plumbers_block | 0 | - | 6.5 ± 0.1 | 6.4 - 6.5 | 1 % | 6.04 ± 0.05 | - | - | 1.18 ± 0.01 | 3.3 |
| stool_circular | 0 | - | 28.7 ± 0.1 | 28.6 - 28.8 | 0 % | 27.70 ± 0.13 | - | - | 5.02 ± 0.04 | 3.2 |

For the four assemblies without a plan the time is the time until the
pipeline gives up, not a planning time.

### Findings (fact)

- **Success is stable:** the same 6 of 10 assemblies get a verified plan in
  all three passes (beam, human_normal, gamepad, duct, car,
  socket_set_small); the same 4 never do.
- **Grasp generation time is reproducible:** SD at most 0.13 min (CV at most
  1.2 %), although its results differ slightly between passes for 6 of 10
  assemblies (e.g. duct part 0: 119 vs 104 feasible move grasps in passes 1
  and 2). Only beam, gamepad, socket_set and socket_set_small have identical
  grasp summaries in all three passes.
- **The loop is where the time varies:** the number of attempts changes
  between passes (human_normal 9 / 9 / 4, gamepad 6 / 1 / 4, cooling_manifold
  7 / 8 / 13), and with it the loop time (SD up to 24.7 min). Where the number
  of attempts was the same in all passes (beam, car), the total time varies by
  2 to 4 %.
- **Parallelism:** grasp generation uses 9 to 14 cores on average, precedence
  planning 4 to 10, every later stage 1.0 (one process).
- **Final check** grows with the mesh size: gamepad (138 330 faces)
  15.3 ± 0.6 min, duct (69 682) 9.6 ± 0.4 min, human_normal (2 711)
  0.69 ± 0.02 min.
- **Memory:** duct peaks at 11.6 to 12.3 GiB (of 15.6 GiB), human_normal at
  8.2 GiB, all others below 6 GiB.
- **Measurement quality:** in all 30 runs the stage CPU times add up to the
  independent total of `/usr/bin/time` within 0.23 s and the stage durations
  to the wall time within 0.6 s; other programs used at most 0.68 of 16
  threads during any stage.

### Interpretation

- The planning time of an assembly is mainly determined by the number of
  loop attempts, which is random from run to run (the planner is not
  deterministic). For the report: give the time with the number of attempts,
  or the time per attempt; report the grasp generation separately, it is
  stable.
- The stages after the grasp generation run on one core; parallel motion
  planning (or planning the paths of different attempts in parallel) is the
  largest lever for a shorter planning time.

### Limitations

- n = 3: the standard deviations are rough estimates (with 2 degrees of
  freedom), and the loop-heavy assemblies would need more runs for a reliable
  mean.
- The order of the assemblies was the same in every pass; effects of the time
  of day (pass 1 evening and night, passes 2 and 3 during the day) cannot be
  separated from effects of the pass.
- The sources of randomness in the planner were not identified.
- Default settings only (50 grasp processes on 16 threads, robot height
  3.3 cm, default places); other settings or places can change both times and
  results.

## Files

| File | Content |
|---|---|
| `report.md`, `results.json`, `summary.csv`, `summary.md`, `environment.json` | this pass, as in the other passes |
| `three_passes.csv` | per assembly: plans (of 3), attempts per pass, total time per pass, mean and SD of total, grasp generation, loop, final check, CPU time, peak PSS |
| `logs/v0.3_exp_<assembly>_flat_prof3/<assembly>/profile/` | raw data of this pass |
