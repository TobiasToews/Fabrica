# Planning time profile, planner v0.3, pass 1 (2026-11-18)

How long the fixture-free ("flat") planning pipeline takes for each of the
ten Fabrica assemblies, stage by stage, and whether it finds a verified plan.
This is pass 1 of 3; passes 2 and 3 repeat it in
`logs/v0.3_profile_2026-11-19_pass2/` and `..._pass3/`.

The generated tables are in [report.md](report.md) (made from `results.json`,
can be made again at any time). This file adds the context and the analysis.

## Setup

| | |
|---|---|
| Planner | v0.3, commit `3d20bb6`, no uncommitted changes in `planning/` |
| Pipeline | `planning/run_planning_flat.py`, unchanged, default settings: `--num-proc 12` (precedence), `--num-proc-grasp 50` (grasp generation), `--max-n-grasp 100`; each stage runs with `OMP_NUM_THREADS=1` |
| Robot height | base 3.3 cm below the table (`FABRICA_PANDA_BASE_Z=-3.3`, the measured value) |
| Places | defaults: assembly at (0, -15) cm, parts laid out around (0, 5) cm |
| Order | plumbers_block (test of the profiler), then beam, human_normal, gamepad, duct, cooling_manifold, car, socket_set, socket_set_small, stool_circular |
| Runs | one at a time, never two in parallel; n = 1 per assembly |
| Machine | AMD Ryzen 7 3700X (8 cores, 16 threads), 15.6 GiB RAM, Ubuntu 20.04.6, kernel 5.15.179-rt84 (PREEMPT_RT); Python 3.10.0, numpy 1.21.4, scipy 1.7.3, trimesh 4.4.1, networkx 2.6.3 |
| Background | desktop session; the LeRobot bridge (`bridge/franka_bridge.py`, started 2026-11-15) was running. Other robot processes were not recorded; their load is included in "other load" |
| Time | 2026-11-18 18:46 to 2026-11-19 00:21 (5.6 h) |

Command:

```bash
conda activate fabrica
export FABRICA_PANDA_BASE_Z=-3.3
python -u profiling/profile_planning.py --assemblies <assemblies> \
    --summary-dir logs/v0.3_profile_2026-11-18
```

Logs of the plans: `logs/v0.3_exp_<assembly>_flat_prof1/<assembly>/`, raw
profiling data in its `profile/` subfolder.

## Method

`profiling/profile_planning.py` runs the pipeline as a subprocess and measures
from outside (details in the docstring of the script):

- **Wall time per stage.** The driver prints `[run_planning_flat] $ <command>`
  right before it starts a stage; every console line gets a timestamp from a
  monotonic clock. A stage lasts from its start line to the next one. The
  driver's own work between two stages (milliseconds to a few seconds) is
  counted to the stage before.
- **CPU time per stage.** Once per second, for every process of the pipeline,
  `utime + stime + cutime + cstime` from `/proc/<pid>/stat`. The sum over all
  live processes does not drop when a worker exits (its time moves to its
  parent). CPU seconds of a stage = sum at its end minus sum at its start,
  linearly interpolated between the neighbouring samples; at the end of the
  run the exact total of `/usr/bin/time` is used.
- **Memory.** Proportional set size (PSS, `/proc/<pid>/smaps_rollup`) summed
  over all processes: pages shared by forked workers are split between them,
  so the sum is the memory really occupied. Units: MiB / GiB.
- **Independent total.** The whole run is wrapped in `/usr/bin/time -v`.
- **Load from other programs.** Busy hardware threads of the machine
  (`/proc/stat`) minus the threads used by the pipeline.
- **Explaining quantities.** Parts, mesh faces, loop attempts, path re-plans,
  failures, result of the final check.

## Results

| assembly | plan | parts | faces | attempts | path re-plans | total [min] | grasp generation [min] | loop [min] | final check [min] | CPU [h] | mean cores | peak PSS [GiB] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beam | yes | 5 | 48 422 | 1 | 0 | 16.1 | 6.7 | 6.1 | 2.8 | 1.74 | 6.5 | 2.9 |
| human_normal | yes | 7 | 2 711 | 9 | 66 | 31.2 | 7.2 | 22.2 | 0.7 | 2.16 | 4.2 | 8.2 |
| gamepad | yes | 6 | 138 330 | 6 | 40 | 63.2 | 7.1 | 40.6 | 14.7 | 2.09 | 2.0 | 2.4 |
| duct | yes | 8 | 69 682 | 2 | 8 | 65.0 | 17.5 | 33.0 | 9.6 | 4.25 | 3.9 | 11.6 |
| cooling_manifold | no | 7 | 49 954 | 7 | 49 | 74.9 | 13.8 | 60.5 | - | 3.73 | 3.0 | 3.7 |
| car | yes | 6 | 44 394 | 1 | 0 | 21.9 | 8.9 | 9.2 | 2.7 | 1.89 | 5.2 | 5.5 |
| socket_set | no | 6 | 2 578 | 17 | 136 | 21.2 | 4.5 | 16.5 | - | 1.21 | 3.4 | 1.5 |
| socket_set_small | yes | 3 | 1 092 | 1 | 0 | 4.3 | 2.7 | 1.2 | 0.2 | 0.58 | 8.1 | 1.3 |
| plumbers_block | no | 5 | 6 488 | - | - | 6.5 | 6.0 | - | - | 1.19 | 11.0 | 3.3 |
| stool_circular | no | 9 | 79 220 | - | - | 28.8 | 27.8 | - | - | 5.06 | 10.5 | 3.2 |

loop = sequence optimization + layout + motion planning, all attempts added.
Per-stage times and cores: [report.md](report.md).

**6 of 10 assemblies get a verified plan.**

### Checks of the measurement (fact)

- In every run the CPU seconds of the stages add up to the total of
  `/usr/bin/time` (largest difference 0.2 s) and the stage durations add up to
  the wall time (largest difference 0.6 s).
- No other planning or training process was running at any start.
- Other programs used on average at most 0.7 of the 16 threads during a stage;
  exception: the grasp generation of stool_circular, 2.6 threads, so that one
  value is slightly too high.

### Runs without a plan (fact, from the logs)

- **plumbers_block, stool_circular:** no move grasp can pick the bolts (parts
  1, 4) or the legs (parts 1, 3, 5, 7) lying flat; the pipeline stops after
  the grasp generation. Same result as with v0.1 on 2026-11-12.
- **socket_set:** the loop used up its 10 changes of a move grasp or spot.
  16 of 17 failures: the straight move of the move arm into a pickup is
  blocked by the hold arm (`no straight move into the goal pose from a pose
  clear of everything (dual-arm after 0.25 cm)`); the loop changed spots and
  move grasps and only once a hold grasp.
- **cooling_manifold:** 5 failures of the same kind for part 2, one of a hold
  arm path; in attempt 7 the layout stage crashed while drawing the PDF
  template: `shapely.errors.GEOSException: TopologyException: Ring edge
  missing` in `save_pickup_template` (`planning/run_pickup_gen.py:277`). The
  layout itself had been found. This is a robustness bug, not a planning
  result.

### Observations (interpretation; n = 1, not statistically tested)

- **The loop drives the time, not the mesh size.** Where the first attempt
  works (beam, car, socket_set_small) planning takes 4 to 22 min. With many
  attempts (human_normal 9, gamepad 6) the loop takes 64 to 71 % of the time.
- **Motion planning, layout and the final check run on one core** (measured
  1.0 mean cores); grasp generation uses 9 to 14 cores, precedence planning 5
  to 10. Motion planning is the stage where parallelism would help most.
- **The final check grows with the mesh size:** gamepad (138 330 faces)
  14.7 min, duct (69 682) 9.6 min, human_normal (2 711) 0.7 min.
- **Memory:** duct peaks at 11.6 GiB of 15.6 GiB; two such runs would not fit
  side by side.
- **System CPU time** is 6 to 22 % of all CPU time (most for plumbers_block,
  where the grasp generation is nearly the whole run). Possibly the 50 grasp
  processes on 16 threads; not examined.
- **Compared with v0.1** (2026-11-12, 8 of 10 verified, socket_set and
  cooling_manifold included): v0.3 plans both arms as robots and checks the
  distance between them. Both new failures are dominated by the hold arm
  blocking the move arm. The comparison is confounded: that batch used 3.0 cm
  robot height and ran several plans in parallel.

### Limitations

- One run per assembly: no spread. Passes 2 and 3 repeat the measurement.
- The duration depends on the course of the loop (failed attempts), so it is
  partly a property of the data, not only of the code.
- Grasp generation starts 50 processes on 16 hardware threads
  (oversubscription); its time is a property of this setting.
- Background programs (desktop, LeRobot bridge) were running; their load is
  included in "other load".
- plumbers_block ran first as a test of the profiler, 8 min before the others,
  in a separate call (`environment.json` has one record per call).

## Files

| File | Content |
|---|---|
| `report.md` | generated tables: runs, wall time and cores per stage, checks, failures |
| `results.json` | everything measured, one entry per run |
| `summary.csv`, `summary.md` | one row per assembly, stage times in minutes |
| `environment.json` | machine, versions, commit, environment variables per call |
| `logs/v0.3_exp_<assembly>_flat_prof1/<assembly>/profile/` | raw data: `console.log` (timestamped output), `samples.csv` (1 Hz samples), `time_v.txt`, `profile.json` |

Evaluate again from the raw data (no new planning run):

```bash
python profiling/profile_planning.py --summary-dir logs/v0.3_profile_2026-11-18 --resummarize
```
