# Guide for coding agents

How to work in this repository. Read [CONTEXT.md](CONTEXT.md) first: it says what was changed, why, how far it
is verified and which decisions are still open.

This is a fork of [Fabrica](README.md) (official release: commit `215a30f`). Most current work is the
fixture-free ("flat") planning pipeline for the single real Franka Panda of the lab. How that pipeline works:
[FLAT_PIPELINE.md](FLAT_PIPELINE.md). How it differs from the original planner:
[FLAT_VS_ORIGINAL.md](FLAT_VS_ORIGINAL.md).

## Rules of the owner

The owner is a junior robotics engineer and reviews the work himself.

- **Check before you state.** Measure or read the code before you claim something; say what you did not verify.
  Write findings as *Fact* / *Interpretation* / *Not examined*, as in `assembly_runs/`. Never present a guessed
  cause as a fact.
- **Nothing on the real robot unless he asks.** Never start `real_robot/run.py`, `place_parts.py`,
  `record_state.py`, `adjust_arm.py` or anything that talks to the robot or to franka-interface on your own.
- **No large planner changes without his permission.** Propose first and start only after an explicit yes. A
  reply that does not answer the proposal is not a yes. This holds for anything that changes what the planner or
  the checks accept (values in `planning/config.py`, `plan_checks.py`, `motion_plan_arm.py`, the grasp
  generation).
- **Visible edits.** Change files with the editor so each diff can be reviewed, not with sed or patch scripts.
- **Commit and push only when asked.** Commit messages in English.
- **Language.** Talk to him in German and use "du". Code, comments, docs and commit messages are English.
- **Explain for a junior.** In code and in answers: purpose first, units and frames, robotics terms in one plain
  sentence.
- Do not kill processes; tell him the PID and let him decide.
- The recording dataset on Hugging Face is private and stays private.

## Environments

| Env | Python | Used for | Keep |
|---|---|---|---|
| `fabrica` | 3.10 | planning, tests, ruff | numpy 1.21.4, scipy 1.7.3, networkx 2.6.3 (numpy 2 breaks matplotlib 3.5.1 and the planning code). Run `pip install --dry-run` before any install. |
| `isaacgym` | 3.8 | `learning/preprocessing/prepare_isaac.sh`, training, `real_robot/run.py` | Isaac Gym Preview 4 only ships bindings up to 3.8. torch 1.13.1+cu117, rl-games 1.6.1 (1.6.5 requires torch >= 2.2.2). |
| `lerobot` | | recording data with `lerobot-record` | Upload a recording only after `lerobot-record` has exited (the parquet footer and `meta/episodes/` are written at the end). |

- For anything that imports frankapy: `source /opt/ros/noetic/setup.bash`, then
  `source ~/tobi/frankapy/catkin_ws/devel/setup.bash` (in this order).
- `cmake<4` is installed in the envs; the system cmake 4.2 breaks the builds of `redmax_py` and catkin.
- Before planning, clear test overrides of the workcell:
  `unset FABRICA_PANDA_BASE_Z FABRICA_ASSEMBLY_CENTER FABRICA_PICKUP_CENTER`.

## Code map of the flat pipeline

| File | Role |
|---|---|
| `planning/run_planning_flat.py` (`.sh`) | the pipeline: runs the stages as subprocesses; the loop that reacts to the cause of a failure |
| `planning/run_grasp_arm_gen.py --flat` | grasps that pick the lying part and insert it; without `--flat` the original generator |
| `planning/flat_grasp.py` | resting poses, flat-pickup test, finger contact on the pad grid, grasp criteria, rating |
| `planning/run_pickup_gen.py` | grasp filter, sequence optimization, layout of the lying parts, 1:1 templates (replaces fixture generation) |
| `planning/run_motion_plan.py` | motion planning; every path of the real (move) arm is checked and re-planned; with `--single-arm` a hold-arm path that cannot be planned is stored as a jump |
| `planning/robot/motion_plan_arm.py` | the planner; `ArmMotionPlanner.strict` is the mode for the real arm |
| `planning/plan_checks.py`, `planning/verify_plan.py` | the rules every pose of the real arm must pass; the final check |
| `planning/config.py` | all limits and their meaning |
| `planning/robot/workcell.py` | robot height, places of the assembly and of the parts, stored per log |
| `planning/robot/panda_condition.py` | Jacobian, singularity and joint-limit distance of the Panda |
| `planning/run_workspace_search.py`, `run_position_test.py` | rate places of the assembly; run the pipeline at many places |
| `real_robot/run.py`, `robot_interface.py` | execution on the robot (`--single-arm`: a human is the hold arm) |

## Running

```bash
conda activate fabrica
bash planning/run_planning_flat.sh <exp_name> <assembly>        # output: logs/<exp_name>/<assembly>/
python planning/run_planning_flat.py --exp-name <exp> --assembly <assembly> --assembly-center X Y --pickup-center X Y
python planning/run_position_test.py --exp-name <exp> --assembly <assembly> --precedence-from <log>
```

A plan is tied to the robot height in `planning/robot/workcell.py` (`get_panda_base_z()`): a new height needs a
new `<exp_name>`. `real_robot/run.py` refuses plans made for another height or without a passed final check.

## Tests and checks

```bash
conda activate fabrica
python planning/test_plan_rules.py         # 39 tests, about 25 s
python planning/test_workspace_search.py   #  9 tests, about 45 s
python planning/test_flat_grasp.py         # 42 tests, about 80 s
ruff check && ruff format --check          # style of the files listed in ruff.toml
```

`ruff check` still reports findings: the restyle is not finished (see CONTEXT.md).

- Run the three test files after every change in `planning/`. Add a test for every new rule or behaviour; a toy
  scene with known dimensions is the pattern (see the existing tests).
- After a change of the planner, the grasps or the checks, verify on several assemblies and include the hard ones
  (beam, human_normal, gamepad, duct, cooling_manifold, car). socket_set_small is only a smoke test. Report per
  assembly, hardest first, and name the ones not run. Write the results into `assembly_runs/`.
- Do not edit `planning/*.py` while a pipeline or a batch runs: every stage is a new process that imports the
  code again. Develop in a scratch copy meanwhile.
- Memory: the machine has 15.9 GB. A motion plan of gamepad needs about 1.4 GB, so at most 6 of them in
  parallel. The grasp generation uses all cores. Run plans one after another unless you checked the memory.

## Code style

- PEP 8: code lines at most 79 characters, comments and docstrings at most 72. No statement on the line of an
  `if`/`for`, no `;`.
- Google-style docstrings with `"""`: summary line, then `Args:`, `Returns:`, `Raises:`. State units (cm, m, rad,
  deg), frames and array shapes. Every sentence must follow from the code or from the earlier documentation.
- Code of the original Fabrica release keeps its own style: in those files only our own functions and lines are
  restyled. Never run `ruff format` on a file that is not listed in `ruff.toml`.
- No type hints unless the owner asks for them.
- The restyle of the existing code is not finished; see CONTEXT.md.

## Git

- `main` tracks `origin` (`TobiasToews/Fabrica`). Experiment logs under `logs/` are several GB and mostly
  untracked: commit a log only when the owner asks. GitHub refuses files over 100 MB (see `.gitignore`).
- `.claude/worktrees/` holds two git worktrees (`gamepad-flat-planning`, `sequence-studio`); do not commit
  `.claude/`.

## Documentation map

| File | Content |
|---|---|
| [CONTEXT.md](CONTEXT.md) | state of the work, history, open decisions |
| [FLAT_PIPELINE.md](FLAT_PIPELINE.md) | the flat pipeline: use, stages, grasps, checks, limits |
| [FLAT_VS_ORIGINAL.md](FLAT_VS_ORIGINAL.md) | differences to the original planner (German) |
| [assembly_runs/](assembly_runs/README.md) | dated run notes: batches on all assemblies, position tests, per-assembly files |
| [REAL_ROBOT_STARTUP.md](REAL_ROBOT_STARTUP.md) | starting the control stack and the robot |
| [RELIABILITY_VLA_PLAN.md](RELIABILITY_VLA_PLAN.md) | proposal for reliability and VLA data collection, not implemented |
| [INSTALL_GUIDE.md](INSTALL_GUIDE.md) | build problems of the simulation and their fixes |
| [USE_WITH_OWN_ASSEMBLY.md](USE_WITH_OWN_ASSEMBLY.md), [ASSEMBLY_VARIANTS.md](ASSEMBLY_VARIANTS.md) | own assemblies; variants of sequences and grasps |

Keep `FLAT_PIPELINE.md` in sync with the code: a change of a stage, a rule or a value goes into it in the same
change.
