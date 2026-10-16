# Assembly variants (different sequences / grasps)

`planning/select_variant.py` picks alternative assembly sequences and grasps from the planned tree (`tree.pkl`) and saves each one as its own **pseudo-assembly** `<assembly>__v<i>`. All later steps (fixture, motion plan, Isaac preprocessing, training, real robot) treat a variant like a normal assembly, so **no other code changes are needed**. One policy can be trained over all variants.

## How it works

- `tree.pkl` is the disassembly tree (`m X` = move arm pulls part X, `h Y` = part Y is held). Every green path is a feasible sequence, and each step has many valid grasps (the *grasp tree*).
- The default pipeline uses one optimized chain (`tree_opt.pkl`). `select_variant.py` runs the same optimizer on a filtered grasp tree and writes a different `tree_opt.pkl` for each variant.
- For each variant it creates:
  - `logs/<exp>/<assembly>__v<i>/` with copied planning files, the new `tree_opt.pkl/.png` and `variant.json` (order and grasp IDs)
  - the symlink `assets/fabrica/<assembly>__v<i> -> <assembly>` (same meshes)
- Variants differ by at least `--min-angle` (default 30°) in the move-grasp orientation of each part. `v0` is always the normal optimum, so it equals the existing plan.

## Workflow

Example: 5 variants of `socket_set_small`. The source must already be fully planned (`run_planning.sh`).

**Terminal 1: planning** (`fabrica` env)
```bash
cd ~/tobi/Fabrica
conda activate fabrica
python planning/select_variant.py --src logs/exp_socket_set_small/socket_set_small --dst-exp exp_socket_set_small_var --n-variants 3
python planning/select_variant.py --src logs/exp_socket_set_small/socket_set_small --dst-exp exp_socket_set_small_var --n-variants 2 --order 2,1 --start-index 3
for v in 0 1 2 3 4; do
  DISPLAY= python planning/run_fixture_gen.py --assembly-dir assets/fabrica/socket_set_small__v$v --log-dir logs/exp_socket_set_small_var/socket_set_small__v$v --optimized
  python planning/run_motion_plan.py --assembly-dir assets/fabrica/socket_set_small__v$v --log-dir logs/exp_socket_set_small_var/socket_set_small__v$v --optimized
done
```
Options for `select_variant.py`:
- `--order 2,1`: fixed assembly order (all parts except the base part). Without it, every variant uses the optimizer's preferred order. For `socket_set_small` both orders score exactly the same and the tie always goes to 1→2, so the other order must be requested explicitly.
- `--start-index 3`: continue numbering (`__v3`, `__v4`, ...) to add variants to an existing experiment. Use the same `--dst-exp`, otherwise `__v0` is overwritten or the names collide during training.
- `--move-grasps 1:687,2:527`: fixed grasp IDs per part (e.g. to keep the current fixture)
- `--min-angle 15`: lower this if fewer variants than requested are found (assemblies with many parts, e.g. `human_normal`)

**Terminal 1: Isaac preprocessing + training** (`isaacgym` env)
```bash
conda activate isaacgym
bash ./learning/preprocessing/prepare_isaac.sh exp_socket_set_small_var
cd learning/isaacgymenvs
python train.py task=FabricaFixPlugTaskAssemble "task.env.assemblies=[socket_set_small__v0,socket_set_small__v1,socket_set_small__v2,socket_set_small__v3,socket_set_small__v4]" experiment=exp_socket_set_small_var task.env.numEnvs=1024 max_iterations=1500 headless=True task.env.franka_friction=1
```
You can also mix variants of different assemblies in one list.

**Real robot**: start the control stack as described in `REAL_ROBOT_STARTUP.md`, then in Terminal B:
```bash
python real_robot/run.py --single-arm --residual --fn logs/exp_socket_set_small_var/socket_set_small__v1/motion.pkl --checkpoint-path learning/isaacgymenvs/runs/exp_socket_set_small_var_<date>/nn/exp_socket_set_small_var.pth
```
Use the `motion.pkl` of the variant whose fixture is mounted.

## What to consider

- **Fixtures**: each variant needs its own printed fixture: `logs/<exp>/<assembly>__v<i>/fixture/fixture.obj`. Only `v0` matches the existing one.
- **`DISPLAY=`** before `run_fixture_gen.py`: without it the `fixture.png` preview crashes with an X11 error (`xcb ... Aborting`) on this PC. The files are still saved, but the exit code is 134.
- **`prepare_isaac.sh` overwrites** `fabrica_pairs.yaml`, `plan_info/` and the Franka URDFs with the experiment you pass it. Prepare and start trainings one after another. Checkpoints and `run.py` don't depend on these files afterwards.
- **`numEnvs × 32` must be divisible by 512** (rl-games assertion). 1024 works.
- **Re-planning the source** (e.g. after changing the robot base height in `planning/robot/workcell.py`) invalidates its variants. Rerun `select_variant.py` and everything after it.
- **Base height**: the motion plans use `get_panda_base_z()` from `workcell.py`. It must match the real setup.
- The variant symlinks and generated Isaac files (`*__v*`) are in `.gitignore`. The `logs/` variant folders are tracked like the other logs.
- **For the VLA dataset**: record the variant name (`variant.json`) and a success flag with every episode.
