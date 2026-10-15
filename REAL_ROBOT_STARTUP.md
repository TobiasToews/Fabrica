# Real robot startup (ICS Panda PC)

Robot IP: `192.168.3.1` · Desk: https://192.168.3.1

## 0. Robot (browser + hardware)

1. Open https://192.168.3.1 → unlock joints.
2. Menu (top right) → **Activate FCI**.
3. Release the external stop button → robot lights must be **blue**.

## 1. Terminal A: control stack (leave open)

```bash
cd ~/tobi/frankapy
bash ./bash_scripts/start_control_pc.sh -i localhost -d tobi/franka-interface -a 192.168.3.1
```

This opens 4 windows (roscore, franka_interface, ROS interface, gripper). **Do not close them.**
The franka_interface window must end at `Will run..` without an error.

## 2. Terminal B: run the robot

```bash
cd ~/tobi/Fabrica
conda activate isaacgym
source /opt/ros/noetic/setup.bash
source ~/tobi/frankapy/catkin_ws/devel/setup.bash
python real_robot/run.py --single-arm --residual --fn logs/exp_socket_set_small/socket_set_small/motion.pkl --checkpoint-path learning/isaacgymenvs/runs/_own_assemblies/exp_socket_set_small_socket_set_small_06-14-03-11/nn/exp_socket_set_small_socket_set_small.pth
```

Quick test (arm goes to home pose, hand on stop button):

```bash
python ~/tobi/frankapy/scripts/reset_arm.py
```

Run the assembly:

```bash
python real_robot/run.py --single-arm --residual --fn logs/exp_socket_set_small/socket_set_small/motion.pkl --checkpoint-path learning/isaacgymenvs/runs/_own_assemblies/exp_socket_set_small_socket_set_small_06-14-03-11/nn/exp_socket_set_small_socket_set_small.pth
```

- `Ready?` → Enter. `[Human hold arm] ...` → do it by hand, then Enter. `Repeat? (y/n)` after insertion.
- Resume after an abort: add `--start-from <step>`.

## TODO first: re-plan for the 32 mm base offset

`planning/robot/workcell.py` now has the robot base 3.2 cm below the board (`get_panda_base_z()`).
The current `motion.pkl` is still from the old height, so re-plan before running on the robot:

```bash
cd ~/tobi/Fabrica
conda activate fabrica
bash ./planning/run_planning.sh exp_socket_set_small socket_set_small
conda activate isaacgym
bash ./learning/preprocessing/prepare_isaac.sh exp_socket_set_small
```

Afterwards, ask Claude to re-check the new `motion.pkl` against the franka-interface virtual walls.

## If something goes wrong

| Symptom | Fix |
|---|---|
| `Unable to register with master node` | roscore window was closed → Terminal A again, with `-s 0` added if franka_interface is still running |
| `FrankaInterface status not ready` / franka_interface window shows `"User stopped"` | Release stop button (lights blue), then in the franka_interface window: `./franka_interface --robot_ip 192.168.3.1 --with_gripper 0 --log 0 --stop_on_error 0` |
| Only the gripper moves, arm stays | Same as above: robot is in "User stopped" |
| `Robot is in collision with virtual walls!` | Pose/plan outside the walls in `~/tobi/franka-interface/franka-interface/include/franka-interface/termination_handler/termination_handler.h` (sides ±0.65 m, floor −0.05 m) |
| `Connection refused` | FCI not activated in Desk |
| `No module named 'isaacgym'` | Isaac Gym folder must stay at `~/tobi/isaacgym` |
| `No module named 'frankapy'` / `franka_interface_msgs` not found | Wrong env or source order: `conda activate isaacgym`, then source `/opt/ros/noetic` **before** `catkin_ws/devel` |

## Training (simulation, optional)

```bash
cd ~/tobi/Fabrica/learning/isaacgymenvs
conda activate isaacgym
python train.py task=FabricaTaskAssemble task.env.assemblies=["socket_set_small"] headless=True
```

Don't Ctrl+C while SDFs are generating on first run. If it then segfaults: `find ~/.isaacgym -name "*.sdf" -size 0 -delete`.
