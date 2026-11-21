# Gamepad: real-robot replication of Fabrica's protocol (R3), started 2026-11-21

Working notes of the real-robot runs that replicate Fabrica's real-world evaluation (paper Sec. 5.4, App. D.4, E.2)
on the gamepad with the fixture-free pipeline. Statements are marked as in the other files of this folder:
**Fact** (read from a log or file, source named), **Interpretation** (a conclusion, not verified unless stated),
**Not examined**.

## Setup

| Item | Value |
|---|---|
| Plan | `logs/v0.3_exp_gamepad_flat_z33_run1/gamepad/` (planner v0.3, commit `3d20bb6`), robot base 3.3 cm below the table, assembly at (0, -20) cm, parts around (0, 10) cm; final check passed, 24 of 24 paths, no workaround |
| Insertion order of the move arm | parts 4, 2, 5, 0, 1 (steps 14, 21, 28, 35, 42); part 3 is held by the human (hold arm) |
| Policy | assembly specialist `learning/isaacgymenvs/runs/v0.3_exp_gamepad_flat_z33_run1_gamepad_1500it_17-19-42-15/nn/v0.3_exp_gamepad_flat_z33_run1_gamepad_1500it.pth` (trained on this plan, 1024 environments, 1500 iterations, seed 42, checkpoint with the best training reward) |
| Execution | `real_robot/run.py --single-arm --residual`, stochastic policy (`real_robot/config.yaml`: `deterministic: False`), trajectory streaming `--dt 0.03333` (default) |
| Control stack | franka-interface branch `fabrica-ics` (commit `4e8c68d`, impedance torque limit 0.5, wider virtual walls); robot on the USB Ethernet adapter `enx00e04c011267` (driver r8152), 192.168.3.10 -> 192.168.3.1 |
| Recording | LeRobot, front camera only, one dataset per run, `tobi/gamepad_flat_v03_z33_1500it_R3_<date>_<time>` |

Commands:

```bash
# recording (lerobot env, ~/tobi/lerobot_franka)
export CAMERAS="{front: {type: intelrealsense, serial_number_or_name: '134222070843', width: 640, height: 480, fps: 30}}"
DISPLAY= lerobot-record --robot.type=franka --robot.mode=observe --robot.cameras="$CAMERAS" --teleop.type=fabrica \
  --dataset.repo_id=tobi/gamepad_flat_v03_z33_1500it_R3_$(date +%Y%m%d_%H%M%S) --dataset.single_task="assemble gamepad" \
  --dataset.episode_time_s=900 --dataset.reset_time_s=60 --dataset.num_episodes=1 --dataset.push_to_hub=false

# execution (isaacgym env, ROS sourced, ~/tobi/Fabrica)
FABRICA_PANDA_BASE_Z=-3.3 python real_robot/run.py --single-arm --residual \
  --fn logs/v0.3_exp_gamepad_flat_z33_run1/gamepad/motion.pkl \
  --checkpoint-path learning/isaacgymenvs/runs/v0.3_exp_gamepad_flat_z33_run1_gamepad_1500it_17-19-42-15/nn/v0.3_exp_gamepad_flat_z33_run1_gamepad_1500it.pth
```

A recording is complete only after `lerobot-record` has exited on its own: end the episode with the right arrow key
(or `n`) in its terminal and wait. Ctrl+C during the episode discards it (`save_episode` is never called,
`lerobot/scripts/lerobot_record.py`).

## Protocol (R3 text of the report, agreed 2026-11-21)

> R3: Fabrica's real-world protocol on the gamepad: three complete runs with an assembly specialist retrained on the
> executed plan, stochastic policy, state-based success detection, up to three trials per step, no vision-language
> model. As in the paper (App. D.4), an intervention is counted when an insertion fails in three consecutive trials;
> the part is then inserted by hand and the run continues. In addition, by our own rule (not stated in the paper),
> re-seating a displaced part and re-placing a dropped part count as interventions. If a pickup leaves the gripper
> empty, the step counts as failed and placing the part by hand as an intervention. A trial ends with the success flag
> or after a 10 s timeout, whereas the paper detects failures from joint deviation and stagnation. Not counted as
> interventions are the hold-arm actions requested from the human and moving the held part back into its original
> position after an insertion, which reproduces the stiff hold of the robot hold arm. All human actions are logged with
> their cause (R4).

Notes on the protocol:

- **Fact:** `run.py` does not limit the number of trials; it asks `Repeat? (y/n)` after every trial. The operator
  answers `n` after the third failed trial.
- **Fact:** the success flag compares the gripper position with the planned goal along the insertion direction
  (`real_robot/robot_interface.py`, `get_rl_obs`); it does not know where the held part really is, and it cannot tell
  whether the gripper holds a part.
- **Fact:** the human holds part 3 at its left end, where the hold gripper grasps in the plan (fingers at x -10 to
  -2 cm, y -29 to -22 cm at the end of step 8). The hold arm does not move between step 8 and step 43
  (`motion.pkl`). The paper also reports parts shifting during insertion (Sec. 5.4, failure d) and assumes a stably held
  base part only for its 3 mm simulation noise.

Comparison values from the paper (gamepad, AS): 87 % successful steps (Table 4), cumulative 33 / 100 / 100 % with
0 / 1 / 2 interventions (Table 5); App. E.2, Table 3 gives 73 % for the same policy type with three trials.

## Runs of 2026-11-21

Five recorded R3 runs. Runs 1, 2 and 5 are the three counted runs (operator's decision, 2026-11-21); runs 3 and 4
failed and are documented with their cause. All five datasets are on the Hugging Face Hub (private, account
`Tobias2503`, repository name = folder name in `~/.cache/huggingface/lerobot/tobi/`), each with a dataset card that
states the run number, the result and, for runs 3 and 4, the cause of the failure.

**Fact**, from `logs/v0.3_exp_gamepad_flat_z33_run1/gamepad/run_logs/insertions_<date>_<time>.jsonl` (one line per
trial: S = success flag, F = 10 s timeout), the ROS logs and the dataset folders:

| R3 run | counted | recording (dataset) | run.py | insertion trials, steps 14 / 21 / 28 / 35 / 42 (parts 4 / 2 / 5 / 0 / 1) | how the run ended |
|---|---|---|---|---|---|
| 1 | yes | `gamepad_flat_v03_z33_1500it_R3_20261121_172259_20261121_172302`, 1 episode, 10 880 frames | 17:23 | S / S / S / FSS / FFF | all five steps |
| 2 | yes | `gamepad_flat_v03_z33_1500it_R3_20261121_173833_20261121_173839`, 1 episode, 10 088 frames | 17:38 | S / F / S / FSS / FFF | all five steps |
| 3 | no | `gamepad_flat_v03_z33_1500it_R3_20261121_174807_20261121_174811`, **no episode** (metadata only) | 17:48 | S / S / S / - / - | robot error state in step 32, then link loss (see below) |
| 4 | no | `gamepad_flat_v03_z33_1500it_R3_20261121_180018_20261121_180022`, **no episode** (metadata only) | 18:00 | S (invalid) / - / - / - / - | two reflexes in step 13, insertion from a wrong start (see below) |
| 5 | yes | `gamepad_flat_v03_z33_1500it_R3_20261121_185757_20261121_185800`, 1 episode, 10 636 frames | 18:58 | S / S / S / S / FFF | all five steps |

Other attempts of the day, not R3 runs (fact):

- 15:51 and 16:10: test runs without recording (15:51: step 14 only, 7 trials, all S; 16:10: S / FF / S / S / F).
- 17:56: attempt without recording between runs 3 and 4; franka-interface reported `status not ready` from its first
  skill on, and `run.py` crashed at the end of step 13 (`FrankaArmCommException: FrankaInterface status not ready for
  10s`).
- Recordings `…_R3_20261121_172148_…` and `…_R3_20261121_180003_…`: started seconds before the next recording, no
  episode; not uploaded.

Not decided yet (operator): the interventions per counted run and the human actions (R4 log). Points to check against
the protocol:

- **Fact:** in runs 1 and 2 step 35 was tried again after a reported success (trials 2 and 3 both S). The protocol
  stops at the first success.
- **Fact:** in run 2 step 21 has a single failed trial and no second one.
- **Fact:** step 42 (part 1) failed in three trials in all three counted runs, which is one intervention per run by the
  protocol.

### Run 3: cause of the failure (fact)

Insertions 14, 21 and 28 succeeded (last trial 17:50:40). The arm then moved on: step 30 (1.1 s path) completed, and
2 s into step 32 (a 6.0 s move-arm path to pick part 0, at most 0.24 rad/s per joint) franka-interface reported
`franka_interface status is not ready` (17:50:55, ROS log `~/.ros/log/e2e950b6-…/rosout.log`). The next two motions
(the end pose of step 32 and the 8.1 s transport of step 34) were terminated with the same message at 17:51:03 and
17:51:11. From 17:53:13 to 17:54:54 the connection to the robot timed out (99x `libfranka: UDP receive: Timeout`). The
recording of this run has no episode. Why franka-interface entered the error state is not in the saved logs (its
console output of that time was not kept); the path of step 32 is slow, so the wrist rotation (see below) is not the
cause here.

### Run 4: cause of the failure (fact)

After restarting the control stack (17:59), step 13 (move arm carries part 4 to its insertion start; joint 7 turns
4.0 rad) was aborted by the robot at 18:01:04 with `motion aborted by reflex! ["joint_velocity_violation"]`. The
following `goto_joints(…, duration=1)` to the end pose of step 13 (`real_robot/run.py:259`) was aborted with
`["tau_J_range_violation"]`. The insertion of part 4 then started at (0.276, -0.399, 0.053) m instead of the planned
insertion start (0.348, -0.450, 0.053) m, 8.7 cm away; the policy moved the part 11 cm and reported success, which is
not valid. The run was stopped. The recording has no episode.

### Run 5 (fact)

Run after another restart of the control stack. All five insertion steps reached; step 42 failed in three trials.
franka-interface reported `status not ready` at 18:59:13, 19:00:00 and 19:00:05 (ROS log `~/.ros/log/bd8d690c-…`)
without stopping the run. The kernel log shows the USB Ethernet adapter of the robot link losing and regaining its
carrier at about 18:54 (`r8152 … enx00e04c011267: carrier off` / `carrier on`), before the run.

## Incident 17:48 to 18:01 (runs 3 and 4): timeline and analysis

### What the logs show (fact)

| Time | Source | Event |
|---|---|---|
| about 17:06, 17:11, 17:13 | kernel log (`dmesg`, about 40 s added to its timestamps to match `journalctl`) | `irq/54-xhci_hcd: page allocation failure: order:3 … alloc_rx_agg … r8152_poll [r8152]`: the driver of the USB Ethernet adapter of the robot link could not allocate receive buffers |
| 17:16:05-17:16:07 | ROS log `~/.ros/log/00192f98-…/rosout.log` | 3x `GripperServer: … libfranka: UDP receive: Timeout` |
| 17:23, 17:38 | insertion logs, recordings | two runs that reach all five insertion steps |
| 17:50:40 | insertion log | last trial of run 3 (step 28) |
| 17:50:55-17:51:11 | ROS log `e2e950b6-…` | run 3: `franka_interface status is not ready` 2 s into step 32 and for the next two motions |
| 17:53:13-17:54:54 | ROS log `~/.ros/log/e2e950b6-…/rosout.log` | 99x `UDP receive: Timeout`: no data from the robot for about 1.7 min |
| 17:55:51-17:56:46 | ROS log `4da7afba-…` | after a restart, `franka_interface status is not ready` from the first skill on; the run crashes at the end of step 13 |
| 18:01:04 | franka-interface console (pasted by the operator) | step 13, streamed joint-impedance path: `motion aborted by reflex! ["joint_velocity_violation"]`, automatic error recovery |
| 18:01:17 | same | the following `goto_joints(…, duration=1)` (`real_robot/run.py:259`) to the end pose of step 13: `motion aborted by reflex! ["tau_J_range_violation"]`, automatic error recovery |
| 18:01:18-18:01:21 | insertion log | step 14 starts at (0.276, -0.399, 0.053) m instead of the planned insertion start (0.348, -0.450, 0.053) m used by every earlier run: 8.7 cm away. The policy moves the part 11 cm and reports "near goal" |
| 17:09-18:07 | kernel log | 6x `BUG: scheduling while atomic: irq/69-s-nvidia` (NVIDIA driver on the PREEMPT_RT kernel), none at the time of the reflexes |

franka-interface reported a control command success rate of 1.00 and 0.99 for the two aborted skills at 18:01.
After the incident the robot answers ping without loss (0.38 ms). The plan and the policy were not changed that day;
`real_robot/` and `planning/` are unchanged since commit `3d20bb6` (2026-11-13, no uncommitted changes), and the
franka-interface binary in `~/tobi/franka-interface/build/` is dated 2026-11-06.

### The wrist rotation in step 13 (fact)

- Plan: in step 13 (move arm carries part 4 to its insertion start) joint 7 turns from -1.77 to +2.26 rad (4.0 rad) in
  12.5 s, at most 1.29 rad/s at `--dt 0.03333` (49 % of its 2.61 rad/s limit). Steps 25 (4.3 rad), 27 (3.6 rad) and 34
  (2.7 rad) have similar rotations; every other path stays below 1 rad/s.
- Recordings of 17:23 and 17:38 (`observation.state` vs. `action`, 30 Hz): during these rotations joint 7 lags up to
  0.33-0.38 rad behind its command and then catches up at up to 2.09 rad/s (80 % of the limit), most in step 25. No
  oscillation is visible at 30 Hz (no sign changes of the joint velocity above 0.05 rad/s).

### Interpretation

- The trigger came from outside the plan and the code: the problems start with an error state of franka-interface
  during a slow path of run 3 (17:50:55), followed by the loss of the robot link at 17:53 (USB Ethernet adapter, after
  several receive-buffer failures of its driver earlier that day).
- Why the run at 18:01, with a working link, still tripped the velocity reflex is open. A changed state of the robot
  after the link loss and the restarts (error state, end-effector setting in Desk, gripper re-initialisation) is
  possible but not shown by any log.
- The wrist rotations of steps 13 and 25 leave little margin (80 % of the velocity limit in good runs), so they are the
  first steps to fail when something is wrong.
- After a reflex, `run.py` continues: it moves to the end pose within 1 s (second reflex) and starts the policy from
  wherever the arm stopped. The "success" of step 14 at 18:01 is therefore not a valid result. The shaking the operator
  saw is likely this sequence of aborted motions and recoveries.

### Not examined

- Vibration above 15 Hz (the recordings have 30 Hz).
- The robot's own error log in Desk for 17:50-18:01.
- Whether the USB adapter dropped packets during the runs of 17:23 and 17:38.

## Suggestions (none implemented)

1. Before the next R3 run: restart cleanly (Desk: clear errors, check the end-effector setting, re-activate FCI; then
   franka-interface and the gripper) and watch `dmesg -w | grep -i r8152` during the run.
2. Connect the robot to the on-board Ethernet port (r8169, PCIe) instead of the USB adapter.
3. Run the transport paths slower with `--dt 0.05` (joint 7 at most 0.86 rad/s in step 13); state it as an execution
   setting in R3. The policy keeps its 30 Hz (`real_robot/config.yaml`).
4. Done (operator's decision): runs 3 and 4 and the attempt of 17:56 are not counted; runs 1, 2 and 5 are the R3 runs.
5. Needs the owner's approval, code on the robot: `run.py` should stop after a reflex instead of continuing with a 1 s
   move and the policy.
6. Needs the owner's approval, planner: prefer grasps that need little wrist rotation while carrying a part.
