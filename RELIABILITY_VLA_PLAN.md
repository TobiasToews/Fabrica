# Plan: a more reliable flat pipeline and data collection for VLA training

Status: proposal only (2026-11-10). Nothing in this file is implemented. It is based on reading the code, the plan
`logs/exp_human_normal_flat_v3/human_normal/motion.pkl` and the one complete recording
(`Tobias2503/human_normal_flat_v3`, 1 episode, 11740 frames, 391 s). Nothing was run on the robot for it.

## Summary

Today `real_robot/run.py` replays a verified plan without feedback: it does not check a grasp or the result of an
insertion, and the recorder does not know what `run.py` is doing. Proposed order of changes:

1. Close the gaps in the plan check.
2. Add checks and a clean stop to `run.py`.
3. Let `run.py` report every step to an own recorder that saves one labelled episode per part, with a success flag.

For the VLA a hybrid is realistic: the VLA picks and carries, the RL policy keeps doing the insertion.

## Findings

From the recording:

| Finding | Numbers |
|---|---|
| Unseated parts went unnoticed | parts 1 and 2 ended 6.5–6.8 mm short of the planned pose after closing pushes of 31 N and 42 N; part 3 1.8 mm short; parts 5, 4, 6 within 0.5 mm |
| Load before the insertion | parts 1 and 2 show about 10 N and 17 N downward load at rest before the insertion starts (cause not examined) |
| Data yield | 1 of about 20 recording attempts on 2026-11-08 was saved: one episode is the whole assembly, any abort discards it |
| Idle time | about 40 % of the frames are standstill (prompts, fixed sleeps); 391 s recorded for 198 s of planned motion |
| Tracking while streaming | largest joint error p95 0.14 rad, max 0.27 rad; TCP error p95 14 mm, max 27 mm (partly lag along the path); the verified clearance is 1 cm |
| Grasp widths | held parts measured 0.5–1.3 mm below the planned width; an empty grasp would close about 10 mm below it (never recorded) |
| Free-motion force noise | up to 17.8 N without contact, so fixed force thresholds below about 20 N do not work during transfers |

From the code:

- `planning/plan_checks.py` (check_path, lines 120–202) has no robot self-collision rule. The verified v3 plan passes
  the hand 3.3 cm from link 5; the wrist camera mount and its cable are not in the model.
- `run.py` streams waypoints without reading the robot state. Franka errors are recovered automatically and the next step
  starts. Ctrl+C exits with code 0; no step index, gripper state or `global_pos_shift` is saved.
- Gripper: the wait is a fixed 2 s sleep, the grasp result and width are never read.
- Insertion: success is "less than about 3 mm left along the axis" (`robot_interface.py:490`); lateral error, force and
  seating are not checked. With `--auto` a failed attempt is repeated without limit (`run.py:207–211`).
  `global_pos_shift` is taken over after every insertion, also after a jam (`run.py:261–266`). After the policy, a stiff
  1 s `goto_joints` pushes to the planned end pose regardless of the outcome (`run.py:258`).
- The policy runs stochastic on the real robot (`real_robot/config.yaml`: `deterministic: False`) with ±3 mm injected
  observation noise (`robot_interface.py:449, 475`).
- 8 of the 24 human hold-arm steps block on `input()` even with `--auto`.
- Recording (`lerobot_franka`): one task string for 6.5 min, episode control by keyboard, gripper action only open/closed
  (the part-specific opening widths are lost), the insertion action is a joint target rebuilt from the Cartesian target,
  no success flag, no plan id, only the front camera, state 36-dim.
- The plan is deterministic (all seeds 0). RL preprocessing reads only `precedence.pkl`, `grasps.pkl`, `tree_opt.pkl`:
  a new layout, path seed or speed needs no retraining; a new grasp or part order does.

## Phases

| Phase | Goal | Robot needed | Effort (estimate) |
|---|---|---|---|
| 0 | close model gaps, test the export on the existing recording | only still images of the wrist camera | 2–3 days |
| 1 | `run.py` checks, stops cleanly, logs | pilot runs | 6–8 working days |
| 2 | recorder driven by `run.py`, fixed dataset format | yes | 1–2 weeks |
| 3 | variation, no hand in the image | yes | 2–3 weeks |
| 4 | learnability test, then VLA as a hybrid | yes | afterwards |

### Phase 0: no robot run

- Self-distance rule in `planning/plan_checks.py`: hand, fingers and carried part against links 0–5. Cost with the existing
  trimesh/fcl objects: 0.7 ms per pose, about 4 s for the whole plan.
- Camera mount as a box behind a flag, for the motion planner and the check only. Not in the grasp generation, otherwise
  grasps and the RL plan info change. STEP file: `real_robot/_tobi_stuff/RealSenseD435_Camera_Mount_on_angle.step`.
  Before that: still images at grasp and pre-insertion poses to see whether the wrist camera sees fingers and part at all.
- Plan manifest next to `motion.pkl` (from `verify_plan.py` iterate_move_paths): per step the 0-based index, frame range,
  task, part id and name, grasp ids; plus hashes of plan, policy and config. The id-to-name table (head, left arm, ...)
  has to be written by hand.
- Export test: cut the existing recording into 12 sub-episodes by matching recorded actions to plan waypoints
  (75 % of the frames equal a plan waypoint exactly; 21 of 21 arm steps were found in order).
- Backup: fork franka-interface and push the branch `fabrica-ics` (TODO in `FLAT_PIPELINE.md`).

### Phase 1: hooks in `run.py`, no rewrite

- Clean stop: replace frankapy's SIGINT handler, try/finally, exit codes, saved state (step, gripper, `global_pos_shift`)
  so that resuming does not mean guessing a step number.
- Event log: per step start, end, part, task, gripper width, attempts, abort reason; as a JSONL file and a ROS topic.
- Franka errors: read the status continuously; a reflex stops the run.
- Grasp check by finger width (band of 3–4 mm around the planned width), with a persistent subscriber and a timeout
  instead of the fixed 2 s sleep.
- Insertion:
  - at most 3 attempts;
  - retract along the reversed insertion path, not a joint-space move from a jammed pose;
  - `global_pos_shift` only after a success;
  - no stiff closing push; hold, then measure remaining distance, lateral error and force. First log only, enforce later.
- Stop while streaming: joint error above 0.35 rad (maximum in the good run: 0.27), stall, or a skill that ended.
- Introduction: every check is a pure function that is first replayed on the existing recording, then 3–5 runs in
  log-only mode, then armed. One dry run without parts.
- Idle time: `--auto`, and bundle or drop the 8 hold-arm prompts (depends on the holder for part 0).
- To clarify: why parts 1 and 2 are loaded before the insertion (height or pose of the base part?); whether the policy
  inserts better deterministic and without the injected noise (simulation first).

### Phase 2: recording

- Own recorder on the LeRobot dataset API (create / add_frame / save_episode) instead of `lerobot-record` with keys.
  Episode boundaries come from the `run.py` events, it also saves on abort, one dataset per run with a load check,
  merged offline.
- Episodes: two per part (pick and carry; insert), cut at the controller switch, with a language label, success flag,
  attempt count and plan id. A failure at part 5 no longer costs parts 1–4.
- Bridge (`lerobot_franka/bridge/franka_bridge.py`): commanded gripper width, the Cartesian insertion target as sent,
  controller mode, robot and camera timestamps, the events.
- Cameras: front and wrist (through `cameras/realsense_server.py`).
- Raw data plus a deterministic export to the training dataset: state and action 8-dim (7 joints + gripper), forces as
  separate features, idle frames trimmed. The current 36-dim state probably exceeds the 32 of SmolVLA and pi0
  (concluded from the code, not run).
- A tmux launcher: `run.py` and the recorder otherwise compete for the terminal. One fixed conda env per script
  (`run.py` and bridge Python 3.8, recorder Python 3.12).

### Phase 3: variation

- 50 runs of one plan are 50 copies of one joint path; a VLA can memorise it without using the image. At most about 10
  identical runs.
- A library of verified plans with different pickup layouts and the same `grasps.pkl` and `tree_opt.pkl`: no RL
  retraining, about 10–15 min planning per variant. Keep some layouts out of the training data for evaluation.
- Let the robot lay out the parts (`real_robot/place_parts.py`) so the printed template can leave the image.
- Part 0 without a hand: a passive holder, or placed by the robot (`planning/plan_place_part.py`). Precondition for any
  run with a learned policy.
- Speed variants only at scale ≤ 1.0 and after a new check.

### Phase 4: training

- Learnability test first: ACT locally (fits the 8 GB GPU), before collecting in bulk or renting a GPU.
- Then SmolVLA or pi0.5 on a rented GPU (10–40 GB).
- Rollout as a hybrid: the RL policy stays in `run.py`, which owns the arm. The bridge needs a safety layer for this
  (speed cap, distance check per target, command-age watchdog).
- Steps: offline, without parts, with parts and holder, one VLA cycle, hybrid; about 20 trials per condition.

## Not recommended

- Enabling frankapy's workspace walls (±0.46 m): the pickup area reaches y = −0.546 m, pickups there would abort.
- A global force threshold around 40 N: the normal run itself reaches 41.9 N. Remove the closing push first, then
  per-skill limits.
- Rewriting `run.py` into a supervisor: hooks are enough.
- A VLA for the insertion: published results without a force signal are poor (figures come from automated page
  summaries and were not cross-checked).

## Open decisions

1. What should the VLA do better than the planner (e.g. pick parts without a template)? This decides the variation needed.
2. Part 0: holder, placed by the robot, or still by hand?
3. Does the wrist camera mount stay as it is? Then its dimensions and pose are needed.
4. Compute for VLA training: rented or the chair's cluster?
5. Who holds the user stop during policy runs?

## Data amount

Rule of thumb from published fine-tunes: about 50 episodes per task, here about 50 assemblies. At a realistic 3–4
assemblies per hour that is 13–17 hours of robot time. The Fabrica paper reports 80 % step success and 15 % complete
assemblies without intervention, which is why per-part episodes matter.

## Not addressed here

Planning failures of `gamepad` and `socket_set` (zero margin between planner and checker for the carried-part clearance,
failure blamed on the wrong part in `run_motion_plan.py:393–435`, hold-arm pickup IK exception in single-arm mode),
time parameterisation of the paths (accelerations up to 30 rad/s² at path corners), part pose perception.
