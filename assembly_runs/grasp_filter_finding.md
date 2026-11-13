# Finding shared by several assemblies: the flat grasp filter leaves parts without a move grasp

See [README.md](README.md) for the setup and for what Fact / Interpretation / Not examined mean.

Eight runs of the batch (beam, plumbers_block, car, gamepad, socket_set, cooling_manifold, duct, stool_circular) stopped in the
first sequence attempt with `[run_flat_seq_opt] No complete sequence`, before any pickup layout or path was planned (3–6 min
after their start; duct and stool_circular later because their precedence planning and grasp generation took longer).
In each of them the grasp filter (`planning/run_pickup_gen.py`, `filter_flat_grasps`) left at least one part with 0 move grasps.

## How the numbers in the assembly files were obtained

Script: session scratchpad `filter_breakdown.py`. For every move grasp in `grasps_unfiltered.pkl` it evaluates the five criteria of
the filter separately, with the pipeline's own functions and values:

| Column | Criterion of the filter |
|---|---|
| walls | all arm poses of the grasp >= `VIRTUAL_WALL_SAFETY` 1.0 cm inside the virtual walls |
| sigma | Jacobian sigma_min >= `MIN_SIGMA_INSERTION` 0.15 at the insertion poses |
| joint | every joint >= `MIN_JOINT_LIMIT_MARGIN` 0.5 rad from its limits at the insertion poses |
| overlap | `grasp_finger_overlap` >= `GRASP_MIN_FINGER_OVERLAP` 0.5 cm |
| flat | the grasp can pick the part lying flat (`MAX_TILT` 30 deg, `GROUND_CLEARANCE` 0.5 cm, `MIN_TIP_ANGLE` 20 deg) |

A grasp can fail several criteria, so the columns do not add up to the number of dropped grasps. "pass" is the number of grasps
that meet all five; it equals the "move N -> M" number printed by the filter in every case that was compared.
"pass (old overlap)" repeats the count with the overlap formula as it was before 2026-11-10.

## Fact: the overlap formula was changed on 2026-11-10 and this changes the filter a lot

- `grasp_finger_overlap` (`planning/plan_checks.py`) returns the depth of the deepest contact point of the grasp, measured from the
  fingertip along the finger axis. Until 2026-11-10 the axis was the line between the centres of the hand mesh and the finger
  meshes, which is tilted by about 6 degrees against the real finger axis. On 2026-11-10 it was replaced by the gripper's own
  axis. In `planning/test_plan_rules.py` (test `finger_overlap`) the new formula returns the constructed depth within 0.01 cm for
  contact points 3 cm to either side; the old one was off by up to about 0.4 cm.
- Effect on the filter, examples from the breakdown: gamepad part 1: 0 of 30 grasps fail the overlap criterion with the old formula,
  30 of 30 with the new one. socket_set part 3: 14 of 100 old, 94 of 100 new.
- Measured on individual grasps of gamepad part 1 (cylinder 2.3 x 2.3 x 3.0 cm): the grasps have 1 to 3 contact points at
  0.27–0.37 cm from the fingertip, while the part itself reaches 0.74–1.02 cm into the gripper from the fingertip.
  For human_normal part 1 the grasps have 10 to 40 contact points and overlaps of 0.6–1.7 cm.

## Interpretation

- With the corrected axis the function measures what it says (depth of the deepest stored contact point). For small round parts
  the grasps store very few contact points, all close to the fingertip, so this depth is much smaller than how far the part really
  sits between the fingers. The old formula overestimated by up to 0.4 cm, which happened to lift many of these grasps above the
  0.5 cm limit.
- So for gamepad and socket_set the "no sequence" result of this batch is a consequence of the corrected formula together with the
  0.5 cm limit, not of the new gripper rules or the strict motion planner.
- For plumbers_block, car, cooling_manifold, duct and stool_circular the result does not depend on the overlap formula: they have
  parts for which no move grasp is flat-pickable at all (see their files). Two kinds of parts are affected: small parts with a
  single resting pose (plumbers_block, car, cooling_manifold, duct) and long thin parts (stool_circular 2.1 x 1.8 x 9.5 cm,
  beam 1.2 x 1.2 x 7.6 cm).

## Not examined

- How the grasp generation chooses and stores contact points, and why small round parts get so few.
- Whether grasps with a contact depth of 0.3 cm but a part depth of 1 cm hold reliably on the real gripper.

## Suggestions (none implemented; the choice changes which grasps are allowed on the real robot)

1. Measure the overlap from geometry instead of from the stored contact points: how far the part reaches between the finger pads
   along the finger axis. This is what the 0.5 cm limit was meant to protect (a gripper that sits too high).
2. Keep the corrected formula and lower `GRASP_MIN_FINGER_OVERLAP` (e.g. to 0.2–0.3 cm). Simple, but keeps a measure that depends
   on how many contact points a grasp happens to store.
3. Go back to the old formula. Restores the earlier behaviour of the filter, with a known error of up to 0.4 cm.
