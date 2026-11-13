"""Tests of the plan rules of the real arm and of its motion planner.

The rules (planning/plan_checks.py) say what every path of the real
robot arm has to keep: distances to the parts, to the table and to the
virtual walls of the robot controller, and a distance to singularities
and joint limits. The pickup layout (run_pickup_gen.py), the motion
planning (run_motion_plan.py) and the final check (verify_plan.py) of
the flat pipeline all use them. The motion planner of the real arm
(planning/robot/motion_plan_arm.py, strict mode) has to produce paths
that pass them. This file tests the rules, that planner, and the cause
that planning/run_motion_plan.py names when it finds no checked path.

No robot and no log folder are needed: a toy scene of boxes and
cylinders is built around arm poses, so every distance is known exactly.
Each rule is tested just inside and just outside its limit.

Usage (fabrica env; the exit code is 1 if a test fails):

    python planning/test_plan_rules.py
    python planning/test_plan_rules.py table dip

The second form runs only the tests whose name contains one of the
words.

Terms used in this file:

* Board frame: the world frame of the planner. Lengths are in cm, z
  points up and the table surface is at z = 0.
* q: the joint angles of the arm in rad. A "full" q has one entry per
  link of the arm chain (9 for the Panda), an "active" q only the 7
  angles of the joints that move.
* open_ratio: how far the gripper is open, from 0 (closed) to 1 (8 cm
  between the fingers). At 0.5 the fingers are 4 cm apart.
* state: a dict {part id: 'pickup' or 'final'}. 'pickup' means that the
  part lies on the table where it is picked, 'final' that it sits at its
  assembled pose. It tells SceneChecker.check_path which parts are
  where.
* violation: one text that check_path returns for a broken rule. It ends
  with the poses of the path that break the rule, for example
  '(poses 3..4 of 80, 2x)'. An empty list means that the path passes.
"""

import os

os.environ['OMP_NUM_THREADS'] = '1'
import sys

# Make the repository root importable, so that the imports from
# `planning.` below work when this file is run as a script.
project_base_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)
sys.path.append(project_base_dir)

import time
import types
import random
import numpy as np
import trimesh

from planning.config import (
    ROBOT_PART_CLEARANCE,
    GRIPPER_TABLE_CLEARANCE,
    TRANSFER_PART_CLEARANCE,
    ASSEMBLY_ZONE_MARGIN,
    ROBOT_SELF_CLEARANCE,
    ASSEMBLY_CLEARANCE,
    ARM_ARM_CLEARANCE,
    MIN_SIGMA,
    MIN_SIGMA_INSERTION,
    MIN_JOINT_LIMIT_MARGIN,
    PLANNER_JOINT_MARGIN,
    PLANNER_SIGMA_MARGIN,
    PLANNER_TABLE_MARGIN,
    PLANNER_MARGIN,
    STRAIGHT_STEP,
    ZONE_PART_TABLE_CLEARANCE,
)
from scipy.spatial.transform import Rotation as R

from planning.plan_checks import (
    SceneChecker,
    grasp_finger_overlap,
    joint_condition,
    ARM_LINKS_CHECKED,
    SELF_LINKS_CHECKED,
    PANDA_JOINT_LIMITS,
    write_failure,
    clear_failure,
    FAILURE_FILE,
    CAUSE_GRASP,
    CAUSE_SPOT,
    CAUSE_INSERTION,
    CAUSE_HOLD_GRASP,
    CAUSE_TEXT,
    WRIST_LINK,
    PICKUP_POSE_SLACK,
)
from planning.robot.motion_plan_arm import (
    ArmMotionPlanner,
    VIRTUAL_WALL_SAFETY,
    HITS_ARM,
    HITS_CARRIED_PART,
)
from planning.run_motion_plan import (
    failure_cause,
    no_path_reason,
    make_move_planner,
    NO_PATH_REASON,
)
from planning.robot.geometry import (
    transform_gripper_meshes,
    get_gripper_finger_names,
)
from planning.robot.util_arm import (
    get_arm_chain,
    get_virtual_wall_margin,
    get_gripper_pos_quat_from_arm_q,
    get_hand_wall_margin,
)
from planning.robot.workcell import (
    get_panda_base_z,
    get_panda_desk_z,
    get_panda_virtual_walls,
)

DOWN = np.array(
    [[1, 0, 0], [0, -1, 0], [0, 0, -1.0]]
)  # flange orientation with the fingers pointing down
REF_XY = np.array(
    [20.0, -30.0]
)  # a well conditioned place in front of the robot (world, cm)
FAR = np.array([-200.0, -200.0])  # where unused toy parts are parked


# ----------------------------------------------------------------------------------------------------------------- helpers


def make_checker():
    dummy = box([1, 1, 1], [*FAR, 0.5])
    return SceneChecker(
        'panda', 'panda', False, {'A': dummy}, {'A': np.eye(4)}, {'A': dummy}
    )


def box(extents, center):
    mesh = trimesh.creation.box(extents)
    mesh.apply_translation(center)
    return mesh


def cylinder(radius, height, xy, z_bottom=0.0):
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=48)
    mesh.apply_translation([xy[0], xy[1], z_bottom + height / 2])
    return mesh


def set_scene(c, lying=None, final=None, zone_center=FAR):
    """lying / final: {part id: mesh in world}; zone_center: xy of the assembly zone (square of ASSEMBLY_ZONE_MARGIN around a point)"""
    c.pickup_meshes = dict(lying or {})
    c.final_meshes = dict(final or {})
    for k in list(c.pickup_meshes) + list(
        c.final_meshes
    ):  # every part needs both entries
        c.pickup_meshes.setdefault(k, box([1, 1, 1], [*FAR, 0.5]))
        c.final_meshes.setdefault(k, box([1, 1, 1], [*FAR, 0.5]))
    zone_center = np.asarray(zone_center, dtype=float)
    c.zone = (
        zone_center - ASSEMBLY_ZONE_MARGIN,
        zone_center + ASSEMBLY_ZONE_MARGIN,
    )


def gripper_vertices(c, q, open_ratio):
    _, tf, _ = c.planner.get_meshes_transforms(q, open_ratio, None, None)
    return {
        n: c.planner.gripper_meshes[n].vertices @ tf[n][:3, :3].T
        + tf[n][:3, 3]
        for n in tf
    }


def gripper_low(c, q, open_ratio):
    return min(
        v[:, 2].min() for v in gripper_vertices(c, q, open_ratio).values()
    )


def ik_down(c, xy, low_z, open_ratio=0.5, q_init=None):
    """full q with the fingers pointing down, the gripper centred at xy and its lowest point at low_z (cm above the table)"""
    q = None
    z = low_z + 11.2
    for _ in range(
        4
    ):  # the offset flange -> fingertip does not depend on the pose; two rounds are exact
        q = c.planner.inverse_kinematics(
            np.array([xy[0], xy[1], z]),
            DOWN,
            q_init=q if q is not None else q_init,
        )
        assert q is not None, f'no IK at {xy}, {low_z}'
        z += low_z - gripper_low(c, q, open_ratio)
    assert abs(gripper_low(c, q, open_ratio) - low_z) < 1e-3
    return q


def gripper_distance(c, q, open_ratio, mesh):
    _, tf, _ = c.planner.get_meshes_transforms(q, open_ratio, None, None)
    for n, t in tf.items():
        c.gripper_mgr.set_transform(n, t)
    return c.gripper_mgr.min_distance_other(c.manager({'m': mesh}))


def has(violations, text):
    return any(text in v for v in violations)


def poses_of(violations, text):
    """(first pose, last pose, count) of the violation containing text"""
    v = next(v for v in violations if text in v)
    inside = v[v.rindex('(poses ') + 7 : -1]  # 'a..b of n, kx'
    span, count = inside.split(' of ')[0], int(inside.split(', ')[1][:-1])
    return int(span.split('..')[0]), int(span.split('..')[1]), count


TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


# -------------------------------------------------------------------------------------------- rules of plan_checks.py


@test
def table_gripper(c):
    set_scene(c)
    ok = c.check_path(
        [ik_down(c, REF_XY, GRIPPER_TABLE_CLEARANCE + 0.1)], 0.5, {}
    )
    bad = c.check_path(
        [ik_down(c, REF_XY, GRIPPER_TABLE_CLEARANCE - 0.1)], 0.5, {}
    )
    assert not has(ok, 'gripper closer than'), ok
    assert has(
        bad, f'gripper closer than {GRIPPER_TABLE_CLEARANCE} cm to the table'
    ), bad


@test
def table_arm(c):
    set_scene(c)
    ok = c.check_path([ik_down(c, REF_XY, 10.0)], 0.5, {})
    assert not has(ok, 'arm closer than'), ok
    rng, found = np.random.default_rng(0), None
    lower, upper = PANDA_JOINT_LIMITS
    for _ in range(
        20000
    ):  # a pose with a link lower than 1 cm above the table
        q = c.chain.active_to_full(rng.uniform(lower, upper))
        arm_tf, _, _ = c.planner.get_meshes_transforms(q, 0.5, None, None)
        z = min(
            (v @ arm_tf[n][:3, :3].T + arm_tf[n][:3, 3])[:, 2].min()
            for n, v in c.arm_hull.items()
        )
        if z < c.table_z + ROBOT_PART_CLEARANCE - 0.2:
            found = q
            break
    assert found is not None
    assert has(
        c.check_path([found], 0.5, {}),
        f'arm closer than {ROBOT_PART_CLEARANCE} cm to the table',
    )


@test
def virtual_walls(c):
    set_scene(c)
    q_in = ik_down(c, REF_XY, 10.0)
    assert get_virtual_wall_margin(c.chain, q_in) > VIRTUAL_WALL_SAFETY + 1
    assert not has(c.check_path([q_in], 0.5, {}), 'virtual wall')
    q, found = q_in, None
    for dy in np.arange(
        0, 40, 0.5
    ):  # move towards the side wall until the margin is used up
        q_next = c.planner.inverse_kinematics(
            np.array([REF_XY[0], REF_XY[1] - dy, 25.0]), DOWN, q_init=q
        )
        if q_next is None:
            continue
        q = q_next
        if get_virtual_wall_margin(c.chain, q) < VIRTUAL_WALL_SAFETY - 0.1:
            found = q
            break
    assert found is not None, 'no pose near the wall found'
    assert has(
        c.check_path([found], 0.5, {}),
        f'closer than {VIRTUAL_WALL_SAFETY} cm to a virtual wall',
    )


@test
def joint_limits(c):
    set_scene(c)
    q = c.chain.active_from_full(ik_down(c, REF_XY, 10.0))
    assert joint_condition(q)[1] > MIN_JOINT_LIMIT_MARGIN + 0.05
    for joint in [1, 4, 7]:
        for side, limit in [
            (0, PANDA_JOINT_LIMITS[0][joint - 1]),
            (1, PANDA_JOINT_LIMITS[1][joint - 1]),
        ]:
            sign = 1 if side == 0 else -1
            q_ok, q_bad = q.copy(), q.copy()
            q_ok[joint - 1] = limit + sign * (MIN_JOINT_LIMIT_MARGIN + 0.01)
            q_bad[joint - 1] = limit + sign * (MIN_JOINT_LIMIT_MARGIN - 0.01)
            assert not has(
                c.check_path([q_ok], 0.5, {}), 'rad to its limit'
            ), (joint, side)
            assert has(
                c.check_path([q_bad], 0.5, {}),
                f'joint {joint} closer than {MIN_JOINT_LIMIT_MARGIN} rad to its limit',
            ), (joint, side)


def pose_with_sigma(c, target):
    """a pose whose Jacobian sigma_min is target (between a well conditioned and a stretched pose), full q"""
    q_good = c.chain.active_from_full(ik_down(c, REF_XY, 10.0))
    q_sing = np.array(
        [0.0, 0.0, 0.0, -0.6, 0.0, 0.6, 0.0]
    )  # arm stretched upwards
    assert joint_condition(q_sing)[0] < target < joint_condition(q_good)[0]
    lo, hi = (
        0.0,
        1.0,
    )  # sigma falls from q_good (0) to q_sing (1); not monotonic in general, so the result is verified
    for _ in range(60):
        mid = (lo + hi) / 2
        if joint_condition(q_good + mid * (q_sing - q_good))[0] > target:
            lo = mid
        else:
            hi = mid
    q = q_good + lo * (q_sing - q_good)
    assert abs(joint_condition(q)[0] - target) < 1e-3
    return c.chain.active_to_full(q)


@test
def singularity(c):
    set_scene(c)
    text = f'too close to a singularity (Jacobian sigma_min < {MIN_SIGMA})'
    assert not has(
        c.check_path([pose_with_sigma(c, MIN_SIGMA + 0.005)], 0.5, {}),
        'singularity',
    )
    assert has(
        c.check_path([pose_with_sigma(c, MIN_SIGMA - 0.005)], 0.5, {}), text
    )


@test
def singularity_while_inserting(c):
    set_scene(
        c,
        lying={'B': box([1, 1, 1], [*FAR, 0.5])},
        final={'B': box([1, 1, 1], [*FAR, 0.5])},
    )
    between = pose_with_sigma(
        c, (MIN_SIGMA + MIN_SIGMA_INSERTION) / 2
    )  # allowed in a transfer, not in an insertion
    assert not has(c.check_path([between], 0.5, {}), 'singularity')
    assert has(
        c.check_path([between], 0.5, {}, carried='B', inserting=True),
        f'sigma_min < {MIN_SIGMA_INSERTION}',
    )
    assert not has(
        c.check_path(
            [pose_with_sigma(c, MIN_SIGMA_INSERTION + 0.005)],
            0.5,
            {},
            carried='B',
            inserting=True,
        ),
        'singularity',
    )


@test
def arm_to_parts(c):
    q = ik_down(c, REF_XY, 10.0)
    arm_tf, _, _ = c.planner.get_meshes_transforms(q, 0.5, None, None)
    top = max(
        (
            (v @ arm_tf[n][:3, :3].T + arm_tf[n][:3, 3])
            for n, v in c.arm_hull.items()
        ),
        key=lambda v: v[:, 2].max(),
    )
    top = top[
        np.argmax(top[:, 2])
    ]  # highest point of the arm links: a box right above it is exactly its gap away
    for gap, expected in [
        (ROBOT_PART_CLEARANCE + 0.1, False),
        (ROBOT_PART_CLEARANCE - 0.1, True),
    ]:
        for where in [
            'pickup',
            'final',
        ]:  # lying and assembled parts count alike
            part = box([3, 3, 3], [top[0], top[1], top[2] + gap + 1.5])
            set_scene(
                c,
                lying={'A': part} if where == 'pickup' else None,
                final={'A': part} if where == 'final' else None,
            )
            violations = c.check_path([q], 0.5, {'A': where})
            assert (
                has(
                    violations,
                    f'closer than {ROBOT_PART_CLEARANCE} cm to part A',
                )
                == expected
            ), (gap, where, violations)


@test
def gripper_to_lying_parts(c):
    for gap, expected in [
        (ROBOT_PART_CLEARANCE + 0.1, False),
        (ROBOT_PART_CLEARANCE - 0.1, True),
    ]:
        set_scene(c, lying={'A': box([12, 12, 2], [*REF_XY, 1.0])})
        q = ik_down(c, REF_XY, 2.0 + gap)
        assert (
            abs(gripper_distance(c, q, 0.5, c.pickup_meshes['A']) - gap) < 1e-3
        )
        violations = c.check_path([q], 0.5, {'A': 'pickup'})
        assert (
            has(
                violations,
                f'closer than {ROBOT_PART_CLEARANCE} cm to lying part A',
            )
            == expected
        ), (gap, violations)


@test
def gripper_to_part_before_grasp(c):
    set_scene(c, lying={'B': box([12, 12, 2], [*REF_XY, 1.0])})
    near = c.check_path(
        [ik_down(c, REF_XY, 2.3)], 0.5, {'B': 'pickup'}, pick_part='B'
    )
    assert near == [], (
        near
    )  # close to the part it is about to pick is fine, also below ROBOT_PART_CLEARANCE
    touch = c.check_path(
        [ik_down(c, REF_XY, 1.7)], 0.5, {'B': 'pickup'}, pick_part='B'
    )
    assert has(touch, 'gripper touches part B before grasping it'), touch
    # an approach that touches only on its last poses (the former exemption around the grasp)
    path = [ik_down(c, REF_XY, z) for z in [8.0, 5.0, 2.3, 1.9, 1.7]]
    violations = c.check_path(path, 0.5, {'B': 'pickup'}, pick_part='B')
    assert poses_of(violations, 'touches part B') == (3, 4, 2), violations


@test
def gripper_to_assembly_contact(c):
    set_scene(
        c, final={'C': box([12, 12, 2], [*REF_XY, 1.0])}, zone_center=REF_XY
    )
    assert (
        c.check_path([ik_down(c, REF_XY, 2.3)], 0.5, {'C': 'final'}) == []
    )  # a single close pose (e.g. at a release) is no contact
    hit = c.check_path([ik_down(c, REF_XY, 1.7)], 0.5, {'C': 'final'})
    assert has(hit, 'gripper hits assembled part C'), hit
    # the part just released has no exemption any more, from the first pose on
    path = [ik_down(c, REF_XY, z) for z in [1.7, 1.9, 2.3, 5.0]]
    released = c.check_path(path, 0.5, {'C': 'final'}, released_part='C')
    assert poses_of(released, 'gripper hits part C (just released)') == (
        0,
        1,
        2,
    ), released


@test
def assembly_clearance_gripper(c):
    part = box([10, 10, 2], [*REF_XY, 1.0])
    set_scene(c, final={'C': part}, zone_center=REF_XY)
    low = 2.0 + ASSEMBLY_CLEARANCE - 0.3  # passes 0.7 cm above the part
    near, far = ik_down(c, REF_XY, low), ik_down(c, REF_XY + [-20.0, 0], low)
    assert (
        abs(gripper_distance(c, near, 0.5, part) - (ASSEMBLY_CLEARANCE - 0.3))
        < 1e-3
        and gripper_distance(c, far, 0.5, part) > ASSEMBLY_CLEARANCE
    )
    text = f'gripper passes closer than {ASSEMBLY_CLEARANCE} cm to assembled part C'
    state = {'C': 'final'}
    assert poses_of(c.check_path([far, near, far], 0.5, state), text) == (
        1,
        1,
        1,
    )  # passing by: not allowed
    assert not has(
        c.check_path([near, far], 0.5, state), 'passes closer'
    )  # leaving after a release
    assert not has(
        c.check_path([far, near], 0.5, state), 'passes closer'
    )  # arriving
    assert not has(
        c.check_path([near, far, near], 0.5, state), 'passes closer'
    )  # leaving and arriving
    assert not has(
        c.check_path([near, near], 0.5, state), 'passes closer'
    )  # staying close
    assert poses_of(
        c.check_path([near, far, near, far, near], 0.5, state), text
    ) == (2, 2, 1)  # coming back in between
    high = ik_down(
        c, REF_XY, 2.0 + ASSEMBLY_CLEARANCE + 0.1
    )  # passing by with the clearance
    assert not has(c.check_path([far, high, far], 0.5, state), 'passes closer')


def carried_scene(c, zone_center=FAR, extra_lying=None, extra_final=None):
    """part B (cylinder, 3 cm high) stands on the table between the open fingers; returns the pose holding it there"""
    part = cylinder(1.5, 3.0, REF_XY)
    set_scene(
        c,
        lying={'B': part, **(extra_lying or {})},
        final=extra_final,
        zone_center=zone_center,
    )
    return ik_down(c, REF_XY, 1.0)


def carry_to(c, q_grasp, xy, part_bottom):
    """pose that holds part B (grasped at q_grasp) with its bottom at part_bottom above the table, at xy"""
    return ik_down(c, xy, 1.0 + part_bottom, q_init=q_grasp)


@test
def carried_part_below_table(c):
    q0 = carried_scene(c)
    violations = c.check_path(
        [q0, carry_to(c, q0, REF_XY, -0.3)], 0.5, {'B': 'pickup'}, carried='B'
    )
    assert has(violations, 'carried part B below the table'), violations
    assert not has(
        c.check_path(
            [q0, carry_to(c, q0, REF_XY, 0.5)],
            0.5,
            {'B': 'pickup'},
            carried='B',
        ),
        'below the table',
    )


@test
def carried_part_height(c):
    text = f'carried part B lower than {TRANSFER_PART_CLEARANCE} cm above the table outside the assembly zone'
    q0 = carried_scene(c)
    up = carry_to(c, q0, REF_XY, TRANSFER_PART_CLEARANCE + 0.5)
    low, high = (
        carry_to(c, q0, REF_XY + [10, 0], TRANSFER_PART_CLEARANCE - 0.2),
        carry_to(c, q0, REF_XY + [10, 0], TRANSFER_PART_CLEARANCE + 0.2),
    )
    state = {'B': 'pickup'}
    assert poses_of(
        c.check_path([q0, up, low], 0.5, state, carried='B'), text
    ) == (2, 2, 1)  # once lifted it must stay up
    assert not has(
        c.check_path([q0, up, high], 0.5, state, carried='B'), 'lower than'
    )
    assert not has(
        c.check_path([q0, low], 0.5, state, carried='B'), 'lower than'
    )  # the lift itself starts on the table
    # inside the assembly zone the part may come down (approach of the insertion)
    q0 = carried_scene(c, zone_center=REF_XY + [10, 0])
    assert not has(
        c.check_path([q0, up, low], 0.5, state, carried='B'), 'lower than'
    )
    # the zone ends ASSEMBLY_ZONE_MARGIN from the assembly: a part edge just outside counts again
    edge = REF_XY + [
        10 + ASSEMBLY_ZONE_MARGIN + 0.3 - 1.5,
        0,
    ]  # cylinder radius 1.5: its rim is 0.3 cm outside the zone
    out = carry_to(c, q0, edge, TRANSFER_PART_CLEARANCE - 0.2)
    assert has(c.check_path([q0, up, out], 0.5, state, carried='B'), text)
    inside = carry_to(c, q0, edge - [0.6, 0], TRANSFER_PART_CLEARANCE - 0.2)
    assert not has(
        c.check_path([q0, up, inside], 0.5, state, carried='B'), 'lower than'
    )


@test
def carried_part_to_lying_parts(c):
    text = (
        f'carried part B closer than {ROBOT_PART_CLEARANCE} cm to lying part A'
    )
    for gap, expected in [
        (ROBOT_PART_CLEARANCE + 0.1, False),
        (ROBOT_PART_CLEARANCE - 0.1, True),
    ]:
        target = REF_XY + [8.0, 0]  # away from where the part lay
        neighbour = box([4, 4, 2], [target[0], target[1], 1.0])
        q0 = carried_scene(c, extra_lying={'A': neighbour})
        up = carry_to(c, q0, REF_XY, 12.0)
        over = carry_to(
            c, q0, target, 2.0 + gap
        )  # part B right above the neighbour, gap above its top
        violations = c.check_path(
            [q0, up, over], 0.5, {'A': 'pickup', 'B': 'pickup'}, carried='B'
        )
        assert has(violations, text) == expected, (gap, violations)
    # it lay among them: lifted straight up beside a taller neighbour (0.5 cm away, 8 cm high) it is close for the whole lift, which is fine
    neighbour = box([2, 2, 8], [REF_XY[0] + 1.5 + 1 + 0.5, REF_XY[1], 4.0])
    state = {'A': 'pickup', 'B': 'pickup'}
    q0 = carried_scene(c, extra_lying={'A': neighbour})
    lift = [q0] + [
        carry_to(c, q0, REF_XY, z) for z in [2.0, 4.0, 6.0, 8.0, 10.0]
    ]  # clear above the neighbour at the end
    assert not has(
        c.check_path(lift, 0.5, state, carried='B'), 'carried part'
    )  # (the gripper is close to this neighbour too; only the carried part is tested here)
    # once it was clear it must stay clear: coming back down beside the neighbour is not allowed
    back = c.check_path(
        lift + [carry_to(c, q0, REF_XY, 6.0)], 0.5, state, carried='B'
    )
    assert poses_of(back, text) == (6, 6, 1), back
    # and it never touches a lying part, also not at the start
    touching = box(
        [2, 2, 8], [REF_XY[0] + 1.5 + 1 - 0.2, REF_XY[1], 4.0]
    )  # overlaps the cylinder by 0.2 cm
    q0 = carried_scene(c, extra_lying={'A': touching})
    assert has(
        c.check_path(
            [q0, carry_to(c, q0, REF_XY, 2.0)], 0.5, state, carried='B'
        ),
        'carried part B hits lying part A',
    )


@test
def carried_part_to_assembly(c):
    target = REF_XY + [12, 0]
    base = box([4, 4, 2], [target[0], target[1], 1.0])
    state = {'C': 'final', 'B': 'pickup'}
    q0 = carried_scene(c, zone_center=target, extra_final={'C': base})
    up = carry_to(c, q0, REF_XY, 12.0)

    def over(gap):
        return carry_to(c, q0, target, 2.0 + gap)

    assert has(
        c.check_path([q0, up, over(-0.3)], 0.5, state, carried='B'),
        'carried part B hits the assembly',
    )
    assert (
        c.check_path([q0, up, over(0.4)], 0.5, state, carried='B') == []
    )  # arriving above the assembly: close is fine
    text = f'carried part passes closer than {ASSEMBLY_CLEARANCE} cm to assembled part C'
    assert poses_of(
        c.check_path([q0, up, over(0.4), up], 0.5, state, carried='B'), text
    ) == (2, 2, 1)  # passing by
    assert not has(
        c.check_path(
            [q0, up, over(ASSEMBLY_CLEARANCE + 0.1), up],
            0.5,
            state,
            carried='B',
        ),
        'passes closer',
    )
    # while inserting the carried part is not checked against the assembly (it ends inside it)
    set_scene(
        c,
        lying={'B': c.pickup_meshes['B']},
        final={'C': base, 'B': cylinder(1.5, 3.0, target, z_bottom=1.0)},
        zone_center=target,
    )
    q_end = ik_down(
        c, target, 2.0
    )  # holds B in its final pose, 1 cm deep in the base
    q_start = ik_down(c, target, 4.0, q_init=q_end)
    assert not has(
        c.check_path(
            [q_start, q_end], 0.5, {'C': 'final'}, carried='B', inserting=True
        ),
        'carried part',
    )


@test
def gripper_to_own_arm(c):
    set_scene(c)
    assert not has(
        c.check_path([ik_down(c, REF_XY, 10.0)], 0.5, {}), 'to the robot'
    )
    rng = np.random.default_rng(1)
    lower, upper = PANDA_JOINT_LIMITS
    near = far = None
    for _ in range(4000):
        q = c.chain.active_to_full(rng.uniform(lower, upper))
        arm_tf, grip_tf, _ = c.planner.get_meshes_transforms(
            q, 0.5, None, None
        )
        for n in SELF_LINKS_CHECKED:
            c.self_mgr.set_transform(n, arm_tf[n])
        for n, t in grip_tf.items():
            c.gripper_mgr.set_transform(n, t)
        d = c.self_mgr.min_distance_other(c.gripper_mgr)
        if 0.05 < d < ROBOT_SELF_CLEARANCE - 0.1 and near is None:
            near = q
        if (
            ROBOT_SELF_CLEARANCE + 0.1 < d < ROBOT_SELF_CLEARANCE + 1.0
            and far is None
        ):
            far = q
        if near is not None and far is not None:
            break
    assert near is not None and far is not None
    text = f'cm to the robot'
    assert has(
        c.check_path([near], 0.5, {}),
        f'closer than {ROBOT_SELF_CLEARANCE} cm to the robot',
    ), 'gripper near its own arm not reported'
    assert not has(c.check_path([far], 0.5, {}), text)


@test
def finger_overlap(c):
    q = ik_down(c, REF_XY, 5.0)
    pos, quat = get_gripper_pos_quat_from_arm_q(c.chain, q, 'panda')
    meshes = transform_gripper_meshes(
        'panda', c.planner.gripper_meshes, pos, quat, np.eye(4), 0.5
    )
    tip = min(
        meshes[n].vertices[:, 2].min()
        for n in get_gripper_finger_names('panda')
    )  # fingers point down
    for depth in [
        0.3,
        0.5,
        1.7,
    ]:  # the part reaches depth cm into the fingers (its highest contact point), wherever it touches sideways
        for side in [-3.0, 0.0, 3.0]:
            contacts = np.array(
                [
                    [REF_XY[0] + side, REF_XY[1], tip + depth],
                    [REF_XY[0] - side, REF_XY[1] + side, tip + depth - 0.2],
                ]
            )
            grasp = types.SimpleNamespace(
                pos=pos, quat=quat, open_ratio=0.5, contact_points=contacts
            )
            assert (
                abs(
                    grasp_finger_overlap(
                        grasp, 'panda', c.planner.gripper_meshes
                    )
                    - depth
                )
                < 0.01
            ), (depth, side)
    assert (
        grasp_finger_overlap(
            types.SimpleNamespace(
                pos=pos, quat=quat, open_ratio=0.5, contact_points=None
            ),
            'panda',
        )
        == 0.0
    )


@test
def carried_part_at_the_limit_up_to_rounding(c):
    """
    a carried part is lifted beside a neighbour that stands ROBOT_PART_CLEARANCE away up to rounding (parts laid out on a grid
    stand exactly 1.0 cm apart; the distance query then returns 0.9999 or 1.0001): that is no violation, a real dip is
    """
    edge = (
        REF_XY[0] + 1.5
    )  # the cylinder of carried_scene (48 sections) has a vertex exactly there
    low = box(
        [2, 2, 1.0], [REF_XY[0] - 1.5 - 0.5 - 1.0, REF_XY[1], 0.5]
    )  # 0.5 cm beside the part, 1 cm high: the part starts close

    def wall(gap, z0, z1):
        return box(
            [2, 6, z1 - z0], [edge + gap + 1.0, REF_XY[1], (z0 + z1) / 2]
        )  # its face `gap` beside the part

    text = (
        f'carried part B closer than {ROBOT_PART_CLEARANCE} cm to lying part'
    )
    state = {'B': 'pickup', 'L': 'pickup', 'W1': 'pickup', 'W2': 'pickup'}
    for upper_gap, violation in [
        (ROBOT_PART_CLEARANCE - 3e-4, False),
        (ROBOT_PART_CLEARANCE + 3e-4, False),
        (ROBOT_PART_CLEARANCE - 0.01, True),
    ]:
        q0 = carried_scene(
            c,
            extra_lying={
                'L': low,
                'W1': wall(ROBOT_PART_CLEARANCE + 2e-4, 4, 8),
                'W2': wall(upper_gap, 8, 14),
            },
        )
        path = [q0] + [
            carry_to(c, q0, REF_XY, bottom)
            for bottom in [1.0, 2.2, 3.0, 6.0, 8.0]
        ]  # clear of L from 2.2, beside W2 from 6.0
        assert (
            has(c.check_path(path, 0.5, state, carried='B'), text) == violation
        ), upper_gap
    # the gripper beside a lying part: the same tolerance
    part = box([6, 6, 2], [*REF_XY, 1.0])
    set_scene(c, lying={'C': part})
    for gap, violation in [
        (ROBOT_PART_CLEARANCE - 3e-4, False),
        (ROBOT_PART_CLEARANCE - 0.01, True),
    ]:
        q = ik_down(c, REF_XY, 2.0 + gap)
        assert abs(gripper_distance(c, q, 0.5, part) - gap) < 1e-4
        assert (
            has(
                c.check_path([q], 0.5, {'C': 'pickup'}),
                f'closer than {ROBOT_PART_CLEARANCE} cm to lying part C',
            )
            == violation
        ), gap


@test
def pickup_with_the_gripper_closed(c):
    """
    a neighbour beside a finger: when the finger closes onto the part it comes closer to that neighbour. The pickup check covers
    the opened gripper (as it arrives) and the closed one (the first pose of the transport).
    """
    q = ik_down(c, REF_XY, 1.0)
    closed, opened = 0.5, 0.6
    names = get_gripper_finger_names('panda')
    vertices = gripper_vertices(c, q, closed)
    u = vertices[names[0]].mean(axis=0) - vertices[names[1]].mean(
        axis=0
    )  # closing direction of the fingers (horizontal)
    u[2] = 0
    u /= np.linalg.norm(u)
    w = np.cross([0, 0, 1.0], u)
    origin = np.array([*REF_XY, 0.0])

    def placed(
        extents, centre
    ):  # a box given along (closing direction, across, up) around the gripper centre
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 3] = (
            u,
            w,
            origin + u * centre[0] + w * centre[1] + [0, 0, centre[2]],
        )
        mesh = trimesh.creation.box(extents)
        mesh.apply_transform(T)
        return mesh

    finger = vertices[names[0]] - origin
    inner, side = (
        (finger @ u).min(),
        (finger @ w).max(),
    )  # inner face of the finger (2 cm from the centre), its side
    part = placed(
        [8 * closed, 1.5, 3.0], [0, 0, 1.5]
    )  # the part to pick: between the closed fingers

    def scene(across):
        neighbour = placed(
            [1.0, 1.0, 1.5], [inner + 0.05 - 0.5, side + across + 0.5, 0.75]
        )  # beside the finger, level with its inner face
        set_scene(c, lying={'P': part, 'N': neighbour})
        return gripper_distance(c, q, closed, neighbour), gripper_distance(
            c, q, opened, neighbour
        )

    across = 1.0
    for _ in range(3):
        across -= scene(across)[0] - (
            ROBOT_PART_CLEARANCE - 0.03
        )  # the closed finger 0.97 cm from the neighbour
    d_closed, d_opened = scene(across)
    assert (
        abs(d_closed - (ROBOT_PART_CLEARANCE - 0.03)) < 0.005
        and d_opened > ROBOT_PART_CLEARANCE + 0.01
    ), (d_closed, d_opened)
    state = {'P': 'pickup', 'N': 'pickup'}
    assert (
        c.check_pickup(q, opened, 'P', state) == []
    )  # the opened gripper alone passes
    both = c.check_pickup(q, opened, 'P', state, closed_ratio=closed)
    assert (
        len(both) == 1
        and 'gripper closed' in both[0]
        and f'closer than {ROBOT_PART_CLEARANCE} cm to lying part N' in both[0]
    ), both
    assert (
        c.check_path([q], closed, state, carried='P') != []
    )  # what the transport would report at its first pose
    scene(across + 0.1)  # 1 mm further away: fine in both states
    assert c.check_pickup(q, opened, 'P', state, closed_ratio=closed) == []
    # an opened gripper that touches the part to pick is still reported, whatever the closed state
    assert has(
        c.check_pickup(q, closed - 0.02, 'P', state, closed_ratio=closed),
        'gripper touches part P before grasping it',
    )
    # pickup_blocked, the test made before an arm pose is searched, looks at the closed gripper as well: the closed finger
    # 0.85 cm from the neighbour (clearly too close), the opened one further than the limit of that test
    for _ in range(3):
        across -= scene(across)[0] - (ROBOT_PART_CLEARANCE - 0.15)
    d_closed, d_opened = scene(across)
    assert (
        abs(d_closed - (ROBOT_PART_CLEARANCE - 0.15)) < 0.005
        and d_opened > ROBOT_PART_CLEARANCE - PICKUP_POSE_SLACK + 0.005
    ), (d_closed, d_opened)
    T = c.gripper_transform(q)
    assert c.pickup_blocked(T, opened, 'P', state) is None
    blocked = c.pickup_blocked(T, opened, 'P', state, closed_ratio=closed)
    assert (
        blocked is not None
        and blocked.startswith('gripper panda_')
        and blocked.endswith(
            f'closer than {ROBOT_PART_CLEARANCE} cm to lying part N'
        )
    ), blocked
    assert has(
        c.check_pickup(q, opened, 'P', state, closed_ratio=closed),
        f'closer than {ROBOT_PART_CLEARANCE} cm to lying part N',
    )


@test
def pickup_tests_before_an_arm_pose_is_searched(c):
    """
    SceneChecker.pickup_blocked: what check_pickup says about the gripper and the wrist, known from the gripper pose alone. It
    names only spots that check_pickup rejects with every arm pose, and it leaves the last PICKUP_POSE_SLACK below the limit
    to the full check.
    """
    q = ik_down(c, REF_XY, 10.0)
    T = c.gripper_transform(q)
    # the wrist link follows the gripper: its pose from the gripper pose is its pose from the arm pose, whatever the arm does
    rng = np.random.RandomState(0)
    for _ in range(5):
        q_any = c.chain.active_to_full(
            c.chain.active_from_full(q) + rng.uniform(-0.4, 0.4, 7)
        )
        arm_tf, _, _ = c.planner.get_meshes_transforms(q_any, 0.5, None, None)
        assert np.allclose(
            c.gripper_transform(q_any) @ c.wrist_offset(),
            arm_tf[WRIST_LINK],
            atol=1e-9,
        )
    far = box(
        [1, 1, 1], [REF_XY[0] + 40, REF_XY[1], 0.5]
    )  # the part to pick, out of the way in these scenes

    # gripper to a lying neighbour: a slab under the fingers, `gap` below them
    for gap, blocked_expected, check_expected in [
        (ROBOT_PART_CLEARANCE + 0.05, False, False),
        (ROBOT_PART_CLEARANCE - 0.05, False, True),
        (ROBOT_PART_CLEARANCE - PICKUP_POSE_SLACK - 0.05, True, True),
    ]:
        slab = box([12, 12, 2], [REF_XY[0], REF_XY[1], 10.0 - gap - 1.0])
        set_scene(c, lying={'P': far, 'N': slab})
        assert abs(gripper_distance(c, q, 0.5, slab) - gap) < 1e-3
        state = {'P': 'pickup', 'N': 'pickup'}
        blocked = c.pickup_blocked(T, 0.5, 'P', state)
        assert (blocked is not None) == blocked_expected, (gap, blocked)
        if blocked is not None:
            assert blocked.startswith('gripper panda_') and blocked.endswith(
                f'closer than {ROBOT_PART_CLEARANCE} cm to lying part N'
            ), blocked
        assert (
            has(
                c.check_pickup(q, 0.5, 'P', state),
                f'closer than {ROBOT_PART_CLEARANCE} cm to lying part N',
            )
            == check_expected
        ), gap
        # the same slab as an assembled part: the gripper has no such distance rule to the assembly, nothing is blocked
        set_scene(c, lying={'P': far}, final={'N': slab})
        assert (
            c.pickup_blocked(T, 0.5, 'P', {'P': 'pickup', 'N': 'final'})
            is None
        )

    # wrist to any part (lying, assembled, or the one that is picked): a box right above the highest point of the wrist link
    arm_tf, _, _ = c.planner.get_meshes_transforms(q, 0.5, None, None)
    wrist = (
        c.arm_hull[WRIST_LINK] @ arm_tf[WRIST_LINK][:3, :3].T
        + arm_tf[WRIST_LINK][:3, 3]
    )
    top = wrist[np.argmax(wrist[:, 2])]
    for gap, blocked_expected in [
        (ROBOT_PART_CLEARANCE + 0.05, False),
        (ROBOT_PART_CLEARANCE - 0.05, False),
        (ROBOT_PART_CLEARANCE - PICKUP_POSE_SLACK - 0.05, True),
    ]:
        above = box([3, 3, 3], [top[0], top[1], top[2] + gap + 1.5])
        for scene, state, name in [
            (
                dict(lying={'P': far, 'W': above}),
                {'P': 'pickup', 'W': 'pickup'},
                'W',
            ),
            (
                dict(lying={'P': far}, final={'W': above}),
                {'P': 'pickup', 'W': 'final'},
                'W',
            ),
            (dict(lying={'P': above}), {'P': 'pickup'}, 'P'),
        ]:
            set_scene(c, **scene)
            blocked = c.pickup_blocked(T, 0.5, 'P', state)
            assert blocked == (
                f'arm {WRIST_LINK} closer than {ROBOT_PART_CLEARANCE} cm to part {name}'
                if blocked_expected
                else None
            ), (gap, state, blocked)
            if (
                gap < ROBOT_PART_CLEARANCE
            ):  # what the full check says with the arm pose: the same rule (it names the nearest arm link)
                assert has(
                    c.check_pickup(q, 0.5, 'P', state),
                    f'closer than {ROBOT_PART_CLEARANCE} cm to part {name}',
                ), (gap, state)


@test
def wall_margin_of_the_hand(c):
    """
    get_hand_wall_margin: the wall margin of the two frames that follow the gripper rigidly (J7, flange), from the gripper pose
    alone. It equals what the arm pose gives for these two frames, so it is never below the margin of the whole arm.
    """
    planes, thresholds = get_panda_virtual_walls()
    base_rot = R.from_euler('xyz', c.chain.base_euler).as_matrix()
    rng = np.random.RandomState(1)
    poses = [
        ik_down(c, REF_XY, 10.0),
        pose_near_the_wall(c, VIRTUAL_WALL_SAFETY),
        c.chain.active_to_full(c.chain.rest_q),
    ]
    poses += [
        c.chain.active_to_full(
            c.chain.active_from_full(poses[0]) + rng.uniform(-0.5, 0.5, 7)
        )
        for _ in range(8)
    ]
    decisive = 0
    for q in poses:
        fk = c.chain.forward_kinematics(q, full_kinematics=True)
        direct = np.inf
        for T, threshold in zip(fk[-2:], thresholds[-2:]):
            pos = base_rot.T @ (T[:3, 3] - c.chain.base_pos) / 100
            direct = min(
                [direct]
                + [
                    abs(pos[axis] - offset) - threshold
                    for axis, offset in planes
                ]
            )
        hand = get_hand_wall_margin(c.chain, 'panda', c.gripper_transform(q))
        whole = get_virtual_wall_margin(c.chain, q)
        assert abs(hand - direct * 100) < 1e-6 and hand >= whole - 1e-9, (
            hand,
            direct * 100,
            whole,
        )
        decisive += abs(hand - whole) < 1e-6
    assert (
        0 < decisive < len(poses)
    )  # for some poses the hand is what is nearest to a wall, for others another joint


# ------------------------------------------------------------------------------- motion planner of the real arm (strict)


def make_planner(c):
    planner = ArmMotionPlanner(
        get_arm_chain('panda', 'move'),
        'panda',
        False,
        virtual_walls=True,
        table_z=get_panda_base_z() + get_panda_desk_z(),
        part_clearance=TRANSFER_PART_CLEARANCE + PLANNER_TABLE_MARGIN,
    )
    planner.strict = True
    ArmMotionPlanner.VISUALIZE = False
    return planner


@test
def planner_rejects_rule_breaking_poses(c):
    set_scene(c)
    planner = make_planner(c)
    _, sample_fn, _, collision_fn = planner.get_fns(None, None, [], 0.5)
    q_ref = c.chain.active_from_full(ik_down(c, REF_XY, 10.0))
    assert not collision_fn(q_ref)
    # joint limits: the free path keeps a reserve, a fixed pose (buffered=False) only the checked value
    limit = PANDA_JOINT_LIMITS[1][6]
    q = q_ref.copy()
    q[6] = limit - (MIN_JOINT_LIMIT_MARGIN - 0.01)
    assert collision_fn(q) and collision_fn(q, buffered=False)
    q[6] = limit - (MIN_JOINT_LIMIT_MARGIN + 0.01)
    assert (
        collision_fn(q)
        and not collision_fn(q, buffered=False)
        and not collision_fn(q, reserve=False)
    )
    q[6] = limit - (MIN_JOINT_LIMIT_MARGIN + PLANNER_JOINT_MARGIN + 0.01)
    assert not collision_fn(q)
    # singularities
    assert collision_fn(
        c.chain.active_from_full(pose_with_sigma(c, MIN_SIGMA - 0.005)),
        buffered=False,
    )
    q_between = c.chain.active_from_full(
        pose_with_sigma(c, MIN_SIGMA + PLANNER_SIGMA_MARGIN / 2)
    )
    assert collision_fn(q_between) and not collision_fn(
        q_between, buffered=False
    )
    # a fixed end pose inside the reserve: the reserve is lowered to it, never below the checked value
    collision_fn.relax([q_between])
    assert not collision_fn(q_between)
    assert collision_fn(
        c.chain.active_from_full(pose_with_sigma(c, MIN_SIGMA - 0.005))
    )
    # samples are drawn inside the joint-limit rule
    random.seed(0)
    np.random.seed(0)
    _, sample_fn, _, collision_fn = planner.get_fns(None, None, [], 0.5)
    for _ in range(200):
        s, m, _ = joint_condition(sample_fn())
        assert (
            m >= MIN_JOINT_LIMIT_MARGIN + PLANNER_JOINT_MARGIN - 1e-9
            and s >= MIN_SIGMA + PLANNER_SIGMA_MARGIN - 1e-9
        )


@test
def planner_keeps_clearance_to_parts(c):
    planner = make_planner(c)
    part = box([12, 12, 2], [*REF_XY, 1.0])
    set_scene(c, lying={'A': part})
    _, _, _, collision_fn = planner.get_fns(None, None, [part], 0.5)

    def q(gap):
        return c.chain.active_from_full(ik_down(c, REF_XY, 2.0 + gap))

    reserve = max(ROBOT_PART_CLEARANCE, ASSEMBLY_CLEARANCE) + PLANNER_MARGIN
    assert collision_fn(q(ROBOT_PART_CLEARANCE - 0.1)) and collision_fn(
        q(reserve - 0.05)
    )  # the free path keeps the checked clearance plus the reserve ...
    assert not collision_fn(q(reserve + 0.05))
    assert not collision_fn(q(0.3), buffered=False) and collision_fn(
        q(-0.3), buffered=False
    )  # ... a straight move only must not touch
    # around the edges and corners of the box the answer follows the exact distance as well (the grown mesh sticks out there)
    _, _, _, free_fn = planner.get_fns(None, None, [], 0.5)
    rng, near, compared = np.random.default_rng(2), 0, 0
    for _ in range(150):
        q_full = ik_down(
            c, REF_XY + rng.uniform(-9, 9, 2), rng.uniform(0.5, 4.5)
        )
        q_active = c.chain.active_from_full(q_full)
        if free_fn(q_active):
            continue  # breaks another rule without any part
        d = gripper_distance(c, q_full, 0.5, part)
        if abs(d - reserve) < 0.02:
            continue  # too close to the limit to call
        compared += 1
        near += abs(d - reserve) < 0.6
        assert collision_fn(q_active) == (d < reserve), (d, q_active)
    assert compared > 100 and near > 15, (compared, near)


def lid_over_hand(c, q, open_ratio, gap):
    """a small plate gap cm above one end of the hand (away from the wrist): the gripper cannot move straight up"""
    hand = next(
        v
        for n, v in gripper_vertices(c, q, open_ratio).items()
        if n not in get_gripper_finger_names('panda')
    )
    centre = c.chain.forward_kinematics(q)[:3, 3]
    end = hand[
        np.argmax(np.linalg.norm(hand[:, :2] - centre[:2], axis=1))
    ]  # the hand is about 20 cm long
    xy = centre[:2] + (end[:2] - centre[:2]) * 0.85
    top = hand[np.linalg.norm(hand[:, :2] - xy, axis=1) < 2.0][:, 2].max()
    return box([2.5, 2.5, 1.0], [xy[0], xy[1], top + gap + 0.5])


def valid_fixed_pose(c, q):
    """a pose the rules accept as a grasp or release pose (the toy poses must not break a rule themselves)"""
    sigma, margin, _ = joint_condition(c.chain.active_from_full(q))
    return (
        sigma >= MIN_SIGMA
        and margin >= MIN_JOINT_LIMIT_MARGIN
        and get_virtual_wall_margin(c.chain, q) >= VIRTUAL_WALL_SAFETY
    )


def straightness(c, path):
    """(largest distance of a pose from the line start -> end [cm], largest change of orientation [rad]) of a path of full qs"""
    fks = [c.chain.forward_kinematics(q) for q in path]
    p0, p1 = fks[0][:3, 3], fks[-1][:3, 3]
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    off = max(
        np.linalg.norm((f[:3, 3] - p0) - ((f[:3, 3] - p0) @ d) * d)
        for f in fks
    )
    rot = max(
        np.arccos(
            np.clip((np.trace(fks[0][:3, :3].T @ f[:3, :3]) - 1) / 2, -1, 1)
        )
        for f in fks
    )
    return off, rot


@test
def planner_straight_move_out(c):
    planner = make_planner(c)
    part = cylinder(
        1.5, 4.0, REF_XY
    )  # stands between the open fingers (4 cm apart)
    set_scene(c, final={'C': part}, zone_center=REF_XY)
    q0 = ik_down(c, REF_XY, 2.0)
    assert 0 < gripper_distance(c, q0, 0.5, part) < 0.6
    _, _, _, collision_fn = planner.get_fns(None, None, [part], 0.5)
    path = planner.plan_path_straight_out(
        q0, collision_fn, np.array([0, 0, 1.0])
    )
    assert path is not None and len(path) > 2
    off, rot = straightness(c, path)
    assert off < 0.02 and rot < 1e-3, (off, rot)
    lows = [gripper_low(c, q, 0.5) for q in path]
    assert np.allclose(np.diff(lows), STRAIGHT_STEP, atol=0.02), np.diff(lows)
    distances = [gripper_distance(c, q, 0.5, part) for q in path]
    assert all(d > 0 for d in distances)  # never touches
    assert (
        distances[-1] >= ROBOT_PART_CLEARANCE and distances[-2] < distances[-1]
    )  # ends where the clearance holds, not earlier than needed
    assert distances[-2] < ROBOT_PART_CLEARANCE + STRAIGHT_STEP + 0.4
    assert c.check_path(path, 0.5, {'C': 'final'}, released_part='C') == []
    # a pose that is clear already does not move
    q_clear = ik_down(c, REF_XY, 12.0)
    assert (
        len(
            planner.plan_path_straight_out(
                q_clear, collision_fn, np.array([0, 0, 1.0])
            )
        )
        == 1
    )
    # blocked: a plate 1.2 cm above the hand; the gripper would have to move 2 cm to be clear of the part
    lid = lid_over_hand(c, q0, 0.5, 1.2)
    _, _, _, collision_fn = planner.get_fns(None, None, [part, lid], 0.5)
    assert not collision_fn(
        c.chain.active_from_full(q0), buffered=False
    )  # the start itself touches nothing
    assert (
        planner.plan_path_straight_out(q0, collision_fn, np.array([0, 0, 1.0]))
        is None
    )


@test
def planner_switch_between_two_parts(c):
    """release part C, move to part B and get ready to grasp it: the planned path passes every rule"""
    planner = make_planner(c)
    xy_c, xy_b = REF_XY, REF_XY + [-12.0, 0.0]
    part_c, part_b = cylinder(1.5, 4.0, xy_c), cylinder(1.9, 3.0, xy_b)
    set_scene(c, lying={'B': part_b}, final={'C': part_c}, zone_center=xy_c)
    q_release = ik_down(c, xy_c, 2.0, open_ratio=0.5)
    q_grasp = ik_down(c, xy_b, 1.2, open_ratio=0.6, q_init=q_release)
    assert valid_fixed_pose(c, q_release) and valid_fixed_pose(c, q_grasp)
    state = {'C': 'final', 'B': 'pickup'}
    random.seed(3)
    np.random.seed(3)
    start = time.time()
    up = np.array([0, 0, 1.0])
    path1, path2 = planner.plan_path_switch(
        q_release,
        q_grasp,
        part_meshes=[part_c, part_b],
        open_ratio=0.5,
        open_ratio_next=0.6,
        retract_start=up,
        retract_goal=up,
        retract_delta=5.0,
        max_speed=4.0,
    )
    assert path1 is not None and path2 is not None, planner.last_failure
    violations = c.check_path(
        path1, 0.5, state, pick_part='B', released_part='C'
    ) + c.check_path(path2, 0.6, state, pick_part='B', released_part='C')
    assert violations == [], violations
    # the gripper changes its width where it is clear of both parts
    q_change = path1[-1]
    for ratio in [0.5, 0.6]:
        assert (
            min(
                gripper_distance(c, q_change, ratio, m)
                for m in [part_c, part_b]
            )
            >= ROBOT_PART_CLEARANCE
        )
    # the first and last centimetres are straight
    n_out = next(
        k
        for k, q in enumerate(path1)
        if gripper_distance(c, q, 0.5, part_c) >= ROBOT_PART_CLEARANCE
    )
    assert straightness(c, path1[: n_out + 1])[0] < 0.05
    assert straightness(c, path2)[0] < 0.05
    print(
        f'      switch planned in {time.time() - start:.1f} s, {len(path1)} + {len(path2)} poses'
    )


@test
def planner_transport_of_a_part(c):
    """lift part B from the table, carry it to the assembly and stop above part C: the planned path passes every rule"""
    planner = make_planner(c)
    xy_c, xy_b = REF_XY, REF_XY + [-12.0, 0.0]
    part_c, part_b = (
        box([4, 4, 2], [xy_c[0], xy_c[1], 1.0]),
        cylinder(1.5, 3.0, xy_b),
    )
    set_scene(c, lying={'B': part_b}, final={'C': part_c}, zone_center=xy_c)
    planner.clearance_exempt_xy = (
        xy_c - (ASSEMBLY_ZONE_MARGIN - PLANNER_TABLE_MARGIN),
        xy_c + (ASSEMBLY_ZONE_MARGIN - PLANNER_TABLE_MARGIN),
    )
    q_grasp = ik_down(c, xy_b, 1.0)
    q_goal = ik_down(
        c, xy_c, 1.0 + 2.0 + 0.6, q_init=q_grasp
    )  # part B 0.6 cm above part C, as before an insertion
    assert valid_fixed_pose(c, q_grasp) and valid_fixed_pose(c, q_goal)
    random.seed(4)
    np.random.seed(4)
    start = time.time()
    up = np.array([0, 0, 1.0])
    path = planner.plan_path_with_grasp(
        q_grasp,
        q_goal,
        move_pickup_mesh=part_b,
        gripper_pickup_transform=c.gripper_transform(q_grasp),
        still_meshes=[part_c],
        open_ratio=0.5,
        retract_start=up,
        retract_goal=up,
        retract_delta=5.0,
        max_speed=4.0,
    )
    assert path is not None, planner.last_failure
    violations = c.check_path(
        path, 0.5, {'C': 'final', 'B': 'pickup'}, carried='B'
    )
    assert violations == [], violations
    print(
        f'      transport planned in {time.time() - start:.1f} s, {len(path)} poses'
    )


@test
def planner_fails_instead_of_planning_without_clearance(c):
    """a release pose that cannot be left straight: no path, the reason names the start"""
    planner = make_planner(c)
    part = cylinder(1.5, 4.0, REF_XY)
    q_release = ik_down(c, REF_XY, 2.0)
    bar = lid_over_hand(c, q_release, 0.5, 1.2)  # 1.2 cm above the hand
    set_scene(c, final={'C': part, 'D': bar}, zone_center=REF_XY)
    q_goal = ik_down(c, REF_XY + [-12.0, 0.0], 10.0, q_init=q_release)
    assert valid_fixed_pose(c, q_release) and valid_fixed_pose(c, q_goal)
    up = np.array([0, 0, 1.0])
    path1, path2 = planner.plan_path_switch(
        q_release,
        q_goal,
        part_meshes=[part, bar],
        open_ratio=0.5,
        open_ratio_next=0.5,
        retract_start=up,
        retract_goal=up,
        retract_delta=5.0,
        max_speed=4.0,
    )
    assert (
        path1 is None and path2 is None and planner.last_failure == 'start'
    ), planner.last_failure


# ------------------------------------------------------------ low insertions, and what a failed plan says about its cause


@test
def carried_part_low_in_the_assembly_zone(c):
    """inside the assembly zone a carried part keeps ZONE_PART_TABLE_CLEARANCE to the table; lower only in the last stretch of its path"""
    text = f'carried part B lower than {ZONE_PART_TABLE_CLEARANCE} cm above the table in the assembly zone'
    target = REF_XY + [10, 0]
    q0 = carried_scene(c, zone_center=target)
    state = {'B': 'pickup'}
    up, over = (
        carry_to(c, q0, REF_XY, TRANSFER_PART_CLEARANCE + 0.5),
        carry_to(c, q0, target, 4.0),
    )

    def low(height):
        return carry_to(c, q0, target, height)

    def check(path, **kw):
        return c.check_path(path, 0.5, state, carried='B', **kw)

    # dipping to the table and rising again: just below the limit is a violation, just above is not
    assert poses_of(
        check([q0, up, over, low(ZONE_PART_TABLE_CLEARANCE - 0.02), over]),
        text,
    ) == (3, 3, 1)
    assert (
        check([q0, up, over, low(ZONE_PART_TABLE_CLEARANCE + 0.02), over])
        == []
    )
    # coming down at the end of the path: the part is set down, or its insertion starts lower than the limit
    assert check([q0, up, over, low(0.5), low(0.1), low(0.0)]) == []
    assert check([q0, up, over, low(0.1)]) == []
    # only the last stretch: low, up again and low again counts the first time
    assert poses_of(
        check([q0, up, over, low(0.1), low(0.5), low(0.1)]), text
    ) == (3, 3, 1)
    # outside the zone the rule is another one: the part lies on the table where it is picked, and is lifted from there
    assert not has(check([q0, up]), 'in the assembly zone')
    # a part that reaches into the zone: its points inside count (cylinder radius 1.5, 0.3 cm of it inside)
    edge = target - [ASSEMBLY_ZONE_MARGIN + 1.5 - 0.3, 0]
    assert poses_of(
        check(
            [
                q0,
                up,
                carry_to(c, q0, edge, 4.0),
                carry_to(c, q0, edge, 0.1),
                carry_to(c, q0, edge, 4.0),
            ]
        ),
        text,
    ) == (3, 3, 1)
    outside = edge - [
        0.6,
        0,
    ]  # 0.3 cm outside: the rule of the transfer height applies there instead
    v = check(
        [
            q0,
            up,
            carry_to(c, q0, outside, 4.0),
            carry_to(c, q0, outside, 0.1),
            carry_to(c, q0, outside, 4.0),
        ]
    )
    assert not has(v, 'in the assembly zone') and has(
        v, 'outside the assembly zone'
    )
    # the insertion itself may end on the table
    set_scene(
        c,
        lying={'B': c.pickup_meshes['B']},
        final={'B': cylinder(1.5, 3.0, target)},
        zone_center=target,
    )
    q_end = ik_down(
        c, target, 1.0
    )  # holds B standing on the table at its final pose
    path = [
        ik_down(c, target, 1.0 + lift, q_init=q_end)
        for lift in [2.0, 0.1, 1.0]
    ] + [q_end]
    assert not has(
        c.check_path(path, 0.5, {}, carried='B', inserting=True),
        'above the table',
    )


@test
def failure_file_names_the_cause(c):
    import json
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix='test_plan_rules_')
    try:

        def read():
            return json.load(open(os.path.join(tmp, FAILURE_FILE)))

        write_failure(tmp, 'run_motion_plan', '3', np.int64(17), 'no way')
        assert read() == {
            'stage': 'run_motion_plan',
            'part': '3',
            'grasp_id': 17,
            'reason': 'no way',
            'cause': CAUSE_GRASP,
        }
        for cause in [
            CAUSE_SPOT,
            CAUSE_INSERTION,
            CAUSE_HOLD_GRASP,
            CAUSE_GRASP,
        ]:
            write_failure(tmp, 'run_motion_plan', '3', 17, 'no way', cause)
            assert read()['cause'] == cause
        write_failure(tmp, 'run_pickup_gen', '0', None, 'the human part')
        assert read()['grasp_id'] is None
        try:
            write_failure(tmp, 'run_motion_plan', '3', 17, 'no way', 'weather')
            raise RuntimeError('an unknown cause was accepted')
        except AssertionError:
            pass
        clear_failure(tmp)
        assert not os.path.exists(os.path.join(tmp, FAILURE_FILE))
        clear_failure(tmp)  # nothing to remove: fine
    finally:
        shutil.rmtree(tmp)
    assert sorted(CAUSE_TEXT) == sorted(
        [CAUSE_GRASP, CAUSE_SPOT, CAUSE_INSERTION, CAUSE_HOLD_GRASP]
    )


def pose_near_the_wall(c, margin):
    """full q, fingers down, 25 cm above the table, whose distance to the side wall is just below margin"""
    q = ik_down(c, REF_XY, 10.0)
    for dy in np.arange(0, 40, 0.1):
        q_next = c.planner.inverse_kinematics(
            np.array([REF_XY[0], REF_XY[1] - dy, 25.0]), DOWN, q_init=q
        )
        if q_next is None:
            continue
        q = q_next
        if get_virtual_wall_margin(c.chain, q) < margin:
            return q
    raise AssertionError('no pose near the wall found')


@test
def planner_names_what_rejects_a_pose(c):
    """collision_fn.hit: the test that rejected the last rejected pose; every name belongs to the arm / gripper or to the carried part"""
    set_scene(c)
    planner = make_planner(c)
    active = c.chain.active_from_full
    plate = box([12, 12, 2], [*REF_XY, 1.0])
    _, _, _, collision_fn = planner.get_fns(None, None, [plate], 0.5)
    assert collision_fn.hit is None

    def above(gap):
        return active(ik_down(c, REF_XY, 2.0 + gap))

    assert not collision_fn(above(3.0)) and collision_fn.hit is None
    seen = []

    def rejected(fn, q, **kw):
        assert fn(q, **kw), kw
        seen.append(fn.hit)
        return fn.hit

    assert rejected(collision_fn, above(0.5)) == 'gripper and part'
    q_limit = above(3.0).copy()
    q_limit[6] = PANDA_JOINT_LIMITS[1][6] - (MIN_JOINT_LIMIT_MARGIN - 0.01)
    assert rejected(collision_fn, q_limit) == 'joint limit'
    assert (
        rejected(collision_fn, active(pose_with_sigma(c, MIN_SIGMA - 0.005)))
        == 'singularity'
    )
    assert (
        rejected(
            collision_fn,
            active(pose_near_the_wall(c, VIRTUAL_WALL_SAFETY - 0.05)),
            buffered=False,
        )
        == 'virtual wall'
    )
    _, _, _, free_fn = planner.get_fns(None, None, [], 0.5)
    assert (
        rejected(free_fn, active(ik_down(c, REF_XY, 0.1)), buffered=False)
        == 'gripper and ground'
    )  # 0.1 cm above the table: in the table slab
    # a pose that passes does not change the name of the last rejected one
    assert not free_fn(above(3.0)) and free_fn.hit == 'gripper and ground'
    # with a part in the gripper
    q0 = carried_scene(c)
    part = c.pickup_meshes['B']
    neighbour = box([2, 2, 8], [REF_XY[0] + 8.0, REF_XY[1], 4.0])
    _, _, _, carry_fn = planner.get_fns(
        part, c.gripper_transform(q0), [neighbour], 0.5
    )

    def at(xy, bottom):
        return active(carry_to(c, q0, xy, bottom))

    assert (
        rejected(carry_fn, at(REF_XY, 0.0)) == 'move part and ground'
    )  # where it lies: inside the planner's table buffer
    assert not carry_fn(at(REF_XY, 1.0))
    assert (
        rejected(carry_fn, at(REF_XY, 1.0), part_clearance=True)
        == 'carried part too close to the table'
    )  # the transfer height, on request
    assert not carry_fn(
        at(REF_XY, TRANSFER_PART_CLEARANCE + PLANNER_TABLE_MARGIN + 0.1),
        part_clearance=True,
    )
    beside = (
        REF_XY + [8.0 - 1.0 - 1.5 - 0.6, 0]
    )  # the cylinder (radius 1.5) 0.6 cm beside the neighbour, at half its height
    assert rejected(carry_fn, at(beside, 2.0)) in [
        'move part and part',
        'gripper and part',
    ]  # (a finger can be the nearer one)
    assert (
        rejected(carry_fn, at(REF_XY + [8.0, 0], 8.0 + 0.6))
        == 'move part and part'
    )  # 0.6 cm above the neighbour: only the part is near it
    assert set(HITS_ARM) & set(HITS_CARRIED_PART) == set() and all(
        name in HITS_ARM + HITS_CARRIED_PART for name in seen
    ), seen
    assert {
        'carried part too close to the table',
        'move part and ground',
        'move part and part',
    } == set(HITS_CARRIED_PART)


@test
def planner_says_why_a_straight_move_fails(c):
    """plan_path_straight_out: last_block names what stopped it and after how many cm; blocked_by_carried_part sorts it"""
    planner = make_planner(c)
    up = np.array([0, 0, 1.0])
    part = cylinder(1.5, 4.0, REF_XY)
    set_scene(c, final={'C': part}, zone_center=REF_XY)
    q0 = ik_down(c, REF_XY, 2.0)
    _, _, _, collision_fn = planner.get_fns(None, None, [part], 0.5)
    assert (
        planner.plan_path_straight_out(q0, collision_fn, up) is not None
        and planner.last_block is None
    )
    assert not planner.blocked_by_carried_part()
    # a step touches something: the plate above the hand
    lid = lid_over_hand(c, q0, 0.5, 1.2)
    _, _, _, collision_fn = planner.get_fns(None, None, [part, lid], 0.5)
    assert planner.plan_path_straight_out(q0, collision_fn, up) is None
    what, moved = planner.last_block
    assert (
        what == 'gripper and part'
        and 1.2 <= moved <= 1.2 + STRAIGHT_STEP + 1e-9
    ), planner.last_block  # the first step that reaches the plate
    assert not planner.blocked_by_carried_part()
    # a step breaks a rule of the arm: towards the virtual wall (as at a pickup spot beside the wall, the hand leaning to it)
    q_wall = pose_near_the_wall(c, VIRTUAL_WALL_SAFETY + 0.3)
    top = gripper_low(c, q_wall, 0.5) - 0.6
    tall = box(
        [2, 2, top], [*c.chain.forward_kinematics(q_wall)[:2, 3], top / 2]
    )  # its top 0.6 cm below the gripper: no contact, but not clear either
    _, _, _, collision_fn = planner.get_fns(None, None, [tall], 0.5)
    assert collision_fn(
        c.chain.active_from_full(q_wall), reserve=False
    ) and not collision_fn(c.chain.active_from_full(q_wall), buffered=False)
    assert (
        planner.plan_path_straight_out(
            q_wall, collision_fn, np.array([0, -1.0, 0])
        )
        is None
    )
    assert (
        planner.last_block[0] == 'virtual wall'
        and 0 < planner.last_block[1] <= 0.5
        and not planner.blocked_by_carried_part()
    ), planner.last_block
    # no clear pose within STRAIGHT_MAX: what keeps it from being clear at the end
    import planning.robot.motion_plan_arm as motion_plan_arm

    saved = motion_plan_arm.STRAIGHT_MAX
    motion_plan_arm.STRAIGHT_MAX = 2.0
    try:
        q0 = carried_scene(c)
        carried = c.pickup_meshes['B']
        _, _, _, carry_fn = planner.get_fns(
            carried, c.gripper_transform(q0), [], 0.5
        )
        q_low = carry_to(
            c, q0, REF_XY, 0.65
        )  # the part 0.65 cm above the table, moved along the table (an insertion that starts low)
        assert (
            planner.plan_path_straight_out(
                q_low, carry_fn, np.array([0, 1.0, 0])
            )
            is None
        )
        assert planner.last_block == (
            'never clear of: move part and ground',
            2.0,
        ), planner.last_block
        assert planner.blocked_by_carried_part()
        # the same with the gripper: a long plate under it
        plate = box([12, 40, 2], [REF_XY[0], REF_XY[1] + 10, 1.0])
        _, _, _, plate_fn = planner.get_fns(None, None, [plate], 0.5)
        assert (
            planner.plan_path_straight_out(
                ik_down(c, REF_XY, 2.5), plate_fn, np.array([0, 1.0, 0])
            )
            is None
        )
        assert (
            planner.last_block == ('never clear of: gripper and part', 2.0)
            and not planner.blocked_by_carried_part()
        ), planner.last_block
    finally:
        motion_plan_arm.STRAIGHT_MAX = saved


@test
def planner_lets_a_carried_part_come_low_in_the_assembly_zone(c):
    """
    the table has a buffer for the carried part (0.78 cm), except in the zone around the assembly: there the part only has to
    stay off the table slab (ZONE_PART_TABLE_CLEARANCE), so that an insertion that starts just above the table has an approach
    """
    planner = make_planner(c)
    active = c.chain.active_from_full
    zone_xy = REF_XY + [10.0, 0.0]
    margin = ASSEMBLY_ZONE_MARGIN - PLANNER_TABLE_MARGIN
    planner.clearance_exempt_xy = (zone_xy - margin, zone_xy + margin)
    q0 = carried_scene(c, zone_center=zone_xy)
    _, _, _, collision_fn = planner.get_fns(
        c.pickup_meshes['B'], c.gripper_transform(q0), [], 0.5
    )

    def at(xy, bottom):
        return active(carry_to(c, q0, xy, bottom))

    buffer_top = planner.ground_mesh_buffered.bounds[1, 2]
    assert (
        abs(planner.ground_mesh.bounds[1, 2] - ZONE_PART_TABLE_CLEARANCE)
        < 1e-9
        and 0.7 < buffer_top < 0.85
    )  # slab and buffer of the planner's table
    # outside the zone: the buffer
    assert (
        collision_fn(at(REF_XY, 0.65))
        and collision_fn.hit == 'move part and ground'
    )
    assert collision_fn(at(REF_XY, buffer_top - 0.03)) and not collision_fn(
        at(REF_XY, buffer_top + 0.03)
    )
    # inside: the slab
    assert not collision_fn(at(zone_xy, 0.65))
    assert not collision_fn(at(zone_xy, ZONE_PART_TABLE_CLEARANCE + 0.03))
    assert (
        collision_fn(at(zone_xy, ZONE_PART_TABLE_CLEARANCE - 0.03))
        and collision_fn.hit == 'move part and ground'
    )
    # a part that sticks out of the zone keeps the buffer (cylinder radius 1.5: its rim 0.3 cm outside, then 0.3 cm inside)
    edge = zone_xy + [margin + 0.3 - 1.5, 0]
    assert collision_fn(at(edge, 0.65)) and not collision_fn(
        at(edge - [0.6, 0], 0.65)
    )
    # the test of the zone itself
    inside, outside = (
        np.array(
            [
                [zone_xy[0], zone_xy[1], 0.0],
                [zone_xy[0] + margin - 0.01, zone_xy[1] - margin + 0.01, 5.0],
            ]
        ),
        np.array([[zone_xy[0] + margin + 0.01, zone_xy[1], 0.0]]),
    )
    assert (
        planner.in_exempt_zone(inside)
        and not planner.in_exempt_zone(outside)
        and not planner.in_exempt_zone(np.vstack([inside, outside]))
    )
    planner.clearance_exempt_xy = None
    assert not planner.in_exempt_zone(inside) and collision_fn(
        at(zone_xy, 0.65)
    )  # no zone: the buffer everywhere


@test
def planner_approaches_an_insertion_that_starts_low(c):
    """
    a part is inserted sideways 0.65 cm above the table (gamepad part 4). In the assembly zone the transport reaches the start
    of that insertion and passes every rule; the same goal outside the zone has no approach, and the reason names the carried part
    """
    import planning.robot.motion_plan_arm as motion_plan_arm

    xy_c, xy_b = REF_XY, REF_XY + [-12.0, 0.0]
    part_c, part_b = (
        box([4, 4, 2], [xy_c[0] + 6.0, xy_c[1], 1.0]),
        cylinder(1.5, 3.0, xy_b),
    )  # the part slides towards part C along x
    margin = ASSEMBLY_ZONE_MARGIN - PLANNER_TABLE_MARGIN
    sideways = np.array([-1.0, 0, 0])  # back out of the insertion
    q_grasp = ik_down(c, xy_b, 1.0)
    q_goal = ik_down(c, xy_c, 1.0 + 0.65, q_init=q_grasp)
    assert valid_fixed_pose(c, q_grasp) and valid_fixed_pose(c, q_goal)

    def plan(planner):
        return planner.plan_path_with_grasp(
            q_grasp,
            q_goal,
            move_pickup_mesh=part_b,
            gripper_pickup_transform=c.gripper_transform(q_grasp),
            still_meshes=[part_c],
            open_ratio=0.5,
            retract_start=np.array([0, 0, 1.0]),
            retract_goal=sideways,
            retract_delta=5.0,
            max_speed=4.0,
        )

    # the zone around the assembly: a path, clean
    set_scene(
        c,
        lying={'B': part_b},
        final={'C': part_c},
        zone_center=xy_c + [3.0, 0],
    )
    c.zone = (
        np.array([xy_c[0] - 5.0, xy_c[1] - 5.0]),
        np.array([xy_c[0] + 11.0, xy_c[1] + 5.0]),
    )  # footprint of B at the goal and of C, plus the margin
    planner = make_planner(c)
    planner.clearance_exempt_xy = (
        c.zone[0] + PLANNER_TABLE_MARGIN,
        c.zone[1] - PLANNER_TABLE_MARGIN,
    )
    random.seed(5)
    np.random.seed(5)
    path = plan(planner)
    assert path is not None, (planner.last_failure, planner.last_block)
    assert (
        c.check_path(path, 0.5, {'C': 'final', 'B': 'pickup'}, carried='B')
        == []
    )
    low = [
        (
            v := c.pickup_meshes['B'].vertices
            @ (
                T := c.gripper_transform(q)
                @ np.linalg.inv(c.gripper_transform(q_grasp))
            )[:3, :3].T
            + T[:3, 3]
        )[:, 2].min()
        for q in path
    ]
    assert (
        abs(low[-1] - 0.65) < 0.01
        and min(low[len(low) // 2 :]) > ZONE_PART_TABLE_CLEARANCE
    )  # it arrives low and never touches the table on the way
    # the same goal with the zone elsewhere (as before 2026-11-11 everywhere): no approach, because of the part itself
    far_zone = xy_c + [40.0, 40.0]
    planner = make_planner(c)
    planner.clearance_exempt_xy = (far_zone - margin, far_zone + margin)
    saved = motion_plan_arm.STRAIGHT_MAX
    motion_plan_arm.STRAIGHT_MAX = 3.0
    try:
        assert plan(planner) is None
    finally:
        motion_plan_arm.STRAIGHT_MAX = saved
    assert (
        planner.last_failure == 'goal' and planner.blocked_by_carried_part()
    ), (planner.last_failure, planner.last_block)
    reason = no_path_reason(planner)
    assert (
        reason
        == 'no path found: no straight move into the goal pose from a pose clear of everything (never clear of: move part and ground after 3.00 cm)'
    ), reason
    assert (
        failure_cause(
            'transport',
            True,
            planner.last_failure,
            planner.blocked_by_carried_part(),
            [reason],
        )
        == CAUSE_INSERTION
    )


@test
def assembly_zone_is_wide_enough_for_gamepad(c):
    """
    ASSEMBLY_ZONE_MARGIN: gamepad part 4 (6 cm long, pushed in sideways 0.65 cm above the table) is clear of its neighbours
    only where its far end is 3.28 cm beyond the footprint of the assembly (measured on exp_gamepad_flat_run4, 2026-11-12).
    That pose has to lie inside the zone of the planner (a little smaller than the checked zone), or the part has no approach.
    """
    reach = 3.28
    planner = make_move_planner(
        get_arm_chain('panda', 'move'),
        'panda',
        False,
        None,
        {'A': box([13.6, 10.6, 3.0], [0.0, -15.0, 1.5])},
    )  # the footprint of gamepad
    lower, upper = planner.clearance_exempt_xy
    assert np.allclose(
        upper,
        [
            6.8 + ASSEMBLY_ZONE_MARGIN - PLANNER_TABLE_MARGIN,
            -9.7 + ASSEMBLY_ZONE_MARGIN - PLANNER_TABLE_MARGIN,
        ],
    )
    part_at_the_clear_pose = np.array(
        [[-0.7, -9.7 + reach - 6.0, 0.65], [0.7, -9.7 + reach, 1.85]]
    )  # corners of part 4 there
    assert planner.in_exempt_zone(part_at_the_clear_pose)
    assert (
        ASSEMBLY_ZONE_MARGIN - PLANNER_TABLE_MARGIN >= reach + 0.2
    )  # with the reserve the planner adds elsewhere
    c.zone = (lower - PLANNER_TABLE_MARGIN, upper + PLANNER_TABLE_MARGIN)
    assert np.all(part_at_the_clear_pose[:, :2] >= c.zone[0]) and np.all(
        part_at_the_clear_pose[:, :2] <= c.zone[1]
    )  # and inside the checked zone


@test
def arms_keep_their_distance(c):
    """
    SceneChecker.other_arm_violations: the arm that moves keeps
    ARM_ARM_CLEARANCE to the other arm, which stands still, except in a
    stretch at the start and at the end of its path; it never touches it,
    also not while inserting. Tested for both arms: the hold arm's
    gripper points down over a spot, the move arm's gripper comes to it
    along x, and the distance of the two arms is measured on the meshes.
    """
    set_scene(c)
    spot = np.array([0.0, -25.0])  # between the two arms (world, cm)
    hold = c.arm_planner('hold')
    q_hold = hold.inverse_kinematics(
        np.array([*spot, 21.2]), DOWN, q_init=None
    )
    assert q_hold is not None, 'the hold arm does not reach the spot'

    def distance(q_move, q_other=q_hold):
        """distance of the two arms (links and grippers), 0 if touching"""
        move_arm = c.placed_arm('move', q_move, 0.5)
        hold_arm = c.placed_arm('hold', q_other, 0.5)
        if move_arm.in_collision_other(hold_arm):
            return 0.0
        return move_arm.min_distance_other(hold_arm)

    # The move gripper at the height of the hold gripper, further and
    # further away along +x (towards the move arm).
    poses, q = [], None
    for dx in np.arange(4.0, 40.0, 0.5):
        q = ik_down(c, spot + [dx, 0.0], 10.0, q_init=q)
        poses.append((distance(q), q))
    touching = next(q for d, q in poses if d == 0.0)
    close = next(q for d, q in poses if 0.2 < d < ARM_ARM_CLEARANCE - 0.2)
    clear = next(q for d, q in poses if d > ARM_ARM_CLEARANCE + 1.0)

    def check(path, inserting=False):
        return c.other_arm_violations(
            'move', path, 0.5, q_hold, 0.5, inserting=inserting
        )

    passes = f'passes closer than {ARM_ARM_CLEARANCE} cm to hold'
    # close only in the middle of the path: not allowed
    assert has(check([clear, close, clear]), passes)
    # close at the start and at the end (leaving, arriving): allowed
    assert check([close, clear, close]) == []
    # touching is never allowed, also at the end of a path
    assert has(check([clear, touching]), 'move arm touches the hold arm')
    # while inserting only contact counts
    assert check([close, close], inserting=True) == []
    assert has(
        check([close, touching], inserting=True),
        'move arm touches the hold arm',
    )
    # check_path applies the rule when it is told where the hold arm is
    assert has(
        c.check_path(
            [clear, close, clear], 0.5, {}, other_arm=(q_hold, 0.5)
        ),
        passes,
    )
    assert not has(c.check_path([clear, close, clear], 0.5, {}), passes)

    # The same rule for a path of the hold arm, the move arm standing at
    # `close`: away from the spot, to it, away again.
    q_far = hold.inverse_kinematics(
        np.array([spot[0] - 20.0, spot[1], 21.2]), DOWN, q_init=q_hold
    )
    assert q_far is not None and distance(close, q_far) > 3.0
    assert has(
        c.other_arm_violations(
            'hold', [q_far, q_hold, q_far], 0.5, close, 0.5
        ),
        f'passes closer than {ARM_ARM_CLEARANCE} cm to move',
    )
    # A part carried by the hold arm counts too. carried=(mesh, None):
    # the part is where the mesh is at the first pose of the path. A box
    # put right at the move gripper touches the move arm.
    gripper_pos = c.gripper_transform(close)[:3, 3]
    at_move_gripper = box([2, 2, 2], gripper_pos)
    assert has(
        c.other_arm_violations(
            'hold', [q_far], 0.5, close, 0.5, carried=(at_move_gripper, None)
        ),
        'hold arm touches the move arm',
    )


@test
def final_check_follows_who_brings_the_first_part(c):
    """
    verify_plan.iterate_move_paths: which parts lie and which stand at the assembly at every path of the move arm. The first
    part gets to the assembly spot by the hold arm, or in the fallback by the move arm itself, which then lets go.
    Every path of the move arm also gets where the hold arm stands meanwhile (other_arm), and iterate_hold_paths gives
    every path of the hold arm with where the move arm stands.
    """
    from planning.verify_plan import iterate_move_paths, iterate_hold_paths

    def P(name):
        return [name]  # a path, only named here

    def move_arm(path, part, task):
        return ['move', 'arm', path, part, task]

    def grip(arm, ratio, task):
        return [arm, 'gripper', ratio, None, task]

    insert_1 = [
        grip('move', 0.4, 'open'),
        move_arm(P('to 1'), None, 'switch'),
        grip('move', 0.3, 'close'),
        move_arm(P('carry 1'), '1', 'transport'),
        move_arm(P('insert 1'), '1', 'assembly'),
        grip('move', 0.4, 'open'),
        move_arm(P('home'), None, 'transport'),
    ]
    start = [
        move_arm(P('rest'), None, 'init'),
        ['hold', 'arm', P('rest'), None, 'init'],
        grip('move', 0.5, 'init'),
        grip('hold', 0.5, 'init'),
    ]

    def seen(motion):
        return [
            (path[0], ratio, state, kwargs)
            for _, path, ratio, state, kwargs in iterate_move_paths(
                motion, ['0', '1']
            )
        ]

    # the hold arm brings part 0
    usual = (
        start
        + [
            grip('hold', 0.6, 'open'),
            ['hold', 'arm', P('h pick'), None, 'transport'],
            grip('hold', 0.5, 'close'),
            ['hold', 'arm', P('h carry'), '0', 'transport'],
        ]
        + insert_1
    )
    lying, standing = (
        {'0': 'pickup', '1': 'pickup'},
        {'0': 'final', '1': 'pickup'},
    )
    holding = ('h carry', 0.5)  # the hold arm holds part 0 at the assembly
    assert seen(usual) == [
        (
            'to 1',
            0.4,
            standing,
            dict(pick_part='1', released_part=None, other_arm=holding),
        ),
        ('carry 1', 0.3, standing, dict(carried='1', other_arm=holding)),
        (
            'insert 1',
            0.3,
            standing,
            dict(carried='1', inserting=True, other_arm=holding),
        ),
        (
            'home',
            0.4,
            {'0': 'final', '1': 'final'},
            dict(pick_part=None, released_part='1', other_arm=holding),
        ),
    ]
    # the hold arm moves while the move arm waits at rest, gripper 0.5
    assert [
        (path[0], ratio, move_q, move_open, part)
        for _, path, ratio, move_q, move_open, part in iterate_hold_paths(
            usual
        )
    ] == [
        ('h pick', 0.6, 'rest', 0.5, None),
        ('h carry', 0.5, 'rest', 0.5, '0'),
    ]
    # fallback: the move arm picks part 0, stands it at the assembly spot, lets go and moves away; then the hold arm takes it
    place_0 = [
        grip('move', 0.45, 'open'),
        move_arm(P('to 0'), None, 'switch'),
        grip('move', 0.35, 'close'),
        move_arm(P('carry 0'), '0', 'transport'),
        grip('move', 0.45, 'open'),
        move_arm(P('away'), None, 'transport'),
        grip('hold', 0.6, 'open'),
        ['hold', 'arm', P('h to 0'), None, 'transport'],
        grip('hold', 0.5, 'close'),
    ]
    resting = ('rest', 0.5)  # the hold arm waits at rest
    holding = ('h to 0', 0.5)  # then holds part 0 where it stands
    assert seen(start + place_0 + insert_1) == [
        (
            'to 0',
            0.45,
            lying,
            dict(pick_part='0', released_part=None, other_arm=resting),
        ),  # everything still lies on the board
        ('carry 0', 0.35, lying, dict(carried='0', other_arm=resting)),
        (
            'away',
            0.45,
            standing,
            dict(pick_part='1', released_part='0', other_arm=resting),
        ),  # part 0 stands at its place and was just released
        (
            'to 1',
            0.4,
            standing,
            dict(pick_part='1', released_part='0', other_arm=holding),
        ),
        ('carry 1', 0.3, standing, dict(carried='1', other_arm=holding)),
        (
            'insert 1',
            0.3,
            standing,
            dict(carried='1', inserting=True, other_arm=holding),
        ),
        (
            'home',
            0.4,
            {'0': 'final', '1': 'final'},
            dict(pick_part=None, released_part='1', other_arm=holding),
        ),
    ]
    # the hold arm takes part 0 while the move arm stands where it went
    # after letting go ('away'), its gripper opened to 0.45
    assert [
        (path[0], ratio, move_q, move_open, part)
        for _, path, ratio, move_q, move_open, part in iterate_hold_paths(
            start + place_0 + insert_1
        )
    ] == [('h to 0', 0.6, 'away', 0.45, None)]


@test
def cause_of_a_failed_plan(c):
    """what has to change after the real arm found no checked path: the grasp, the spot where the part lies, or nothing can help"""
    lying = 'carried part 2 closer than 1.0 cm to lying part 5 (poses 3..4 of 80, 2x)'
    gripper_lying = 'after opening: gripper panda_leftfinger closer than 1.0 cm to lying part 7 (poses 60..61 of 62, 2x)'
    wall = 'closer than 1.0 cm to a virtual wall (poses 10..12 of 80, 3x)'
    # no path at all: which straight move is missing
    assert (
        failure_cause('switch', False, 'goal', False, []) == CAUSE_SPOT
    )  # into the grasp of the next part
    assert (
        failure_cause('switch', False, 'start', False, []) == CAUSE_GRASP
    )  # out of the release pose (the caller blames the released part)
    assert failure_cause('switch', False, 'path', False, []) == CAUSE_GRASP
    assert (
        failure_cause('transport', True, 'start', False, []) == CAUSE_SPOT
    )  # out of the pickup with the part
    assert failure_cause('transport', True, 'start', True, []) == CAUSE_SPOT
    assert (
        failure_cause('transport', False, 'start', False, []) == CAUSE_GRASP
    )  # leaving the last release pose with the empty gripper
    assert (
        failure_cause('transport', True, 'goal', True, []) == CAUSE_INSERTION
    )  # to the insertion start, stopped by the part itself
    assert (
        failure_cause('transport', True, 'goal', False, []) == CAUSE_GRASP
    )  # ... stopped by the arm or the gripper
    assert (
        failure_cause('transport', False, 'goal', True, []) == CAUSE_GRASP
    )  # nothing carried: no insertion
    assert (
        failure_cause('transport', True, 'path', True, []) == CAUSE_GRASP
        and failure_cause('transport', True, 'path', False, []) == CAUSE_GRASP
    )
    # paths were found, but all attempts fail the checks
    assert (
        failure_cause('transport', True, None, False, [lying, lying])
        == CAUSE_SPOT
    )
    assert (
        failure_cause('switch', False, None, False, [gripper_lying])
        == CAUSE_SPOT
    )
    assert (
        failure_cause('transport', True, None, False, [lying, wall])
        == CAUSE_GRASP
    )
    assert (
        failure_cause(
            'transport', True, None, False, [lying, NO_PATH_REASON['path']]
        )
        == CAUSE_GRASP
    )  # one attempt found no path at all
    assert (
        failure_cause('transport', True, None, True, []) == CAUSE_GRASP
        and failure_cause('switch', False, None, False, [wall]) == CAUSE_GRASP
    )

    # the text of a missing path names what stopped the straight move
    def fake(failure, block):
        return types.SimpleNamespace(last_failure=failure, last_block=block)

    assert (
        no_path_reason(fake('goal', ('virtual wall', 0.25)))
        == NO_PATH_REASON['goal'] + ' (virtual wall after 0.25 cm)'
    )
    assert (
        no_path_reason(
            fake('start', ('no straight lift of the carried part', None))
        )
        == NO_PATH_REASON['start'] + ' (no straight lift of the carried part)'
    )
    assert no_path_reason(fake('goal', None)) == NO_PATH_REASON['goal']
    assert (
        no_path_reason(fake('path', ('virtual wall', 0.25)))
        == NO_PATH_REASON['path']
    )  # the free path: no straight move involved
    assert no_path_reason(fake(None, None)) == 'no path found'


@test
def planner_of_the_move_arm(c):
    """make_move_planner: the planner run_motion_plan.py plans with and run_pickup_gen.py tests pickup spots with"""
    from planning.robot.workcell import get_dual_arm_box

    final = {
        'A': box([4, 6, 2], [10.0, -20.0, 1.0]),
        'B': box([2, 2, 5], [14.0, -12.0, 2.5]),
    }
    box_move, _ = get_dual_arm_box('panda')
    planner = make_move_planner(
        get_arm_chain('panda', 'move'), 'panda', False, box_move, final
    )
    assert (
        planner.strict
        and planner.virtual_walls
        and planner.arm_box is box_move
    )
    assert (
        planner.table_z == get_panda_base_z() + get_panda_desk_z()
        and planner.part_clearance
        == TRANSFER_PART_CLEARANCE + PLANNER_TABLE_MARGIN
    )
    lower, upper = (
        planner.clearance_exempt_xy
    )  # the footprint of the assembly plus the zone margin, a little smaller than the checked zone
    margin = ASSEMBLY_ZONE_MARGIN - PLANNER_TABLE_MARGIN
    assert np.allclose(lower, [8.0 - margin, -23.0 - margin]) and np.allclose(
        upper, [15.0 + margin, -11.0 + margin]
    )
    assert planner.part_buffers() == (
        max(ROBOT_PART_CLEARANCE, ASSEMBLY_CLEARANCE) + PLANNER_MARGIN,
        PLANNER_MARGIN,
    )
    planner.strict = False
    assert planner.part_buffers() == (0.5, 0.5)


@test
def planner_takes_inflated_meshes_from_outside(c):
    """add_buffered / drop_buffered: the inflated version of a part that only moved is handed over instead of computed again"""
    from planning.robot.geometry import offset_mesh

    planner = make_planner(c)
    part = box([4, 4, 2], [*REF_XY, 1.0])
    moved = part.copy().apply_translation([10.0, 0, 0])
    buffer = planner.part_buffers()[0]
    inflated = offset_mesh(part, buffer)
    assert np.allclose(
        inflated.bounds,
        [
            [REF_XY[0] - 2 - buffer, REF_XY[1] - 2 - buffer, -buffer],
            [REF_XY[0] + 2 + buffer, REF_XY[1] + 2 + buffer, 2 + buffer],
        ],
    )  # every face moved out by the buffer
    probe = trimesh.collision.CollisionManager()
    probe.add_object(
        'probe',
        box(
            [0.2, 0.2, 0.2],
            [REF_XY[0] + 10.0 + 2.0 + buffer - 0.2, REF_XY[1], 1.0],
        ),
    )  # inside the buffer of the moved part
    # the inflated mesh moves with the part: handed over, it is used as it is
    planner.add_buffered(
        moved, buffer, inflated.copy().apply_translation([10.0, 0, 0])
    )
    assert planner.part_manager([moved], buffer).in_collision_other(probe)
    assert not planner.part_manager([moved], 0.0).in_collision_other(
        probe
    )  # the part itself is 1 cm away
    # (a wrong one would be used too: this is what is handed over, not a new computation)
    planner.add_buffered(moved, buffer, inflated)
    assert not planner.part_manager([moved], buffer).in_collision_other(probe)
    # forgotten again: the planner computes it itself, with the same result as the one that moved along
    planner.drop_buffered(moved)
    assert all(key[0] != id(moved) for key in planner._buffer_cache)
    assert planner.part_manager([moved], buffer).in_collision_other(probe)
    assert np.allclose(
        planner._buffer_cache[(id(moved), buffer)][1].vertices,
        inflated.vertices + [10.0, 0, 0],
        atol=1e-9,
    )
    planner.drop_buffered(
        part
    )  # a mesh the planner has never seen: nothing happens


@test
def planner_releases_the_collision_structures_of_earlier_calls(c):
    """
    get_fns builds collision structures for the still parts (for large meshes hundreds of MB). The functions it returns refer
    to each other, so what an earlier call built is only freed by the garbage collector; get_fns runs it before it builds
    new ones. Tested with the automatic runs switched off: without that step every call would leave its structures behind
    (gamepad: 218 MB per call, eight plans at once filled the memory of the machine).
    """
    import gc

    planner = make_planner(c)
    parts = [
        box([4, 4, 2], [*REF_XY, 1.0]),
        box([2, 2, 2], [REF_XY[0] + 10.0, REF_XY[1], 1.0]),
    ]

    def managers():
        return sum(
            isinstance(obj, trimesh.collision.CollisionManager)
            for obj in gc.get_objects()
        )

    planner.get_fns(
        None, None, parts, 0.5
    )  # the inflated meshes are computed and kept: not what is counted here
    gc.collect()
    kept = managers()  # the planner's own (arm, gripper)
    gc.disable()
    try:
        planner.get_fns(None, None, parts, 0.5)
        one_call = managers() - kept
        assert one_call > 0
        for _ in range(4):
            planner.get_fns(None, None, parts, 0.5)
        assert managers() - kept == one_call, (
            managers() - kept,
            one_call,
        )  # only what the last call built is still there
        # what a caller holds stays: the functions of one call are used while the next call is made
        held = planner.get_fns(None, None, parts, 0.5)
        q = c.chain.active_from_full(ik_down(c, REF_XY, 10.0))
        before = held[3](q)
        planner.get_fns(None, None, parts[:1], 0.6)
        assert managers() - kept >= one_call and held[3](q) == before
    finally:
        gc.enable()


if __name__ == '__main__':
    words = sys.argv[1:]
    checker = make_checker()
    failed = 0
    for fn in TESTS:
        if words and not any(w in fn.__name__ for w in words):
            continue
        start = time.time()
        try:
            fn(checker)
            print(f'PASS  {fn.__name__}  ({time.time() - start:.1f} s)')
        except Exception as e:
            failed += 1
            import traceback

            print(f'FAIL  {fn.__name__}: {type(e).__name__}: {e}')
            traceback.print_exc(limit=3)
    print(
        f'{"ALL TESTS PASSED" if failed == 0 else f"{failed} TEST(S) FAILED"}'
    )
    sys.exit(1 if failed else 0)
