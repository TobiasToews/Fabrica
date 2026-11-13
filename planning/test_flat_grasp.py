"""Tests of the grasps of the fixture-free (flat) pipeline.

Tested are planning/flat_grasp.py, the flat mode of the grasp generation
(run_grasp_arm_gen.py --flat) and the parts of run_pickup_gen.py that
use them. No robot, no planning run: the tests use toy parts (boxes,
cylinders) with known dimensions, so every expected number can be
derived by hand (it is given in the comments).

Every test is a function marked with the decorator `test`. The main
block at the end of the file calls each test with one shared scene
checker (test_plan_rules.make_checker) and prints PASS or FAIL.

Usage (fabrica env, about 1 min; exit code 1 if a test fails):

    python planning/test_flat_grasp.py
    python planning/test_flat_grasp.py pad rating

The second line runs only the tests whose name contains one of the
words.
"""

import os

os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)
sys.path.append(project_base_dir)

import json
import pickle
import shutil
import tempfile
import time
import types
import numpy as np
import networkx as nx
import trimesh
from scipy.spatial.transform import Rotation as R

import planning.config as config
import planning.flat_grasp as flat_grasp
from planning.config import (
    GRASP_MIN_FINGER_OVERLAP,
    PAD_CELL,
    PAD_CLOSING,
    PAD_STORE_CELL,
    FLAT_RESAMPLE_PT,
    FLAT_RATING_WEIGHTS,
    MIN_SIGMA_INSERTION,
    MIN_JOINT_LIMIT_MARGIN,
)
from scipy.spatial import ConvexHull
from planning.flat_grasp import (
    hull_faces,
    resting_poses,
    SUPPORT_TOL,
    FACE_MERGE_ANGLE,
)
from planning.flat_grasp import (
    MAX_TILT,
    RATING_TILT,
    GROUND_CLEARANCE,
    WRIST_CLEARANCE,
    WRIST_LINK,
    MIN_TIP_ANGLE,
    MIN_TIP_ANGLE_FALLBACK,
    get_flat_rots,
    get_flat_pickup_candidates,
    wrist_points,
    finger_pads,
    PartFaces,
    pad_surface,
    pad_contact,
    set_pad_contact,
    move_grasp_criteria,
    flat_move_grasps,
    part_center_of_mass,
    insertion_score,
    insertion_ranks,
    rate_flat_grasp,
    approach_direction,
    select_grasps,
    holds_with_room,
    lying_grasp_candidates,
)
from assets.transform import mat_to_pos_quat
from planning.robot.util_grasp import compute_antipodal_pairs
from planning.plan_checks import (
    grasp_finger_overlap,
    joint_condition,
    PANDA_JOINT_LIMITS,
)
from assets.transform import get_transform_matrix_quat
from planning.config import OPEN_RATIO_REST
from planning.plan_checks import CAUSE_GRASP, CAUSE_SPOT, CAUSE_INSERTION
from planning.robot.geometry import (
    load_gripper_meshes,
    transform_gripper_meshes,
    get_gripper_meshes_transforms,
    get_gripper_finger_names,
    get_arm_meshes_transforms,
    offset_mesh,
)
from planning.robot.motion_plan_arm import ArmMotionPlanner
from planning.robot.util_arm import get_arm_chain
from planning.robot.workcell import get_dual_arm_box
from planning.run_motion_plan import (
    get_pickup_arm_q,
    get_pickup_gripper_pose,
    get_back_retract_dir,
    make_move_planner,
    assembly_moves_blocked,
)
from planning.robot.motion_plan_arm import VIRTUAL_WALL_SAFETY
from planning.robot.util_arm import (
    get_virtual_wall_margin,
    get_gripper_pos_quat_from_arm_q,
    get_hand_wall_margin,
)
from planning.plan_checks import PICKUP_POSE_SLACK
import planning.run_pickup_gen as run_pickup_gen
import io
import contextlib
from planning.robot.util_grasp import Grasp, get_gripper_pos_quat
from planning.run_grasp_arm_gen import GraspArmGenerator
from planning.run_pickup_gen import (
    flat_rots_of,
    filter_flat_grasps,
    first_hold_nodes,
    place_grasps_of_first_part,
    finger_footprints,
    fingers_clear,
    PickupMoves,
    pickup_ik_solutions,
    load_banned_spots,
    ban_spot,
    clear_banned_spots,
    spot_key,
    BANNED_SPOTS_FILE,
    PART_GAP,
)
from planning.config import ROBOT_PART_CLEARANCE
from planning.test_plan_rules import (
    make_checker,
    box,
    cylinder,
    set_scene,
    ik_down,
    has,
    poses_of,
    pose_with_sigma,
    carried_scene,
    carry_to,
    REF_XY,
    DOWN,
)

ASSETS = os.path.join(project_base_dir, 'assets')
# The contact is measured on the visual finger meshes (as in the
# original pipeline).
VISUAL = load_gripper_meshes('panda', ASSETS, visual=True)
COLLISION = load_gripper_meshes('panda', ASSETS)
# The gripping pad of each finger: the flat inner face that presses on
# the part (flat_grasp.finger_pads).
PADS = finger_pads('panda', VISUAL)
# The pad along the finger: from 0.015 to 1.765 cm behind the fingertip.
PAD_A = PADS[0]['a']
# The pad is 1.754 cm wide across the finger.
PAD_WIDTH = PADS[0]['b'][1] - PADS[0]['b'][0]
# Unit vectors along the x, y and z axis.
X, Y, Z = np.eye(3)


# -------------------------------------------------------------- helpers


def toy_grasp(center, approach, closing, width, grasp_id=0, part_id='A'):
    """Makes a grasp from where the fingers are and how the hand comes.

    The fingers close along `closing` over `width` cm around `center`,
    and the hand comes along `approach`.

    Args:
        center (array-like): The grasp centre, the point in the middle
            between the two fingers at the level of their tips (the tips
            end 0.15 mm before it), shape (3,), in cm in the frame in
            which the toy part is given.
        approach (array-like): Direction in which the hand comes, from
            the hand towards the fingertips, shape (3,).
        closing (array-like): Direction of the line between the two
            fingers, shape (3,).
        width (float): Distance between the two fingers in cm.
        grasp_id (int): Number of the grasp, stored in it.
        part_id (str): Id of the grasped part, stored in the grasp.

    Returns:
        Grasp: The grasp with the gripper pose (`pos` in cm, `quat` with
        w first) and `open_ratio` = width / 8.
    """
    approach, closing = (
        np.asarray(approach, dtype=float),
        np.asarray(closing, dtype=float),
    )
    # The panda gripper opens 8 cm.
    open_ratio = width / 8.0
    pos, quat = get_gripper_pos_quat(
        'panda',
        np.asarray(center, dtype=float),
        -approach,
        closing,
        open_ratio,
    )
    return Grasp(part_id, grasp_id, pos, quat, open_ratio)


def fingertip(grasp, meshes=VISUAL):
    """Returns how far the fingertips reach along the approach.

    Args:
        grasp (Grasp): The grasp that gives the gripper pose and how far
            the gripper is open.
        meshes (dict): The gripper meshes by name, as returned by
            load_gripper_meshes (VISUAL or COLLISION).

    Returns:
        float: The largest position of a finger vertex along the
        approach direction of the grasp in cm (a scalar, measured from
        the origin of the frame of the grasp).
    """
    approach = approach_direction(grasp)
    transformed = transform_gripper_meshes(
        'panda', meshes, grasp.pos, grasp.quat, np.eye(4), grasp.open_ratio
    )
    return max(
        (transformed[name].vertices @ approach).max()
        for name in get_gripper_finger_names('panda')
    )


def contact_of(grasp, mesh):
    """Measures the contact of the finger pads with a part.

    Args:
        grasp (Grasp): The grasp. Its contact_points, contact_depths and
            contact_area are set (flat_grasp.set_pad_contact).
        mesh (trimesh.Trimesh): The part in the frame of the grasp.

    Returns:
        Grasp: The same grasp object, with the contact stored in it.
    """
    set_pad_contact(grasp, PADS, 'panda', VISUAL, PartFaces(mesh))
    return grasp


# The toy part at its final pose: 6 cm along x, 2 along y, 4 high, top
# face at z = 9. Its resting poses (poses in which it lies on the board
# by itself):
# * on a 6 x 4 face: centre of mass 1 cm high, 2 cm from the edge, tips
#   over at atan(2 / 1) = 63.4 deg;
# * on a 6 x 2 face, as it stands: 2 cm high, 1 cm from the edge,
#   26.6 deg;
# * on a 2 x 4 end face: 3 cm high, 1 cm from the edge, 18.4 deg, not
#   stable.
BRICK = box([6, 2, 4], [0, 0, 7])


def top_grasp(depth, x=0.0, grasp_id=0):
    """Makes a grasp of the brick from above.

    The fingers are on the 6 x 4 faces. This grasp picks the brick
    standing as it is.

    Args:
        depth (float): How far the grasp centre is below the top face
            in cm.
        x (float): The x coordinate of the grasp centre in cm.
        grasp_id (int): Number of the grasp.

    Returns:
        Grasp: The grasp for the brick at its final pose.
    """
    return toy_grasp([x, 0, 9 - depth], -Z, Y, 2.0, grasp_id)


def side_grasp(depth, grasp_id=0):
    """Makes a grasp of the brick from the side (+y).

    The fingers are on the top and the bottom face. This grasp picks
    the brick lying on a 6 x 4 face.

    Args:
        depth (float): How far the grasp centre is behind the side face
            at y = 1 in cm.
        grasp_id (int): Number of the grasp.

    Returns:
        Grasp: The grasp for the brick at its final pose.
    """
    return toy_grasp([0, 1 - depth, 7], -Y, Z, 4.0, grasp_id)


def end_grasp(depth, grasp_id=0):
    """Makes a grasp of the brick from the end (+x).

    The fingers are on the 6 x 4 faces. This grasp picks the brick only
    standing on an end face (not a stable pose).

    Args:
        depth (float): How far the grasp centre is behind the end face
            at x = 3 in cm.
        grasp_id (int): Number of the grasp.

    Returns:
        Grasp: The grasp for the brick at its final pose.
    """
    return toy_grasp([3 - depth, 0, 7], -X, Y, 2.0, grasp_id)


def below_grasp(depth, grasp_id=0):
    """Makes a grasp of the brick from below.

    This grasp picks the brick when it lies upside down.

    Args:
        depth (float): How far the grasp centre is above the bottom face
            in cm.
        grasp_id (int): Number of the grasp.

    Returns:
        Grasp: The grasp for the brick at its final pose.
    """
    return toy_grasp([0, 0, 5 + depth], Z, Y, 2.0, grasp_id)


def leaning_grasp(angle=45.0, grasp_id=0):
    """Makes a grasp of the brick with a hand that leans.

    The hand leans about the line between the fingers. The grasp centre
    is 1 cm below the top face.

    Args:
        angle (float): How far the hand leans from straight above in
            deg.
        grasp_id (int): Number of the grasp.

    Returns:
        Grasp: The grasp for the brick at its final pose.
    """
    return toy_grasp(
        [0, 0, 8.0],
        R.from_euler('y', angle, degrees=True).apply(-Z),
        Y,
        2.0,
        grasp_id,
    )


def low_grasp(grasp_id=0):
    """Makes a grasp that holds the brick but cannot pick it lying.

    The hand comes from above, the fingertips 0.2 cm above the bottom
    face. The grasp holds the brick well, but cannot take it from the
    board in any resting pose: standing, the fingertips would be closer
    than GROUND_CLEARANCE to the board; in every other pose the hand
    comes from the side or from below and would be in the board.

    Args:
        grasp_id (int): Number of the grasp.

    Returns:
        Grasp: The grasp for the brick at its final pose.
    """
    return top_grasp(3.8, grasp_id=grasp_id)


# The test functions of this file, in the order in which they run.
TESTS = []


def test(fn):
    """Registers a function as a test (used as a decorator).

    Args:
        fn (callable): The test function. It is called with the scene
            checker as its only argument.

    Returns:
        callable: The same function, unchanged.
    """
    TESTS.append(fn)
    return fn


# ------------------------------- resting poses and the flat-pickup test


@test
def resting_poses_of_a_box(c):
    """Tests the resting poses of the brick and their tipping angles.

    A resting pose is a pose in which the part lies on the board by
    itself. Its tipping angle says by how many degrees the part can be
    tilted before it tips over. The expected angles are derived in the
    comment above BRICK.

    Args:
        c (SceneChecker): The shared scene checker (not used here).
    """
    rots = get_flat_rots(BRICK)
    assert sorted(round(r['tip_angle'], 1) for r in rots) == [
        26.6,
        26.6,
        63.4,
        63.4,
    ], [r['tip_angle'] for r in rots]
    assert not any(r.get('fallback') for r in rots)
    # Margin and height of the centre of mass belong to the same pose:
    # the tipping angle is atan(margin / height). The brick is turned
    # into the pose with a 4x4 transform made of the rotation alone.
    for r in rots:
        lying = BRICK.copy().apply_transform(
            np.block([[r['rot'], np.zeros((3, 1))], [np.zeros((1, 3)), 1]])
        )
        height = lying.center_mass[2] - lying.bounds[0, 2]
        assert (
            abs(np.degrees(np.arctan2(r['margin'], height)) - r['tip_angle'])
            < 1e-6
        )
        assert (round(r['tip_angle'], 1), round(height, 3)) in [
            (63.4, 1.0),
            (26.6, 2.0),
        ]
    # The end faces (18.4 deg) are only offered on request, and marked.
    more = get_flat_rots(BRICK, with_fallback=True)
    assert len(more) == 6 and [
        round(r['tip_angle'], 1) for r in more if r.get('fallback')
    ] == [18.4, 18.4]
    # The stable ones come first, unchanged.
    assert [r['tip_angle'] for r in more[:4]] == [r['tip_angle'] for r in rots]
    # A stricter limit drops poses; the fallback only reaches down to
    # MIN_TIP_ANGLE_FALLBACK.
    assert len(get_flat_rots(BRICK, min_tip_angle=30.0)) == 2
    assert (
        MIN_TIP_ANGLE_FALLBACK
        <= min(r['tip_angle'] for r in more)
        < MIN_TIP_ANGLE
    )


@test
def resting_poses_without_a_stable_one(c):
    """Tests that a part without a stable pose gets marked poses.

    A long cylinder (radius 1 cm, 8 cm long) rolls when it lies and tips
    at atan(1 / 4) = 14 deg when it stands: it gets the standing poses,
    marked as fallback.

    Args:
        c (SceneChecker): The shared scene checker (not used here).
    """
    rots = get_flat_rots(
        trimesh.creation.cylinder(radius=1.0, height=8.0, sections=48)
    )
    assert (
        len(rots) == 2
        and all(r.get('fallback') for r in rots)
        and all(abs(r['tip_angle'] - 14.04) < 0.1 for r in rots)
    ), rots
    # A short one stands safely (atan(1 / 1.5) = 33.7 deg): no fallback
    # pose is added, with or without the request.
    short = trimesh.creation.cylinder(radius=1.0, height=3.0, sections=48)
    assert [bool(r.get('fallback')) for r in get_flat_rots(short)] == [
        False,
        False,
    ]
    assert len(get_flat_rots(short, with_fallback=True)) == 2
    # A pin that tips below MIN_TIP_ANGLE_FALLBACK when it stands has no
    # resting pose at all.
    assert (
        get_flat_rots(
            trimesh.creation.cylinder(radius=0.5, height=8.0, sections=48)
        )
        == []
    )


def pose_check(mesh, rot):
    """Tests a resting pose from scratch, as a second opinion.

    The whole mesh is turned by `rot` about the middle of its bounding
    box. What is within SUPPORT_TOL of its lowest point touches the
    board. The convex hull of those points seen from above (the smallest
    convex outline around them) is the area the part stands on.

    Args:
        mesh (trimesh.Trimesh): The part, lengths in cm.
        rot (np.ndarray): Rotation matrix of the resting pose, shape
            (3, 3).

    Returns:
        tuple[float, float] | None: The margin (distance in cm of the
        centre of mass, seen from above, to the edge of that area) and
        the tipping angle in deg. None if the centre of mass is not over
        the area.
    """
    local = mesh.copy()
    local.apply_translation(-mesh.bounds.mean(axis=0))
    rest, com = local.vertices @ rot.T, rot @ local.center_mass
    low = rest[:, 2].min()
    support = rest[rest[:, 2] < low + SUPPORT_TOL][:, :2]
    # Each row of `edges` is one edge of the area as (unit normal x,
    # unit normal y, offset): normal @ p + offset is the distance of the
    # point p from that edge, negative inside the area. The margin is
    # the distance of the centre of mass to the nearest edge.
    edges = ConvexHull(support).equations
    margin = -np.max(edges[:, :2] @ com[:2] + edges[:, 2])
    return (
        (float(margin), float(np.degrees(np.arctan2(margin, com[2] - low))))
        if margin > 0
        else None
    )


def same_rots(a, b, tol=1e-9):
    """Says whether two lists of resting poses are the same.

    Args:
        a (list[dict]): Resting poses as returned by get_flat_rots.
        b (list[dict]): The poses to compare them with, in the same
            order.
        tol (float): Largest allowed difference between two entries of
            the rotation matrices.

    Returns:
        bool: True if both lists are equally long and every pair of
        poses agrees in rotation, tipping angle, margin, share ('prob')
        and fallback mark.
    """
    return len(a) == len(b) and all(
        np.allclose(r['rot'], q['rot'], atol=tol)
        and abs(r['tip_angle'] - q['tip_angle']) <= 1e-6
        and abs(r['margin'] - q['margin']) <= 1e-6
        and abs(r['prob'] - q['prob']) <= 1e-6
        and bool(r.get('fallback')) == bool(q.get('fallback'))
        for r, q in zip(a, b)
    )


@test
def resting_poses_are_a_property_of_the_shape(c):
    """Tests that resting poses depend only on the shape of a part.

    get_flat_rots returns the faces of the convex hull (the smallest
    convex shape around the part) over which the centre of mass stands.
    The same part gives the same poses wherever it stands, also a part
    that is not closed, and a hull with sliver triangles (triangles
    without an area) is no problem. The routine of trimesh used before
    lost poses at some places and did not return for such hulls.

    Args:
        c (SceneChecker): The shared scene checker (not used here).
    """
    # The faces of the brick's hull: 6, each with its 4 corners. Their
    # shares are the solid angles of the rectangles seen from the
    # middle: 4 atan(a b / (2 h sqrt(4 h^2 + a^2 + b^2))) / (4 pi) for a
    # rectangle a x b at the distance h.
    points = np.asarray(BRICK.vertices, dtype=float)
    faces = hull_faces(points, BRICK.center_mass)

    def share(a, b, h):
        """Returns the share of a rectangle a x b at the distance h."""
        return (
            4
            * np.arctan(a * b / (2 * h * np.sqrt(4 * h * h + a * a + b * b)))
            / (4 * np.pi)
        )

    # The brick: 6 along x, 2 along y, 4 along z.
    expected = {
        (1, 0, 0): share(2, 4, 3),
        (0, 1, 0): share(6, 4, 1),
        (0, 0, 1): share(6, 2, 2),
    }
    assert len(faces) == 6 and abs(sum(f[2] for f in faces) - 1.0) < 1e-9
    for normal, corners, part in faces:
        axis = tuple(int(v) for v in np.abs(np.round(normal)))
        assert (
            np.allclose(np.abs(normal), axis, atol=1e-12)
            and len(corners) == 4
            and abs(part - expected[axis]) < 1e-9
        ), (normal, part)
        # The corners of the face lie on its plane, outermost along the
        # normal.
        assert np.allclose(points[corners] @ normal, (points @ normal).max())
    # The faces are ordered by direction.
    assert [tuple(np.round(f[0])) for f in faces] == sorted(
        tuple(np.round(f[0])) for f in faces
    )
    # Every pose that is reported holds when it is tested from scratch,
    # with the reported numbers. The pyramid rests on its base and on
    # its four sides, never on its tip.
    pyramid = trimesh.creation.cone(radius=3.0, height=2.0, sections=4)
    for mesh in [
        BRICK,
        trimesh.creation.cylinder(radius=1.0, height=3.0, sections=48),
        box([5, 3, 1], [40, -20, 0.5]),
        pyramid,
    ]:
        rots = get_flat_rots(mesh, min_tip_angle=MIN_TIP_ANGLE_FALLBACK)
        assert len(rots) > 0
        for r in rots:
            margin, tip = pose_check(mesh, r['rot'])
            assert (
                abs(margin - r['margin']) < 1e-5
                and abs(tip - r['tip_angle']) < 1e-5
                and abs(np.linalg.det(r['rot']) - 1) < 1e-9
            )
        # The most stable pose comes first.
        assert [r['tip_angle'] for r in rots] == sorted(
            (r['tip_angle'] for r in rots), reverse=True
        )
    assert (
        len(get_flat_rots(pyramid, min_tip_angle=MIN_TIP_ANGLE_FALLBACK)) == 5
    )
    # resting_poses: every face over which the centre of mass stands,
    # whatever its tipping angle (the brick: all 6; the limits of the
    # tipping angle are applied by get_flat_rots). `middle` is the brick
    # with the middle of its bounding box at the origin: x +-3, y +-1,
    # z +-2.
    middle = points - (points.min(axis=0) + points.max(axis=0)) / 2
    assert sorted(
        round(r['tip_angle'], 1) for r in resting_poses(middle, np.zeros(3))
    ) == [18.4, 18.4, 26.6, 26.6, 63.4, 63.4]
    # The centre of mass 0.1 cm from the end face +x: on the four long
    # faces it stands 0.1 cm from the edge, on the two end faces 1 cm
    # (half the width of the brick).
    beside = resting_poses(middle, np.array([2.9, 0.0, 0.0]))
    assert sorted(round(r['margin'], 3) for r in beside) == [
        0.1,
        0.1,
        0.1,
        0.1,
        1.0,
        1.0,
    ], [r['margin'] for r in beside]
    # The pyramid with its centre of mass moved over one side (x = 2,
    # 0.5 above the base): it stands over the base and over that side
    # only; the faces it does not stand over are no resting poses.
    cone = np.asarray(pyramid.vertices, dtype=float)
    cone = cone - (cone.min(axis=0) + cone.max(axis=0)) / 2
    leaning = resting_poses(cone, np.array([2.0, 0.0, -0.5]))
    assert (
        len(resting_poses(cone, np.array([0.0, 0.0, -0.5]))) == 5
        and 1 <= len(leaning) < 5
        and all(r['margin'] > 0 for r in leaning)
    )
    # The brick moved to other places gives the same poses.
    reference = get_flat_rots(BRICK, with_fallback=True)
    for shift in [[5.0, -15.0, 0.0], [-12.5, 32.5, 0.0], [0.013, 517.3, -2.7]]:
        assert same_rots(
            get_flat_rots(
                BRICK.copy().apply_translation(shift), with_fallback=True
            ),
            reference,
        ), shift
    # A mesh that is not closed (one face of a box missing): its centre
    # of mass must not depend on the place either.
    open_box = box([5, 3, 2], [0, 0, 1])
    open_box.update_faces(np.arange(len(open_box.faces)) != 0)
    assert not open_box.is_watertight
    reference_open = get_flat_rots(open_box, with_fallback=True)
    assert len(reference_open) > 0
    for shift in [[5.0, -15.0, 0.0], [-12.5, 32.5, 0.0]]:
        assert same_rots(
            get_flat_rots(
                open_box.copy().apply_translation(shift), with_fallback=True
            ),
            reference_open,
        ), shift
    # A flat face cut into many triangles, with points on its edges
    # (slivers without an area in the hull): still 6 faces, and the
    # poses of the plain brick. `slivers` adds two points a hair outside
    # two faces.
    cut = BRICK.copy().subdivide().subdivide()
    slivers = trimesh.Trimesh(
        np.vstack(
            [
                cut.vertices,
                [[1.0, 1.0, 9.0 + 1e-10], [-2.0, -1.0, 5.0 - 1e-10]],
            ]
        ),
        cut.faces,
        process=False,
    )
    assert (
        len(
            hull_faces(
                np.asarray(slivers.vertices, dtype=float), BRICK.center_mass
            )
        )
        == 6
    )
    assert [
        round(r['tip_angle'], 3)
        for r in get_flat_rots(cut, with_fallback=True)
    ] == [round(r['tip_angle'], 3) for r in reference]

    # Triangles of one face whose normals differ by less than
    # FACE_MERGE_ANGLE are one face. A corner of the top face lifted by
    # 0.003 cm (0.09 deg between its two triangles; the corner is 1.9 cm
    # from the diagonal) leaves the brick its pose on that face, with
    # all four corners as the support although they are up to 0.003 cm
    # apart in height (more than SUPPORT_TOL). Lifted by 0.05 cm
    # (1.5 deg) the brick would rock on the diagonal, and that pose is
    # gone.
    def lifted(delta):
        """Returns the brick with a top corner lifted by delta cm."""
        vertices = np.asarray(BRICK.vertices, dtype=float).copy()
        # The corner with the largest x + y + z is a corner of the top
        # face.
        vertices[np.argmax(vertices @ [1.0, 1.0, 1.0])] += [0, 0, delta]
        return trimesh.Trimesh(vertices, BRICK.faces, process=False)

    def up(rots):
        """Counts the poses that rest on the top face."""
        # Row 2 of the rotation is the direction of the brick that
        # points up in the resting pose: -z if the top face is down.
        return sum(r['rot'][2] @ np.array([0, 0, -1.0]) > 0.999 for r in rots)

    assert FACE_MERGE_ANGLE == 0.1 and SUPPORT_TOL == 1e-3
    assert (
        up(get_flat_rots(BRICK)) == 1
        and up(get_flat_rots(lifted(0.003))) == 1
        and up(get_flat_rots(lifted(0.05), with_fallback=True)) == 0
    )
    assert (
        abs(
            [
                r
                for r in get_flat_rots(lifted(0.003))
                if r['rot'][2] @ np.array([0, 0, -1.0]) > 0.999
            ][0]['tip_angle']
            - 26.6
        )
        < 0.1
    )
    merged = hull_faces(
        np.asarray(lifted(0.003).vertices, dtype=float), BRICK.center_mass
    )
    # The two triangles count as one face with four corners.
    assert (
        len(merged) == 6
        and abs(sum(f[2] for f in merged) - 1.0) < 1e-9
        and sorted(len(f[1]) for f in merged) == [4] * 6
    )
    assert (
        len(
            hull_faces(
                np.asarray(lifted(0.05).vertices, dtype=float),
                BRICK.center_mass,
            )
        )
        == 7
    )
    # Asking twice gives the same answer, and a mark set on one answer
    # is not in the next (get_flat_rots keeps the poses of a part and
    # hands out copies).
    marked = get_flat_rots(BRICK, with_fallback=True)
    assert any(r.get('fallback') for r in marked) and not any(
        r.get('fallback') for r in get_flat_rots(BRICK)
    )
    assert same_rots(get_flat_rots(BRICK, with_fallback=True), marked)
    # This cylinder stands only in a pose that tips at 14 deg.
    rolling = trimesh.creation.cylinder(radius=1.0, height=8.0, sections=48)
    assert [bool(r.get('fallback')) for r in get_flat_rots(rolling)] == [
        True,
        True,
    ]
    # With that limit they are stable poses.
    assert [
        bool(r.get('fallback'))
        for r in get_flat_rots(rolling, min_tip_angle=MIN_TIP_ANGLE_FALLBACK)
    ] == [False, False]
    # A centre of mass 0.005 cm beyond the end face +x: it does not
    # stand over the four long faces (0.005 cm outside their edge), so
    # they are no resting poses; over the end face -x it stands 1 cm
    # from the edge.
    outside = resting_poses(middle, np.array([3.005, 0.0, 0.0]))
    assert all(r['margin'] > 0 for r in outside) and any(
        np.allclose(r['rot'][2], [1.0, 0, 0]) and r['margin'] == 1.0
        for r in outside
    )
    # The parts of the assembly beam: 4 poses each wherever the assembly
    # stands (the routine used before gave part 1 two of them at the
    # default place, one to three elsewhere, and did not return for
    # parts 2 and 6 with the assembly at (+5, -30)).
    from planning.run_pickup_gen import load_final_part_cfg

    beam = load_final_part_cfg(
        os.path.join(project_base_dir, 'assets', 'fabrica', 'beam'), 'panda'
    )['mesh']
    for part, tips in [
        ('0', [46.6, 46.6, 37.6, 35.8]),
        ('1', [46.6, 46.6, 37.6, 35.8]),
        ('2', [43.3, 43.0, 40.6, 40.6]),
        ('3', [43.3, 43.0, 40.6, 40.6]),
        ('6', [52.5, 52.3, 32.5, 32.5]),
    ]:
        at_default = get_flat_rots(beam[part], with_fallback=True)
        assert [round(r['tip_angle'], 1) for r in at_default] == tips, (
            part,
            [r['tip_angle'] for r in at_default],
        )
        for r in at_default:
            assert (
                abs(pose_check(beam[part], r['rot'])[1] - r['tip_angle'])
                < 1e-4
            )
        for shift in [[5.0, -15.0, 0.0], [12.5, -5.0, 0.0], [-2.5, 50.0, 0.0]]:
            assert same_rots(
                get_flat_rots(
                    beam[part].copy().apply_translation(shift),
                    with_fallback=True,
                ),
                at_default,
                tol=1e-7,
            ), (part, shift)


@test
def flat_pickup_tilt(c):
    """Tests which resting poses a grasp can pick the brick in.

    get_flat_pickup_candidates takes a grasp (defined for the part at
    its assembled pose) and returns the resting poses in which the
    gripper can take the part from the board. Each candidate carries its
    tilt: the angle between the direction in which the hand comes and
    straight down. The tilt is measured but not limited.

    Args:
        c (SceneChecker): The shared scene checker (not used here).
    """
    rots = get_flat_rots(BRICK)
    cands = get_flat_pickup_candidates(
        top_grasp(1.0), BRICK, rots, 'panda', COLLISION
    )
    # One resting pose, 4 yaw alternatives (the lying part turned about
    # the vertical axis, as alternatives for the arm).
    assert len(cands) == 4 and [x['yaw_offset'] for x in cands] == [
        0,
        180,
        90,
        -90,
    ]
    assert all(
        abs(x['tilt']) < 1e-6 and round(x['tip_angle'], 1) == 26.6
        for x in cands
    )
    # The resting pose keeps the brick standing: the approach direction
    # of the grasp points straight down.
    for x in cands:
        assert np.allclose(
            x['rot'] @ approach_direction(top_grasp(1.0)), -Z, atol=1e-9
        )
    # In the first candidate the fingers are along x.
    assert np.allclose(
        cands[0]['rot'] @ Y, [1, 0, 0], atol=1e-9
    ) or np.allclose(cands[0]['rot'] @ Y, [-1, 0, 0], atol=1e-9)
    # The matching of the two poses: a grasp from the side picks the
    # brick lying on a 6 x 4 face, one from below picks it upside down.
    side = get_flat_pickup_candidates(
        side_grasp(1.0), BRICK, rots, 'panda', COLLISION
    )
    assert len(side) == 4 and all(
        round(x['tip_angle'], 1) == 63.4 and abs(x['tilt']) < 1e-6
        for x in side
    )
    below = get_flat_pickup_candidates(
        below_grasp(1.0), BRICK, rots, 'panda', COLLISION
    )
    assert len(below) == 4 and all(
        np.allclose(x['rot'] @ Z, -Z, atol=1e-9) for x in below
    )
    # One from the end would need the brick standing on its end (not
    # stable).
    assert (
        get_flat_pickup_candidates(
            end_grasp(1.0), BRICK, rots, 'panda', COLLISION
        )
        == []
    )
    end = get_flat_pickup_candidates(
        end_grasp(1.0),
        BRICK,
        get_flat_rots(BRICK, with_fallback=True),
        'panda',
        COLLISION,
    )
    assert len(end) == 4 and all(
        x['fallback'] and round(x['tip_angle'], 1) == 18.4 for x in end
    )
    # There is no limit on how far the hand leans: the angle is only
    # measured. The two clearances are switched off here; they are
    # tested below.
    assert MAX_TILT == 180.0
    off = dict(clearance=-100.0, wrist_clearance=-100.0)
    for angle in [25.0, 35.0, 45.0, 75.0, 90.0, 120.0, 179.0]:
        cands = get_flat_pickup_candidates(
            leaning_grasp(angle), BRICK, rots, 'panda', COLLISION, **off
        )
        # Every resting pose of the brick, 4 yaw alternatives each.
        assert len(cands) == 16, angle
        # The smallest tilt: the brick standing as it is, or upside
        # down.
        assert (
            abs(min(x['tilt'] for x in cands) - min(angle, 180 - angle)) < 1e-6
        ), angle
        # The most upright pose comes first.
        assert abs(cands[0]['tilt'] - min(x['tilt'] for x in cands)) < 1e-6
    # A caller can still ask for a limit.
    for angle, accepted in [
        (25.0, True),
        (29.5, True),
        (30.5, False),
        (35.0, False),
    ]:
        cands = get_flat_pickup_candidates(
            leaning_grasp(angle),
            BRICK,
            rots,
            'panda',
            COLLISION,
            max_tilt=30.0,
            **off,
        )
        assert (len(cands) > 0) == accepted, angle
        if accepted:
            assert len(cands) == 4 and abs(cands[0]['tilt'] - angle) < 1e-6


@test
def flat_pickup_clearance(c):
    """Tests the clearance of the gripper above the board.

    A grasp can pick a lying part only if the whole gripper stays
    GROUND_CLEARANCE above the board. The brick stands 4 cm high: a
    grasp `depth` below its top has the fingertips 4 - depth above the
    board.

    Args:
        c (SceneChecker): The shared scene checker (not used here).
    """
    rots = get_flat_rots(BRICK)
    for depth in [1.0, 3.0, 3.4]:
        cands = get_flat_pickup_candidates(
            top_grasp(depth), BRICK, rots, 'panda', COLLISION
        )
        assert len(cands) == 4, depth
        # The fingertip ends 0.15 mm above the grasp centre.
        assert abs(cands[0]['gripper_min_z'] - (4 - depth)) < 0.03, (
            depth,
            cands[0]['gripper_min_z'],
        )
    assert GROUND_CLEARANCE == 0.5
    # The fingertips 0.55 cm above the board.
    assert (
        len(
            get_flat_pickup_candidates(
                top_grasp(3.45), BRICK, rots, 'panda', COLLISION
            )
        )
        == 4
    )
    # The fingertips 0.45 cm above the board.
    assert (
        get_flat_pickup_candidates(
            top_grasp(3.55), BRICK, rots, 'panda', COLLISION
        )
        == []
    )
    assert (
        len(
            get_flat_pickup_candidates(
                top_grasp(3.55), BRICK, rots, 'panda', COLLISION, clearance=0.4
            )
        )
        == 4
    )
    # The "never" grasp of the scenarios below.
    assert (
        get_flat_pickup_candidates(
            low_grasp(),
            BRICK,
            get_flat_rots(BRICK, with_fallback=True),
            'panda',
            COLLISION,
        )
        == []
    )
    # Candidates are sorted: stable poses before fallback poses, upright
    # before tilted.
    cands = get_flat_pickup_candidates(
        leaning_grasp(20.0),
        BRICK,
        get_flat_rots(BRICK, with_fallback=True),
        'panda',
        COLLISION,
        clearance=-100.0,
        wrist_clearance=-100.0,
    )
    keys = [(x['fallback'], round(x['tilt'], 1)) for x in cands]
    assert (
        len(cands) == 24
        and keys == sorted(keys)
        and keys[0] == (False, 20.0)
        and keys[-1] == (True, 110.0)
    ), keys


@test
def flat_pickup_wrist(c):
    """Tests the clearance of the wrist above the board.

    The arm link behind the hand (link 7) must stay WRIST_CLEARANCE
    above the board: this is what stops a hand that leans too far. The
    hand is bolted to this link, so where the link is follows from the
    gripper pose alone (flat_grasp.wrist_points).

    Args:
        c (SceneChecker): The shared scene checker. Its arm model
            (c.chain, c.planner) gives link 7 and an arm pose.
    """
    assert WRIST_CLEARANCE == 1.0 and WRIST_LINK == 'panda_link7'
    wrist = wrist_points('panda')
    # Only the gripper of the real arm has the test.
    assert wrist_points('robotiq-85') is None
    # The points are link 7 of the arm model, wherever the arm is: for
    # arm poses q, link 7 placed by the forward kinematics (the pose of
    # every link that follows from the joint angles) coincides with the
    # points placed by the gripper pose of q.
    hull = c.planner.arm_meshes[WRIST_LINK].convex_hull.vertices
    np.random.seed(3)
    for _ in range(5):
        q = c.chain.active_to_full(
            np.random.uniform(
                PANDA_JOINT_LIMITS[0] + 0.1, PANDA_JOINT_LIMITS[1] - 0.1
            )
        )
        link = get_arm_meshes_transforms(c.planner.arm_meshes, c.chain, q)[
            WRIST_LINK
        ]
        gripper = get_transform_matrix_quat(
            *get_gripper_pos_quat_from_arm_q(c.chain, q, 'panda')
        )
        assert np.allclose(
            wrist @ gripper[:3, :3].T + gripper[:3, 3],
            hull @ link[:3, :3].T + link[:3, 3],
            atol=1e-9,
        )
    # The link sits behind the gripper base (up to 5.5 cm) and reaches
    # up to 9 cm sideways (its flange side).
    assert (
        -5.6 < wrist[:, 2].min() < -5.4
        and wrist[:, 2].max() < 0.0
        and 8.5 < np.linalg.norm(wrist[:, :2], axis=1).max() < 9.5
    )
    # An upright hand: the wrist is far above the board (gripper base
    # 11.24 cm above the grasp centre, the link behind it).
    rots = get_flat_rots(BRICK)
    upright = get_flat_pickup_candidates(
        top_grasp(1.0), BRICK, rots, 'panda', COLLISION
    )[0]
    assert (
        abs(upright['wrist_min_z'] - (3.0 + 11.24)) < 0.02
        and upright['gripper_min_z'] < upright['wrist_min_z']
    )
    # The further the hand leans, the lower the wrist: at 75 deg it is
    # still above the limit, at 85 deg not (the gripper itself would
    # fit).
    heights = []
    for angle in [0.0, 30.0, 60.0, 75.0]:
        cands = get_flat_pickup_candidates(
            leaning_grasp(angle), BRICK, rots, 'panda', COLLISION
        )
        assert len(cands) == 4 and abs(cands[0]['tilt'] - angle) < 1e-6, angle
        heights.append(cands[0]['wrist_min_z'])
    assert (
        heights == sorted(heights, reverse=True)
        and heights[-1] >= WRIST_CLEARANCE
    ), heights
    assert (
        get_flat_pickup_candidates(
            leaning_grasp(85.0), BRICK, rots, 'panda', COLLISION
        )
        == []
    )
    without = get_flat_pickup_candidates(
        leaning_grasp(85.0),
        BRICK,
        rots,
        'panda',
        COLLISION,
        wrist_clearance=-100.0,
    )
    assert (
        len(without) == 4
        and without[0]['gripper_min_z'] >= GROUND_CLEARANCE
        and without[0]['wrist_min_z'] < WRIST_CLEARANCE
    )
    # The limit is the given one: just below the height of the wrist it
    # passes, just above not.
    z = heights[-1]
    assert (
        len(
            get_flat_pickup_candidates(
                leaning_grasp(75.0),
                BRICK,
                rots,
                'panda',
                COLLISION,
                wrist_clearance=z - 0.01,
            )
        )
        == 4
    )
    assert (
        get_flat_pickup_candidates(
            leaning_grasp(75.0),
            BRICK,
            rots,
            'panda',
            COLLISION,
            wrist_clearance=z + 0.01,
        )
        == []
    )
    # The same height as the plan checks measure for the arm: an arm
    # pose at the pickup of a leaning grasp. rot4 becomes the pose of
    # the lying brick: on the table, in reach of the arm.
    cand = get_flat_pickup_candidates(
        leaning_grasp(60.0), BRICK, rots, 'panda', COLLISION
    )[0]
    rot4 = np.block([[cand['rot'], np.zeros((3, 1))], [np.zeros((1, 3)), 1]])
    lying = BRICK.copy().apply_transform(rot4)
    rot4[:3, 3] = [
        REF_XY[0] - lying.bounds[:, 0].mean(),
        REF_XY[1] - lying.bounds[:, 1].mean(),
        -lying.bounds[0, 2],
    ]
    grasp = leaning_grasp(60.0)
    q = get_pickup_arm_q(
        c.planner,
        grasp,
        rot4,
        np.eye(4),
        arm_q_init=c.chain.active_to_full(c.chain.rest_q),
        optimizer='least_squares',
        regularization=1.0,
    )
    assert q is not None
    link = get_arm_meshes_transforms(c.planner.arm_meshes, c.chain, q)[
        WRIST_LINK
    ]
    assert (
        abs(
            (hull @ link[:3, :3].T + link[:3, 3])[:, 2].min()
            - c.table_z
            - cand['wrist_min_z']
        )
        < 0.01
    )


# -------------------------------------------------------------------------------------- finger contact on the pad grid


@test
def finger_pads_geometry(c):
    assert len(PADS) == 2 and {p['name'] for p in PADS} == set(
        get_gripper_finger_names('panda')
    )
    for p in PADS:  # measured on the finger mesh: the flat inner face, 1.75 cm long and 1.754 cm wide, starting 0.15 mm behind the fingertip
        assert (
            abs(p['a'][0] - 0.015) < 0.003 and abs(p['a'][1] - 1.765) < 0.003
        ), p['a']
        assert (
            abs((p['b'][1] - p['b'][0]) - 1.754) < 0.003
            and abs(p['b'][0] + p['b'][1]) < 0.003
        ), p['b']
        for u, v in [
            ('along', 'across'),
            ('along', 'normal'),
            ('across', 'normal'),
        ]:
            assert abs(p[u] @ p[v]) < 1e-9
        assert all(
            abs(np.linalg.norm(p[k]) - 1) < 1e-9
            for k in ['along', 'across', 'normal']
        )
    # the pads move with the gripper: they face each other, 8 * open_ratio apart (minus 0.17 mm of the mesh), and start at the fingertip
    for grasp in [
        top_grasp(1.0),
        toy_grasp([3, -2, 12], [1, 1, 0] / np.sqrt(2), Z, 5.0),
    ]:
        transforms = get_gripper_meshes_transforms(
            'panda', VISUAL, grasp.pos, grasp.quat, np.eye(4), grasp.open_ratio
        )
        frames = []
        for p in PADS:
            T = transforms[p['name']]
            frames.append(
                {k: T[:3, :3] @ p[k] for k in ['along', 'across', 'normal']}
            )
            frames[-1]['origin'] = T[:3, :3] @ p['origin'] + T[:3, 3]
        assert np.allclose(
            frames[0]['normal'], -frames[1]['normal'], atol=1e-9
        )
        gap = (frames[1]['origin'] - frames[0]['origin']) @ frames[0]['normal']
        assert (
            abs(gap - 8 * grasp.open_ratio) < 0.03
            and gap < 8 * grasp.open_ratio
        ), gap
        for f in frames:
            assert np.allclose(
                f['along'], -approach_direction(grasp), atol=1e-9
            )  # from the fingertip towards the hand
            assert (
                abs(f['origin'] @ approach_direction(grasp) - fingertip(grasp))
                < 1e-6
            )
    # a gripper without two flat inner finger faces gets no pads (the original contact sampling is used then)
    edge_on = trimesh.creation.box([1, 1, 3])
    edge_on.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 4, Z)
    )  # an edge, not a face, points to the other finger
    assert finger_pads('panda', {name: edge_on for name in VISUAL}) is None
    assert finger_pads('panda', {'panda_hand': VISUAL['panda_hand']}) is None


@test
def pad_contact_on_a_box(c):
    """fingers on the two 6 x 4 faces: the contact reaches from the fingertip up to the top face of the brick"""
    for depth in [0.2, 0.3, 0.5, 1.0, 1.7]:
        grasp = contact_of(top_grasp(depth), BRICK)
        tip_z = -fingertip(grasp)  # the approach is -z
        assert (
            abs(tip_z - (9 - depth + 0.01)) < 1e-6
        )  # the fingertip is 0.1 mm above the grasp centre
        reach = (
            9 - tip_z
        )  # the part reaches this far into the fingers: depth - 0.01
        assert (
            len(grasp.contact_depths) == 2
            and abs(grasp.contact_depths[0] - grasp.contact_depths[1]) < 1e-9
        )
        assert (
            reach - PAD_CELL - 1e-6 <= grasp.contact_depths[0] <= reach + 1e-6
        ), (depth, grasp.contact_depths)  # a grid of PAD_CELL
        area = (
            2 * (reach - PAD_A[0]) * PAD_WIDTH
        )  # both pads, full width (the brick is 6 cm long)
        assert (
            abs(grasp.contact_area - area) < 2 * PAD_CELL * PAD_WIDTH + 0.01
        ), (depth, grasp.contact_area, area)
        assert (
            abs(grasp_finger_overlap(grasp, 'panda') - grasp.contact_depths[0])
            < 1e-12
        )
    # deeper than the pad is long: the whole pad holds the part (2 x 1.75 x 1.754 = 6.14 cm^2)
    full = contact_of(top_grasp(2.5), BRICK)
    assert (
        abs(full.contact_depths[0] - PAD_A[1]) <= PAD_CELL
        and abs(full.contact_area - 2 * (PAD_A[1] - PAD_A[0]) * PAD_WIDTH)
        < 0.1
    )
    # a narrow part (0.6 cm along the pad width) gives a narrow contact
    narrow = contact_of(top_grasp(1.0), box([0.6, 2, 4], [0, 0, 7]))
    assert (
        abs(narrow.contact_area - 2 * (1.0 - 0.01 - PAD_A[0]) * 0.6)
        < 2 * (PAD_CELL * 0.6 + PAD_CELL * 1.0) + 0.01
    ), narrow.contact_area
    # the fingers do not reach the part: no contact, no contact points (such a candidate is dropped by the generator)
    away = contact_of(top_grasp(-0.5), BRICK)
    assert (
        away.contact_depths == [0.0, 0.0]
        and away.contact_area == 0.0
        and len(away.contact_points) == 0
    )
    assert grasp_finger_overlap(away, 'panda') == 0.0


@test
def pad_contact_weaker_finger(c):
    """a part that sits 1 mm off the middle touches one pad only: the grasp counts as held by the weaker finger"""
    grasp = contact_of(top_grasp(1.0), box([6, 2, 4], [0, 0.1, 7]))
    assert (
        sorted(grasp.contact_depths)[0] == 0.0
        and abs(sorted(grasp.contact_depths)[1] - 0.99) <= PAD_CELL
    ), grasp.contact_depths
    assert (
        len(grasp.contact_points) > 0
        and grasp_finger_overlap(grasp, 'panda') == 0.0
    )
    assert (
        abs(grasp.contact_area - (0.99 - PAD_A[0]) * PAD_WIDTH)
        < PAD_CELL * PAD_WIDTH + 0.01
    )  # one pad
    # 0.2 mm off: within the 0.4 mm the fingers close (PAD_CLOSING), both pads hold
    both = contact_of(top_grasp(1.0), box([6, 2, 4], [0, 0.02, 7]))
    assert min(both.contact_depths) > 0.9

    # grasp_finger_overlap: the weaker finger of grasps with depths, the old measure for grasps without
    def make(**kw):
        return types.SimpleNamespace(
            pos=grasp.pos, quat=grasp.quat, open_ratio=grasp.open_ratio, **kw
        )

    assert (
        grasp_finger_overlap(
            make(
                contact_points=grasp.contact_points, contact_depths=[0.8, 0.25]
            ),
            'panda',
        )
        == 0.25
    )
    assert (
        grasp_finger_overlap(
            make(contact_points=grasp.contact_points, contact_depths=[]),
            'panda',
        )
        == 0.0
    )
    old = grasp_finger_overlap(
        make(contact_points=grasp.contact_points), 'panda', COLLISION
    )  # deepest stored point, of either finger
    assert abs(old - 0.99) < 2 * PAD_CELL, old


@test
def pad_contact_first_touch(c):
    """
    the lower half of the part is wider than the upper half the grasp was made for: the fingers stop on the wider half, so only
    that half holds (0.5 cm of the pad), not the 1.5 cm the opening of the grasp suggests
    """
    upper = box([6, 2.0, 1.0], [0, 0, 8.5])  # z 8..9
    tip = (
        9 - 1.5 + 0.01
    )  # top_grasp(1.5): the fingertips reach 0.49 cm below the step at z = 8
    for width, depth in [
        (2.2, 8 - tip),
        (2.06, 9 - tip),
        (2.0, 9 - tip),
    ]:  # 1 mm wider on each side / 0.3 mm (within PAD_CLOSING) / flush
        part = trimesh.util.concatenate(
            [upper, box([6, width, 3.0], [0, 0, 6.5])]
        )  # z 5..8
        grasp = contact_of(top_grasp(1.5), part)
        assert all(
            depth - PAD_CELL - 1e-6 <= d <= depth + 1e-6
            for d in grasp.contact_depths
        ), (width, grasp.contact_depths, depth)
    # and the other way round: the pads stand 1 mm away from a narrower lower half, the upper half holds alone (rows 0.49..1.49)
    part = trimesh.util.concatenate([upper, box([6, 1.8, 3.0], [0, 0, 6.5])])
    grasp = contact_of(top_grasp(1.5), part)
    assert (
        abs(grasp.contact_depths[0] - (9 - tip)) <= PAD_CELL
        and abs(grasp.contact_area - 2 * 1.0 * PAD_WIDTH)
        < 2 * PAD_CELL * PAD_WIDTH + 0.01
    )


@test
def pad_contact_on_a_cylinder(c):
    """a round bar of radius r touches a flat pad along a strip: where the surface is at most 0.4 mm away, half width sqrt(2 r c - c^2)"""
    for radius in [0.5, 1.0, 2.0]:
        bar = trimesh.creation.cylinder(
            radius=radius, height=6.0, sections=720
        )  # axis along z first
        bar.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, Y)
        )  # along x: across the fingers
        bar.apply_translation([0, 0, 7])
        grasp = contact_of(
            toy_grasp([0, 0, 7 - 1.0], -Z, Y, 2 * radius), bar
        )  # the fingertips 1 cm below the axis
        # the strip is counted from the line where the pad touches first; there the surface is c = PAD_CLOSING closer than at its edge
        half = np.sqrt(
            2 * radius * PAD_CLOSING - PAD_CLOSING**2
        )  # 0.20, 0.28, 0.40 cm
        assert (
            abs(grasp.contact_depths[0] - (1.0 - 0.01 + half)) <= 2 * PAD_CELL
        ), (radius, grasp.contact_depths, 1.0 - 0.01 + half)
        assert abs(grasp.contact_depths[0] - grasp.contact_depths[1]) < 1e-9
        assert (
            abs(grasp.contact_area - 2 * 2 * half * PAD_WIDTH)
            < 2 * 2 * PAD_CELL * PAD_WIDTH + 0.02
        ), (radius, grasp.contact_area, 4 * half * PAD_WIDTH)
        # the bar standing between the fingers (top at z = 9): the strip runs along the finger, from the fingertip to the top of the bar
        standing = trimesh.creation.cylinder(
            radius=radius, height=6.0, sections=720
        )
        standing.apply_translation([0, 0, 6])
        grasp = contact_of(
            toy_grasp([0, 0, 9 - 1.0], -Z, Y, 2 * radius), standing
        )
        assert abs(grasp.contact_depths[0] - 0.99) <= PAD_CELL
        width = min(2 * half, PAD_WIDTH)
        assert (
            abs(grasp.contact_area - 2 * (0.99 - PAD_A[0]) * width)
            < 2 * (PAD_CELL * width + 2 * PAD_CELL * 1.0) + 0.02
        ), (radius, grasp.contact_area)


def pad_surface_slow(pad, finger_transform, mesh, closing, cell):
    """pad_surface written the plain way: every face against every grid point (no pre-selection, no bounding boxes, no chunks)"""
    from planning.config import PAD_PENETRATION

    rot = finger_transform[:3, :3]
    origin = rot @ pad['origin'] + finger_transform[:3, 3]
    along, across, normal = (
        rot @ pad['along'],
        rot @ pad['across'],
        rot @ pad['normal'],
    )
    (a0, a1), (b0, b1) = pad['a'], pad['b']
    n_a, n_b = int(round((a1 - a0) / cell)), int(round((b1 - b0) / cell))
    pa, pb = np.meshgrid(
        a0 + (np.arange(n_a) + 0.5) * cell,
        b0 + (np.arange(n_b) + 0.5) * cell,
        indexing='ij',
    )
    height = np.full((n_a, n_b), np.inf)
    for triangle, face_normal in zip(
        np.asarray(mesh.triangles), np.asarray(mesh.face_normals)
    ):
        if face_normal @ normal >= -1e-6:
            continue  # does not face the pad
        g, a, b = (
            (triangle - origin) @ normal,
            (triangle - origin) @ along,
            (triangle - origin) @ across,
        )
        det = (b[1] - b[2]) * (a[0] - a[2]) + (a[2] - a[1]) * (b[0] - b[2])
        if abs(det) < 1e-12:
            continue
        w0 = ((b[1] - b[2]) * (pa - a[2]) + (a[2] - a[1]) * (pb - b[2])) / det
        w1 = ((b[2] - b[0]) * (pa - a[2]) + (a[0] - a[2]) * (pb - b[2])) / det
        w2 = 1 - w0 - w1
        h = w0 * g[0] + w1 * g[1] + w2 * g[2]
        inside = (
            (w0 >= -1e-9)
            & (w1 >= -1e-9)
            & (w2 >= -1e-9)
            & (h <= closing)
            & (h >= -PAD_PENETRATION)
        )
        height[inside] = np.minimum(height[inside], h[inside])
    return height


@test
def pad_surface_matches_the_plain_computation(c):
    """the fast rasterisation (face pre-selection, bounding boxes, chunks) gives the same surface as the plain loops"""
    rng = np.random.RandomState(3)
    ball = trimesh.creation.icosphere(subdivisions=3, radius=1.3)
    capsule = trimesh.creation.capsule(height=2.0, radius=0.8, count=[16, 16])
    tilted = trimesh.creation.box([3, 1.5, 2.5])
    tilted.apply_transform(
        trimesh.transformations.rotation_matrix(0.3, [1, 2, 3])
    )
    compared = hits = 0
    for mesh in [
        ball,
        capsule,
        tilted,
        BRICK.copy().apply_translation([0, 0, -7]),
    ]:
        faces = PartFaces(mesh)
        for _ in range(6):
            approach = R.random(random_state=rng).apply(Z)
            closing = np.cross(approach, R.random(random_state=rng).apply(Z))
            closing /= np.linalg.norm(closing)
            width = (
                2
                * np.abs(mesh.vertices @ closing).max()
                * rng.uniform(0.9, 1.02)
            )  # the pads near the surface, in front of or behind it
            grasp = toy_grasp(
                rng.uniform(-0.3, 0.3, 3) - 0.5 * approach,
                approach,
                closing,
                min(width, 7.5),
            )
            transforms = get_gripper_meshes_transforms(
                'panda',
                VISUAL,
                grasp.pos,
                grasp.quat,
                np.eye(4),
                grasp.open_ratio,
            )
            for pad in PADS:
                slow = pad_surface_slow(
                    pad, transforms[pad['name']], mesh, PAD_CLOSING, PAD_CELL
                )
                for chunk in [200000, 37]:  # 37: many small chunks
                    fast, _ = pad_surface(
                        pad,
                        transforms[pad['name']],
                        faces,
                        PAD_CLOSING,
                        PAD_CELL,
                        chunk=chunk,
                    )
                    assert np.array_equal(
                        np.isfinite(fast), np.isfinite(slow)
                    ), 'different grid points in contact'
                    assert np.allclose(
                        fast[np.isfinite(fast)],
                        slow[np.isfinite(slow)],
                        atol=1e-12,
                    )
                compared += 1
                hits += int(np.isfinite(slow).any())
    assert compared == 48 and hits >= 12, (
        compared,
        hits,
    )  # enough of the random grasps touch their part


@test
def pad_contact_points_and_determinism(c):
    grasp = contact_of(top_grasp(1.0), BRICK)
    again = contact_of(top_grasp(1.0), BRICK)
    assert (
        np.array_equal(grasp.contact_points, again.contact_points)
        and grasp.contact_depths == again.contact_depths
        and grasp.contact_area == again.contact_area
    )
    points = np.asarray(grasp.contact_points)
    # stored: one point per PAD_STORE_CELL cell (1 mm^2) of the contact, plus the deepest one of each finger
    cells = grasp.contact_area / PAD_STORE_CELL**2
    assert abs(len(points) - (cells + 2)) < 0.15 * cells, (len(points), cells)
    assert set(np.sign(np.round(points[:, 1], 6))) == {
        -1.0,
        1.0,
    }  # on both pads
    assert (
        np.allclose(np.abs(points[:, 1]), np.abs(points[0, 1]), atol=1e-9)
        and 0.95 < abs(points[0, 1]) <= 1.0 + 1e-9
    )  # on the pad planes, closed onto the part
    assert (
        np.abs(points[:, 0]).max() <= PAD_WIDTH / 2 + 1e-9
    )  # inside the pad, across
    tip_z = 9 - 1.0 + 0.01
    assert (
        points[:, 2].min() >= tip_z + PAD_A[0] - 1e-9
        and points[:, 2].max() <= 9.0 + 1e-9
    )  # between the fingertip and the top of the brick
    assert (
        abs((points[:, 2].max() - tip_z) - grasp.contact_depths[0]) < 1e-9
    )  # the deepest stored point is the depth
    # the stored contact moves with the grasp: the same grasp on the same part, both turned and shifted, gives the same numbers
    T = trimesh.transformations.rotation_matrix(
        0.7, [1, 2, 3], point=[1, 0, 2]
    )
    T[:3, 3] += [3, -2, 5]
    moved_grasp = toy_grasp(
        T[:3, :3] @ [0, 0, 8.0] + T[:3, 3], T[:3, :3] @ -Z, T[:3, :3] @ Y, 2.0
    )
    moved = contact_of(moved_grasp, BRICK.copy().apply_transform(T))
    assert (
        np.allclose(
            moved.contact_depths, grasp.contact_depths, atol=PAD_CELL + 1e-9
        )
        and abs(moved.contact_area - grasp.contact_area) < 0.1
    )


# ------------------------------------------------------------------------------------------------ criteria of a move grasp


def arm_grasp(c, q, depth, open_ratio=0.25, grasp_id=0):
    """the grasp of the real arm at pose q and the brick it holds: the part reaches `depth` cm into the fingers"""
    pos, quat = get_gripper_pos_quat_from_arm_q(c.chain, q, 'panda')
    grasp = Grasp('A', grasp_id, pos, quat, open_ratio)
    grasp.arm_q = q
    rot = R.from_quat(
        np.asarray(quat)[[1, 2, 3, 0]]
    ).as_matrix()  # columns: across, closing, approach
    along = (
        fingertip(grasp) - pos @ rot[:, 2]
    )  # fingertip in the gripper frame
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rot, pos
    part = box(
        [6, 8 * open_ratio, 4], [0, 0, along - depth + 2.0]
    ).apply_transform(
        T
    )  # its face towards the hand `depth` behind the fingertip
    return contact_of(grasp, part), part


@test
def move_grasp_criteria_each(c):
    q = ik_down(c, REF_XY, 10.0)
    grasp, part = arm_grasp(c, q, 1.0)
    rots = get_flat_rots(part)
    assert sorted(round(r['tip_angle'], 1) for r in rots) == [
        26.6,
        26.6,
        63.4,
        63.4,
    ]
    assert abs(grasp_finger_overlap(grasp, 'panda') - 1.0) <= PAD_CELL
    all_true = {
        'walls': True,
        'conditioned': True,
        'overlap': True,
        'flat': True,
    }
    assert (
        move_grasp_criteria([grasp], part, rots, 'panda', COLLISION, c.chain)
        == all_true
    )
    # overlap: GRASP_MIN_FINGER_OVERLAP of the weaker finger
    for depth, ok in [
        (GRASP_MIN_FINGER_OVERLAP + 0.05, True),
        (GRASP_MIN_FINGER_OVERLAP - 0.05, False),
    ]:
        g, p = arm_grasp(c, q, depth)
        assert move_grasp_criteria(
            [g], p, get_flat_rots(p), 'panda', COLLISION, c.chain
        ) == {**all_true, 'overlap': ok}, depth
    # flat: only resting poses where the hand would come from the side
    lying = [r for r in rots if round(r['tip_angle'], 1) == 63.4]
    assert move_grasp_criteria(
        [grasp], part, lying, 'panda', COLLISION, c.chain
    ) == {**all_true, 'flat': False}
    assert move_grasp_criteria(
        [grasp],
        part,
        rots,
        'panda',
        COLLISION,
        c.chain,
        max_tilt=30.0,
        clearance=3.5,
    ) == {**all_true, 'flat': False}  # 3 cm above the board
    # walls: one pose of the insertion path too close to a virtual wall
    q_wall, near_wall = q, None
    for dy in np.arange(0, 40, 0.5):
        q_next = c.planner.inverse_kinematics(
            np.array([REF_XY[0], REF_XY[1] - dy, 25.0]), DOWN, q_init=q_wall
        )
        if q_next is None:
            continue
        q_wall = q_next
        if (
            get_virtual_wall_margin(c.chain, q_wall)
            < VIRTUAL_WALL_SAFETY - 0.1
        ):
            near_wall = q_wall
            break
    assert near_wall is not None
    second = grasp.copy()
    second.arm_q = near_wall
    result = move_grasp_criteria(
        [grasp, second], part, rots, 'panda', COLLISION, c.chain
    )
    assert result['walls'] is False and result['overlap'] and result['flat']
    # conditioned: a pose of the path near a singularity, or with a joint near its limit
    second.arm_q = pose_with_sigma(c, MIN_SIGMA_INSERTION - 0.005)
    assert (
        move_grasp_criteria(
            [grasp, second], part, rots, 'panda', COLLISION, c.chain
        )['conditioned']
        is False
    )
    q_limit = c.chain.active_from_full(q).copy()
    q_limit[0] = PANDA_JOINT_LIMITS[1][0] - (MIN_JOINT_LIMIT_MARGIN - 0.01)
    second.arm_q = c.chain.active_to_full(q_limit)
    assert joint_condition(q_limit)[1] < MIN_JOINT_LIMIT_MARGIN
    assert (
        move_grasp_criteria(
            [grasp, second], part, rots, 'panda', COLLISION, c.chain
        )['conditioned']
        is False
    )
    # another arm than the real one: only the grasp itself is judged
    assert (
        move_grasp_criteria(
            [grasp, second], part, rots, 'panda', COLLISION, None
        )
        == all_true
    )


@test
def flat_move_grasps_and_fallback_poses(c):
    """the less stable resting poses are used for a part only if none of its grasps fits a stable one"""

    def as_move(grasp):
        return [contact_of(grasp, BRICK)]

    top, side, end, low, shallow, lean = (
        as_move(g)
        for g in [
            top_grasp(1.0, grasp_id=1),
            side_grasp(1.0, grasp_id=2),
            end_grasp(1.0, grasp_id=3),
            low_grasp(4),
            top_grasp(0.2, grasp_id=5),
            leaning_grasp(45.0, grasp_id=6),
        ]
    )

    def ids(moves):
        return [m[0].grasp_id for m in moves]

    kept, rots, fallback = flat_move_grasps(
        [top, side, end, low, shallow, lean], BRICK, 'panda', COLLISION
    )
    assert (
        ids(kept) == [1, 2, 6] and fallback is False and len(rots) == 4
    )  # the end grasp would need the unstable pose: not kept; a leaning hand is fine
    kept, rots, fallback = flat_move_grasps(
        [top, side, end, low, shallow, lean],
        BRICK,
        'panda',
        COLLISION,
        max_tilt=30.0,
    )
    assert (
        ids(kept) == [1, 2]
    )  # with the tilt limit of the earlier pipeline the leaning hand is dropped
    kept, rots, fallback = flat_move_grasps(
        [end, low, shallow], BRICK, 'panda', COLLISION
    )
    assert ids(kept) == [3] and fallback is True and len(rots) == 6
    kept, rots, fallback = flat_move_grasps(
        [end, low, shallow], BRICK, 'panda', COLLISION, allow_fallback=False
    )
    assert kept == [] and fallback is False and len(rots) == 4
    kept, rots, fallback = flat_move_grasps(
        [low, shallow], BRICK, 'panda', COLLISION
    )
    assert (
        kept == [] and fallback is False and len(rots) == 4
    )  # nothing fits any pose: no fallback is reported
    kept, rots, fallback = flat_move_grasps([], BRICK, 'panda', COLLISION)
    assert kept == [] and fallback is False and len(rots) == 4
    # the later stages read the decision per part from the settings of grasps.pkl
    settings = {'settings': {'flat_pickup': {'fallback_parts': ['B']}}}
    assert (
        len(flat_rots_of(BRICK, 'A', settings)) == 4
        and len(flat_rots_of(BRICK, 'B', settings)) == 6
    )
    assert (
        len(flat_rots_of(BRICK, 'B', {'settings': {}})) == 4
        and len(flat_rots_of(BRICK, 'B', {'settings': {'flat_pickup': {}}}))
        == 4
    )


# --------------------------------------------------------------------------------------------- rating and choice of grasps


@test
def centre_of_mass(c):
    assert np.allclose(part_center_of_mass(BRICK), [0, 0, 7])
    lopsided = trimesh.util.concatenate(
        [box([2, 2, 2], [0, 0, 1]), box([2, 2, 2], [4, 0, 1])]
    )  # two cubes
    assert np.allclose(part_center_of_mass(lopsided), [2, 0, 1])
    open_box = BRICK.copy()
    open_box.update_faces(
        np.arange(len(open_box.faces)) >= 2
    )  # one face missing: no volume, the surface centroid is used
    assert not open_box.is_watertight
    com = part_center_of_mass(open_box)
    assert np.all(np.isfinite(com)) and np.linalg.norm(com - [0, 0, 7]) < 1.0


@test
def rating_terms(c):
    com = part_center_of_mass(BRICK)

    def lying(grasp):
        return get_flat_pickup_candidates(
            grasp,
            BRICK,
            get_flat_rots(BRICK),
            'panda',
            COLLISION,
            clearance=-100.0,
            wrist_clearance=-100.0,
        )[0]

    def rate(grasp, depths, area, **kw):
        grasp.contact_depths, grasp.contact_area = depths, area
        return rate_flat_grasp(grasp, lying(grasp), com, 'panda', **kw)

    assert abs(sum(FLAT_RATING_WEIGHTS.values()) - 1.0) < 1e-12
    best = rate(top_grasp(1.0), [1.0, 1.0], 2.0)
    assert sorted(best) == [
        'balance',
        'grip',
        'insertion',
        'rating',
        'upright',
    ] and all(abs(v - 1.0) < 1e-6 for v in best.values()), best
    # grip: half from the depth of the weaker finger (full at 1 cm), half from the contact area (full at 2 cm^2)
    assert (
        abs(
            rate(top_grasp(1.0), [0.3, 1.0], 0.5)['grip']
            - (0.5 * 0.3 + 0.5 * 0.25)
        )
        < 1e-12
    )
    assert (
        rate(top_grasp(1.0), [2.0, 2.0], 6.0)['grip'] == 1.0
        and rate(top_grasp(1.0), [0.0, 1.0], 0.0)['grip'] == 0.0
    )
    # balance: the centre of mass beside the line between the fingers (full when on it, zero from 3 cm)
    for x, balance in [
        (0.0, 1.0),
        (1.5, 0.5),
        (-1.5, 0.5),
        (3.0, 0.0),
        (2.9, 1 - 2.9 / 3),
    ]:
        assert (
            abs(
                rate(top_grasp(1.0, x=x), [1.0, 1.0], 2.0)['balance'] - balance
            )
            < 1e-9
        ), x
    off_along_closing = toy_grasp(
        [0, 0.4, 8.0], -Z, Y, 2.0
    )  # shifted along the line between the fingers: still balanced
    assert (
        abs(rate(off_along_closing, [1.0, 1.0], 2.0)['balance'] - 1.0) < 1e-9
    )
    # upright: the hand straight above (1) ... from the side, RATING_TILT = 90 deg (0)
    assert RATING_TILT == 90.0
    for angle, upright in [
        (0.0, 1.0),
        (30.0, 2 / 3),
        (45.0, 0.5),
        (81.0, 0.1),
        (90.0, 0.0),
    ]:
        assert (
            abs(
                rate(leaning_grasp(angle), [1.0, 1.0], 2.0)['upright']
                - upright
            )
            < 1e-6
        ), angle
    # insertion: the rank by the move score of the original pipeline (0 best ... 1 worst); equal scores share a rank
    assert (
        rate(top_grasp(1.0), [1.0, 1.0], 2.0, insertion_rank=1.0)['insertion']
        == 0.0
    )
    assert (
        abs(
            rate(top_grasp(1.0), [1.0, 1.0], 2.0, insertion_rank=0.25)[
                'rating'
            ]
            - (1.0 - 0.25 * FLAT_RATING_WEIGHTS['insertion'])
        )
        < 1e-6
    )
    assert insertion_ranks([3.0, 1.0, 2.0]).tolist() == [
        1.0,
        0.0,
        0.5,
    ] and insertion_ranks([0.0, 0.0, 0.0]).tolist() == [0.0, 0.0, 0.0]
    assert (
        insertion_ranks([1.0, 1.0, 2.0, 5.0, 5.0]).tolist()
        == [0.0, 0.0, 0.5, 0.75, 0.75]
        and insertion_ranks([4.0]).tolist() == [0.0]
        and len(insertion_ranks([])) == 0
    )
    # the rating is the weighted sum, and a real pad contact rates deeper grasps higher
    r = rate(top_grasp(1.0, x=1.5), [0.5, 0.6], 1.0, insertion_rank=0.5)
    assert (
        abs(
            r['rating']
            - sum(FLAT_RATING_WEIGHTS[k] * r[k] for k in FLAT_RATING_WEIGHTS)
        )
        < 1e-12
    )
    ratings = []
    for depth in [0.3, 0.5, 0.8, 1.2]:
        grasp = contact_of(top_grasp(depth), BRICK)
        ratings.append(
            rate_flat_grasp(grasp, lying(grasp), com, 'panda')['rating']
        )
    assert ratings == sorted(ratings) and ratings[0] < ratings[-1], ratings
    # a grasp without the pad measurement (older grasps.pkl) is rated by the other terms only
    bare = top_grasp(1.0)
    assert rate_flat_grasp(bare, lying(bare), com, 'panda')['grip'] == 0.0


@test
def insertion_score_of_the_original_pipeline(c):
    """torque of the insertion forces about the centre of mass: 0 for a grasp around the axis of the forces, larger beside it"""
    G = nx.DiGraph()
    G.add_node('base', path=None, com=np.zeros(3))
    G.add_node(
        'A',
        path=np.array([[0, 0, 5.0, 0, 0, 0], [0, 0, 0.0, 0, 0, 0]]),
        com=np.array([0, 0, 7.0]),
    )
    G.add_edge(
        'base',
        'A',
        contact_points=np.array(
            [[1, 1, 5.0], [-1, 1, 5.0], [1, -1, 5.0], [-1, -1, 5.0]]
        ),
    )  # the brick stands on the base
    grasp = types.SimpleNamespace(
        part_id='A',
        contact_points=np.array(
            [[0, 1, 8.0], [0, -1, 8.0], [0.5, 1, 8.5], [-0.5, -1, 8.5]]
        ),
    )
    assert insertion_score(grasp, G) < 1e-12
    grasp.contact_points = (
        grasp.contact_points + [2.0, 0, 0]
    )  # 2 cm beside the centre of mass: torque 2, divided by the 4 contact points
    assert abs(insertion_score(grasp, G) - 0.5) < 1e-12
    assert (
        insertion_score(
            types.SimpleNamespace(
                part_id='base', contact_points=grasp.contact_points
            ),
            G,
        )
        == 0.0
    )  # the base part is not inserted
    assert (
        insertion_score(
            types.SimpleNamespace(part_id='A', contact_points=[]), G
        )
        == 0.0
    )


@test
def selection_spreads_over_directions(c):
    assert np.allclose(approach_direction(top_grasp(1.0)), -Z) and np.allclose(
        approach_direction(side_grasp(1.0)), -Y
    )
    # 6 grasps from above, 3 from the side, 1 diagonal
    ratings = [0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.50, 0.40, 0.30, 0.20]
    directions = [-Z] * 6 + [-Y] * 3 + [[0.7, 0, -0.7]]
    assert select_grasps(ratings, directions, None) == list(
        range(10)
    ) and select_grasps(ratings, directions, 10) == list(range(10))
    assert select_grasps(ratings[::-1], directions[::-1], 20) == list(
        range(9, -1, -1)
    )  # fewer than n: all, best first
    assert select_grasps(ratings, directions, 5) == [
        0,
        6,
        9,
        1,
        7,
    ]  # the directions take turns, each giving its best
    assert select_grasps(ratings, directions, 8) == [
        0,
        6,
        9,
        1,
        7,
        2,
        8,
        3,
    ]  # an empty direction is skipped
    assert (
        select_grasps(ratings, directions, 1) == [0]
        and select_grasps(ratings, directions, 0) == []
    )
    assert select_grasps([], [], 5) == []
    # slightly different directions fall into the same group; ties are broken by the index (deterministic)
    near = [
        R.from_euler('x', a, degrees=True).apply(-Z) for a in [0, 10, -15, 20]
    ]
    assert select_grasps([0.5, 0.5, 0.5, 0.5], near, 2) == [0, 1]
    assert select_grasps([0.5, 0.7, 0.5, 0.7], near, 3) == [1, 3, 0]
    many = np.random.RandomState(0)
    ratings, directions = (
        many.rand(300),
        R.random(300, random_state=many).apply(Z),
    )
    chosen = select_grasps(ratings, directions, 100)
    assert len(chosen) == len(set(chosen)) == 100 and chosen == select_grasps(
        ratings, directions, 100
    )
    groups = {
        tuple(np.round(d / np.abs(d).max()).astype(int)) for d in directions
    }
    assert {
        tuple(
            np.round(directions[i] / np.abs(directions[i]).max()).astype(int)
        )
        for i in chosen
    } == groups  # every direction is represented


# ------------------------------------------------------------------------- the flat mode of the grasp generation (--flat)

config.FLAT_MIN_GRASPS = 3  # the scripted scenarios below are written for a second round below 3 usable grasps, whatever planning/config.py sets


def toy_generator(
    rounds, first_held=False, infeasible=(), part=BRICK, lying=()
):
    """
    GraspArmGenerator.generate_grasps_flat on the brick with scripted candidates instead of sampled ones (rounds: one list of
    grasps per call of antipodal_candidates; lying: one list per call of lying_candidates, none if not given) and without an
    arm: every candidate is reachable unless its id is in `infeasible`.
    The pad contact, the flat-pickup test, the criteria, the rating and the selection are the real ones.
    """
    gen = object.__new__(GraspArmGenerator)
    gen.flat, gen.pads, gen.part_faces = True, PADS, {}
    gen.gripper_type, gen.arm_type = (
        'panda',
        'toy',
    )  # not the real arm: no virtual walls, no conditioning
    gen.gripper_meshes, gen.gripper_meshes_visual = COLLISION, VISUAL
    gen.part_meshes, gen.part_final_transforms = {'A': part}, {'A': np.eye(4)}
    gen.arm_chains = {'move': None, 'hold': None}
    gen.n_surface_pt = 200
    gen.G_preced = nx.DiGraph()
    gen.G_preced.add_node(
        'A',
        path=None
        if first_held
        else np.array([[0, 0, 5.0, 0, 0, 0], [0, 0, 0.0, 0, 0, 0]]),
        com=np.array([0, 0, 7.0]),
        parts_after=[] if first_held else ['base'],
    )
    gen.calls, gen.checked, gen.lying_calls = [], [], []

    def antipodal_candidates(part_id, part_mesh, sample_budget, grasp_id):
        gen.calls.append((sample_budget, grasp_id))
        made = []
        for i, make in enumerate(rounds[len(gen.calls) - 1]):
            made.append(make(grasp_id + i))
        return made

    def lying_candidates(
        part_id, part_mesh, flat_rots, sample_budget, grasp_id
    ):
        gen.lying_calls.append(
            (
                sample_budget,
                grasp_id,
                sorted({round(r['tip_angle'], 1) for r in flat_rots}),
            )
        )
        made = []
        if len(gen.lying_calls) <= len(lying):
            for i, make in enumerate(lying[len(gen.lying_calls) - 1]):
                made.append(make(grasp_id + i))
                made[-1].source = 'lying'
        return made

    gen.lying_candidates = lying_candidates

    def check_grasp_feasible(grasp, part_id, verbose=False):
        gen.checked.append(grasp.grasp_id)
        if grasp.grasp_id in infeasible:
            return None
        return {'move': [grasp.copy()], 'hold': grasp.copy()}

    gen.antipodal_candidates, gen.check_grasp_feasible = (
        antipodal_candidates,
        check_grasp_feasible,
    )
    return gen


def TOP(depth, x=0.0):
    return lambda grasp_id: top_grasp(depth, x, grasp_id)


def SIDE(depth):
    return lambda grasp_id: side_grasp(depth, grasp_id)


def END(depth):
    return lambda grasp_id: end_grasp(depth, grasp_id)


LOW = low_grasp  # holds the brick, but cannot take it from the board in any resting pose


def LEAN(angle):
    return lambda grasp_id: leaning_grasp(angle, grasp_id)


def move_ids(grasps):
    return [g[0].grasp_id for g in grasps['move']]


def hold_ids(grasps):
    return [g.grasp_id for g in grasps['hold']]


@test
def generator_keeps_only_usable_move_grasps(c):
    """one round is enough: no re-sampling; move grasps must pick the brick lying flat and hold it deep enough, hold grasps need not"""
    gen = toy_generator(
        [
            [
                TOP(1.0),
                TOP(0.8, 1.0),
                TOP(0.2),
                SIDE(1.0),
                LOW,
                END(1.0),
                TOP(1.2),
                TOP(-0.5),
                LEAN(45.0),
                LEAN(85.0),
            ]
        ],
        infeasible=[6],
    )
    grasps = gen.generate_grasps('A', max_n_grasp=100, n_proc=1)
    assert gen.calls == [(200, 0)], (
        gen.calls
    )  # the surface point budget of the pipeline, ids from 0
    assert (
        sorted(gen.checked) == [0, 1, 2, 3, 4, 5, 6, 8, 9]
    )  # 7 has no contact with the part: dropped before the (expensive) feasibility check
    # 2: too shallow, 4: fingertips too close to the board, 5: only in the unstable pose, 6: not reachable, 9: the wrist would be in the board;
    # 8: a hand leaning 45 deg is kept (no tilt limit)
    assert sorted(move_ids(grasps)) == [0, 1, 3, 8]
    assert sorted(hold_ids(grasps)) == [0, 1, 2, 3, 4, 5, 8, 9]
    assert all(
        hasattr(g[0], 'contact_depths') and hasattr(g[0], 'flat_rating')
        for g in grasps['move']
    )
    assert all(len(g.contact_points) > 0 for g in grasps['hold'])
    assert move_ids(grasps) == sorted(
        move_ids(grasps),
        key=lambda i: (
            -[g[0].flat_rating for g in grasps['move'] if g[0].grasp_id == i][
                0
            ]
        ),
    )  # best first


@test
def generator_resamples_when_too_few(c):
    """fewer than FLAT_MIN_GRASPS usable grasps: one more round with FLAT_RESAMPLE_PT pairs; its ids go on, only flat ones are checked"""
    assert (
        config.FLAT_MIN_GRASPS == 3 and FLAT_RESAMPLE_PT == 600
    )  # the limit of these scenarios (set above), the budget of the second round
    first = [
        TOP(0.2),
        TOP(0.2, 1.0),
        TOP(1.0),
        TOP(0.9),
        END(1.0),
        LOW,
    ]  # 2 usable
    second = [TOP(1.0, 0.5), LOW, TOP(0.7), SIDE(1.0), LOW, END(0.8)]
    gen = toy_generator([first, second])
    grasps = gen.generate_grasps('A', max_n_grasp=100, n_proc=1)
    assert gen.calls == [(200, 0), (FLAT_RESAMPLE_PT, 6)], gen.calls
    assert (
        sorted(gen.checked) == [0, 1, 2, 3, 4, 5, 6, 8, 9, 11]
    )  # of the second round only the ones that can pick the brick lying flat (11: in the unstable pose)
    assert sorted(move_ids(grasps)) == [
        2,
        3,
        6,
        8,
        9,
    ]  # the end grasps (4, 11) are not needed: stable poses have grasps
    ids = move_ids(grasps) + hold_ids(grasps)
    assert (
        len(set(move_ids(grasps))) == len(move_ids(grasps))
        and len(set(hold_ids(grasps))) == len(hold_ids(grasps))
        and max(ids) == 11
    )
    # exactly FLAT_MIN_GRASPS usable grasps: no second round
    gen = toy_generator([[TOP(1.0), TOP(0.9), TOP(0.8), LOW], second])
    assert (
        sorted(move_ids(gen.generate_grasps('A', max_n_grasp=100, n_proc=1)))
        == [0, 1, 2]
        and len(gen.calls) == 1
    )


@test
def generator_fallback_poses(c):
    """no grasp fits a stable resting pose, even after the second round: the unstable poses are admitted for this part (switch)"""
    rounds = [[END(1.0), LOW, TOP(0.2)], [END(0.8), LOW, END(0.2)]]
    gen = toy_generator(rounds)
    grasps = gen.generate_grasps('A', max_n_grasp=100, n_proc=1)
    assert len(gen.calls) == 2 and sorted(gen.checked) == [0, 1, 2, 3, 5]
    assert sorted(move_ids(grasps)) == [0, 3]  # 5: too shallow
    config.FLAT_FALLBACK_POSES = False
    try:
        gen = toy_generator(rounds)
        grasps = gen.generate_grasps('A', max_n_grasp=100, n_proc=1)
        assert (
            move_ids(grasps) == []
            and sorted(gen.checked) == [0, 1, 2]
            and sorted(hold_ids(grasps)) == [0, 1, 2]
        )
    finally:
        config.FLAT_FALLBACK_POSES = True


@test
def generator_selects_the_best_rated(c):
    """more usable grasps than max_n_grasp: the best rated are kept, taking turns between the approach directions"""
    tops = [
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        0.45,
    ]  # ids 0..7: deeper is better (up to 1 cm)
    sides = [0.5, 0.6, 0.9, 0.7]  # ids 8..11
    rounds = [[TOP(d) for d in tops] + [SIDE(d) for d in sides]]
    results = []
    for _ in range(2):
        np.random.seed(0)
        gen = toy_generator(rounds)
        results.append(gen.generate_grasps('A', max_n_grasp=6, n_proc=1))
    grasps = results[0]
    assert move_ids(grasps) == [6, 10, 5, 11, 4, 9], move_ids(
        grasps
    )  # top 1.0, side 0.9, top 0.9, side 0.7, top 0.8, side 0.6
    assert move_ids(results[1]) == move_ids(grasps) and hold_ids(
        results[1]
    ) == hold_ids(grasps)  # reproducible
    assert (
        len(hold_ids(grasps)) == 6 and len(set(hold_ids(grasps))) == 6
    )  # hold grasps: a random choice, as before
    # max_n_grasp None: everything usable is kept
    assert (
        len(
            move_ids(
                toy_generator(rounds).generate_grasps(
                    'A', max_n_grasp=None, n_proc=1
                )
            )
        )
        == 12
    )


@test
def generator_first_held_part(c):
    """the part that is held first also gets the best hold grasps that can pick it lying flat (max_n_grasp / 2 more); other parts do not"""
    rounds = [
        [TOP(0.4 + 0.1 * i) for i in range(6)] + [LOW, LOW, END(1.0), END(0.7)]
    ]  # ids 0..5 pick the brick in a stable pose, 6..9 do not
    holds = {}
    for first_held in [True, False]:
        np.random.seed(1)
        gen = toy_generator(rounds, first_held=first_held)
        holds[first_held] = hold_ids(
            gen.generate_grasps('A', max_n_grasp=4, n_proc=1)
        )
    random_four = holds[False]
    assert (
        len(random_four) == 4
        and len(set(random_four)) == 4
        and set(random_four) <= set(range(10))
    )  # the random choice of the original pipeline
    assert (
        holds[True][:4] == random_four
    )  # the same choice for the part held first ...
    best_flat = [i for i in [5, 4, 3, 2, 1, 0] if i not in random_four][
        :2
    ]  # ... plus the deepest grasps that pick it lying flat
    assert holds[True][4:] == best_flat, (holds, best_flat)
    # with few hold grasps all are kept anyway
    gen = toy_generator(rounds, first_held=True)
    assert sorted(
        hold_ids(gen.generate_grasps('A', max_n_grasp=100, n_proc=1))
    ) == list(range(10))


@test
def candidates_made_for_the_lying_part(c):
    """
    lying_grasp_candidates: the second direction of the matching. On the brick every number is known: its faces are axis
    parallel, so the fingers close horizontally in every resting pose and the hand can come from straight above.
    """
    assert config.FLAT_LYING_PT == 200 and config.FLAT_LYING_ANGLES == (
        0.0,
        20.0,
        -20.0,
    )
    rots = get_flat_rots(BRICK)
    np.random.seed(0)
    cands = lying_grasp_candidates(
        BRICK, rots, 'panda', COLLISION, 120, config.FLAT_LYING_ANGLES
    )
    assert len(cands) > 200 and {k for _, _, _, k in cands} == {
        0,
        1,
        2,
        3,
    }  # every resting pose gets its share
    seen_tilts, n_nominal = set(), 0
    for pos, quat, open_ratio, k in cands:
        assert round(8 * open_ratio, 6) in [
            2.0,
            4.0,
            6.0,
        ]  # across the 2, 4 or 6 cm of the brick
        rot = rots[k]['rot']
        rest = np.eye(4)
        rest[:3, :3], rest[2, 3] = rot, -(BRICK.vertices @ rot.T)[:, 2].min()
        grasp_rot = R.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()
        closing, approach = (
            rot @ grasp_rot @ Y,
            rot @ grasp_rot @ Z,
        )  # in the lying pose it was made for
        assert abs(closing[2]) < 1e-6  # the fingers close horizontally
        tilt = np.degrees(np.arccos(np.clip(-approach[2], -1, 1)))
        assert min(abs(tilt - t) for t in [0.0, 20.0]) < 1e-4, (
            tilt
        )  # straight above, or leaning 20 deg about the line between the fingers
        seen_tilts.add(round(tilt))
        # the gripper where the part lies: its lowest point keeps the clearance, and it is as low as allowed or the pair sits in the middle of the pad
        gripper = get_transform_matrix_quat(pos, quat)
        lying_pos, lying_quat = mat_to_pos_quat(rest @ gripper)
        low = min(
            min(
                m.bounds[0, 2]
                for m in transform_gripper_meshes(
                    'panda', COLLISION, lying_pos, lying_quat, np.eye(4), ratio
                ).values()
            )
            for ratio in [open_ratio, open_ratio + 0.1]
        )
        assert low >= GROUND_CLEARANCE, low
        grasp = contact_of(Grasp('A', 0, pos, quat, open_ratio), BRICK)
        at_the_board = (
            low < GROUND_CLEARANCE + 0.03
        )  # the fingers as low as the board allows; else the pair sits 1.6 or 0.9 cm deep in the pad of a tall face
        n_nominal += not at_the_board
        # it picks the brick lying flat, in the pose it was made for or a better one, and its pads touch the brick
        flat = get_flat_pickup_candidates(
            grasp, BRICK, rots, 'panda', COLLISION
        )
        assert (
            len(flat) > 0
            and flat[0]['tilt'] <= tilt + 1e-4
            and len(grasp.contact_points) > 0
        )
    assert seen_tilts == {0, 20} and 0 < n_nominal < len(
        cands
    )  # both kinds: as low as the board allows, and higher on tall faces
    # most of them hold the brick deep enough (the few others touch it near a corner)
    deep = sum(
        min(contact_of(Grasp('A', 0, pos, quat, ratio), BRICK).contact_depths)
        >= GRASP_MIN_FINGER_OVERLAP
        for pos, quat, ratio, _ in cands
    )
    assert deep > 0.9 * len(cands), (deep, len(cands))
    # the hand directions asked for; no pairs or no resting pose: no candidates; same seed, same candidates
    np.random.seed(1)
    upright = lying_grasp_candidates(
        BRICK, rots[:1], 'panda', COLLISION, 40, (0.0,)
    )
    assert len(upright) > 0 and all(k == 0 for _, _, _, k in upright)
    for pos, quat, _, _ in upright:
        assert np.allclose(
            rots[0]['rot']
            @ R.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()
            @ Z,
            -Z,
            atol=1e-6,
        )
    assert (
        lying_grasp_candidates(BRICK, rots, 'panda', COLLISION, 0, (0.0,))
        == []
        and lying_grasp_candidates(BRICK, [], 'panda', COLLISION, 40, (0.0,))
        == []
    )
    np.random.seed(1)
    again = lying_grasp_candidates(
        BRICK, rots[:1], 'panda', COLLISION, 40, (0.0,)
    )
    assert len(again) == len(upright) and all(
        np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])
        for a, b in zip(upright, again)
    )
    # a hand that would lie almost flat is not made: leaning 80 deg about the line between the fingers
    assert (
        lying_grasp_candidates(
            BRICK, rots[:1], 'panda', COLLISION, 40, (80.0,)
        )
        == []
    )
    # faces that slope: a bar with a parallelogram as cross-section, 2 cm high, its two long sides sloping by `slope` deg
    from shapely.geometry import Polygon

    def sloped_bar(slope, base):
        shift = 2.0 / np.tan(np.radians(slope))
        bar = trimesh.creation.extrude_polygon(
            Polygon([(0, 0), (base, 0), (base + shift, 2.0), (shift, 2.0)]),
            height=4.0,
        )  # cross-section in x and y, 4 cm long along z
        bar.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, X)
        )  # the height of the cross-section upwards
        return bar

    def lying(bar, seed):
        np.random.seed(seed)
        bar_rots = get_flat_rots(bar)
        rows = []
        for pos, quat, _, k in lying_grasp_candidates(
            bar, bar_rots, 'panda', COLLISION, 200, (0.0,)
        ):
            grasp_rot = R.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()
            closing, approach = (
                bar_rots[k]['rot'] @ grasp_rot @ Y,
                bar_rots[k]['rot'] @ grasp_rot @ Z,
            )
            rows.append(
                (
                    np.degrees(np.arcsin(abs(closing[2]))),
                    np.degrees(np.arccos(np.clip(-approach[2], -1, 1))),
                )
            )
        return np.array(rows)

    steep = lying(
        sloped_bar(60.0, 3.0), 2
    )  # sides sloping 60 deg: the line between fingers on them is 30 deg from horizontal
    assert (
        len(steep) > 20 and np.allclose(steep[:, 0], steep[:, 1], atol=1e-4)
    )  # the hand is as upright as the pair allows: it leans as much as the line between the fingers
    assert set(np.round(steep[:, 0]).astype(int)) == {0, 30}, set(
        np.round(steep[:, 0]).astype(int)
    )
    shallow = lying(
        sloped_bar(30.0, 4.0), 2
    )  # sides sloping 30 deg: that line would be 60 deg from horizontal, more than LYING_MAX_AXIS_TILT
    assert len(shallow) > 5 and set(np.round(shallow[:, 0]).astype(int)) == {
        0
    }, set(np.round(shallow[:, 0]).astype(int))  # only across its ends
    # the pair search gives up instead of running forever when a mesh has no pair it accepts (here: none above the given height)
    assert compute_antipodal_pairs(
        BRICK, sample_budget=20, ground_z=100.0, max_rounds=3
    ).shape == (0, 2, 3)
    # the generator marks them and counts their ids on
    gen = object.__new__(GraspArmGenerator)
    gen.gripper_type, gen.gripper_meshes, gen.antipodal_thres = (
        'panda',
        COLLISION,
        0.95,
    )
    np.random.seed(1)
    made = gen.lying_candidates('A', BRICK, rots[:1], 40, 17)
    assert [g.grasp_id for g in made] == list(
        range(17, 17 + len(made))
    ) and all(g.source == 'lying' and g.part_id == 'A' for g in made)

    def poses(grasps):
        return {
            tuple(np.round(np.concatenate([pos, quat]), 5))
            for pos, quat in grasps
        }

    assert poses((pos, quat) for pos, quat, _, _ in upright) < poses(
        (g.pos, g.quat) for g in made
    )  # the three hand directions of the pipeline include the upright one
    assert len(made) > 2 * len(upright)


@test
def generator_matches_both_poses(c):
    """
    the matching in the generator: candidates made at the assembled pose and candidates made for the lying part go through the
    same tests and the same rating; of the second kind only those that pick the part lying flat are checked at the assembled pose
    """
    assembled = [
        TOP(0.4),
        TOP(0.5),
        SIDE(0.6),
        LOW,
        END(1.0),
    ]  # ids 0..4: usable 0, 1, 2
    lying = [
        [TOP(1.0), TOP(0.9, 1.0), LOW, TOP(0.2)]
    ]  # ids 5..8 (made for the stable poses): 5 and 6 usable, 7 not flat, 8 too shallow
    gen = toy_generator([assembled], lying=lying)
    grasps = gen.generate_grasps('A', max_n_grasp=100, n_proc=1)
    assert gen.calls == [(200, 0)] and gen.lying_calls == [
        (config.FLAT_LYING_PT, 5, [26.6, 63.4])
    ]  # after the first round, for the stable resting poses
    assert sorted(gen.checked) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        8,
    ]  # 7 cannot pick the brick lying flat: not checked at the assembled pose
    assert sorted(move_ids(grasps)) == [0, 1, 2, 5, 6]
    assert (
        move_ids(grasps)[0] == 5
    )  # the deepest grasp above the centre of mass is rated best, wherever it was made
    by_id = {g[0].grasp_id: g[0] for g in grasps['move']}
    assert [getattr(by_id[i], 'source', None) for i in [0, 1, 2, 5, 6]] == [
        None,
        None,
        None,
        'lying',
        'lying',
    ]
    assert sorted(hold_ids(grasps)) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        8,
    ]  # hold grasps: everything that was checked
    # enough usable grasps with them: no second round; without them the second round is needed
    few = [TOP(0.4), LOW, END(1.0)]
    gen = toy_generator([few, [TOP(0.7)]], lying=[[TOP(1.0), TOP(0.9, 1.0)]])
    assert (
        sorted(move_ids(gen.generate_grasps('A', max_n_grasp=100, n_proc=1)))
        == [0, 3, 4]
        and len(gen.calls) == 1
    )
    gen = toy_generator([few, [TOP(0.7)]])
    assert sorted(
        move_ids(gen.generate_grasps('A', max_n_grasp=100, n_proc=1))
    ) == [0, 3] and gen.calls == [(200, 0), (FLAT_RESAMPLE_PT, 3)]
    # the second round counts its ids on after the lying candidates
    gen = toy_generator([few, [TOP(0.7)]], lying=[[TOP(1.0)]])
    assert sorted(
        move_ids(gen.generate_grasps('A', max_n_grasp=100, n_proc=1))
    ) == [0, 3, 4] and gen.calls == [(200, 0), (FLAT_RESAMPLE_PT, 4)]
    # no grasp for a stable pose, from either side: candidates are made for the less stable poses too (second call), ids going on
    gen = toy_generator(
        [[LOW, TOP(0.2)], [LOW]], lying=[[LOW], [END(1.0), END(0.2)]]
    )
    grasps = gen.generate_grasps('A', max_n_grasp=100, n_proc=1)
    assert gen.lying_calls == [
        (config.FLAT_LYING_PT, 2, [26.6, 63.4]),
        (config.FLAT_LYING_PT, 4, [18.4]),
    ], gen.lying_calls
    assert move_ids(grasps) == [
        4
    ]  # the end grasp, in the unstable pose; 5 is too shallow
    # switched off: only the candidates of the assembled pose, as before
    config.FLAT_LYING_PT = 0
    try:
        gen = toy_generator([assembled], lying=lying)
        assert (
            sorted(
                move_ids(gen.generate_grasps('A', max_n_grasp=100, n_proc=1))
            )
            == [0, 1, 2]
            and gen.lying_calls == []
        )
    finally:
        config.FLAT_LYING_PT = 200


@test
def hold_grasps_that_leave_room(c):
    """
    hold grasps are chosen at random; for every part inserted while this part is held, hold grasps whose gripper is not in the
    way of that insertion are added if the random choice has too few (car: 3 of 441 hold grasps of the chassis leave room for the body)
    """

    def hold(grasp_id, *in_the_way):
        return types.SimpleNamespace(
            grasp_id=grasp_id,
            parts_in_collision_hold={'fix': [], 'move': list(in_the_way)},
        )

    def ids(grasps):
        return [g.grasp_id for g in grasps]

    holds = [
        hold(0, 'B', 'C'),
        hold(1, 'B'),
        hold(2, 'B', 'C'),
        hold(3, 'C'),
        hold(4),
        hold(5, 'B'),
        hold(6, 'C', 'D'),
        hold(7, 'B', 'C', 'D'),
    ]
    # room for B: 3, 4, 6; for C: 1, 4, 5; for D: 0 .. 5
    assert holds_with_room(holds, [], ['B'], 1) == [
        holds[4]
    ]  # the one that is in nobody's way first
    assert ids(holds_with_room(holds, [], ['B'], 2)) == [3, 4] and ids(
        holds_with_room(holds, [], ['B'], 5)
    ) == [3, 4, 6]  # all there are
    assert (
        ids(holds_with_room(holds, [holds[3]], ['B'], 2)) == [4]
        and holds_with_room(holds, [holds[3], holds[6]], ['B'], 2) == []
    )  # the chosen ones count
    assert (
        holds_with_room(
            holds, [holds[0], holds[2], holds[7]], ['B', 'C', 'D'], 0
        )
        == []
    )  # switched off
    # several parts: a grasp added for one part counts for the others; the result keeps the order of the grasps
    assert ids(holds_with_room(holds, [holds[0]], ['B', 'C', 'D'], 1)) == [
        4
    ]  # 4 leaves room for B and C, 0 was chosen and leaves room for D
    assert ids(holds_with_room(holds, [holds[0]], ['B', 'C', 'D'], 2)) == [
        1,
        3,
        4,
    ]  # B: 4, 3; C: 4 and 1; D: 0, 4
    assert ids(holds_with_room(holds, [holds[7]], ['C', 'B'], 2)) == [
        1,
        3,
        4,
    ]  # the order of the parts does not matter here
    # no hold grasp leaves room for a part: nothing to add for it
    assert (
        holds_with_room([hold(0, 'B'), hold(1, 'B')], [], ['B'], 3) == []
        and holds_with_room([], [], ['B'], 3) == []
    )
    # grasps without the collision record (not from check_grasp_feasible) are in nobody's way
    assert ids(
        holds_with_room([types.SimpleNamespace(grasp_id=9)], [], ['B'], 1)
    ) == [9]
    # in the generator: only when the random choice cut the hold grasps, and for the parts that are not there yet
    assert config.FLAT_HOLD_ROOM == 5

    def with_parts(first_held):
        gen = toy_generator(
            [[TOP(0.4 + 0.1 * i) for i in range(8)]], first_held=first_held
        )
        gen.G_preced.add_node('B', parts_after=['A'] if first_held else [])
        feasible = gen.check_grasp_feasible

        def check(grasp, part_id, verbose=False):
            result = feasible(grasp, part_id, verbose)
            result['hold'].parts_in_collision_hold = {
                'fix': [],
                'move': [] if grasp.grasp_id in [6, 7] else ['B'],
            }  # only 6 and 7 leave room for B
            return result

        gen.check_grasp_feasible = check
        return gen

    np.random.seed(5)
    cut = hold_ids(
        with_parts(False).generate_grasps('A', max_n_grasp=3, n_proc=1)
    )
    np.random.seed(5)
    random_three = hold_ids(
        toy_generator(
            [[TOP(0.4 + 0.1 * i) for i in range(8)]]
        ).generate_grasps('A', max_n_grasp=3, n_proc=1)
    )
    assert cut[:3] == random_three and sorted(set(cut)) == sorted(
        set(random_three) | {6, 7}
    ), (cut, random_three)  # the random three, plus the two with room
    assert (
        len(
            hold_ids(
                with_parts(False).generate_grasps(
                    'A', max_n_grasp=100, n_proc=1
                )
            )
        )
        == 8
    )  # no cut: all hold grasps are kept anyway
    # B is there before A (A is inserted onto B): B cannot be inserted while A is held, so no hold grasp is added for it
    gen = toy_generator([[TOP(0.4 + 0.1 * i) for i in range(8)]])
    gen.G_preced.add_node('B', parts_after=[])
    gen.G_preced.nodes['A']['parts_after'] = ['B']
    np.random.seed(5)
    assert (
        hold_ids(gen.generate_grasps('A', max_n_grasp=3, n_proc=1))
        == random_three
    )


@test
def generator_real_candidates(c):
    """antipodal_candidates on the brick: grasps across its 2, 4 or 6 cm, a second call continues the ids with new surface points"""
    gen = object.__new__(GraspArmGenerator)
    gen.gripper_type, gen.antipodal_thres, gen.n_angle, gen.offset_delta = (
        'panda',
        0.95,
        10,
        0.0,
    )
    np.random.seed(0)
    first = gen.antipodal_candidates('A', BRICK, 30, 0)
    second = gen.antipodal_candidates('A', BRICK, 30, len(first))
    assert len(first) > 30 and len(second) > 30
    assert [g.grasp_id for g in first + second] == list(
        range(len(first) + len(second))
    )
    for grasp in first + second:
        assert (
            round(8 * grasp.open_ratio, 6) in [2.0, 4.0, 6.0]
            and grasp.part_id == 'A'
        )
        closing = R.from_quat(np.asarray(grasp.quat)[[1, 2, 3, 0]]).apply(Y)
        assert np.isclose(
            np.abs(closing).max(), 1.0, atol=1e-6
        )  # the fingers close along an axis of the brick

    def centres(grasps):
        return {tuple(np.round(g.pos, 6)) for g in grasps}

    assert (
        len(centres(first) & centres(second)) == 0
    )  # new surface points, no candidate twice
    np.random.seed(0)
    again = gen.antipodal_candidates('A', BRICK, 30, 0)
    assert len(again) == len(first) and all(
        np.allclose(a.pos, b.pos) and np.allclose(a.quat, b.quat)
        for a, b in zip(first, again)
    )  # same seed, same candidates
    # the pad contact of the generator is the grid measurement
    gen.pads, gen.part_faces, gen.gripper_meshes_visual = PADS, {}, VISUAL
    gen.part_meshes, gen.part_final_transforms = {'A': BRICK}, {'A': np.eye(4)}
    measured = gen.compute_contact_points_chunk(
        [top_grasp(1.0), top_grasp(-0.5)], 'A'
    )
    assert (
        abs(measured[0].contact_depths[0] - 0.99) <= PAD_CELL
        and len(measured[1].contact_points) == 0
    )


# ------------------------------------------------------------------------------------------ the part that is held first


@test
def first_hold_nodes_of_the_grasp_tree(c):
    """
    sequence optimisation: the hold grasp of the first step takes its part from the board, so it must pick the part lying flat.
    The nodes whose grasp cannot are removed; there is no other way for the first part to get to the assembly spot.
    """
    holds = {
        'A': {
            'hold': {
                5: top_grasp(1.0, grasp_id=5),
                6: low_grasp(6),
                7: side_grasp(1.0, grasp_id=7),
                8: end_grasp(1.0, grasp_id=8),
            }
        },
        'B': {'hold': {1: low_grasp(1), 2: top_grasp(0.8, grasp_id=2)}},
    }
    meshes = {'A': BRICK, 'B': BRICK}
    settings = {
        'gripper': 'panda',
        'settings': {'flat_pickup': {'fallback_parts': []}},
    }
    tree = nx.DiGraph()
    first = [
        (('A',), 'A', 5),
        (('A',), 'A', 6),
        (('A',), 'A', 7),
        (('A',), 'A', 8),
        (('B',), 'B', 1),
        (('B',), 'B', 2),
    ]
    later = [
        (('A', 'B'), 'A', 6),
        (('A', 'B'), 'B', 1),
        (('B',), 'A', 6),
    ]  # two parts remain, or the part held is not the one that remains
    tree.add_nodes_from(first + later)
    first_holds, infeasible = first_hold_nodes(
        tree.nodes, holds, meshes, settings, COLLISION
    )
    assert sorted(first_holds) == sorted(first)
    assert sorted(infeasible) == [
        (('A',), 'A', 6),
        (('A',), 'A', 8),
        (('B',), 'B', 1),
    ]  # low grasps; the end grasp needs the unstable pose
    # a part that may lie in its less stable poses (the grasp generation decided so): the end grasp picks it there
    settings['settings']['flat_pickup']['fallback_parts'] = ['A']
    assert sorted(
        first_hold_nodes(tree.nodes, holds, meshes, settings, COLLISION)[1]
    ) == [(('A',), 'A', 6), (('B',), 'B', 1)]
    # the clearance of the caller is used
    assert (
        sorted(
            first_hold_nodes(
                tree.nodes, holds, meshes, settings, COLLISION, clearance=0.1
            )[1]
        )
        == []
    )
    # none of the first hold grasps can: all are infeasible (run_flat_seq_opt reports that there is no sequence)
    only_low = [(('A',), 'A', 6), (('B',), 'B', 1)]
    assert first_hold_nodes(
        only_low + later, holds, meshes, settings, COLLISION
    ) == (only_low, only_low)
    assert first_hold_nodes(later, holds, meshes, settings, COLLISION) == (
        [],
        [],
    )


@test
def place_grasp_of_the_first_part(c):
    """
    fallback: no first hold grasp picks the first part lying flat, so the move arm brings it to the assembly spot. The grasps
    for that: move grasps of the part that pick it lying flat and whose arm pose at the assembly spot is fine, best rated first.
    """
    rots = get_flat_rots(BRICK)

    def move(grasp, rating=None, arm_q='q'):
        grasp.arm_q = arm_q
        if rating is not None:
            grasp.flat_rating = rating
        return [grasp]

    moves = [
        move(top_grasp(0.5, grasp_id=1), 0.6),
        move(top_grasp(1.0, grasp_id=2), 0.9),
        move(low_grasp(3), 0.95),
        move(side_grasp(1.0, grasp_id=4), 0.8),
        move(end_grasp(1.0, grasp_id=5), 0.99),
        move(top_grasp(1.0, x=1.0, grasp_id=6), 0.85, arm_q=None),
    ]
    places = place_grasps_of_first_part(moves, BRICK, rots, 'panda', COLLISION)
    assert (
        [g.grasp_id for g, _ in places] == [2, 4, 1]
    )  # best rated first; 3 cannot pick the brick lying, 5 only in the unstable pose, 6 has no arm pose
    by_id = {
        m[0].grasp_id: m for m in moves
    }  # the form SequenceOptimizer turns the grasps into
    assert [
        g.grasp_id
        for g, _ in place_grasps_of_first_part(
            by_id, BRICK, rots, 'panda', COLLISION
        )
    ] == [2, 4, 1]
    assert (
        all(len(cands) == 4 for _, cands in places)
        and round(places[1][1][0]['tip_angle'], 1) == 63.4
    )  # with the lying poses each one picks (4: on a 6 x 4 face)
    # the arm pose with the part standing at the assembly spot is tested by the caller
    assert [
        g.grasp_id
        for g, _ in place_grasps_of_first_part(
            moves,
            BRICK,
            rots,
            'panda',
            COLLISION,
            final_pose_ok=lambda g: g.grasp_id != 2,
        )
    ] == [4, 1]
    assert (
        place_grasps_of_first_part(
            moves,
            BRICK,
            rots,
            'panda',
            COLLISION,
            final_pose_ok=lambda g: False,
        )
        == []
    )
    # the part may lie in its less stable poses: the end grasp is the best rated then
    assert [
        g.grasp_id
        for g, _ in place_grasps_of_first_part(
            moves,
            BRICK,
            get_flat_rots(BRICK, with_fallback=True),
            'panda',
            COLLISION,
        )
    ] == [5, 2, 4, 1]
    # grasps without a rating (not from the flat grasp generation): the most upright hand first, then the id
    plain = [
        move(leaning_grasp(30.0, grasp_id=7)),
        move(top_grasp(1.0, grasp_id=9)),
        move(top_grasp(0.8, grasp_id=8)),
    ]
    assert [
        g.grasp_id
        for g, _ in place_grasps_of_first_part(
            plain, BRICK, rots, 'panda', COLLISION
        )
    ] == [8, 9, 7]
    assert (
        place_grasps_of_first_part([], BRICK, rots, 'panda', COLLISION) == []
    )


@test
def first_part_rules_with_an_empty_assembly(c):
    """
    Bringing the first part to the assembly spot: no part is assembled yet, so the rules about the assembly have nothing to
    check, wherever the later parts will stand. The part that was just put down is still protected when the gripper leaves.
    """
    target = REF_XY + [12, 0]
    later = box(
        [4, 4, 2], [target[0], target[1], 1.0]
    )  # part C will be assembled here, later
    q0 = carried_scene(c, zone_center=target, extra_final={'C': later})
    up = carry_to(c, q0, REF_XY, 12.0)
    through = carry_to(
        c, q0, target, 0.5
    )  # the carried part B inside the place of C
    assert (
        c.check_path(
            [q0, up, through, up],
            0.5,
            {'B': 'pickup', 'C': 'pickup'},
            carried='B',
        )
        == []
    )  # C still lies on the table (far away)
    assembled = c.check_path(
        [q0, up, through, up], 0.5, {'B': 'pickup', 'C': 'final'}, carried='B'
    )
    assert has(assembled, 'carried part B hits the assembly'), (
        assembled
    )  # the same path once C is assembled
    # after putting the first part down: it is the only assembled part, and the gripper must not touch it while leaving
    set_scene(
        c, final={'B': box([12, 12, 2], [*target, 1.0])}, zone_center=target
    )
    path = [ik_down(c, target, z) for z in [1.7, 1.9, 2.3, 5.0]]
    released = c.check_path(path, 0.5, {'B': 'final'}, released_part='B')
    assert poses_of(released, 'gripper hits part B (just released)') == (
        0,
        1,
        2,
    ), released
    assert (
        c.check_path(
            [ik_down(c, target, z) for z in [2.3, 3.5, 5.0]],
            0.5,
            {'B': 'final'},
            released_part='B',
        )
        == []
    )  # leaving without touching


# ------------------------------------------------------------------------------------ pickup layout: the quick finger test


@test
def distance_between_lying_parts(c):
    """
    PART_GAP: two lying parts are at least as far apart as the motion planner wants a carried part from every other part
    (checked clearance plus the planner's reserve). A part lifted straight up from its spot is then clear of its neighbours from
    the first pose on (until 2026-11-11 the gap was 0.5 cm: cooling_manifold part 4 came within 0.81 cm of a neighbour).
    The value itself, 2 cm, was set by the user on 2026-11-12: room for the gripper to turn between the parts.
    """
    assert PART_GAP == 2.0
    planner = move_planner()
    assert PART_GAP >= planner.part_buffers()[0] > ROBOT_PART_CLEARANCE
    # two bricks that stand that far apart (plus 0.1 mm: at exactly the limit rounding decides), one of them in the gripper:
    # for the planner the carried one is clear of the other on the whole way up
    grasp, mat = top_grasp(1.0), brick_at(0.0, 5.0)
    moves = PickupMoves(planner, {'A': BRICK})
    mesh = moves.mesh_at('A', mat)
    np.random.seed(0)
    q = pickup_ik_solutions({}, planner, grasp, 'A', mat, np.eye(4), False)[0][
        1
    ]
    gripper_pickup = get_pickup_gripper_pose(grasp, mat, np.eye(4))
    fk = c.chain.forward_kinematics(q)

    def lifted(lift):
        return c.chain.active_from_full(
            planner.inverse_kinematics(
                fk[:3, 3] + [0, 0, lift], fk[:3, :3], q_init=q
            )
        )

    def blocked_by_the_neighbour(gap, lift):
        neighbour = moves.mesh_at(
            'A', brick_at(0.0, 5.0 + 2.0 + gap)
        )  # the bricks are 2 cm thick along y, the fingers close along y
        _, _, _, collision_fn = planner.get_fns(
            mesh, gripper_pickup, [neighbour], grasp.open_ratio
        )
        return collision_fn(
            lifted(lift), reserve=False
        ) and collision_fn.hit in ['move part and part', 'gripper and part']

    for lift in [
        1.0,
        2.0,
        4.0,
        6.0,
    ]:  # (on the table itself the part is inside the planner's table buffer, whatever lies beside it)
        assert not blocked_by_the_neighbour(
            PART_GAP + 0.01 + 2.65 + 0.4, lift
        ), lift  # also the finger (2.65 cm thick, opened by 0.4 cm) is clear
    assert not blocked_by_the_neighbour(
        PART_GAP + 0.01, 4.0 + 6.0
    )  # high above it
    # as close as the layout allows: the fingers are in the way, but not the part (the layout checks the fingers at every spot)
    neighbour = moves.mesh_at(
        'A', brick_at(PART_GAP + 0.01 + 6.0, 5.0)
    )  # beside the end of the brick (6 cm long along x), where no finger is
    _, _, _, collision_fn = planner.get_fns(
        mesh, gripper_pickup, [neighbour], grasp.open_ratio
    )
    assert all(
        not collision_fn(lifted(lift), reserve=False)
        for lift in [1.0, 2.0, 4.0, 6.0]
    )
    # 0.5 cm apart at the end, as before 2026-11-11: the part itself is too close to its neighbour for the planner
    neighbour = moves.mesh_at('A', brick_at(0.5 + 6.0, 5.0))
    _, _, _, collision_fn = planner.get_fns(
        mesh, gripper_pickup, [neighbour], grasp.open_ratio
    )
    assert (
        collision_fn(lifted(2.0), reserve=False)
        and collision_fn.hit == 'move part and part'
    ), collision_fn.hit
    assert not collision_fn(
        lifted(2.0), buffered=False
    )  # (no contact: the checks would accept the lift, the planner asks for more)


@test
def finger_footprints_for_the_wide_search(c):
    """the footprints of the fingers at a pickup (opened and closed) and the quick test that they stay clear of lying neighbours"""
    from shapely.geometry import box as rectangle

    grasp = top_grasp(1.0)  # fingers close along y over 2 cm, hand from above
    fingers = finger_footprints(grasp, np.eye(4), 'panda', COLLISION)
    assert len(fingers) == 2
    spans = sorted(
        [f.bounds for f in fingers], key=lambda b: b[1]
    )  # (min x, min y, max x, max y), the finger at -y first
    for (x0, y0, x1, y1), side in zip(spans, [-1, 1]):
        assert (
            abs(x0 + 1.05) < 0.03 and abs(x1 - 1.05) < 0.03
        )  # 2.1 cm wide across
        inner, outer = (y1, y0) if side < 0 else (y0, y1)
        assert abs(abs(inner) - 0.99) < 0.02  # closed on the 2 cm part ...
        assert (
            abs(abs(outer) - (0.99 + 2.65 + 0.4)) < 0.03
        )  # ... to the outer face of the finger with the gripper opened by 0.1 (4 mm per finger)
    # the footprints follow the pose of the lying part: turned 90 deg about the vertical and moved
    pose = trimesh.transformations.rotation_matrix(np.pi / 2, Z)
    pose[:3, 3] = [5.0, -3.0, 0.0]
    moved = sorted(
        [f.bounds for f in finger_footprints(grasp, pose, 'panda', COLLISION)],
        key=lambda b: b[0],
    )
    assert (
        abs(moved[1][0] - (5.0 + 0.99)) < 0.02
        and abs(moved[0][2] - (5.0 - 0.99)) < 0.02
    )  # now the fingers close along x, around x = 5
    assert all(
        abs(b[1] - (-3.0 - 1.05)) < 0.03 and abs(b[3] - (-3.0 + 1.05)) < 0.03
        for b in moved
    )
    # the quick test: 1 cm between the finger footprints and the footprints of the other parts
    near, far = (
        rectangle(1.05 + 0.95, 1.5, 4.0, 3.0),
        rectangle(1.05 + 1.08, 1.5, 4.0, 3.0),
    )  # beside the finger at +y
    assert (
        fingers_clear(fingers, [], 1.0)
        and fingers_clear(fingers, [far], 1.0)
        and not fingers_clear(fingers, [near], 1.0)
    )
    assert not fingers_clear(fingers, [far, near], 1.0) and fingers_clear(
        fingers, [near], 0.9
    )
    # footprints seen from above: clear on the board means clear in space (the distance in space is never smaller)
    manager = trimesh.collision.CollisionManager()
    for name in get_gripper_finger_names('panda'):
        for ratio in [grasp.open_ratio, grasp.open_ratio + 0.1]:
            T = get_gripper_meshes_transforms(
                'panda', COLLISION, grasp.pos, grasp.quat, np.eye(4), ratio
            )[name]
            manager.add_object(f'{name}{ratio}', COLLISION[name], T)
    for neighbour_box, flat_distance in [(near, 0.95), (far, 1.08)]:
        x0, y0, x1, y1 = neighbour_box.bounds
        neighbour = box(
            [x1 - x0, y1 - y0, 12.0], [(x0 + x1) / 2, (y0 + y1) / 2, 6.0]
        )
        in_space = manager.min_distance_single(neighbour)
        assert (
            in_space >= min(f.distance(neighbour_box) for f in fingers) - 1e-6
            and abs(in_space - flat_distance) < 0.2
        ), (in_space, flat_distance)


# ------------------------------------------------------- pickup layout: the straight moves at a spot, spots that are banned


def brick_at(x, y, yaw=0.0):
    """pose (4x4, applied to the brick where its grasps are defined) that stands it on the table at (x, y), turned by yaw deg"""
    mat = trimesh.transformations.rotation_matrix(np.radians(yaw), Z)
    mat[:3, 3] = [x, y, -5.0]
    return mat


def move_planner():
    ArmMotionPlanner.VISUALIZE = False
    box_move, _ = get_dual_arm_box('panda')
    return make_move_planner(
        get_arm_chain('panda', 'move'),
        'panda',
        False,
        box_move,
        {'Z': box([2, 2, 2], [*REF_XY, 1.0])},
    )  # the assembly: far from the spots of these tests


UP = np.array([0, 0, 1.0])


@test
def pickup_spot_meshes(c):
    """PickupMoves.mesh_at: the part where it lies, with the inflated versions the planner needs (computed once per part, moved along)"""
    planner = move_planner()
    moves = PickupMoves(planner, {'A': BRICK})
    mesh = moves.mesh_at('A', brick_at(0.0, 5.0))
    assert np.allclose(mesh.bounds, [[-3, 4, 0], [3, 6, 4]])
    buffers = planner.part_buffers()
    assert len(set(buffers)) == 2 and sorted(moves.inflated) == sorted(
        ('A', buffer) for buffer in buffers
    )
    for buffer in (
        buffers
    ):  # what the planner would compute itself for the mesh where it lies
        handed = planner._buffer_cache[(id(mesh), buffer)][1]
        assert np.allclose(
            handed.vertices, offset_mesh(mesh, buffer).vertices, atol=1e-9
        ) and np.array_equal(handed.faces, offset_mesh(mesh, buffer).faces)
        assert np.allclose(
            handed.bounds,
            [
                [-3 - buffer, 4 - buffer, -buffer],
                [3 + buffer, 6 + buffer, 4 + buffer],
            ],
        )
    # the same part at another spot, turned: nothing is inflated again, the inflated meshes turn with it
    first = {key: value for key, value in moves.inflated.items()}
    turned = moves.mesh_at('A', brick_at(-3.0, 8.0, 90.0))
    assert (
        all(moves.inflated[key] is first[key] for key in first)
        and len(moves.inflated) == 2
    )
    for buffer in buffers:
        assert np.allclose(
            planner._buffer_cache[(id(turned), buffer)][1].bounds,
            [
                [-4 - buffer, 5 - buffer, -buffer],
                [-2 + buffer, 11 + buffer, 4 + buffer],
            ],
        )
    # a spot that is not taken is forgotten
    moves.drop(turned)
    assert all(key[0] != id(turned) for key in planner._buffer_cache) and all(
        (id(mesh), buffer) in planner._buffer_cache for buffer in buffers
    )


@test
def pickup_spot_straight_moves(c):
    """
    PickupMoves.at_spot: can the arm move straight into the grasp and straight up with the part? The answer and its reason are
    those of the motion planner, which is asked afterwards for the same spot (plan_path_switch, plan_path_with_grasp).
    """
    planner = move_planner()
    moves = PickupMoves(planner, {'A': BRICK})
    rest = c.chain.active_to_full(c.chain.rest_q)
    np.random.seed(0)
    mat, grasp = (
        brick_at(0.0, 5.0),
        top_grasp(1.0),
    )  # fingers closed at 0.25, opened to 0.35 for the approach
    mesh = moves.mesh_at('A', mat)
    gripper_pickup = get_pickup_gripper_pose(grasp, mat, np.eye(4))
    q = pickup_ik_solutions({}, planner, grasp, 'A', mat, np.eye(4), False)[0][
        1
    ]  # an arm pose that passes the checks of the pose itself

    def transport(others):
        return planner.plan_path_with_grasp(
            q,
            rest,
            move_pickup_mesh=mesh,
            gripper_pickup_transform=gripper_pickup,
            still_meshes=others,
            open_ratio=grasp.open_ratio,
            retract_start=UP,
            retract_goal=get_back_retract_dir(planner, rest),
            retract_delta=5.0,
            max_speed=4.0,
        )

    def switch(q_goal, meshes):
        return planner.plan_path_switch(
            rest,
            q_goal,
            part_meshes=meshes,
            open_ratio=OPEN_RATIO_REST,
            open_ratio_next=0.35,
            retract_start=get_back_retract_dir(planner, rest),
            retract_goal=get_back_retract_dir(planner, q_goal),
            retract_delta=5.0,
            max_speed=4.0,
        )

    # the collision tests of the planner are set up for the spot when the first arm pose is tested, once
    calls, get_fns = [], planner.get_fns
    planner.get_fns = (
        lambda move_mesh, pickup_transform, still, open_ratio, **kw: (
            calls.append(
                (move_mesh is not None, len(still), round(open_ratio, 3))
            ),
            get_fns(move_mesh, pickup_transform, still, open_ratio, **kw),
        )[1]
    )
    try:
        test = moves.at_spot(grasp, mesh, gripper_pickup, [], OPEN_RATIO_REST)
        assert calls == []
        assert (
            test(q) is None and test(q) is None
        )  # a free spot in the middle of the table: both moves exist
        assert (
            calls == [(False, 1, 0.35), (True, 0, 0.25)]
        )  # approach: the opened gripper, the brick is a still part; lift: the brick in the closed gripper
        # a gripper that arrives less opened than the grasp needs: the approach must also end where that gripper is clear
        calls.clear()
        assert moves.at_spot(grasp, mesh, gripper_pickup, [], 0.3)(q) is None
        assert calls == [(False, 1, 0.35), (False, 1, 0.3), (True, 0, 0.25)]
    finally:
        planner.get_fns = get_fns
    path1, path2 = switch(q, [mesh])
    assert (
        path1 is not None and path2 is not None and transport([]) is not None
    )  # the planner agrees
    # a small plate 1 cm above one end of the brick, beside the fingers: the gripper gets in, the brick does not get out
    plate = box([1.3, 2.0, 0.5], [2.85, 5.0, 5.25])
    verdict = moves.at_spot(
        grasp, mesh, gripper_pickup, [plate], OPEN_RATIO_REST
    )(q)
    assert verdict in [
        'no straight move up with the part (move part and part after 1.00 cm)',
        'no straight move up with the part (move part and part after 1.25 cm)',
    ], verdict
    assert (
        transport([plate]) is None
        and planner.last_failure == 'start'
        and verdict
        == f'no straight move up with the part ({planner.last_block[0]} after {planner.last_block[1]:.2f} cm)'
    )
    path1, path2 = switch(q, [mesh, plate])
    assert path1 is not None and path2 is not None
    # the plate 2.5 cm above that end: the brick gets clear of it and of the table (1 cm up), but cannot be lifted to the
    # transport height (3.5 cm) below it
    higher = box([1.3, 2.0, 0.5], [2.95, 5.0, 6.75])
    assert (
        moves.at_spot(grasp, mesh, gripper_pickup, [higher], OPEN_RATIO_REST)(
            q
        )
        == 'no straight lift of the part to the transport height'
    )
    assert (
        transport([higher]) is None
        and planner.last_failure == 'start'
        and planner.last_block
        == ('no straight lift of the carried part', None)
    )
    # beside the virtual wall, the hand leaning 20 deg with its back to the wall (duct part 2 on 2026-11-11): the arm pose at the
    # pickup passes every check, but backing out along the hand axis reaches the wall before the gripper is clear of the part
    grasp = leaning_grasp(20.0)
    seen = {}
    for x in [-2.5, -3.0, -3.5, -3.75, -4.0]:
        np.random.seed(
            int(-100 * x)
        )  # the arm poses found for a spot must not depend on what was planned before (random IK starts)
        mat = brick_at(x, 5.0, 180.0)
        solutions = pickup_ik_solutions(
            {}, planner, grasp, 'A', mat, np.eye(4), False
        )
        if len(solutions) == 0:
            continue  # none of the IK starts gave a pose that passes the checks of the pose itself
        q_spot = solutions[0][1]
        assert (
            get_virtual_wall_margin(c.chain, q_spot) >= VIRTUAL_WALL_SAFETY
            and solutions[0][0] >= 1
        )  # wall, singularity and joint limits are fine at the pose
        assert (
            np.linalg.norm(
                get_back_retract_dir(planner, q_spot)
                - [-np.sin(np.radians(20)), 0, np.cos(np.radians(20))]
            )
            < 1e-3
        )  # the hand axis leans towards -x
        spot_mesh = moves.mesh_at('A', mat)
        verdict = moves.at_spot(
            grasp,
            spot_mesh,
            get_pickup_gripper_pose(grasp, mat, np.eye(4)),
            [],
            OPEN_RATIO_REST,
        )(q_spot)
        path1, path2 = switch(q_spot, [spot_mesh])
        if verdict is None:
            assert path1 is not None and path2 is not None, (
                x,
                planner.last_failure,
                planner.last_block,
            )
        else:
            assert path1 is None and planner.last_failure == 'goal', (
                x,
                planner.last_failure,
            )
            assert (
                verdict
                == f'no straight move into the grasp ({planner.last_block[0]} after {planner.last_block[1]:.2f} cm)'
            ), (x, verdict, planner.last_block)
        seen[x] = verdict
        moves.drop(spot_mesh)
    free, blocked = (
        [x for x in seen if seen[x] is None],
        [x for x in seen if seen[x] is not None],
    )
    assert len(free) > 0 and len(blocked) > 0, seen  # both cases occur
    assert all(
        seen[x].startswith(
            'no straight move into the grasp (virtual wall after '
        )
        for x in blocked
    ), seen
    assert max(blocked) < min(free), (
        seen
    )  # free further from the wall, blocked nearer to it


def world_grasp(planner, grasp, mat, q, offset=(0.0, 0.0, 0.0)):
    """
    the toy grasp for the brick standing at mat as a grasp of the real arm: gripper pose in world coordinates and the arm pose
    for it; offset (cm): the same grasp with the gripper moved by it (a pose of the insertion path), None if the arm cannot reach it
    """
    pose = get_pickup_gripper_pose(grasp, mat, np.eye(4))
    pose[:3, 3] += offset
    if np.any(np.asarray(offset) != 0):
        fk = planner.arm_chain.forward_kinematics(q)
        q = planner.inverse_kinematics(
            fk[:3, 3] + np.asarray(offset), fk[:3, :3], q_init=q
        )
        if q is None:
            return None
    made = Grasp('A', grasp.grasp_id, *mat_to_pos_quat(pose), grasp.open_ratio)
    made.arm_q = np.array(q, dtype=float)
    return made


@test
def straight_moves_at_the_assembly(c):
    """
    assembly_moves_blocked: can the arm bring the part straight to the insertion start, and leave straight after the release?
    The answer and its reason are those of the motion planner, asked for the same poses (plan_path_with_grasp, plan_path).
    """
    ArmMotionPlanner.VISUALIZE = False
    rest = c.chain.active_to_full(c.chain.rest_q)
    mat, grasp = brick_at(0.0, 5.0), top_grasp(1.0)
    brick = BRICK.copy().apply_transform(
        mat
    )  # the brick at its assembled pose: standing on the table at (0, 5), 4 cm high
    planner = make_move_planner(
        get_arm_chain('panda', 'move'),
        'panda',
        False,
        get_dual_arm_box('panda')[0],
        {'A': brick},
    )
    np.random.seed(0)
    q = pickup_ik_solutions({}, planner, grasp, 'A', mat, np.eye(4), False)[0][
        1
    ]
    final, start = (
        world_grasp(planner, grasp, mat, q),
        world_grasp(planner, grasp, mat, q, (0, 0, 5.0)),
    )  # inserted from 5 cm above
    assert start is not None and np.allclose(
        get_gripper_pos_quat_from_arm_q(c.chain, start.arm_q, 'panda')[0],
        start.pos,
        atol=1e-3,
    )
    gripper_final = get_transform_matrix_quat(final.pos, final.quat)
    above = world_grasp(
        planner, grasp, mat, q, (0, 0, 17.0)
    )  # where the transport of the planner starts in this test: high above the assembly

    def transport(others):
        return planner.plan_path_with_grasp(
            above.arm_q,
            start.arm_q,
            move_pickup_mesh=brick,
            gripper_pickup_transform=gripper_final,
            still_meshes=others,
            open_ratio=grasp.open_ratio,
            retract_start=None,
            retract_goal=UP,
            retract_delta=5.0,
            max_speed=4.0,
        )

    def leave(q_from, meshes, open_ratio):
        return planner.plan_path(
            q_from,
            rest,
            part_meshes=meshes,
            open_ratio=open_ratio,
            retract_start=get_back_retract_dir(planner, q_from),
            retract_goal=get_back_retract_dir(planner, rest),
            retract_delta=5.0,
            max_speed=4.0,
        )

    # nothing in the way: both moves exist, and the planner plans both
    assert assembly_moves_blocked(planner, [final, start], brick, []) is None
    assert (
        transport([]) is not None
        and leave(final.arm_q, [brick], 0.35) is not None
    )
    # a small plate 1 cm above one end of the brick where the insertion starts: the brick cannot come straight down to that
    # pose from a pose clear of the plate (tested backwards: straight up from the insertion start)
    plate = box([1.3, 2.0, 0.5], [3.25, 5.0, 10.25])
    verdict = assembly_moves_blocked(planner, [final, start], brick, [plate])
    assert verdict in [
        'no straight move to the insertion start (move part and part after 1.00 cm)',
        'no straight move to the insertion start (move part and part after 1.25 cm)',
    ], verdict
    assert (
        transport([plate]) is None
        and planner.last_failure == 'goal'
        and verdict
        == f'no straight move to the insertion start ({planner.last_block[0]} after {planner.last_block[1]:.2f} cm)'
    )
    # a part that is not inserted (one pose only, or an insertion path that does not move): only the release is tested
    once = assembly_moves_blocked(planner, [final], brick, [plate])
    assert once == assembly_moves_blocked(
        planner, [final, final.copy()], brick, [plate]
    ) and (
        once is None
        or once.startswith('no straight move away after the release (')
    )
    assert assembly_moves_blocked(planner, [final], brick, []) is None
    # the worker of the grasp generation builds that scene itself: the parts at their assembled poses, and of them those
    # that are assembled before this part (G_preced 'parts_after')
    gen = object.__new__(GraspArmGenerator)
    gen.arm_type, gen.gripper_type, gen.has_ft_sensor = (
        'panda',
        'panda',
        {'move': False, 'hold': False},
    )
    gen.part_meshes = {'A': BRICK, 'B': box([1.3, 2.0, 0.5], [0, 0, 0])}
    gen.part_final_transforms = {
        'A': mat,
        'B': trimesh.transformations.translation_matrix([3.25, 5.0, 10.25]),
    }
    gen.G_preced = nx.DiGraph()
    gen.G_preced.add_node('A', parts_after=['B'])
    gen.G_preced.add_node('B', parts_after=[])
    # the planner and the scene are built once (the generator does that before it starts its worker processes, which then
    # share them), with the inflated meshes of all parts ready
    scene_planner, scene_final = gen.assembly_scene()
    assert sorted(scene_final) == ['A', 'B'] and scene_planner.strict
    assert np.allclose(scene_final['A'].bounds, brick.bounds) and np.allclose(
        scene_final['B'].bounds, plate.bounds
    )
    assert all(
        (id(mesh), buffer) in scene_planner._buffer_cache
        for mesh in scene_final.values()
        for buffer in scene_planner.part_buffers()
    )
    assert gen.assembly_moves_chunk([[final, start], [final]], 'A') == [
        verdict,
        once,
    ]
    assert gen.assembly_scene()[0] is scene_planner
    # the collision structures the planner builds for a grasp are gone before the next grasp is tested. They are held in
    # reference cycles, so only the garbage collector releases them; the worker runs it after every grasp (here the automatic
    # runs are switched off: without that step every grasp would leave its structures behind, with duct 60 MB per grasp)
    import gc

    def managers():
        return sum(
            isinstance(obj, trimesh.collision.CollisionManager)
            for obj in gc.get_objects()
        )

    gc.collect()
    n_before = managers()
    gc.disable()
    try:
        gen.assembly_moves_chunk([[final, start]] * 5, 'A')
        n_after = managers()
    finally:
        gc.enable()
    assert n_after <= n_before, (n_before, n_after)
    gen.G_preced.nodes['A'][
        'parts_after'
    ] = []  # the plate is assembled later: it is not there when the brick comes
    assert gen.assembly_moves_chunk([[final, start], [final]], 'A') == [
        None,
        None,
    ]
    # the part that is held first is not inserted, whatever path the precedence plan gives it (plumbers_block, duct: a slide
    # along the table, which the arm could not drive: the part would rub on the table). Only its release is tested.
    slide = world_grasp(
        planner, grasp, mat, q, (3.0, 0, 0)
    )  # the "insertion" would start 3 cm beside the assembled pose, on the table
    assert slide is not None
    assert (
        assembly_moves_blocked(planner, [final, slide], brick, [])
        == 'no straight move to the insertion start (move part and ground after 0.25 cm)'
    )
    assert (
        assembly_moves_blocked(
            planner, [final, slide], brick, [], inserted=False
        )
        is None
    )
    assert (
        assembly_moves_blocked(
            planner, [final, start], brick, [plate], inserted=False
        )
        == once
    )  # the plate would block the approach: not looked at
    assert gen.assembly_moves_chunk([[final, slide]], 'A') == [
        None
    ]  # nothing is assembled before the brick: it can be the first part
    gen.G_preced.nodes['A']['parts_after'] = [
        'B'
    ]  # with a part before it the brick has to be inserted along that path
    gen.part_final_transforms['B'] = (
        trimesh.transformations.translation_matrix([-20.0, 5.0, 0.25])
    )  # far from the brick
    gen._assembly_scene = None  # another assembly: the scene is built again
    assert gen.assembly_moves_chunk([[final, slide]], 'A') == [
        'no straight move to the insertion start (move part and ground after 0.25 cm)'
    ]

    # the release beside the virtual wall, the hand leaning 20 deg with its back to the wall (as in pickup_spot_straight_moves):
    # the brick comes straight down from above, which works, but backing out along the hand axis reaches the wall before the
    # opened gripper is clear of the brick. Same move as the approach to a pickup there, so PickupMoves gives the same reason.
    planner = move_planner()
    moves = PickupMoves(planner, {'A': BRICK})
    grasp = leaning_grasp(20.0)
    seen = {}
    for x in [-2.5, -3.0, -3.5, -3.75, -4.0]:
        np.random.seed(int(-100 * x))
        mat = brick_at(x, 5.0, 180.0)
        solutions = pickup_ik_solutions(
            {}, planner, grasp, 'A', mat, np.eye(4), False
        )
        if len(solutions) == 0:
            continue
        q_spot = solutions[0][1]
        final, start = (
            world_grasp(planner, grasp, mat, q_spot),
            world_grasp(planner, grasp, mat, q_spot, (0, 0, 5.0)),
        )
        if start is None:
            continue
        spot_mesh = moves.mesh_at('A', mat)
        verdict = assembly_moves_blocked(
            planner, [final, start], spot_mesh, []
        )
        pickup = moves.at_spot(
            grasp,
            spot_mesh,
            get_pickup_gripper_pose(grasp, mat, np.eye(4)),
            [],
            OPEN_RATIO_REST,
        )(q_spot)
        path = leave(
            q_spot,
            [spot_mesh],
            min(grasp.open_ratio + config.RETRACT_OPEN_RATIO, 1.0),
        )
        if verdict is None:
            assert path is not None and (
                pickup is None
                or not pickup.startswith('no straight move into the grasp')
            ), (x, pickup, planner.last_failure, planner.last_block)
            # the approach follows the insertion direction (straight up here), not the leaning hand: a plate 1 cm above the
            # +x end of the brick where the insertion starts, over its last 2 mm. Straight up the brick hits it; along the
            # hand axis (up and towards -x) it would slip past
            over_the_end = assembly_moves_blocked(
                planner,
                [final, start],
                spot_mesh,
                [box([1.3, 2.0, 0.5], [x + 3.45, 5.0, 10.25])],
            )
            assert over_the_end in [
                'no straight move to the insertion start (move part and part after 1.00 cm)',
                'no straight move to the insertion start (move part and part after 1.25 cm)',
            ], (x, over_the_end)
        else:
            assert path is None and planner.last_failure == 'start', (
                x,
                planner.last_failure,
            )
            assert (
                verdict
                == f'no straight move away after the release ({planner.last_block[0]} after {planner.last_block[1]:.2f} cm)'
            ), (x, verdict, planner.last_block)
            assert pickup == verdict.replace(
                'no straight move away after the release',
                'no straight move into the grasp',
            ), (x, verdict, pickup)
        seen[x] = verdict
        moves.drop(spot_mesh)
    free, blocked = (
        [x for x in seen if seen[x] is None],
        [x for x in seen if seen[x] is not None],
    )
    assert len(free) > 0 and len(blocked) > 0, seen
    assert all(
        seen[x].startswith(
            'no straight move away after the release (virtual wall after '
        )
        for x in blocked
    ), seen
    assert max(blocked) < min(free), seen


@test
def generator_drops_grasps_without_the_moves_at_the_assembly(c):
    """
    generate_grasps_flat for the real arm: the move grasps that pass everything else are tested for the straight moves at the
    assembly in the worker processes (assembly_moves_chunk, scripted here); the ones that fail are dropped and counted, whatever
    the order in which the processes finish. Hold grasps are not concerned.
    """
    q = ik_down(c, REF_XY, 10.0)
    grasp, part = arm_grasp(c, q, 1.0)

    def at(grasp_id):
        made = grasp.copy()
        made.grasp_id = grasp_id
        return made

    def generator(reasons, n_grasps=7):
        gen = toy_generator([[at] * n_grasps, []], part=part)
        gen.arm_type, gen.arm_chains, gen.has_ft_sensor = (
            'panda',
            {'move': c.chain, 'hold': None},
            {'move': False, 'hold': False},
        )  # the real arm: walls, conditioning, moves at the assembly
        gen.tested = []

        def assembly_moves_chunk(moves, part_id):
            time.sleep(
                0.3 if moves[0][0].grasp_id == 0 else 0.0
            )  # the chunk with grasp 0 finishes last
            return [reasons.get(move[0].grasp_id) for move in moves]

        gen.assembly_moves_chunk = assembly_moves_chunk
        return gen

    reasons = {
        1: 'no straight move away after the release (joint limit after 0.75 cm)',
        4: 'no straight move to the insertion start (virtual wall after 0.50 cm)',
    }
    for n_proc in [1, 3]:
        gen, out = generator(reasons), io.StringIO()
        with contextlib.redirect_stdout(out):
            grasps = gen.generate_grasps('A', max_n_grasp=100, n_proc=n_proc)
        assert sorted(move_ids(grasps)) == [0, 2, 3, 5, 6], (
            n_proc,
            move_ids(grasps),
        )
        assert sorted(hold_ids(grasps)) == [0, 1, 2, 3, 4, 5, 6]
        assert (
            '7 feasible move grasps, 5 usable lying flat' in out.getvalue()
            and '2 more dropped (no straight move of the arm to the insertion start or away after the release)'
            in out.getvalue()
        ), out.getvalue()
    # none can do the moves at the assembly: no move grasp, and the reason is printed
    gen, out = generator({i: reasons[1] for i in range(7)}), io.StringIO()
    with contextlib.redirect_stdout(out):
        grasps = gen.generate_grasps('A', max_n_grasp=100, n_proc=2)
    assert move_ids(grasps) == [] and len(hold_ids(grasps)) == 7
    assert (
        'no usable move grasp' in out.getvalue()
        and f'the 7 that pass these fail at the assembly, e.g. {reasons[1]}'
        in out.getvalue()
    ), out.getvalue()
    assert (
        gen._assembly_scene is not None
    )  # the planner for the test was built before the worker processes started
    # no more worker processes than cores for this test, however many the generator is given (each runs collision tests):
    # 40 grasps and 64 processes asked for; every grasp is given the size of the chunk it was tested in as its reason
    gen, out = generator({}, n_grasps=40), io.StringIO()
    gen.assembly_moves_chunk = lambda moves, part_id: (
        [f'chunk of {len(moves)}'] * len(moves)
    )
    with contextlib.redirect_stdout(out):
        gen.generate_grasps('A', max_n_grasp=100, n_proc=64)
    n_chunks = min(64, os.cpu_count())
    assert (
        f'the 40 that pass these fail at the assembly, e.g. chunk of {len(range(0, 40, n_chunks))}'
        in out.getvalue()
    ), out.getvalue()
    # the parts of an assembly are processed at the same time: with 4 parts each of them gets a quarter of the cores
    gen, out = generator({}, n_grasps=40), io.StringIO()
    gen.assembly_moves_chunk = lambda moves, part_id: (
        [f'chunk of {len(moves)}'] * len(moves)
    )
    for name in ['B', 'C', 'D']:
        gen.part_meshes[name], gen.part_final_transforms[name] = (
            box([1, 1, 1], [REF_XY[0] + 30, REF_XY[1], 0.5]),
            np.eye(4),
        )
    with contextlib.redirect_stdout(out):
        gen.generate_grasps('A', max_n_grasp=100, n_proc=64)
    n_chunks = max(1, min(64, os.cpu_count() // 4))
    assert (
        f'the 40 that pass these fail at the assembly, e.g. chunk of {len(range(0, 40, n_chunks))}'
        in out.getvalue()
    ), out.getvalue()
    # another arm than the real one: the test is not made
    gen = toy_generator([[at] * 3, []], part=part)
    gen.assembly_moves_chunk = lambda moves, part_id: [reasons[1]] * len(moves)
    with contextlib.redirect_stdout(io.StringIO()):
        assert sorted(
            move_ids(gen.generate_grasps('A', max_n_grasp=100, n_proc=1))
        ) == [0, 1, 2]


@test
def hold_arm_picks_the_first_part_only_where_it_reaches(c):
    """
    run_pickup_gen.hold_pickup_q: a spot for the first part counts only
    if the hold arm has an arm pose there that puts the gripper where the
    hold grasp wants it, and its arm and opened gripper touch nothing
    (until 2026-11-13 every free spot counted: a human took the part).
    """
    ArmMotionPlanner.VISUALIZE = False
    hold = ArmMotionPlanner(
        get_arm_chain('panda', 'hold'),
        'panda',
        False,
        get_dual_arm_box('panda')[1],
    )
    grasp = top_grasp(1.0)

    def q_at(x, y, others=()):
        mat = brick_at(x, y)
        brick = BRICK.copy().apply_transform(mat)
        return run_pickup_gen.hold_pickup_q(
            hold,
            grasp,
            mat,
            np.eye(4),
            [brick, *others],
            False,
            c.chain,
            False,
        )

    # In front of the hold arm (its base stands at x = -45, y = 20): an
    # arm pose, with the gripper where the grasp wants it.
    q = q_at(-20.0, 0.0)
    assert q is not None
    gripper = get_transform_matrix_quat(
        *get_gripper_pos_quat_from_arm_q(
            hold.arm_chain, q, 'panda', has_ft_sensor=False
        )
    )
    wanted = get_pickup_gripper_pose(grasp, brick_at(-20.0, 0.0), np.eye(4))
    assert np.allclose(gripper[:3, 3], wanted[:3, 3], atol=0.05)
    # More than a metre from its base: out of reach.
    assert q_at(40.0, -40.0) is None
    # Reachable, but something is where the hand would be: no pose. The
    # plate goes through the middle of the hand, above the brick (wider
    # than the hand: the collision test compares surfaces, so a box that
    # lies entirely inside the hand would not count).
    _, grip_tf, _ = hold.get_meshes_transforms(
        q, min(grasp.open_ratio + config.RETRACT_OPEN_RATIO, 1.0), None, None
    )
    T_hand = grip_tf['panda_hand']
    hand = hold.gripper_meshes['panda_hand'].vertices
    hand = hand @ T_hand[:3, :3].T + T_hand[:3, 3]
    blocker = box([30.0, 30.0, 1.0], hand.mean(axis=0))
    assert blocker.bounds[0, 2] > 4.0  # above the brick (0 to 4 cm high)
    assert q_at(-20.0, 0.0, [blocker]) is None


@test
def layout_searches_no_arm_pose_where_the_hand_is_behind_a_wall(c):
    """
    pickup_ik_solutions: wrist frame and flange follow the gripper, so their wall margin is known before an arm pose is searched
    (get_hand_wall_margin). Clearly behind a wall, at the pickup or on the lift above it: no search, no arm pose. Within
    PICKUP_POSE_SLACK of the limit the search runs as before (and finds nothing that passes).
    """
    planner = move_planner()
    grasp = top_grasp(1.0)

    def hand_margin(mat, lift=0.0):
        return get_hand_wall_margin(
            c.chain,
            'panda',
            get_pickup_gripper_pose(grasp, mat, np.eye(4))
            + np.array(
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, lift], [0, 0, 0, 0]]
            ),
        )

    calls, real = [], run_pickup_gen.get_pickup_arm_q
    run_pickup_gen.get_pickup_arm_q = lambda *args, **kwargs: (
        calls.append(1),
        real(*args, **kwargs),
    )[1]

    def searched(mat):
        calls.clear()
        np.random.seed(0)
        solutions = pickup_ik_solutions(
            {}, planner, grasp, 'A', mat, np.eye(4), False
        )
        return len(calls), solutions

    try:
        # along x towards the wall behind the board: the margin of the hand falls 1 cm per cm
        margins = {
            x: hand_margin(brick_at(x, 5.0))
            for x in np.arange(0.0, -40.0, -0.5)
        }
        assert (
            margins[0.0] > VIRTUAL_WALL_SAFETY + 5
            and min(margins.values()) < VIRTUAL_WALL_SAFETY - 5
        )
        inside = max(
            x for x, m in margins.items() if m < VIRTUAL_WALL_SAFETY + 3.0
        )  # the first spot nearer than 3 cm above the limit ...
        slope = (margins[inside - 0.5] - margins[inside]) / -0.5
        assert abs(slope - 1.0) < 1e-6, slope

        def at_margin(margin):
            return brick_at(inside - (margins[inside] - margin) / slope, 5.0)

        n_starts = 2 + run_pickup_gen.N_PICKUP_IK_SAMPLES
        # 3 cm inside the limit: the search runs; whatever it finds is inside the walls
        n, solutions = searched(at_margin(VIRTUAL_WALL_SAFETY + 3.0))
        assert n == n_starts and all(
            get_virtual_wall_margin(c.chain, s[1]) >= VIRTUAL_WALL_SAFETY
            for s in solutions
        )
        # 0.05 cm behind the limit, less than PICKUP_POSE_SLACK: the search still runs, and no arm pose passes
        assert (
            abs(
                hand_margin(at_margin(VIRTUAL_WALL_SAFETY - 0.05))
                - (VIRTUAL_WALL_SAFETY - 0.05)
            )
            < 1e-6
        )
        assert searched(at_margin(VIRTUAL_WALL_SAFETY - 0.05)) == (
            n_starts,
            [],
        )
        # further behind than the slack: no search at all
        assert searched(
            at_margin(VIRTUAL_WALL_SAFETY - PICKUP_POSE_SLACK - 0.05)
        ) == (0, [])
        assert searched(at_margin(VIRTUAL_WALL_SAFETY - 10.0)) == (0, [])
        # the lift counts as well: a pickup high up whose pose is inside the walls, but 10 cm higher the flange is behind the upper wall
        high = None
        for height in np.arange(40.0, 140.0, 1.0):
            mat = brick_at(0.0, 5.0)
            mat[2, 3] += height
            if (
                hand_margin(mat) >= VIRTUAL_WALL_SAFETY + 0.5
                and hand_margin(mat, max(run_pickup_gen.LIFT_CHECK_HEIGHTS))
                < VIRTUAL_WALL_SAFETY - PICKUP_POSE_SLACK - 0.5
            ):
                high = mat
                break
        assert high is not None
        assert searched(high) == (0, [])
    finally:
        run_pickup_gen.get_pickup_arm_q = real


@test
def banned_spots_of_the_layout(c):
    """spots where motion planning failed because of the place: stored per part and grasp"""
    tmp = tempfile.mkdtemp(prefix='test_flat_grasp_')
    try:
        assert load_banned_spots(tmp) == set()
        assert ban_spot(tmp, '2', np.int64(1603), np.array([-3.75, 5.0])) == [
            ('2', 1603, -3.75, 5.0)
        ]
        ban_spot(tmp, '2', 1603, [-3.75, 5.0])  # a second time: still once
        ban_spot(tmp, '7', 12, (1.2500000001, -2.5))
        assert load_banned_spots(tmp) == {
            ('2', 1603, -3.75, 5.0),
            ('7', 12, 1.25, -2.5),
        }
        with open(os.path.join(tmp, BANNED_SPOTS_FILE)) as fp:
            assert json.load(fp) == [
                ['2', 1603, -3.75, 5.0],
                ['7', 12, 1.25, -2.5],
            ]
        # the key the layout builds for a spot of its grid finds them; another grasp or spot of the same part is not banned
        grid_x = np.arange(-31.25, 31.25, 1.25)
        xy = np.array([grid_x[np.argmin(np.abs(grid_x + 3.75))], 5.0])
        banned = load_banned_spots(tmp)
        assert spot_key('2', np.int64(1603), xy) in banned and spot_key(
            2, 1603, xy
        ) == spot_key('2', 1603, xy)
        assert all(
            k not in banned
            for k in [
                spot_key('2', 1604, xy),
                spot_key('2', 1603, xy + [1.25, 0]),
                spot_key('2', 1603, xy + [0, 1.25]),
                spot_key('3', 1603, xy),
            ]
        )
        clear_banned_spots(tmp)
        assert load_banned_spots(tmp) == set() and not os.path.exists(
            os.path.join(tmp, BANNED_SPOTS_FILE)
        )
        clear_banned_spots(tmp)  # nothing to remove: fine
    finally:
        shutil.rmtree(tmp)


@test
def loop_reacts_to_the_cause_of_a_failure(c):
    """run_planning_flat.py: drop the grasp, move the part to another spot, or stop"""
    from planning.run_planning_flat import next_step, ban_hold_grasp
    from planning.plan_checks import CAUSE_HOLD_GRASP

    def failure(**kw):
        return {
            'stage': 'run_motion_plan',
            'part': '2',
            'grasp_id': 5,
            'reason': 'why',
            **kw,
        }

    assert next_step(failure(cause=CAUSE_GRASP)) == 'ban grasp'
    assert next_step(failure(cause=CAUSE_SPOT)) == 'ban spot'
    assert (
        next_step(failure(cause=CAUSE_INSERTION)) == 'stop'
    )  # no grasp and no spot can help
    assert (
        next_step(failure()) == 'ban grasp'
    )  # a failure file without a cause (written before causes existed)
    assert (
        next_step(failure(cause=CAUSE_GRASP, grasp_id=None)) == 'stop'
        and next_step(failure(cause=CAUSE_SPOT, grasp_id=None)) == 'stop'
    )  # not a part of the real arm
    assert next_step(None) == 'stop'  # the stage failed without saying why
    # The hold arm cannot do its part with its grasp: that hold grasp goes.
    assert next_step(failure(cause=CAUSE_HOLD_GRASP)) == 'ban hold grasp'
    assert (
        next_step(failure(cause=CAUSE_HOLD_GRASP, grasp_id=None)) == 'stop'
    )  # no hold grasp named: nothing to drop

    # ban_hold_grasp removes the hold grasp and the pairs that use it, and
    # nothing else (not the move grasp with the same id, not other parts).
    def g(grasp_id):
        return types.SimpleNamespace(grasp_id=grasp_id)

    tmp = tempfile.mkdtemp()
    try:
        grasps = {
            'grasps': {
                '0': {'move': [[g(3)]], 'hold': [g(3), g(4)]},
                '1': {'move': [[g(7)]], 'hold': [g(3)]},
            },
            'grasp_id_pairs': {
                ('1', '0'): [(7, 3), (7, 4)],
                ('0', '1'): [(3, 3)],
            },
        }
        with open(os.path.join(tmp, 'grasps.pkl'), 'wb') as fp:
            pickle.dump(grasps, fp)
        ban_hold_grasp(tmp, '0', 3)
        with open(os.path.join(tmp, 'grasps.pkl'), 'rb') as fp:
            after = pickle.load(fp)
    finally:
        shutil.rmtree(tmp)
    assert [h.grasp_id for h in after['grasps']['0']['hold']] == [4]
    assert [m[0].grasp_id for m in after['grasps']['0']['move']] == [3]
    assert [h.grasp_id for h in after['grasps']['1']['hold']] == [3]
    assert after['grasp_id_pairs'] == {
        ('1', '0'): [(7, 4)],  # part 0 held with grasp 3: gone
        ('0', '1'): [(3, 3)],  # part 1 held with its own grasp 3: kept
    }


# ------------------------------------------------------------------------------- the grasp filter on a real planning log


@test
def filter_reproduces_a_verified_plan(c):
    """
    The grasp filter now uses flat_grasp.move_grasp_criteria. On the unfiltered grasps of a verified plan (human_normal v6, made
    with the earlier code, a finger overlap of 0.5 cm and hands leaning at most 30 deg) it must keep exactly the grasps the
    earlier filter kept when it is given those two values; without the tilt limit it keeps those and more.
    """
    log_dir = os.path.join(
        project_base_dir, 'logs', 'old_exp_human_normal_flat_v6', 'human_normal'
    )
    if not os.path.exists(os.path.join(log_dir, 'grasps_unfiltered.pkl')):
        print(
            '      (skipped: logs/old_exp_human_normal_flat_v6 is not on this machine)'
        )
        return
    with open(os.path.join(log_dir, 'grasps.pkl'), 'rb') as fp:
        stored = pickle.load(fp)
    with open(os.path.join(log_dir, 'workcell.json'), 'r') as fp:
        assert (
            json.load(fp)['banned_move_grasps'] == []
        )  # nothing was removed from grasps.pkl after the filter
    tmp = tempfile.mkdtemp(prefix='test_flat_grasp_')
    saved = flat_grasp.GRASP_MIN_FINGER_OVERLAP
    try:
        for name in ['grasps_unfiltered.pkl', 'precedence.pkl']:
            shutil.copy(os.path.join(log_dir, name), os.path.join(tmp, name))
        shutil.copy(
            os.path.join(log_dir, 'grasps_unfiltered.pkl'),
            os.path.join(tmp, 'grasps.pkl'),
        )
        flat_grasp.GRASP_MIN_FINGER_OVERLAP = stored['settings'][
            'flat_pickup'
        ]['min_finger_overlap']
        assert stored['settings']['flat_pickup']['max_tilt'] == 30.0
        empty = filter_flat_grasps(
            os.path.join(
                project_base_dir, 'assets', 'fabrica', 'human_normal'
            ),
            tmp,
            max_tilt=30.0,
        )
        with open(os.path.join(tmp, 'grasps.pkl'), 'rb') as fp:
            new = pickle.load(fp)
        shutil.copy(
            os.path.join(log_dir, 'grasps_unfiltered.pkl'),
            os.path.join(tmp, 'grasps.pkl'),
        )
        filter_flat_grasps(
            os.path.join(
                project_base_dir, 'assets', 'fabrica', 'human_normal'
            ),
            tmp,
        )  # no tilt limit
        with open(os.path.join(tmp, 'grasps.pkl'), 'rb') as fp:
            lifted = pickle.load(fp)
    finally:
        flat_grasp.GRASP_MIN_FINGER_OVERLAP = saved
        shutil.rmtree(tmp)
    assert (
        empty == [] and new['settings']['flat_pickup']['fallback_parts'] == []
    )
    more = 0
    for part in stored['grasps']:
        kept = sorted(g[0].grasp_id for g in stored['grasps'][part]['move'])
        assert (
            sorted(g[0].grasp_id for g in new['grasps'][part]['move']) == kept
        ), part
        assert sorted(
            g.grasp_id for g in new['grasps'][part]['hold']
        ) == sorted(g.grasp_id for g in stored['grasps'][part]['hold']), part
        without_limit = {g[0].grasp_id for g in lifted['grasps'][part]['move']}
        assert set(kept) <= without_limit, part
        more += len(without_limit) - len(kept)
    assert new['grasp_id_pairs'] == stored['grasp_id_pairs']
    assert (
        more > 0 and lifted['settings']['flat_pickup']['max_tilt'] == 180.0
    )  # hands that lean further than 30 deg are usable now


@test
def layout_of_a_planned_assembly(c):
    """
    run_pickup_gen (the layout) with the grasps and the sequence of a verified plan (socket_set_small): every part gets a
    spot; the arm pose stored for each pickup passes the full pickup check in the scene at its pick; the tests made before an
    arm pose is searched (pickup_blocked, wall margin of the hand) say nothing against the spots that were taken; the lying
    parts keep PART_GAP and stay out of the assembly zone.
    """
    from shapely.geometry import MultiPoint, box as shapely_box
    from assets.transform import get_transform_matrix
    from planning.robot.geometry import load_part_meshes
    from planning.robot.workcell import use_log_workcell
    from planning.run_pickup_gen import load_final_part_cfg
    from planning.run_seq_opt import SequenceOptimizer
    from planning.plan_checks import SceneChecker

    log_dir = os.path.join(
        project_base_dir,
        'logs',
        'old_exp_socket_set_small_flat_run2',
        'socket_set_small',
    )
    if not os.path.exists(os.path.join(log_dir, 'tree_opt.pkl')):
        print(
            '      (skipped: logs/old_exp_socket_set_small_flat_run2 is not on this machine)'
        )
        return
    assembly_dir = os.path.join(
        project_base_dir, 'assets', 'fabrica', 'socket_set_small'
    )

    def lay_out():
        """runs the layout on a copy of the log, returns what it wrote"""
        tmp = tempfile.mkdtemp(prefix='test_flat_grasp_')
        try:
            for name in [
                'grasps.pkl',
                'precedence.pkl',
                'tree_opt.pkl',
                'stats.json',
            ]:
                shutil.copy(
                    os.path.join(log_dir, name), os.path.join(tmp, name)
                )
            use_log_workcell(
                tmp
            )  # default places of the assembly and of the pickup area
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                assert run_pickup_gen.run_pickup_gen(
                    assembly_dir, tmp
                ), out.getvalue()
            written = {}
            for name in [
                'flat_pickup.json',
                'pickup.json',
                'pickup_q.json',
                'pickup_q_hold.json',
            ]:
                with open(os.path.join(tmp, 'fixture', name)) as fp:
                    written[name] = json.load(fp)
            for name in ['grasps.pkl', 'precedence.pkl', 'tree_opt.pkl']:
                with open(os.path.join(tmp, name), 'rb') as fp:
                    written[name] = pickle.load(fp)
            return written, out.getvalue()
        finally:
            shutil.rmtree(tmp)

    written, _ = lay_out()
    flat, pickup_q = written['flat_pickup.json'], written['pickup_q.json']
    pickup = {
        k: get_transform_matrix(v) for k, v in written['pickup.json'].items()
    }
    grasps, G = written['grasps.pkl'], written['precedence.pkl']
    tree = written['tree_opt.pkl']
    sequence, grasps_sequence = SequenceOptimizer(G, grasps).get_sequence(tree)
    sequence, grasps_sequence = (
        sequence[::-1],
        grasps_sequence[::-1],
    )  # in the order of the assembly: (part moved, part held)
    first = sequence[0][1]
    assert (
        sorted(flat) == sorted(pickup) == sorted(G.nodes)
        and flat[first]['arm'] == 'hold'
    )
    assert sorted(pickup_q) == sorted(part for part, _ in sequence) and all(
        flat[part]['arm'] == 'move' for part, _ in sequence
    )
    # The hold arm picks the first part: the layout stored its arm pose,
    # and that pose holds the gripper where the hold grasp wants it.
    pickup_q_hold = written['pickup_q_hold.json']
    assert sorted(pickup_q_hold) == [first]
    cfg_first = load_final_part_cfg(assembly_dir, 'panda')
    hold_gripper = get_transform_matrix_quat(
        *get_gripper_pos_quat_from_arm_q(
            get_arm_chain('panda', 'hold'),
            np.array(pickup_q_hold[first]),
            'panda',
            has_ft_sensor=False,
        )
    )
    wanted = get_pickup_gripper_pose(
        grasps_sequence[0][1],
        pickup[first],
        get_transform_matrix_quat(
            cfg_first['pos'][first], cfg_first['quat'][first]
        ),
    )
    assert np.allclose(hold_gripper[:3, 3], wanted[:3, 3], atol=0.05)

    # Where the hold arm can pick the first part on no spot, the move arm
    # brings it (fallback 'place'), and no hold-arm pose is stored.
    real = run_pickup_gen.hold_pickup_q
    run_pickup_gen.hold_pickup_q = lambda *args, **kwargs: None
    try:
        fallback, out = lay_out()
    finally:
        run_pickup_gen.hold_pickup_q = real
    assert fallback['flat_pickup.json'][first]['arm'] == 'place', out
    assert fallback['pickup_q_hold.json'] == {}
    assert first in fallback['pickup_q.json']  # the move arm's pose there
    assert 'the hold arm can pick it on none of the' in out
    cfg = load_final_part_cfg(assembly_dir, 'panda')
    raw = {
        k[4:]: v
        for k, v in load_part_meshes(assembly_dir, transform='none').items()
    }
    checker = SceneChecker('panda', 'panda', False, raw, pickup, cfg['mesh'])
    assembled = {first}
    for (part, _), (grasps_move, _) in zip(sequence, grasps_sequence):
        grasp = grasps_move[0]
        assert grasp.grasp_id == flat[part]['grasp_id']
        state = {k: ('final' if k in assembled else 'pickup') for k in raw}
        q, opened = (
            np.array(pickup_q[part]),
            min(grasp.open_ratio + config.RETRACT_OPEN_RATIO, 1.0),
        )
        gripper = get_pickup_gripper_pose(
            grasp,
            pickup[part],
            get_transform_matrix_quat(cfg['pos'][part], cfg['quat'][part]),
        )
        assert np.allclose(
            checker.gripper_transform(q), gripper, atol=1e-3
        )  # the stored arm pose holds the gripper where the grasp wants it
        assert (
            checker.check_pickup(
                q, opened, part, state, closed_ratio=grasp.open_ratio
            )
            == []
        ), part
        assert (
            checker.pickup_blocked(
                gripper, opened, part, state, closed_ratio=grasp.open_ratio
            )
            is None
        ), part
        hand, whole = (
            get_hand_wall_margin(checker.chain, 'panda', gripper),
            get_virtual_wall_margin(checker.chain, q),
        )
        assert hand >= whole - 1e-3 and whole >= VIRTUAL_WALL_SAFETY, (
            part,
            hand,
            whole,
        )  # 1e-3: how exactly the arm pose reaches the gripper pose
        assembled.add(part)
    # the lying parts: PART_GAP between their footprints, none in the zone around the assembly
    footprint = {
        k: MultiPoint(
            raw[k]
            .copy()
            .apply_transform(pickup[k])
            .convex_hull.vertices[:, :2]
        ).convex_hull
        for k in raw
    }
    bounds = np.array([mesh.bounds for mesh in cfg['mesh'].values()])
    zone = shapely_box(
        *(bounds[:, 0, :2].min(axis=0) - config.ASSEMBLY_ZONE_MARGIN),
        *(bounds[:, 1, :2].max(axis=0) + config.ASSEMBLY_ZONE_MARGIN),
    )
    parts = sorted(raw)
    assert all(
        footprint[a].distance(footprint[b]) >= PART_GAP - 1e-6
        for i, a in enumerate(parts)
        for b in parts[:i]
    )
    assert all(not footprint[k].intersects(zone) for k in parts)


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
