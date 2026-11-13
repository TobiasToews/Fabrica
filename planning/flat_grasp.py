'''
Grasps for the fixture-free (flat) pipeline: one grasp has to pick its part lying flat on the table AND hold it at the assembled
pose. Shared by the grasp generation (run_grasp_arm_gen.py --flat), the grasp filter, the sequence optimisation and the layout
(run_pickup_gen.py), and by the plan checks.

  resting poses and the flat-pickup test       get_flat_rots, get_flat_pickup_candidates
  finger contact on a fixed grid of the pads   finger_pads, PartFaces, pad_contact, set_pad_contact
  criteria of a move grasp                     move_grasp_criteria (grasp generation and grasp filter use the same)
  rating and choice of the grasps that are kept  rate_flat_grasp, select_grasps
'''
import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import types
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

from assets.transform import get_transform_matrix_quat, mat_to_pos_quat
from planning.config import RETRACT_OPEN_RATIO, GRASP_MIN_FINGER_OVERLAP, MIN_SIGMA_INSERTION, MIN_JOINT_LIMIT_MARGIN, ROBOT_PART_CLEARANCE, \
    PAD_CELL, PAD_STORE_CELL, PAD_CLOSING, PAD_PENETRATION, FLAT_RATING_WEIGHTS, FLAT_RATING_DEPTH, FLAT_RATING_AREA, FLAT_RATING_LEVER
from planning.robot.geometry import transform_gripper_meshes, get_gripper_meshes_transforms, get_gripper_finger_names, get_gripper_basis_directions
from planning.robot.util_grasp import get_grasp_info_from_gripper_state

MAX_TILT = 180.0 # max angle [deg] between gripper approach direction and straight down: 180 = no limit (was 30 until 2026-11-11). What keeps a leaning hand off the board are GROUND_CLEARANCE (gripper) and WRIST_CLEARANCE (the arm link behind it)
MIN_TIP_ANGLE_FALLBACK = 10.0 # deg, resting poses allowed for a part that cannot be picked in a pose tipping only above MIN_TIP_ANGLE (marked on the template)
MIN_TIP_ANGLE = 20.0 # min tilt [deg] of the board/part before the part tips over (support margin vs. height of the center of mass)
GROUND_CLEARANCE = 0.5 # min height [cm] of the (opened) gripper above the board
WRIST_CLEARANCE = ROBOT_PART_CLEARANCE # min height [cm] above the board of the arm link that carries the hand: the plan checks demand it of every arm link (planning/plan_checks.py), and where this link is follows from the gripper pose alone
WRIST_LINK = 'panda_link7'
YAW_OFFSETS = [0, 180, 90, -90] # rotations [deg] of a resting part about the vertical axis to try for the arm IK
PAD_PLANE_TOL = 0.02 # cm, faces of a finger this close to its inner-most plane (and parallel to it) form the gripping pad
RATING_TILT = 90.0 # deg, rating of a grasp: a hand leaning this much (from the side) gets 0 for "upright", a hand from straight above 1


# ------------------------------------------------------------------------------- resting poses and the flat-pickup test

FACE_MERGE_ANGLE = 0.1 # deg: triangles of the convex hull whose normals differ by less belong to one face to lie on
SUPPORT_TOL = 1e-3 # cm: points this close to the lowest point of a resting part touch the board


def hull_faces(vertices, center_of_mass, hull=None):
    '''
    The faces of the convex hull of the points `vertices` (n, 3): [(outward unit normal, corner indices, share)], one entry
    per face. A flat face of a CAD part is cut into many triangles, among them slivers without an area whose normal is
    noise; triangles whose normals differ by less than FACE_MERGE_ANGLE are one face, its normal is their mean weighted with
    their areas, its corners are all their corners (indices into vertices). share: the fraction of all directions in which
    the face is the lowest one seen from center_of_mass (its solid angle / 4 pi), the chance to land on it for a part dropped
    at random. Sorted by the direction of the normal: the order does not depend on how the hull was triangulated.
    hull: scipy's ConvexHull of vertices if it is at hand.
    '''
    hull = ConvexHull(vertices) if hull is None else hull
    simplices = hull.simplices
    tri = vertices[simplices] # (n, 3, 3)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = np.linalg.norm(cross, axis=1) / 2
    normals = hull.equations[:, :3] / np.linalg.norm(hull.equations[:, :3], axis=1)[:, None] # outward
    a, b, c = (tri[:, i] - np.asarray(center_of_mass, dtype=float) for i in range(3))
    la, lb, lc = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1), np.linalg.norm(c, axis=1)
    dot = lambda u, v: np.einsum('ij,ij->i', u, v)
    # solid angle of each triangle seen from the centre of mass (Van Oosterom and Strackee)
    angles = 2 * np.arctan2(np.abs(dot(a, np.cross(b, c))), la * lb * lc + dot(a, b) * lc + dot(a, c) * lb + dot(b, c) * la)
    # 1. triangles with the same normal up to 3 decimals; 2. of these groups, the ones closer than FACE_MERGE_ANGLE
    keys, group = np.unique(np.round(normals, 3) + 0.0, axis=0, return_inverse=True)
    group = group.reshape(-1)
    limit = np.cos(np.radians(FACE_MERGE_ANGLE))
    directions, faces = np.zeros((0, 3)), [] # of the faces found so far: unit normal; [area-weighted sum of normals, triangle indices]
    for k in np.argsort(-np.bincount(group, weights=areas, minlength=len(keys)), kind='stable'): # large groups first: they set the direction of a face
        members = np.nonzero(group == k)[0]
        weighted = (normals[members] * areas[members][:, None]).sum(axis=0)
        if np.linalg.norm(weighted) < 1e-12: continue # only slivers
        near = np.nonzero(directions @ (weighted / np.linalg.norm(weighted)) > limit)[0]
        if len(near) > 0:
            faces[near[0]][0] = faces[near[0]][0] + weighted
            faces[near[0]][1] = np.concatenate([faces[near[0]][1], members])
            directions[near[0]] = faces[near[0]][0] / np.linalg.norm(faces[near[0]][0])
        else:
            faces.append([weighted, members])
            directions = np.vstack([directions, weighted / np.linalg.norm(weighted)])
    result = [(directions[i], np.unique(simplices[members]), float(angles[members].sum() / (4 * np.pi))) for i, (_, members) in enumerate(faces)]
    return sorted(result, key=lambda face: tuple(np.round(face[0], 3) + 0.0))


def get_flat_rots(mesh, min_tip_angle=MIN_TIP_ANGLE, with_fallback=False):
    '''
    Stable resting rotations of a mesh lying on the board (about the origin), with the support margin and tipping angle.
    Poses that tip between MIN_TIP_ANGLE_FALLBACK and min_tip_angle are added (marked 'fallback') if the part has no stable pose,
    or if with_fallback is set: the part has stable poses, but no grasp can pick it in any of them (e.g. the only stable pose is
    the assembled pose upside down, where the gripper would have to come from below).

    A part can lie on every face of its convex hull over which its centre of mass stands (resting_poses). How a part can lie
    is a property of its shape and must not depend on where the part stands: everything is computed for a copy with the
    middle of its bounding box at the origin (also the centre of mass: for a mesh that is not closed the volume integral
    depends on the origin), the numbers are rounded, the poses ordered by tipping angle and face direction.
    Until 2026-11-12 trimesh.poses.compute_stable_poses was used on the mesh where it stands. It lost poses through rounding
    (beam part 1: of its 4 poses 2 at the default place of the assembly), differently at different places, and for a hull
    with a sliver triangle it never returned (beam parts 2 and 6 with the assembly at (+5, -30)).
    '''
    vertices = np.asarray(mesh.vertices, dtype=float)
    middle = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    local = mesh.copy()
    local.apply_translation(-middle)
    com = np.asarray(local.center_mass, dtype=float)
    points = np.asarray(local.vertices, dtype=float)
    key = (points.shape, points.tobytes(), com.tobytes())
    if key not in _RESTING_POSES:
        if len(_RESTING_POSES) >= 64: _RESTING_POSES.clear() # a few parts per process; never grows without bound
        _RESTING_POSES[key] = resting_poses(points, com)
    flat_rots = [dict(r) for r in _RESTING_POSES[key]]
    stable = [r for r in flat_rots if r['tip_angle'] >= min_tip_angle]
    if (len(stable) == 0 or with_fallback) and min_tip_angle > MIN_TIP_ANGLE_FALLBACK: # e.g. a small cylinder: rolls lying, but stands upright
        fallback = [r for r in flat_rots if MIN_TIP_ANGLE_FALLBACK <= r['tip_angle'] < min_tip_angle]
        for r in fallback: r['fallback'] = True
        stable = stable + fallback
    return stable


_RESTING_POSES = {} # (points, centre of mass) -> resting_poses: the same part is asked several times in a stage


def resting_poses(points, com):
    '''
    All poses in which the points (n, 3; a part, the middle of its bounding box at the origin) rest on a face of their convex
    hull with the centre of mass com over what touches the board: [{'rot', 'prob', 'margin', 'tip_angle'}], the most stable
    first. rot (3x3, about the origin) turns the face down; prob: the share of the face (hull_faces); margin: distance (cm)
    of the centre of mass, seen from above, to the edge of the support; tip_angle: by how many degrees the part can be
    tilted before it tips over that edge.
    '''
    hull = ConvexHull(points)
    flat_rots = []
    for normal, corners, share in hull_faces(points, com, hull):
        # the point of the face plane under the centre of mass has to belong to the part (to its hull): else the centre of
        # mass is not over anything that touches the board. Most faces of a rounded part end here.
        height = (points[corners] @ normal).mean() - com @ normal # of the centre of mass above the face plane
        foot = com + height * normal
        if np.max(hull.equations[:, :3] @ foot + hull.equations[:, 3]) > 10 * SUPPORT_TOL: continue
        # the rotation that puts this face on the board: z axis = minus its normal, x and y as trimesh chose them
        z = -normal
        x = np.array([-z[1], z[0], 0.0])
        x = np.array([1.0, 0.0, 0.0]) if np.linalg.norm(x) < 1e-6 else x / np.linalg.norm(x)
        y = np.cross(z, x)
        rot = np.array([x, y / np.linalg.norm(y), z])
        rest, com_rest = points @ rot.T, rot @ com
        low = rest[:, 2].min()
        touching = np.union1d(corners, np.nonzero(rest[:, 2] < low + SUPPORT_TOL)[0]) # the corners of the face, and what else touches the board
        support = rest[touching][:, :2]
        if len(support) < 3: continue
        try:
            edges = ConvexHull(support).equations
        except Exception: # degenerate (line) support
            continue
        margin = -np.max(edges[:, :2] @ com_rest[:2] + edges[:, 2])
        if margin <= 0: continue # the centre of mass does not stand over this face: the part does not stay on it
        tip_angle = np.degrees(np.arctan2(margin, com_rest[2] - low))
        flat_rots.append({'rot': rot, 'prob': round(share, 6), 'margin': round(float(margin), 6), 'tip_angle': round(float(tip_angle), 6)})
    flat_rots.sort(key=lambda r: -r['tip_angle']) # stable sort: equal tipping angles stay in the order of the face directions
    return flat_rots


_WRIST_POINTS = {}

def wrist_points(gripper_type):
    '''
    Corner points (n, 3) of the arm link that carries the hand (panda link 7, convex hull of its collision mesh), in the frame of
    the gripper base. The hand is bolted to this link, so where the link is follows from the gripper pose alone, whatever the
    arm does behind it. None for grippers that the flat pipeline does not plan on the panda arm (no wrist test then).
    Without a force-torque sensor between flange and hand (the flat pipeline plans without one).
    '''
    if gripper_type not in _WRIST_POINTS:
        points = None
        if gripper_type == 'panda':
            from planning.robot.geometry import load_arm_meshes, get_arm_meshes_transforms
            from planning.robot.util_arm import get_arm_chain, get_gripper_pos_quat_from_arm_q
            chain = get_arm_chain('panda', 'move')
            q = chain.active_to_full(chain.rest_q)
            meshes = load_arm_meshes('panda', os.path.join(project_base_dir, 'assets'), visual=False, convex=True)
            link = get_arm_meshes_transforms(meshes, chain, q)[WRIST_LINK]
            relative = np.linalg.inv(get_transform_matrix_quat(*get_gripper_pos_quat_from_arm_q(chain, q, gripper_type))) @ link
            points = meshes[WRIST_LINK].convex_hull.vertices @ relative[:3, :3].T + relative[:3, 3]
        _WRIST_POINTS[gripper_type] = points
    return _WRIST_POINTS[gripper_type]


def get_flat_pickup_candidates(grasp, part_mesh, flat_rots, gripper_type, gripper_meshes, max_tilt=MAX_TILT, clearance=GROUND_CLEARANCE, wrist_clearance=WRIST_CLEARANCE):
    '''
    For a grasp (defined at the final pose), the resting rotations of the part in which the gripper can take it from the board:
    the whole gripper stays `clearance` above the board and the arm link behind it `wrist_clearance` (see wrist_points). There
    is no limit on how far the hand leans (max_tilt 180), a hand that leans too far fails one of the two.
    The yaw first aligns the gripper fingers with the x axis (as in the fixture), then is rotated by YAW_OFFSETS as alternatives
    for the arm IK. Stable resting poses come before fallback poses, upright hands before leaning ones.
    '''
    grasp_rot = R.from_quat(grasp.quat[[1, 2, 3, 0]]).as_matrix()
    wrist = wrist_points(gripper_type)
    candidates = []
    for flat_rot in flat_rots:
        approach = flat_rot['rot'] @ grasp_rot @ np.array([0, 0, 1])
        tilt = np.degrees(np.arccos(np.clip(-approach[2], -1, 1)))
        if tilt > max_tilt: continue

        l2r = flat_rot['rot'] @ grasp_rot @ np.array([0, -1, 0])
        yaw = -np.arctan2(l2r[1], l2r[0])
        rot = R.from_euler('z', yaw).as_matrix() @ flat_rot['rot']

        # the heights do not depend on the yaw
        pickup_mat = np.eye(4)
        pickup_mat[:3, :3] = rot
        pickup_mat[2, 3] = -(part_mesh.vertices @ rot.T)[:, 2].min() # rest on the board
        gripper_mat = pickup_mat @ get_transform_matrix_quat(grasp.pos, grasp.quat)
        wrist_z = float((wrist @ gripper_mat[2, :3] + gripper_mat[2, 3]).min()) if wrist is not None else None
        if wrist_z is not None and wrist_z < wrist_clearance: continue
        gripper_pos, gripper_quat = mat_to_pos_quat(gripper_mat)
        min_z = np.inf
        for open_ratio in [grasp.open_ratio, min(grasp.open_ratio + RETRACT_OPEN_RATIO, 1.0)]:
            meshes = transform_gripper_meshes(gripper_type, gripper_meshes, gripper_pos, gripper_quat, np.eye(4), open_ratio)
            min_z = min(min_z, min(mesh.bounds[0, 2] for mesh in meshes.values()))
        if min_z < clearance: continue

        for i, yaw_offset in enumerate(YAW_OFFSETS):
            candidates.append({'rot': R.from_euler('z', yaw_offset, degrees=True).as_matrix() @ rot, 'tilt': float(tilt), 'tip_angle': flat_rot['tip_angle'], 'margin': flat_rot['margin'], 'fallback': bool(flat_rot.get('fallback', False)),
                               'prob': flat_rot['prob'], 'gripper_min_z': float(min_z), 'wrist_min_z': wrist_z, 'yaw_offset': yaw_offset, 'yaw_rank': i})
    candidates.sort(key=lambda c: (c['fallback'], round(c['tilt'], 1), -c['tip_angle'], c['yaw_rank']))
    return candidates


LYING_MAX_AXIS_TILT = 45.0 # deg, point pairs for the lying part: the line between the two fingers at most this far from horizontal
LYING_MIN_HAND_UP = 0.2 # the hand axis of such a candidate points at least this much upwards (cosine of its tilt): not a hand that lies almost flat
LYING_PAD_END = 1.6 # cm from the fingertip: how deep a point pair may sit in the gripping pad (the pad ends at 1.765 cm)
LYING_PAD_MIDDLE = 0.9 # cm from the fingertip: the middle of the pad (the generator of the original pipeline puts the pair at the fingertip)


def lying_grasp_candidates(part_mesh, flat_rots, gripper_type, gripper_meshes, n_pairs, angles, antipodal_thres=0.95, clearance=GROUND_CLEARANCE):
    '''
    Grasp candidates made for the part lying on the board: the second direction of the matching between lying and assembled
    pose. The candidates of the original generator are made at the assembled pose and then tested on the lying part; these are
    made on the lying part and have to be tested at the assembled pose afterwards (reachable, free of collisions, insertion).
    A grasp is the pose of the gripper relative to the part, so one grasp serves both poses.

    For every resting pose of flat_rots (get_flat_rots; each gets an equal share of n_pairs antipodal point pairs, sampled on
    the lying part): pairs whose line is at most LYING_MAX_AXIS_TILT from horizontal. For each pair the hand comes from as
    straight above as the pair allows, and leaning by each of `angles` (deg) about the line between the fingers. The fingers
    reach down as far as the board allows (lowest point of the gripper `clearance` above it), but not deeper than the pair at
    LYING_PAD_END from the fingertip; on a tall face, where that is reached, the pair in the middle of the pad is a second
    candidate. (A pair that lies lower than the fingertips may go is kept: the pad can still cover the face above it.)
    Returns [(pos, quat, open_ratio, index of the resting pose)], the gripper pose in the frame of part_mesh.
    '''
    from planning.robot.geometry import get_gripper_open_ratio
    from planning.robot.util_grasp import compute_antipodal_pairs, get_gripper_pos_quat
    candidates = []
    if len(flat_rots) == 0 or n_pairs <= 0:
        return candidates
    budget = int(np.ceil(n_pairs / len(flat_rots)))
    for k, flat_rot in enumerate(flat_rots):
        to_rest = np.eye(4)
        to_rest[:3, :3] = flat_rot['rot']
        to_rest[2, 3] = -(part_mesh.vertices @ flat_rot['rot'].T)[:, 2].min() # lying on the board (z = 0)
        mesh_rest = part_mesh.copy().apply_transform(to_rest)
        to_part = np.linalg.inv(to_rest)
        # pairs anywhere on the lying part (no height limit: the pad can cover the face above a low pair); finger room as in the original
        pairs = compute_antipodal_pairs(mesh_rest, sample_budget=budget, antipodal_thres=antipodal_thres, ground_z=-2.0, max_rounds=20)
        for pair in pairs:
            open_ratio = get_gripper_open_ratio(gripper_type, pair)
            if open_ratio is None or open_ratio > 0.95: continue
            l2r = (pair[1] - pair[0]) / np.linalg.norm(pair[1] - pair[0])
            if abs(l2r[2]) > np.sin(np.radians(LYING_MAX_AXIS_TILT)): continue
            up = np.array([0, 0, 1.0]) - l2r[2] * l2r # the direction from the fingers to the hand that is closest to straight up
            up /= np.linalg.norm(up)
            for angle in angles:
                base = R.from_rotvec(np.radians(angle) * l2r).apply(up)
                if base[2] < LYING_MIN_HAND_UP: continue
                centre = pair.mean(axis=0)
                pos, quat = get_gripper_pos_quat(gripper_type, centre, base, l2r, open_ratio)
                low = np.inf
                for ratio in [open_ratio, min(open_ratio + RETRACT_OPEN_RATIO, 1.0)]:
                    meshes = transform_gripper_meshes(gripper_type, gripper_meshes, pos, quat, np.eye(4), ratio)
                    low = min(low, min(mesh.bounds[0, 2] for mesh in meshes.values()))
                # get_gripper_pos_quat puts the pair at the fingertip. Moving the gripper by s along the hand axis lifts every point
                # of it by s * base[2]; with s < 0 the pair sits -s behind the fingertip, in the pad
                deepest = max((clearance + 0.02 - low) / base[2], -LYING_PAD_END) # s of the lowest gripper that is allowed
                shifts = [deepest] + ([-LYING_PAD_MIDDLE] if deepest < -LYING_PAD_MIDDLE - 0.1 else [])
                for shift in shifts:
                    pos_shifted = pos + base * shift
                    grasp_mat = to_part @ get_transform_matrix_quat(pos_shifted, quat)
                    pos_part, quat_part = mat_to_pos_quat(grasp_mat)
                    candidates.append((pos_part, quat_part, float(open_ratio), k))
    return candidates


# ----------------------------------------------------------------------------- finger contact on a fixed grid of the pads

def finger_pads(gripper_type, gripper_meshes):
    '''
    The gripping pad of each finger: the rectangle spanned by the faces on the inner-most plane of the finger (the plane that
    faces the other finger). Returned in the frame of the finger mesh, so that it moves with the transforms of
    get_gripper_meshes_transforms: a list of {'name', 'origin' (on the pad plane, at the fingertip, in the middle of the finger),
    'along' (unit vector from the fingertip towards the hand), 'across', 'normal' (towards the other finger),
    'a': (start, end) of the pad along the finger [cm from the fingertip], 'b': (start, end) across}.
    None if the gripper has no two fingers with such a plane (then the contact is sampled as in the original pipeline).
    '''
    names = get_gripper_finger_names(gripper_type)
    if len(names) != 2 or any(name not in gripper_meshes for name in names):
        return None
    reference = get_gripper_meshes_transforms(gripper_type, gripper_meshes, np.zeros(3), np.array([1.0, 0, 0, 0]), np.eye(4), 0.5)
    base_direction, l2r_direction = [np.array(v, dtype=float) for v in get_gripper_basis_directions(gripper_type)]
    axis = -base_direction / np.linalg.norm(base_direction) # from the hand towards the fingertips
    world = {name: gripper_meshes[name].vertices @ reference[name][:3, :3].T + reference[name][:3, 3] for name in names}
    tip = max((vertices @ axis).max() for vertices in world.values())

    pads = []
    for name in names:
        other = names[1] if name == names[0] else names[0]
        inward = l2r_direction * np.sign((world[other].mean(axis=0) - world[name].mean(axis=0)) @ l2r_direction)
        across = np.cross(axis, inward)
        rot, trans = reference[name][:3, :3], reference[name][:3, 3]
        mesh = gripper_meshes[name]
        triangles = mesh.triangles @ rot.T + trans
        inner = (world[name] @ inward).max()
        on_pad = ((mesh.face_normals @ rot.T) @ inward > 0.99) & ((triangles @ inward).min(axis=1) > inner - PAD_PLANE_TOL)
        if not on_pad.any():
            return None
        corners = triangles[on_pad].reshape(-1, 3)
        a, b = tip - corners @ axis, corners @ across
        origin = inner * inward + tip * axis # the gripper frame of the reference state is the world frame
        pads.append({'name': name, 'origin': rot.T @ (origin - trans), 'along': rot.T @ (-axis), 'across': rot.T @ across, 'normal': rot.T @ inward,
                     'a': (float(a.min()), float(a.max())), 'b': (float(b.min()), float(b.max()))})
    return pads


class PartFaces:
    '''the triangles of a part mesh (in the pose where the grasps are defined) and their normals, for pad_contact'''

    def __init__(self, mesh):
        self.triangles = np.asarray(mesh.triangles, dtype=float)
        self.normals = np.asarray(mesh.face_normals, dtype=float)
        self.centers = self.triangles.mean(axis=1) # bounding sphere of every face: to skip the faces far from a pad
        self.radii = np.linalg.norm(self.triangles - self.centers[:, None], axis=2).max(axis=1)


def pad_surface(pad, finger_transform, part, closing=PAD_CLOSING, cell=PAD_CELL, chunk=200000):
    '''
    The part surface in front of one finger pad, on the pad grid: for every grid point (cell centre) the distance [cm] from the
    pad plane to the nearest part face that faces the pad (negative: the face lies behind the pad plane, the pad of the gripper
    state would be inside the part by that much), inf where no such face lies between PAD_PENETRATION behind and `closing` in
    front of the pad. Returns (height (n_a, n_b), frame of the pad in the frame of the part: origin, along, across, normal).
    '''
    rot = finger_transform[:3, :3]
    origin = rot @ pad['origin'] + finger_transform[:3, 3]
    along, across, normal = rot @ pad['along'], rot @ pad['across'], rot @ pad['normal']
    (a0, a1), (b0, b1) = pad['a'], pad['b']
    n_a, n_b = int(round((a1 - a0) / cell)), int(round((b1 - b0) / cell))
    height = np.full((n_a, n_b), np.inf)

    # faces that face the pad, cross the layer around it and project onto it (first a cheap test with the bounding spheres)
    low = lambda x: np.minimum(np.minimum(x[:, 0], x[:, 1]), x[:, 2]) # over the 3 corners (much faster than x.min(axis=1))
    high = lambda x: np.maximum(np.maximum(x[:, 0], x[:, 1]), x[:, 2])
    offset = part.centers - (origin + 0.5 * (a0 + a1) * along + 0.5 * (b0 + b1) * across)
    reach = 0.5 * np.hypot(a1 - a0, b1 - b0) + max(closing, PAD_PENETRATION)
    close = np.flatnonzero((np.einsum('ij,ij->i', offset, offset) <= (reach + part.radii) ** 2) & (part.normals @ normal < -1e-6))
    rel = part.triangles[close] - origin
    gap, a, b = rel @ normal, rel @ along, rel @ across # (faces, 3): distance in front of the pad, position on the pad
    a_low, a_high, b_low, b_high = low(a), high(a), low(b), high(b)
    near = (low(gap) <= closing) & (high(gap) >= -PAD_PENETRATION) & (a_high >= a0) & (a_low <= a1) & (b_high >= b0) & (b_low <= b1)
    gap, a, b, a_low, a_high, b_low, b_high = gap[near], a[near], b[near], a_low[near], a_high[near], b_low[near], b_high[near]
    # grid points (cell centres) inside the bounding box of the projection of each face
    i0 = np.maximum(np.ceil((a_low - a0) / cell - 0.5).astype(int), 0)
    i1 = np.minimum(np.floor((a_high - a0) / cell - 0.5).astype(int), n_a - 1)
    j0 = np.maximum(np.ceil((b_low - b0) / cell - 0.5).astype(int), 0)
    j1 = np.minimum(np.floor((b_high - b0) / cell - 0.5).astype(int), n_b - 1)
    det = (b[:, 1] - b[:, 2]) * (a[:, 0] - a[:, 2]) + (a[:, 2] - a[:, 1]) * (b[:, 0] - b[:, 2])
    seen = (i0 <= i1) & (j0 <= j1) & (np.abs(det) >= 1e-12) # not between the grid points, not edge-on
    gap, a, b, i0, j0, det = gap[seen], a[seen], b[seen], i0[seen], j0[seen], det[seen]
    n_i, n_j = (i1[seen] - i0 + 1), (j1[seen] - j0 + 1)
    count = n_i * n_j # grid points to test per face

    # all (face, grid point) pairs at once, in chunks of about `chunk` pairs
    ends = np.cumsum(count)
    start = 0
    while start < len(count):
        stop = max(int(np.searchsorted(ends, (ends[start - 1] if start > 0 else 0) + chunk, side='right')), start + 1)
        c = count[start:stop]
        face = np.repeat(np.arange(start, stop), c)
        k = np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c) # index of the grid point within the box of its face
        i, j = i0[face] + k // n_j[face], j0[face] + k % n_j[face]
        pa, pb = a0 + (i + 0.5) * cell, b0 + (j + 0.5) * cell
        ta, tb, g = a[face], b[face], gap[face]
        w0 = ((tb[:, 1] - tb[:, 2]) * (pa - ta[:, 2]) + (ta[:, 2] - ta[:, 1]) * (pb - tb[:, 2])) / det[face] # barycentric coordinates
        w1 = ((tb[:, 2] - tb[:, 0]) * (pa - ta[:, 2]) + (ta[:, 0] - ta[:, 2]) * (pb - tb[:, 2])) / det[face]
        w2 = 1 - w0 - w1
        h = w0 * g[:, 0] + w1 * g[:, 1] + w2 * g[:, 2] # distance of the face in front of the grid point
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9) & (h <= closing) & (h >= -PAD_PENETRATION)
        # the nearest face per grid point (sorted: np.minimum.at is slow, and an assignment with repeated indices is not defined)
        point, h = i[inside] * n_b + j[inside], h[inside]
        order = np.lexsort((h, point))
        point, h = point[order], h[order]
        first = np.ones(len(point), dtype=bool)
        first[1:] = point[1:] != point[:-1]
        flat = height.reshape(-1) # a view
        flat[point[first]] = np.minimum(flat[point[first]], h[first])
        start = stop
    return height, (origin, along, across, normal)


def pad_contact(pads, finger_transforms, part, closing=PAD_CLOSING, cell=PAD_CELL):
    '''
    Contact of the two finger pads with a part, on a fixed grid over each pad (cells of `cell` cm).
    A grid point is in contact if the part surface that faces the pad lies at most `closing` in front of it: the pad presses into
    the part when the fingers close by that much (the contact test of the original pipeline, there with random points on the
    whole finger). If part of the surface lies behind the pad plane (the fingers would stop there before they reach the opening
    of the grasp), the contact is counted from that first touching point: within `closing` of it.
    finger_transforms: mesh name -> 4x4 of the gripper state (get_gripper_meshes_transforms); part: PartFaces.
    Returns {'points': (n, 3) contact points on the pads (every PAD_STORE_CELL, plus the deepest one of each finger),
             'depths': per finger, how far the contact reaches from the fingertip along the finger [cm] (0: no contact),
             'area': contact area of both pads [cm^2]}
    '''
    points, depths, area = [], [], 0.0
    for pad in pads:
        height, (origin, along, across, normal) = pad_surface(pad, finger_transforms[pad['name']], part, closing, cell)
        if not np.isfinite(height).any():
            depths.append(0.0)
            continue
        level = min(float(height.min()), 0.0) + closing # the pad plane when the finger has closed onto the part
        rows, cols = np.nonzero(height <= level)
        a0, b0 = pad['a'][0], pad['b'][0]
        depths.append(float(a0 + (rows.max() + 0.5) * cell))
        area += len(rows) * cell ** 2
        step = max(int(round(PAD_STORE_CELL / cell)), 1)
        keep = (rows % step == step // 2) & (cols % step == step // 2) # a coarser grid is stored (scores use positions and counts)
        if not keep.any(): keep[:] = True # a contact strip narrower than the stored grid
        deepest = np.argmax(rows)
        stored = np.vstack([np.stack([rows[keep], cols[keep]], axis=1), [rows[deepest], cols[deepest]]])
        points.append(origin + level * normal + (a0 + (stored[:, :1] + 0.5) * cell) * along + (b0 + (stored[:, 1:] + 0.5) * cell) * across)
    return {'points': np.vstack(points) if len(points) > 0 else np.zeros((0, 3)), 'depths': depths, 'area': float(area)}


def set_pad_contact(grasp, pads, gripper_type, gripper_meshes, part):
    '''computes the pad contact of a grasp and stores it: contact_points (as before, but on the pad grid), contact_depths, contact_area'''
    transforms = get_gripper_meshes_transforms(gripper_type, gripper_meshes, grasp.pos, grasp.quat, np.eye(4), grasp.open_ratio)
    contact = pad_contact(pads, transforms, part)
    grasp.contact_points = contact['points']
    grasp.contact_depths = contact['depths']
    grasp.contact_area = contact['area']
    return grasp


# ------------------------------------------------------------------------------------------------ criteria of a move grasp

def move_grasp_criteria(grasp_move, part_mesh, flat_rots, gripper_type, gripper_meshes, arm_chain=None, max_tilt=MAX_TILT, clearance=GROUND_CLEARANCE):
    '''
    What a move grasp must meet in the flat pipeline, as {criterion: passed}. grasp_move: the grasp along the insertion path
    (list of Grasp, [0] = assembled pose). arm_chain: the move arm of the real robot (panda), None for other arms.
      walls        every arm pose of the grasp stays inside the virtual walls of franka-interface
      conditioned  every arm pose is away from singularities and joint limits (insertion values)
      overlap      the part reaches at least GRASP_MIN_FINGER_OVERLAP into the fingers
      flat         the grasp can pick the part lying flat in one of flat_rots
    '''
    from planning.plan_checks import grasp_finger_overlap
    from planning.robot.motion_plan_arm import VIRTUAL_WALL_SAFETY
    from planning.robot.panda_condition import joint_condition
    from planning.robot.util_arm import get_virtual_wall_margin
    criteria = {'walls': True, 'conditioned': True}
    if arm_chain is not None:
        criteria['walls'] = all(get_virtual_wall_margin(arm_chain, g.arm_q) >= VIRTUAL_WALL_SAFETY for g in grasp_move)
        conditions = [joint_condition(arm_chain.active_from_full(g.arm_q)) for g in grasp_move]
        criteria['conditioned'] = all(sigma >= MIN_SIGMA_INSERTION and margin >= MIN_JOINT_LIMIT_MARGIN for sigma, margin, _ in conditions)
    criteria['overlap'] = grasp_finger_overlap(grasp_move[0], gripper_type, gripper_meshes) >= GRASP_MIN_FINGER_OVERLAP
    criteria['flat'] = len(get_flat_pickup_candidates(grasp_move[0], part_mesh, flat_rots, gripper_type, gripper_meshes, max_tilt, clearance)) > 0
    return criteria


def flat_move_grasps(moves, part_mesh, gripper_type, gripper_meshes, arm_chain=None, max_tilt=MAX_TILT, clearance=GROUND_CLEARANCE, min_tip_angle=MIN_TIP_ANGLE, allow_fallback=True):
    '''
    The move grasps of a part that meet move_grasp_criteria, and the resting poses of the part they were tested with: the stable
    ones. If no grasp fits a stable pose and allow_fallback is set, the less stable resting poses are admitted for this part.
    moves: list of move grasps (each the list of Grasp along the insertion path). Returns (grasps kept, resting poses, fallback used).
    '''
    flat_rots = get_flat_rots(part_mesh, min_tip_angle)
    passes = lambda grasp_move, rots: all(move_grasp_criteria(grasp_move, part_mesh, rots, gripper_type, gripper_meshes, arm_chain, max_tilt, clearance).values())
    kept = [grasp_move for grasp_move in moves if passes(grasp_move, flat_rots)]
    if len(kept) == 0 and len(moves) > 0 and allow_fallback:
        rots_fallback = get_flat_rots(part_mesh, min_tip_angle, with_fallback=True)
        if len(rots_fallback) > len(flat_rots):
            kept_fallback = [grasp_move for grasp_move in moves if passes(grasp_move, rots_fallback)]
            if len(kept_fallback) > 0:
                return kept_fallback, rots_fallback, True
    return kept, flat_rots, False


# --------------------------------------------------------------------------------------------- rating and choice of grasps

def part_center_of_mass(mesh):
    '''centre of mass of a part mesh (the surface centroid if the mesh is not closed, where the volume integral is unreliable)'''
    return np.asarray(mesh.center_mass if mesh.is_watertight else mesh.centroid, dtype=float)


def insertion_score(grasp, G_preced):
    '''the move score of the original pipeline (torque of the insertion contact forces about the centre of mass; smaller is better)'''
    from planning.run_seq_opt import SequenceOptimizer
    if G_preced.nodes[grasp.part_id]['path'] is None or grasp.contact_points is None or len(grasp.contact_points) == 0:
        return 0.0
    return float(SequenceOptimizer._get_move_grasp_score(types.SimpleNamespace(G_preced=G_preced), grasp))


def insertion_ranks(scores):
    '''
    Rank in [0, 1] of every grasp of a part by its insertion score (0: the best, i.e. smallest score; 1: the worst).
    Equal scores share a rank, so grasps that the score does not tell apart are not ordered by it.
    '''
    scores = np.asarray(scores, dtype=float)
    smaller = np.searchsorted(np.sort(scores), scores, side='left') # number of grasps with a strictly smaller score
    return smaller / max(len(scores) - 1, 1)


def rate_flat_grasp(grasp, candidate, center_of_mass, gripper_type, insertion_rank=0.0):
    '''
    Rating in [0, 1] of a grasp that can pick its part lying flat (larger is better), from four terms:
      grip       how much of the pads holds the part: depth of the weaker finger and contact area
      balance    the centre of mass is close to the line between the fingers when the part is lifted (little torque about it)
      upright    the gripper comes from straight above
      insertion  rank of the grasp by the move score of the original pipeline (0 = best of the part's grasps, 1 = worst)
    candidate: the lying pose (from get_flat_pickup_candidates) the grasp would use. Returns {'rating', 'grip', 'balance', 'upright', 'insertion'}.
    '''
    depths = getattr(grasp, 'contact_depths', None) or [0.0]
    area = getattr(grasp, 'contact_area', None) or 0.0
    grip = 0.5 * min(min(depths) / FLAT_RATING_DEPTH, 1.0) + 0.5 * min(area / FLAT_RATING_AREA, 1.0)

    info = get_grasp_info_from_gripper_state(gripper_type, grasp.pos, grasp.quat, grasp.open_ratio)
    offset = candidate['rot'] @ (info['grasp_center'] - center_of_mass) # from the centre of mass to the grasp centre, part lying
    closing = candidate['rot'] @ info['l2r_direction']
    lever = abs(offset[0] * closing[1] - offset[1] * closing[0]) / max(np.linalg.norm(closing[:2]), 1e-9) # horizontal distance of the centre of mass from the line between the fingers
    balance = 1.0 - min(lever / FLAT_RATING_LEVER, 1.0)

    upright = 1.0 - min(candidate['tilt'] / RATING_TILT, 1.0)
    terms = {'grip': float(grip), 'balance': float(balance), 'upright': float(upright), 'insertion': float(1.0 - insertion_rank)}
    terms['rating'] = float(sum(FLAT_RATING_WEIGHTS[k] * terms[k] for k in FLAT_RATING_WEIGHTS))
    return terms


def holds_with_room(holds, chosen, other_parts, min_count):
    '''
    Hold grasps to keep in addition to those already chosen, so that every other part can be inserted while this part is held.
    A hold grasp leaves room for a part if its gripper is not in the way of that part's insertion (the part is not in
    parts_in_collision_hold['move']). For every part of other_parts: if fewer than min_count of the chosen hold grasps leave
    room for it, hold grasps that do are added until min_count is reached or none is left; those first whose gripper is in the
    way of the fewest parts. holds: all feasible hold grasps of the part; chosen: the ones kept so far.
    Returns the grasps to add, in the order of holds.
    '''
    blocked = lambda hold: set(getattr(hold, 'parts_in_collision_hold', None)['move']) if getattr(hold, 'parts_in_collision_hold', None) else set()
    kept = {hold.grasp_id for hold in chosen}
    added = set()
    for part in other_parts:
        free = [hold for hold in holds if part not in blocked(hold)]
        count = sum(hold.grasp_id in kept or hold.grasp_id in added for hold in free)
        for hold in sorted([hold for hold in free if hold.grasp_id not in kept and hold.grasp_id not in added], key=lambda hold: (len(blocked(hold)), hold.grasp_id)):
            if count >= min_count: break
            added.add(hold.grasp_id)
            count += 1
    return [hold for hold in holds if hold.grasp_id in added]


def approach_direction(grasp):
    '''unit vector along which the gripper approaches the part (from the hand towards the fingertips), frame of the grasp'''
    return R.from_quat(np.asarray(grasp.quat)[[1, 2, 3, 0]]).apply([0.0, 0.0, 1.0])


def select_grasps(ratings, directions, n):
    '''
    Indices of at most n grasps: the best rated, spread over the approach directions. The grasps are grouped by direction
    (26 groups: along an axis, a face diagonal or a space diagonal); the groups take turns, best group first, each giving its
    best remaining grasp. Deterministic (ties are broken by the index).
    '''
    ratings = np.asarray(ratings, dtype=float)
    if n is None or len(ratings) <= n:
        return sorted(range(len(ratings)), key=lambda i: (-ratings[i], i))
    groups = {}
    for i, direction in enumerate(directions):
        d = np.asarray(direction, dtype=float)
        key = tuple(np.round(d / max(np.abs(d).max(), 1e-9)).astype(int)) # nearest of the 26 directions with components -1, 0, 1
        groups.setdefault(key, []).append(i)
    queues = [sorted(members, key=lambda i: (-ratings[i], i)) for members in groups.values()]
    queues.sort(key=lambda q: (-ratings[q[0]], q[0]))
    chosen = []
    while len(chosen) < n:
        for queue in queues:
            if len(queue) > 0 and len(chosen) < n:
                chosen.append(queue.pop(0))
    return chosen
