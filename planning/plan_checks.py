'''
Checks of what the real (move) arm executes, shared by run_pickup_gen.py (pickup poses), run_motion_plan.py (re-plan a path
that fails) and verify_plan.py (final check). Exact geometry, no planner buffers.

Rules (clearances in planning/config.py):
  arm links 2-7 vs every part (lying or assembled, except the carried one) >= ROBOT_PART_CLEARANCE
  gripper vs lying parts >= ROBOT_PART_CLEARANCE, except the part being picked (no penetration, contact allowed near its grasp)
  gripper vs assembled parts: no penetration, except the part just released (contact allowed near its release)
  carried part vs lying parts >= ROBOT_PART_CLEARANCE, vs assembled parts no penetration (not checked while inserting)
  carried part above the table: never below it; outside the assembly zone, once lifted, >= TRANSFER_PART_CLEARANCE
  gripper >= GRIPPER_TABLE_CLEARANCE, arm links >= ROBOT_PART_CLEARANCE above the table
  franka-interface virtual walls >= VIRTUAL_WALL_SAFETY
  every move grasp: finger overlap with the part >= GRASP_MIN_FINGER_OVERLAP (see grasp_finger_overlap)
  every pose: smallest singular value of the arm Jacobian >= MIN_SIGMA (MIN_SIGMA_INSERTION while inserting),
              every joint >= MIN_JOINT_LIMIT_MARGIN from its limits
'''
import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from assets.transform import get_transform_matrix_quat
from planning.config import ROBOT_PART_CLEARANCE, GRIPPER_TABLE_CLEARANCE, TRANSFER_PART_CLEARANCE, GRASP_MIN_FINGER_OVERLAP, CONTACT_RADIUS, \
    MIN_SIGMA, MIN_SIGMA_INSERTION, MIN_JOINT_LIMIT_MARGIN
from planning.robot.geometry import load_gripper_meshes, transform_gripper_meshes, get_gripper_finger_names
from planning.robot.motion_plan_arm import ArmMotionPlanner, VIRTUAL_WALL_SAFETY
from planning.robot.util_arm import get_arm_chain, get_gripper_pos_quat_from_arm_q, get_virtual_wall_margin
from planning.robot.workcell import get_panda_base_z, get_panda_desk_z

PANDA_JOINT_LIMITS = (np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
                      np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]))
PANDA_DH = [(0, 0, 0.333), (0, -np.pi / 2, 0), (0, np.pi / 2, 0.316), (0.0825, np.pi / 2, 0), (-0.0825, -np.pi / 2, 0.384), (0, np.pi / 2, 0), (0.088, np.pi / 2, 0)]


def panda_jacobian(q):
    '''6x7 geometric Jacobian (m, base frame) of the panda end effector (flange + hand), q: 7 joint angles'''
    def dh(a, alpha, d, theta):
        ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
        return np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1]])
    T, frames = np.eye(4), []
    for (a, alpha, d), theta in zip(PANDA_DH, q):
        T = T @ dh(a, alpha, d, theta)
        frames.append(T.copy())
    p = (T @ dh(0, 0, 0.107 + 0.1034, 0))[:3, 3]
    return np.array([np.concatenate([np.cross(F[:3, 2], p - F[:3, 3]), F[:3, 2]]) for F in frames]).T


def joint_condition(q):
    '''(smallest singular value of the Jacobian, smallest distance of any joint to its limits [rad], that joint 1..7)'''
    q = np.asarray(q)[:7]
    sigma = np.linalg.svd(panda_jacobian(q), compute_uv=False).min()
    margins = np.minimum(q - PANDA_JOINT_LIMITS[0], PANDA_JOINT_LIMITS[1] - q)
    return sigma, margins.min(), int(np.argmin(margins)) + 1


ARM_LINKS_CHECKED = ['panda_link2', 'panda_link3', 'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7']


def grasp_finger_overlap(grasp, gripper_type, gripper_meshes=None):
    '''
    How far (cm) the part reaches into the fingers, measured from the fingertip along the finger, from the grasp's contact
    points (final pose). With the gripper e cm too high the overlap shrinks by e, so this is the tolerance of the grasp.
    '''
    if grasp.contact_points is None or len(grasp.contact_points) == 0:
        return 0.0
    if gripper_meshes is None:
        gripper_meshes = load_gripper_meshes(gripper_type, os.path.join(project_base_dir, 'assets'))
    meshes = transform_gripper_meshes(gripper_type, gripper_meshes, grasp.pos, grasp.quat, np.eye(4), grasp.open_ratio)
    fingers = np.vstack([meshes[name].vertices for name in get_gripper_finger_names(gripper_type)])
    finger_dirs = np.array(fingers.mean(axis=0) - meshes[[n for n in meshes if n not in get_gripper_finger_names(gripper_type)][0]].vertices.mean(axis=0))
    axis = finger_dirs / np.linalg.norm(finger_dirs) # from the hand towards the fingertips
    tip = (fingers @ axis).max()
    return float(tip - (np.array(grasp.contact_points) @ axis).min())


class SceneChecker:
    '''Exact clearance checks of the move arm in the real workcell'''

    def __init__(self, arm_type, gripper_type, has_ft_sensor, part_meshes_raw, pickup_mats, final_meshes):
        assert arm_type == 'panda', 'the real-robot checks are written for the panda workcell'
        self.chain = get_arm_chain(arm_type, 'move')
        self.planner = ArmMotionPlanner(self.chain, gripper_type, has_ft_sensor)
        self.gripper_type = gripper_type
        self.has_ft_sensor = has_ft_sensor
        self.table_z = get_panda_base_z() + get_panda_desk_z()
        self.part_ids = list(part_meshes_raw.keys())
        self.pickup_meshes = {k: part_meshes_raw[k].copy().apply_transform(pickup_mats[k]) for k in self.part_ids}
        self.final_meshes = final_meshes
        bounds = np.array([mesh.bounds for mesh in final_meshes.values()])
        self.zone = (bounds[:, 0, :2].min(axis=0) - TRANSFER_PART_CLEARANCE, bounds[:, 1, :2].max(axis=0) + TRANSFER_PART_CLEARANCE)
        self.arm_mgr = trimesh.collision.CollisionManager()
        for name in ARM_LINKS_CHECKED:
            self.arm_mgr.add_object(name, self.planner.arm_meshes[name])
        self.gripper_mgr = trimesh.collision.CollisionManager()
        for name, mesh in self.planner.gripper_meshes.items():
            self.gripper_mgr.add_object(name, mesh)
        self.arm_hull = {name: self.planner.arm_meshes[name].convex_hull.vertices for name in ARM_LINKS_CHECKED}
        self.gripper_hull = {name: mesh.convex_hull.vertices for name, mesh in self.planner.gripper_meshes.items()}

    def set_pickups(self, part_meshes_raw, pickup_mats):
        self.pickup_meshes = {k: part_meshes_raw[k].copy().apply_transform(pickup_mats[k]) for k in pickup_mats}

    def part_mesh(self, part_id, where):
        return self.final_meshes[part_id] if where == 'final' else self.pickup_meshes[part_id]

    def manager(self, meshes):
        mgr = trimesh.collision.CollisionManager()
        for name, mesh in meshes.items():
            mgr.add_object(name, mesh)
        return mgr

    def gripper_transform(self, q):
        return get_transform_matrix_quat(*get_gripper_pos_quat_from_arm_q(self.chain, q, self.gripper_type, has_ft_sensor=self.has_ft_sensor))

    def check_path(self, path, open_ratio, state, carried=None, pick_part=None, released_part=None, inserting=False, label=''):
        '''
        path: list of full (9) or active (7) arm qs of the move arm
        state: {part_id: 'pickup' (lying) / 'final' (assembled)} before the path, without the carried part
        carried: part in the gripper (it lies at its pickup pose when the path starts, unless inserting)
        pick_part: part picked at the end of this path (lying); released_part: part released at its start (assembled)
        returns a list of violation strings (empty: ok)
        '''
        path = [np.asarray(q) for q in path]
        path = [self.chain.active_to_full(q) if len(q) == 7 else q for q in path]
        lying = {k for k, v in state.items() if v == 'pickup' and k != carried}
        assembled = {k for k, v in state.items() if v == 'final' and k != carried}
        all_parts = self.manager({k: self.part_mesh(k, state[k]) for k in lying | assembled})
        lying_strict = self.manager({k: self.pickup_meshes[k] for k in lying if k != pick_part})
        assembled_mgr = self.manager({k: self.final_meshes[k] for k in assembled if k != released_part})
        pick_mgr = self.manager({pick_part: self.pickup_meshes[pick_part]}) if pick_part is not None else None
        released_mgr = self.manager({released_part: self.final_meshes[released_part]}) if released_part is not None else None
        if carried is not None:
            # the carried part follows the gripper: transport paths start at its pickup pose, insertion paths end at its final pose
            carried_mesh = self.final_meshes[carried] if inserting else self.pickup_meshes[carried]
            T_grip_ref = self.gripper_transform(path[-1] if inserting else path[0])
            carried_mgr = self.manager({'carried': carried_mesh})
            carried_hull = carried_mesh.convex_hull.vertices
        T_grip_start, T_grip_end = self.gripper_transform(path[0]), self.gripper_transform(path[-1])

        violations = {}
        def add(key, k):
            violations.setdefault(key, []).append(k)

        lifted = False
        for k, q in enumerate(path):
            arm_tf, grip_tf, _ = self.planner.get_meshes_transforms(q, open_ratio, None, None)
            for name in ARM_LINKS_CHECKED: self.arm_mgr.set_transform(name, arm_tf[name])
            for name, tf in grip_tf.items(): self.gripper_mgr.set_transform(name, tf)
            T_grip = self.gripper_transform(q)
            near_start = np.linalg.norm(T_grip[:3, 3] - T_grip_start[:3, 3]) < CONTACT_RADIUS
            near_end = np.linalg.norm(T_grip[:3, 3] - T_grip_end[:3, 3]) < CONTACT_RADIUS

            # table
            z_grip = min((verts @ grip_tf[n][:3, :3].T + grip_tf[n][:3, 3])[:, 2].min() for n, verts in self.gripper_hull.items())
            z_arm = min((verts @ arm_tf[n][:3, :3].T + arm_tf[n][:3, 3])[:, 2].min() for n, verts in self.arm_hull.items())
            if z_grip < self.table_z + GRIPPER_TABLE_CLEARANCE: add(f'gripper closer than {GRIPPER_TABLE_CLEARANCE} cm to the table', k)
            if z_arm < self.table_z + ROBOT_PART_CLEARANCE: add(f'arm closer than {ROBOT_PART_CLEARANCE} cm to the table', k)
            # walls
            if get_virtual_wall_margin(self.chain, q) < VIRTUAL_WALL_SAFETY: add(f'closer than {VIRTUAL_WALL_SAFETY} cm to a virtual wall', k)
            # singularities and joint limits (all joints)
            sigma, margin, joint = joint_condition(self.chain.active_from_full(q))
            min_sigma = MIN_SIGMA_INSERTION if inserting else MIN_SIGMA
            if sigma < min_sigma: add(f'too close to a singularity (Jacobian sigma_min < {min_sigma})', k)
            if margin < MIN_JOINT_LIMIT_MARGIN: add(f'joint {joint} closer than {MIN_JOINT_LIMIT_MARGIN} rad to its limit', k)
            # arm links vs parts
            if len(lying | assembled) > 0:
                d, names = self.arm_mgr.min_distance_other(all_parts, return_names=True)
                if d < ROBOT_PART_CLEARANCE: add(f'arm {names[0]} closer than {ROBOT_PART_CLEARANCE} cm to part {names[1]}', k)
            # gripper vs lying parts
            if len(lying_strict._objs) > 0:
                d, names = self.gripper_mgr.min_distance_other(lying_strict, return_names=True)
                if d < ROBOT_PART_CLEARANCE: add(f'gripper {names[0]} closer than {ROBOT_PART_CLEARANCE} cm to lying part {names[1]}', k)
            if pick_mgr is not None and not near_end and self.gripper_mgr.in_collision_other(pick_mgr):
                add(f'gripper hits part {pick_part} (to be picked) away from its grasp', k)
            # gripper vs assembled parts
            if len(assembled_mgr._objs) > 0:
                hit, names = self.gripper_mgr.in_collision_other(assembled_mgr, return_names=True)
                if hit: add(f'gripper hits assembled part {sorted(names)[0][1]}', k)
            if released_mgr is not None and not near_start and self.gripper_mgr.in_collision_other(released_mgr):
                add(f'gripper hits part {released_part} (just released) away from its release', k)
            # carried part
            if carried is not None:
                T_part = T_grip @ np.linalg.inv(T_grip_ref)
                carried_mgr.set_transform('carried', T_part)
                v = carried_hull @ T_part[:3, :3].T + T_part[:3, 3]
                if v[:, 2].min() < self.table_z - 0.05: add(f'carried part {carried} below the table', k)
                outside = ~np.all((v[:, :2] >= self.zone[0]) & (v[:, :2] <= self.zone[1]), axis=1)
                if not inserting:
                    low = v[outside, 2].min() - self.table_z if outside.any() else np.inf
                    if low >= TRANSFER_PART_CLEARANCE - 1e-3: lifted = True
                    elif lifted and outside.any(): add(f'carried part {carried} lower than {TRANSFER_PART_CLEARANCE} cm above the table outside the assembly zone', k)
                    if len(lying_strict._objs) > 0:
                        d, names = carried_mgr.min_distance_other(lying_strict, return_names=True)
                        if d < ROBOT_PART_CLEARANCE and not near_start: add(f'carried part {carried} closer than {ROBOT_PART_CLEARANCE} cm to lying part {names[1]}', k)
                    if len(assembled_mgr._objs) > 0 and carried_mgr.in_collision_other(assembled_mgr):
                        add(f'carried part {carried} hits the assembly', k)
        return [f'{label}{key} (poses {ks[0]}..{ks[-1]} of {len(path)}, {len(ks)}x)' for key, ks in violations.items()]

    def check_pickup(self, q, open_ratio, part_id, state):
        '''robot at the pickup pose of part_id (lying), state as in check_path: clearance to all other parts'''
        return self.check_path([q], open_ratio, state, pick_part=part_id, label=f'pickup of part {part_id}: ')


FAILURE_FILE = 'planning_failure.json'


def clear_failure(log_dir):
    path = os.path.join(log_dir, FAILURE_FILE)
    if os.path.exists(path): os.remove(path)


def write_failure(log_dir, stage, part, grasp_id, reason):
    '''tells run_planning_flat.py which move grasp made planning fail, so it can try another sequence without it'''
    import json
    with open(os.path.join(log_dir, FAILURE_FILE), 'w') as fp:
        json.dump({'stage': stage, 'part': part, 'grasp_id': None if grasp_id is None else int(grasp_id), 'reason': reason}, fp, indent=4)
    print(f'[{stage}] FAILED for part {part} (move grasp {grasp_id}): {reason}')
