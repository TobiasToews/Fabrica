import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(project_base_dir)

import gc
import numpy as np
import trimesh
import fcl
import os
from scipy.spatial.transform import Rotation as R, Slerp

from assets.transform import get_transform_matrix_quat
from planning.path.rrt_connect import rrt_connect
from planning.path.rrt_star import rrt_star
from planning.path.meta import direct_path
from planning.path.smoothing import smooth_path
from planning.robot.util_arm import get_gripper_pos_quat_from_arm_q, get_ik_target_orientation, get_gripper_basis_directions, get_virtual_wall_margin
from planning.robot.geometry import load_arm_meshes, load_gripper_meshes, transform_arm_meshes, transform_gripper_meshes, get_arm_meshes_transforms, get_gripper_meshes_transforms, get_buffered_arm_meshes, get_buffered_gripper_meshes, get_buffered_meshes, offset_mesh, get_gripper_grasp_base_offset
from utils.common import TimeStamp
from planning.config import TRANSFER_LIFT_DELTA, ROBOT_PART_CLEARANCE, ASSEMBLY_CLEARANCE, MIN_SIGMA, MIN_JOINT_LIMIT_MARGIN, \
    PLANNER_MARGIN, PLANNER_JOINT_MARGIN, PLANNER_SIGMA_MARGIN, STRAIGHT_STEP, STRAIGHT_MAX, STRAIGHT_MAX_JOINT_STEP
from planning.robot.panda_condition import PANDA_JOINT_LIMITS, panda_jacobian, joint_limit_margin


DUAL_ARM_COL_CHECK = True
SELF_ARM_INTERSECT_CHECK = True
MAX_LIN_SPEED = 5 # cm/s
MAX_ANG_SPEED = 1 # rad/s
N_IK_RESTART = 20 # chain's default is 3
MAX_RETRACT_JOINT_STEP = 0.5 # rad, largest joint change between two retract poses
VIRTUAL_WALL_SAFETY = 1.0 # cm, extra distance to franka-interface's virtual walls (tracking error)
# what the collision test of a pose can find (collision_fn.hit), by who causes it: the arm and the gripper (their pose follows from
# the grasp), or the part in the gripper (its pose at an insertion start or at its pickup spot does not depend on the grasp)
HITS_ARM = ['joint limit', 'singularity', 'virtual wall', 'arm and ground', 'gripper and ground', 'arm and gripper', 'arm and part', 'arm and move part',
            'gripper and part', 'arm self-intersection', 'dual-arm']
HITS_CARRIED_PART = ['carried part too close to the table', 'move part and ground', 'move part and part']


def interpolate_angles(start_angle, end_angle, num_steps, bidirectional):
    if bidirectional: # if joint angle can take [-2pi, 2pi]
        angular_distance = (end_angle - start_angle) % (2 * np.pi)
        if angular_distance > np.pi:
            angular_distance -= 2 * np.pi
        step_size = angular_distance / num_steps
        interpolated_angles = []
        current_angle = start_angle
        for _ in range(num_steps + 1):  # including the start and end angles
            interpolated_angles.append(current_angle)
            current_angle = (current_angle + step_size) % (2 * np.pi)
    else:
        interpolated_angles = np.linspace(start_angle, end_angle, num_steps + 1)
    return np.array(interpolated_angles)


def interpolate_qs(arm_type, start_q, end_q, num_steps):
    interpolated_qs = np.zeros((num_steps + 1, len(start_q)))
    for i in range(len(start_q)):
        if arm_type == 'xarm7':
            bidirectional = i in [0, 2, 4, 6]
        elif arm_type == 'panda':
            bidirectional = False
        elif arm_type == 'ur5e':
            bidirectional = i in [0, 1, 3, 4, 5]
        else:
            raise NotImplementedError
        interpolated_qs[:, i] = interpolate_angles(start_q[i], end_q[i], num_steps, bidirectional)
    return np.array(interpolated_qs)


def interpolate_arm_path(arm_type, original_path, distance_fn, max_lin_speed=MAX_LIN_SPEED, max_ang_speed=MAX_ANG_SPEED):
    fps = 30
    segment_lin_dist = max_lin_speed / fps
    segment_ang_dist = max_ang_speed / fps
    interpolated_path = np.array([original_path[0]])

    for i in range(1, len(original_path)):
        lin_dist_i = distance_fn(original_path[i - 1], original_path[i])
        lin_interp_steps = int(lin_dist_i // segment_lin_dist)
        ang_diff_i = np.abs(original_path[i] - original_path[i - 1])
        ang_diff_i = np.minimum(ang_diff_i, 2 * np.pi - ang_diff_i)
        ang_dist_i = np.max(ang_diff_i)
        ang_interp_steps = int(ang_dist_i // segment_ang_dist)
        interp_steps = max(lin_interp_steps, ang_interp_steps)
        if interp_steps > 1:
            interp_qs = interpolate_qs(arm_type, original_path[i - 1], original_path[i], interp_steps)[1:] # ignore start
            interpolated_path = np.vstack([interpolated_path, interp_qs])
        else:
            interpolated_path = np.vstack([interpolated_path, original_path[i]])
    return interpolated_path


def interpolate_transformations(T1, T2, num_steps):
    # Extract rotation and translation components
    R1, t1 = T1[:3, :3], T1[:3, 3]
    R2, t2 = T2[:3, :3], T2[:3, 3]
    
    # Convert rotation matrices to quaternions
    rot1 = R.from_matrix(R1)
    rot2 = R.from_matrix(R2)
    
    # Define the SLERP object
    slerp = Slerp([0, 1], R.from_quat([rot1.as_quat(), rot2.as_quat()]))
    
    # Generate interpolation points
    t_vals = np.linspace(0, 1, num_steps + 1)  # N+1 points including start and end
    interpolated_rots = slerp(t_vals)  # SLERP interpolation
    
    # Interpolate translation vectors linearly
    interpolated_translations = [(1 - alpha) * t1 + alpha * t2 for alpha in t_vals]
    
    # Combine into transformation matrices
    interpolated_transformations = []
    for R_i, t_i in zip(interpolated_rots.as_matrix(), interpolated_translations):
        T_i = np.eye(4)
        T_i[:3, :3] = R_i
        T_i[:3, 3] = t_i
        interpolated_transformations.append(T_i)
    
    return interpolated_transformations


class ArmMotionPlanner: # NOTE: all input/output q is full unless specified active

    VISUALIZE = True # debug viewers on planning warnings; run_motion_plan.py --no-viz turns them off

    def __init__(self, arm_chain, gripper_type, has_ft_sensor=False, arm_box=None, optimizer=None, stamp=None, virtual_walls=False, table_z=None, part_clearance=None, clearance_exempt_xy=None):
        self.arm_chain = arm_chain
        self.arm_type = arm_chain.arm_type
        self.arm_rest_q_active = arm_chain.rest_q
        # self.step_size = 5
        self.step_size = 0.5 # TODO: check
        self.min_num_steps = 10
        asset_folder = os.path.join(project_base_dir, 'assets')
        self.arm_meshes = load_arm_meshes(self.arm_type, asset_folder, visual=False, convex=True)
        self.arm_meshes_buffered = get_buffered_arm_meshes(self.arm_meshes)
        self.gripper_type = gripper_type
        self.has_ft_sensor = has_ft_sensor
        self.gripper_meshes = load_gripper_meshes(gripper_type, asset_folder, has_ft_sensor=has_ft_sensor, visual=False) # NOTE: has_ft_sensor may have issue if other arm is different
        self.gripper_meshes_buffered = get_buffered_gripper_meshes(self.gripper_type, self.gripper_meshes)
        self.arm_box = arm_box
        self.virtual_walls = virtual_walls # stay inside franka-interface's virtual walls (real robot arm, panda only)
        self.table_z = table_z # world z of the real table surface (cm); with part_clearance, a carried part stays this far above it during transport
        self.part_clearance = part_clearance
        self.clearance_exempt_xy = clearance_exempt_xy # (xy lower, xy upper) around the assembly where carried parts may come low (insertion approach)
        # strict: plan for the real arm so that the paths pass planning/plan_checks.py (panda only). Every pose keeps the joint-limit,
        # singularity and wall rules; a path is [straight move out] + [free path that keeps the clearances to all parts] + [straight
        # move in]; the straight moves only have to be contact free and end where the clearances hold; nothing falls back silently.
        self.strict = False
        self.last_failure = None # strict: why the last plan_path* call returned None ('start', 'goal' or 'path')
        self.last_block = None # strict: what stopped the last straight move, see plan_path_straight_out
        self._buffer_cache = {} # (id(mesh), buffer) -> (mesh, inflated mesh); the mesh is kept so that its id stays unique
        self._self_pairs = None # pairs of arm links that are tested against each other, see arm_self_collision
        self.optimizer = optimizer
        if stamp is None:
            self.stamp = TimeStamp()
        else:
            self.stamp = stamp

        self.gripper_col_manager = trimesh.collision.CollisionManager()
        for name, mesh in self.gripper_meshes.items():
            self.gripper_col_manager.add_object(name, mesh)

        self.arm_col_manager_buffered = trimesh.collision.CollisionManager()
        self.gripper_col_manager_buffered = trimesh.collision.CollisionManager()
        self.other_col_manager_buffered = trimesh.collision.CollisionManager()
        for name, mesh in self.arm_meshes_buffered.items():
            self.arm_col_manager_buffered.add_object(name, mesh)
            self.other_col_manager_buffered.add_object(name, mesh)
        for name, mesh in self.gripper_meshes_buffered.items():
            self.gripper_col_manager_buffered.add_object(name, mesh)
            self.other_col_manager_buffered.add_object(name, mesh)
        
        self.ground_col_manager = trimesh.collision.CollisionManager()
        self.ground_col_manager_buffered = trimesh.collision.CollisionManager()
        self.ground_mesh = trimesh.creation.box((100, 100, 0.4)) # NOTE: 0.2cm safety
        self.ground_mesh_buffered = get_buffered_meshes(self.ground_mesh, buffer=1.0)
        self.ground_col_manager.add_object('ground', self.ground_mesh)
        self.ground_col_manager_buffered.add_object('ground', self.ground_mesh_buffered)
        self.box_mesh = None
        if self.arm_box is not None:
            box_lower, box_upper = self.arm_box
            box_inner_mesh = trimesh.creation.box(bounds=np.vstack([box_lower - 0.1, box_upper]))
            box_outer_mesh = trimesh.creation.box(bounds=np.vstack([box_lower - 1.0, box_upper + 1.0]))
            self.box_mesh = box_outer_mesh.difference(box_inner_mesh)
            self.ground_col_manager.add_object('box', self.box_mesh)
            self.ground_col_manager_buffered.add_object('box', self.box_mesh)

            # arm_meshes = transform_arm_meshes(self.arm_meshes, self.arm_chain, self.arm_chain.active_to_full(self.arm_chain.rest_q))
            # trimesh.Scene([self.box_mesh, *arm_meshes.values()]).show()

    def transform_meshes(self, q, open_ratio, move_pickup_mesh=None, gripper_pickup_transform=None, arm_chain=None, has_ft_sensor=None, buffered=False):
        if arm_chain is None: arm_chain = self.arm_chain
        if has_ft_sensor is None: has_ft_sensor = self.has_ft_sensor

        # calculate arm transform
        arm_meshes_i = transform_arm_meshes(self.arm_meshes if not buffered else self.arm_meshes_buffered, arm_chain, q)

        # calculate gripper transform
        gripper_pos, gripper_quat = get_gripper_pos_quat_from_arm_q(arm_chain, q, self.gripper_type, has_ft_sensor=has_ft_sensor)
        gripper_meshes_i = transform_gripper_meshes(self.gripper_type, self.gripper_meshes if not buffered else self.gripper_meshes_buffered, gripper_pos, gripper_quat, np.eye(4), open_ratio)

        # calculate move part transform
        if move_pickup_mesh is None:
            move_mesh_i = None
        else:
            gripper_transform = get_transform_matrix_quat(gripper_pos, gripper_quat)
            move_transform = gripper_transform @ np.linalg.inv(gripper_pickup_transform)
            move_mesh_i = move_pickup_mesh.copy()
            move_mesh_i.apply_transform(move_transform)

        return arm_meshes_i, gripper_meshes_i, move_mesh_i

    def get_meshes_transforms(self, q, open_ratio, move_pickup_mesh, gripper_pickup_transform):
        arm_meshes_transforms = get_arm_meshes_transforms(self.arm_meshes, self.arm_chain, q)
        gripper_pos, gripper_quat = get_gripper_pos_quat_from_arm_q(self.arm_chain, q, self.gripper_type, has_ft_sensor=self.has_ft_sensor)
        gripper_meshes_transforms = get_gripper_meshes_transforms(self.gripper_type, self.gripper_meshes, gripper_pos, gripper_quat, np.eye(4), open_ratio)
        if move_pickup_mesh is None:
            move_transform_global = None
        else:
            gripper_transform = get_transform_matrix_quat(gripper_pos, gripper_quat)
            move_transform_global = gripper_transform @ np.linalg.inv(gripper_pickup_transform)
        return arm_meshes_transforms, gripper_meshes_transforms, move_transform_global

    def get_meshes_transforms_active(self, q, open_ratio, move_pickup_mesh, gripper_pickup_transform):
        return self.get_meshes_transforms(self.arm_chain.active_to_full(q), open_ratio, move_pickup_mesh, gripper_pickup_transform)

    def get_meshes_transforms_other(self, arm_chain_other, q_other, open_ratio_other, has_ft_sensor_other):
        arm_meshes_transforms = get_arm_meshes_transforms(self.arm_meshes, arm_chain_other, q_other)
        gripper_pos, gripper_quat = get_gripper_pos_quat_from_arm_q(arm_chain_other, q_other, self.gripper_type, has_ft_sensor=has_ft_sensor_other)
        gripper_meshes_transforms = get_gripper_meshes_transforms(self.gripper_type, self.gripper_meshes, gripper_pos, gripper_quat, np.eye(4), open_ratio_other)
        return arm_meshes_transforms, gripper_meshes_transforms

    def apply_transforms_to_col_manager(self, col_manager, transforms):
        for name, mesh in transforms.items():
            col_manager.set_transform(name, mesh)

    def arm_self_collision(self):
        '''
        True if two links of the (inflated) arm that may collide intersect, at the transforms last set on arm_col_manager_buffered.
        Same result as in_collision_internal() filtered by check_colliding_links(), but only those link pairs are tested and each
        test stops at the first contact (neighbouring inflated links always overlap and produced hundreds of contacts per call).
        '''
        if self._self_pairs is None:
            names = list(self.arm_col_manager_buffered._objs)
            self._self_pairs = [(self.arm_col_manager_buffered._objs[a]['obj'], self.arm_col_manager_buffered._objs[b]['obj'])
                                for i, a in enumerate(names) for b in names[i + 1:] if self.arm_chain.check_colliding_links(a, b)]
        for obj_a, obj_b in self._self_pairs:
            if fcl.collide(obj_a, obj_b, fcl.CollisionRequest(), fcl.CollisionResult()) > 0:
                return True
        return False

    def part_buffers(self):
        '''
        By how much (cm) the still parts are inflated for the gripper / carried part, and for the arm. strict: gripper and carried
        part keep the checked clearance to every part, the arm (inflated by 1 cm itself) only needs the reserve.
        '''
        if self.strict:
            return max(ROBOT_PART_CLEARANCE, ASSEMBLY_CLEARANCE) + PLANNER_MARGIN, PLANNER_MARGIN
        return 0.5, 0.5

    def part_manager(self, meshes, buffer):
        '''collision manager of the still meshes, each inflated by buffer (cm); the inflated meshes are computed once per mesh'''
        manager = trimesh.collision.CollisionManager()
        for idx, mesh in enumerate(meshes):
            if buffer > 0:
                key = (id(mesh), buffer)
                if key not in self._buffer_cache: # strict: grown by at least buffer everywhere, see offset_mesh
                    self._buffer_cache[key] = (mesh, offset_mesh(mesh, buffer) if self.strict else get_buffered_meshes(mesh, buffer=buffer))
                mesh = self._buffer_cache[key][1]
            manager.add_object(f'part_still_{idx}', mesh)
        return manager

    def visualize_col_managers(self, col_managers, other_meshes=[]):
        if not ArmMotionPlanner.VISUALIZE: return
        meshes = {}
        meshes.update(self.arm_meshes_buffered)
        meshes.update(self.gripper_meshes_buffered)
        meshes_viz = []
        for col_manager in col_managers:
            transforms = {}
            for name in col_manager._objs:
                o = col_manager._objs[name]["obj"]
                transform_fcl = o.getTransform()
                transforms[name] = np.eye(4)
                transforms[name][:3, 3] = transform_fcl.getTranslation()
                transforms[name][:3, :3] = transform_fcl.getRotation()
            meshes_viz_i = {name: meshes[name].copy().apply_transform(transforms[name]) for name in transforms}
            meshes_viz.extend(list(meshes_viz_i.values()))
        trimesh.Scene(meshes_viz + list(other_meshes)).show()

    def visualize_meshes(self, q, open_ratio, move_pickup_mesh=None, gripper_pickup_transform=None, still_meshes=None, other_arm_chain=None, other_q=None, other_open_ratio=None, other_has_ft_sensor=None, buffered=False):
        if not ArmMotionPlanner.VISUALIZE: return
        arm_meshes_i, gripper_meshes_i, move_mesh_i = self.transform_meshes(q, open_ratio, move_pickup_mesh, gripper_pickup_transform, has_ft_sensor=self.has_ft_sensor, buffered=buffered)
        all_meshes = [*arm_meshes_i.values(), *gripper_meshes_i.values()]
        if move_mesh_i is not None:
            all_meshes.append(move_mesh_i)
        if still_meshes is not None:
            all_meshes.extend(still_meshes)
        if other_arm_chain is not None:
            arm_meshes_other_i, gripper_meshes_other_i, _ = self.transform_meshes(other_q, other_open_ratio, arm_chain=other_arm_chain, has_ft_sensor=other_has_ft_sensor, buffered=buffered)
            all_meshes.extend([*arm_meshes_other_i.values(), *gripper_meshes_other_i.values()])
        # if self.box_mesh is not None:
        #     all_meshes.append(self.box_mesh)
        trimesh.Scene(all_meshes).show()

    def visualize_meshes_active(self, q, *args, **kwargs):
        return self.visualize_meshes(self.arm_chain.active_to_full(q), *args, **kwargs)

    def get_fns(self, move_pickup_mesh, gripper_pickup_transform, still_meshes, open_ratio, arm_chain_other=None, arm_q_other=None, open_ratio_other=None, has_ft_sensor_other=False, buffered=True, verbose=False):

        # The collision structures built by an earlier call are held in reference cycles (the functions returned below refer to
        # each other), so Python frees them only when its garbage collector runs. Left to itself that is too late for large
        # meshes: every call left 218 MB behind for gamepad (two parts with 52000 faces), 76 MB for beam, and eight motion
        # plans of gamepad at once filled the 15 GB of the machine. Collected here, the memory of a process stays constant.
        # What a caller still holds is not touched.
        gc.collect()

        # move part collision manager
        if move_pickup_mesh is not None:
            move_col_manager = trimesh.collision.CollisionManager()
            move_col_manager.add_object('move', move_pickup_mesh)
        else:
            move_col_manager = None

        # still part collision managers: exact, and inflated for the gripper / carried part and for the (already inflated) arm
        part_buffer, part_buffer_arm = self.part_buffers()
        part_col_manager = self.part_manager(still_meshes, 0.0)
        part_col_manager_buffered = self.part_manager(still_meshes, part_buffer)
        part_col_manager_arm = self.part_manager(still_meshes, part_buffer_arm)

        # strict: limits of the cheap rules. The free path keeps the reserve, the fixed poses (grasp, release, insertion start) and the
        # straight moves only the checked values; relax() lowers the reserve to what the two ends of the free path reach.
        limits = {'joint': MIN_JOINT_LIMIT_MARGIN + PLANNER_JOINT_MARGIN, 'sigma': MIN_SIGMA + PLANNER_SIGMA_MARGIN, 'wall': VIRTUAL_WALL_SAFETY + PLANNER_MARGIN}
        limits_checked = {'joint': MIN_JOINT_LIMIT_MARGIN, 'sigma': MIN_SIGMA, 'wall': VIRTUAL_WALL_SAFETY}
        def pose_values(q):
            return {'joint': joint_limit_margin(q)[0], 'sigma': np.linalg.svd(panda_jacobian(q), compute_uv=False).min(),
                    'wall': get_virtual_wall_margin(self.arm_chain, self.arm_chain.active_to_full(q)) if self.virtual_walls else np.inf}
        def relax(qs):
            for q in qs:
                for key, value in pose_values(q).items():
                    limits[key] = max(limits_checked[key], min(limits[key], value))

        # other arm collision manager
        if arm_chain_other is not None and DUAL_ARM_COL_CHECK:
            assert arm_q_other is not None and open_ratio_other is not None
            arm_transforms_other, gripper_transforms_other = self.get_meshes_transforms_other(arm_chain_other, arm_q_other, open_ratio_other, has_ft_sensor_other)
            self.apply_transforms_to_col_manager(self.other_col_manager_buffered, arm_transforms_other)
            self.apply_transforms_to_col_manager(self.other_col_manager_buffered, gripper_transforms_other)
            other_col_manager_buffered = self.other_col_manager_buffered
        else:
            other_col_manager_buffered = None

        # consider gripper base offset in distance function
        tcp_offset = get_gripper_grasp_base_offset(self.gripper_type, open_ratio)

        # the joint and TCP positions of a pose, as one array (n + 1, 3): path smoothing asks for the same poses again and again
        points_cache = {}
        def pose_points(q):
            key = np.asarray(q, dtype=float).tobytes()
            points = points_cache.get(key)
            if points is None:
                fk = self.arm_chain.forward_kinematics_active(q, full_kinematics=True)
                tcp = fk[-1][:3, 3] + fk[-1][:3, 0 if self.arm_type == 'ur5e' else 2] * tcp_offset
                points = np.array([f[:3, 3] for f in fk] + [tcp])
                if len(points_cache) >= 50000: points_cache.clear()
                points_cache[key] = points
            return points

        def distance_fn(q1, q2, use_all_joints=True, average=True): # NOTE: use all joints or only the endeffector joint
            distances = np.linalg.norm(pose_points(q1) - pose_points(q2), axis=1) # joints ..., TCP
            if not use_all_joints:
                return distances[-1]
            return distances.sum() / len(distances) if average else distances.sum()

        move_hull_vertices = move_pickup_mesh.convex_hull.vertices if move_pickup_mesh is not None else None

        def hit(what):
            collision_fn.hit = what # what rejected the last pose that was rejected (one of HITS_ARM / HITS_CARRIED_PART), read by plan_path_straight_out
            if verbose:
                print('collision detected: ' + what)
            return True

        def collision_fn(q, buffered=buffered, move_ground_buffer=True, part_clearance=False, reserve=True):

            # real arm: joint limits, singularities and walls first (cheap; most rejected samples stop here)
            if self.strict:
                active = limits if buffered and reserve else limits_checked
                if joint_limit_margin(q)[0] < active['joint'] - 1e-9:
                    return hit('joint limit')
                if np.linalg.svd(panda_jacobian(q), compute_uv=False).min() < active['sigma'] - 1e-9:
                    return hit('singularity')
                if self.virtual_walls and get_virtual_wall_margin(self.arm_chain, self.arm_chain.active_to_full(q)) < active['wall'] - 1e-9:
                    return hit('virtual wall')

            # carried part too close to the table (only while transporting, the pickup and the assembly approach are exempt)
            if part_clearance and move_hull_vertices is not None and self.part_clearance is not None:
                _, _, move_transform_global = self.get_meshes_transforms_active(q, open_ratio, move_pickup_mesh, gripper_pickup_transform)
                if self.vertices_too_low(move_hull_vertices @ move_transform_global[:3, :3].T + move_transform_global[:3, 3]):
                    return hit('carried part too close to the table')

            # the real robot stops at franka-interface's virtual walls
            if not self.strict and self.virtual_walls and not self.check_virtual_walls(self.arm_chain.active_to_full(q)):
                return hit('virtual wall')

            # transform arm and gripper meshes
            arm_transforms_i, gripper_transforms_i, move_transform_global = self.get_meshes_transforms_active(q, open_ratio, move_pickup_mesh, gripper_pickup_transform)
            self.apply_transforms_to_col_manager(self.arm_col_manager_buffered, arm_transforms_i)
            self.apply_transforms_to_col_manager(self.gripper_col_manager_buffered, gripper_transforms_i)
            self.apply_transforms_to_col_manager(self.gripper_col_manager, gripper_transforms_i)
            if move_transform_global is not None:
                self.apply_transforms_to_col_manager(move_col_manager, {'move': move_transform_global})

            # check collision between arm and ground
            _, objs_in_collision = self.arm_col_manager_buffered.in_collision_other(self.ground_col_manager, return_names=True)
            for obj_pair in objs_in_collision:
                if self.arm_chain.get_base_link_name() in obj_pair: continue
                return hit('arm and ground')

            # check collision between gripper and ground
            gripper_col_manager = self.gripper_col_manager_buffered if buffered else self.gripper_col_manager
            if gripper_col_manager.in_collision_other(self.ground_col_manager):
                return hit('gripper and ground')

            # check collision between move part and ground. The table has a buffer for the carried part, except in the zone around the
            # assembly: there the part only has to stay off the table slab itself (0.2 cm, as the gripper), because an insertion can
            # start lower than the buffer (a part that slides in just above the table)
            if move_col_manager is not None:
                ground_buffer = move_ground_buffer and not self.in_exempt_zone(move_hull_vertices @ move_transform_global[:3, :3].T + move_transform_global[:3, 3])
                if move_col_manager.in_collision_other(self.ground_col_manager_buffered if ground_buffer else self.ground_col_manager):
                    return hit('move part and ground')

            # check collision between arm and grippers
            _, objs_in_collision = self.arm_col_manager_buffered.in_collision_other(self.gripper_col_manager_buffered, return_names=True)
            for obj_pair in objs_in_collision:
                if 'ft_sensor' in obj_pair or self.arm_chain.get_eef_link_name() in obj_pair: continue
                return hit('arm and gripper')

            # check collision between arm and parts
            part_col_manager_i = part_col_manager_buffered if buffered else part_col_manager
            if self.arm_col_manager_buffered.in_collision_other(part_col_manager_arm if buffered else part_col_manager):
                return hit('arm and part')

            # check collision between arm and move part
            if move_col_manager is not None and self.arm_col_manager_buffered.in_collision_other(move_col_manager):
                return hit('arm and move part')

            # strict, with buffers: the grown parts stick out at edges and corners, so a hit is confirmed by the exact distance
            # (the slower distance query only runs for poses next to a part)
            confirm = self.strict and buffered

            # check collision between gripper and parts (NOTE: unbuffered gripper)
            if self.gripper_col_manager.in_collision_other(part_col_manager_i) and \
                    (not confirm or self.gripper_col_manager.min_distance_other(part_col_manager) < part_buffer):
                return hit('gripper and part')

            # check collision between move part and parts
            if move_col_manager is not None and part_col_manager_i.in_collision_other(move_col_manager) and \
                    (not confirm or move_col_manager.min_distance_other(part_col_manager) < part_buffer):
                return hit('move part and part')

            # check arm self-intersection
            if SELF_ARM_INTERSECT_CHECK and self.arm_self_collision():
                return hit('arm self-intersection')

            # check dual-arm collision
            if other_col_manager_buffered is not None:
                if other_col_manager_buffered.in_collision_other(self.arm_col_manager_buffered) or \
                    other_col_manager_buffered.in_collision_other(self.gripper_col_manager_buffered) or \
                    (move_col_manager is not None and other_col_manager_buffered.in_collision_other(move_col_manager)):
                    return hit('dual-arm')

            if verbose:
                print('no collision detected')

            return False

        collision_fn.hit = None
        collision_fn.relax = relax

        active_bounds = np.array([link.bounds for link_idx, link in enumerate(self.arm_chain.links) if self.arm_chain.active_links_mask[link_idx]], dtype=float)
        def sample_fn():
            while True:
                if self.strict: # draw only where the joint-limit rule allows
                    lower = np.maximum(active_bounds[:, 0], PANDA_JOINT_LIMITS[0] + limits['joint'])
                    upper = np.minimum(active_bounds[:, 1], PANDA_JOINT_LIMITS[1] - limits['joint'])
                    q = np.random.uniform(lower, upper)
                else:
                    q = self.arm_chain.sample_joint_angles_active()
                if not collision_fn(q):
                    return q

        def extend_fn(q1, q2):
            q_dist = distance_fn(q1, q2)
            num_steps = int(np.ceil(q_dist / self.step_size))
            if num_steps < self.min_num_steps: num_steps = self.min_num_steps
            interpolated_qs = interpolate_qs(self.arm_type, q1, q2, num_steps)
            for i in range(num_steps):
                yield interpolated_qs[i + 1]
        
        return distance_fn, sample_fn, extend_fn, collision_fn

    def check_virtual_walls(self, q):
        '''q: full, True if q is inside the virtual walls (always True without virtual walls)'''
        return not self.virtual_walls or get_virtual_wall_margin(self.arm_chain, q) >= VIRTUAL_WALL_SAFETY

    def inverse_kinematics(self, pos, ori, q_init=None, n_restart=None, optimizer=None, regularization_parameter=None):
        if n_restart is None: n_restart = N_IK_RESTART
        if optimizer is None: optimizer = self.optimizer
        if q_init is None: q_init = self.arm_chain.active_to_full(self.arm_rest_q_active)
        q, success = self.arm_chain.inverse_kinematics(pos, ori, orientation_mode='all', initial_position=q_init, n_restart=n_restart, optimizer=optimizer, regularization_parameter=regularization_parameter)
        return q if success else None

    def inverse_kinematics_collision_free_with_grasp(self, pos, ori, move_pickup_mesh, gripper_pickup_transform, still_meshes, open_ratio, q_init=None, arm_chain_other=None, arm_q_other=None, open_ratio_other=None, has_ft_sensor_other=False, n_restart=3, n_restart_inner=None, optimizer=None, regularization_parameter=None, verbose=False):
        if optimizer is None: optimizer = self.optimizer
        success = False
        n_trial = 0
        _, _, _, collision_fn = self.get_fns(move_pickup_mesh, gripper_pickup_transform, still_meshes, open_ratio, arm_chain_other=arm_chain_other, arm_q_other=arm_q_other, open_ratio_other=open_ratio_other, has_ft_sensor_other=has_ft_sensor_other, verbose=False)
        while not success and n_trial < n_restart:
            q = self.inverse_kinematics(pos, ori, q_init=q_init, n_restart=n_restart_inner, optimizer=optimizer, regularization_parameter=regularization_parameter)
            if q is not None:
                q_active = self.arm_chain.active_from_full(q)
                if collision_fn(q_active):
                    msg = '[inverse_kinematics_collision_free_with_grasp] planned IK in collision'
                    msg += ', retrying' if n_trial + 1 < n_restart else ', failed'
                    if verbose: self.stamp.print(msg)
                else:
                    success = True
            n_trial += 1
            q_init = self.arm_chain.sample_joint_angles()
        return q if success else None

    def inverse_kinematics_collision_free(self, pos, ori, part_meshes, open_ratio, q_init=None, arm_chain_other=None, arm_q_other=None, open_ratio_other=None, has_ft_sensor_other=False, n_restart=3, n_restart_inner=None, optimizer=None, regularization_parameter=None, verbose=False):
        if optimizer is None: optimizer = self.optimizer
        return self.inverse_kinematics_collision_free_with_grasp(pos, ori, None, None, part_meshes, open_ratio, q_init, arm_chain_other, arm_q_other, open_ratio_other, has_ft_sensor_other, n_restart, n_restart_inner, optimizer, regularization_parameter, verbose)

    def plan_path_straight_out(self, q, collision_fn, direction):
        '''
        strict: move the gripper straight along direction (same orientation) from q (full) until the pose keeps the clearances to
        all parts (collision_fn with buffers). Every step must be contact free and keep the checked joint-limit, singularity and
        wall rules. Returns the path of full qs starting with q ([q] if q is clear already), or None; then last_block says why:
        (what, cm moved), what = one of HITS_ARM / HITS_CARRIED_PART (a step breaks a rule or touches something), the same with
        'never clear of: ' in front (no clear pose within STRAIGHT_MAX), or 'no arm pose' (no IK, or the arm would flip).
        '''
        self.last_block = None
        fk = self.arm_chain.forward_kinematics(q)
        pos, ori = fk[:3, 3].copy(), fk[:3, :3].copy()
        path = [np.array(q, dtype=float)]
        while collision_fn(self.arm_chain.active_from_full(path[-1]), reserve=False):
            not_clear = getattr(collision_fn, 'hit', None)
            moved = len(path) * STRAIGHT_STEP
            if moved > STRAIGHT_MAX:
                self.last_block = (f'never clear of: {not_clear}', moved - STRAIGHT_STEP)
                return None
            q_next = self.inverse_kinematics(pos + direction * moved, ori, q_init=path[-1]) # from the first pose: no drift
            if q_next is None or np.abs(q_next - path[-1]).max() > STRAIGHT_MAX_JOINT_STEP:
                self.last_block = ('no arm pose', moved)
                return None
            if collision_fn(self.arm_chain.active_from_full(q_next), buffered=False, move_ground_buffer=False):
                self.last_block = (getattr(collision_fn, 'hit', None), moved)
                return None
            path.append(q_next)
        return path

    def blocked_by_carried_part(self):
        '''True if the last straight move (plan_path_straight_out) failed because of the carried part, not of the arm or the gripper'''
        return self.last_block is not None and str(self.last_block[0]).replace('never clear of: ', '') in HITS_CARRIED_PART

    def in_exempt_zone(self, vertices):
        '''True if all points (n, 3) lie in the zone around the assembly where a carried part may come low (clearance_exempt_xy)'''
        if self.clearance_exempt_xy is None:
            return False
        lower, upper = self.clearance_exempt_xy
        return bool(np.all((vertices[:, :2] >= lower) & (vertices[:, :2] <= upper)))

    def add_buffered(self, mesh, buffer, buffered):
        '''
        Register the inflated version of a still mesh that was computed elsewhere, so that part_manager does not compute it again:
        the inflated mesh of a part moves with the part (run_pickup_gen.py tests the same part at many spots).
        '''
        self._buffer_cache[(id(mesh), buffer)] = (mesh, buffered)

    def drop_buffered(self, mesh):
        '''forget the inflated versions of a mesh that is not used any more (the cache holds a reference to every mesh it has seen)'''
        for key in [key for key in self._buffer_cache if key[0] == id(mesh)]:
            del self._buffer_cache[key]

    def plan_path_retract(self, q, collision_fn, retract, retract_delta):
        if self.strict:
            return self.plan_path_straight_out(q, collision_fn, retract)
        q_retract = q.copy()
        path_retract = [q_retract]
        while collision_fn(self.arm_chain.active_from_full(q_retract), buffered=False, move_ground_buffer=False):
            fk = self.arm_chain.forward_kinematics(q_retract)
            pos, ori = fk[:3, 3], fk[:3, :3]
            q_next = self.inverse_kinematics(pos + retract * retract_delta, ori, q_init=q_retract)
            if q_next is None or np.abs(q_next - q_retract).max() > MAX_RETRACT_JOINT_STEP: # no IK, or the arm would flip between two retract poses (unchecked sweep)
                return None
            q_retract = q_next
            path_retract.append(q_retract)
        return path_retract
    
    def check_part_too_low(self, q, open_ratio, move_pickup_mesh, gripper_pickup_transform):
        '''q: active, True if the carried part is closer than part_clearance to the table (outside the zone around the assembly)'''
        _, _, move_transform_global = self.get_meshes_transforms_active(q, open_ratio, move_pickup_mesh, gripper_pickup_transform)
        vertices = move_pickup_mesh.convex_hull.vertices @ move_transform_global[:3, :3].T + move_transform_global[:3, 3]
        return self.vertices_too_low(vertices)

    def vertices_too_low(self, vertices):
        low = vertices[:, 2] < self.table_z + self.part_clearance
        if self.clearance_exempt_xy is not None:
            lower, upper = self.clearance_exempt_xy
            low &= ~np.all((vertices[:, :2] >= lower) & (vertices[:, :2] <= upper), axis=1)
        return bool(np.any(low))

    def plan_path_lift(self, q, check_fn, collision_fn=None, max_lift=30.0):
        '''Lift the gripper straight up (same orientation) from q (full) until check_fn(q_active) is False, every step collision free; path of full qs or None'''
        path = [q.copy()]
        lifted = 0.0
        while check_fn(self.arm_chain.active_from_full(path[-1])):
            if lifted >= max_lift:
                return None
            fk = self.arm_chain.forward_kinematics(path[-1])
            q_next = self.inverse_kinematics(fk[:3, 3] + np.array([0, 0, TRANSFER_LIFT_DELTA]), fk[:3, :3], q_init=path[-1])
            if q_next is None or not self.check_virtual_walls(q_next):
                return None
            if collision_fn is not None and any(collision_fn(self.arm_chain.active_from_full(path[-1] + t * (q_next - path[-1]))) for t in np.linspace(0.25, 1, 4)):
                return None
            path.append(q_next)
            lifted += TRANSFER_LIFT_DELTA
        return path

    def plan_path_with_grasp(self, q_start, q_goal, move_pickup_mesh, gripper_pickup_transform, still_meshes, open_ratio, arm_chain_other=None, arm_q_other=None, open_ratio_other=None, has_ft_sensor_other=False, retract_start=None, retract_goal=None, retract_delta=0.5, max_speed=None, verbose=False):
        '''
        Plan arm reaching with moving part in held and a single gripper open ratio
        ----------------------------------
        q_start: start q of arm (full)
        q_goal: goal q of arm (full)
        move_pickup_mesh: mesh to move at pickup pose
        still_meshes: list of still meshes
        open_ratio: open ratio of gripper
        '''
        distance_fn, sample_fn, extend_fn, collision_fn = self.get_fns(move_pickup_mesh, gripper_pickup_transform, still_meshes, open_ratio, arm_chain_other=arm_chain_other, arm_q_other=arm_q_other, open_ratio_other=open_ratio_other, has_ft_sensor_other=has_ft_sensor_other, verbose=False)
        collision_fn_unbuffered = lambda q: collision_fn(q, buffered=False, move_ground_buffer=False)
        self.last_failure, self.last_block = None, None

        path_start_retract = []
        q_start_active = self.arm_chain.active_from_full(q_start)
        if collision_fn(q_start_active):
            if retract_start is None:
                if verbose:
                    self.stamp.print('[plan_path_with_grasp] start is in collision')
                    self.visualize_meshes(q_start, open_ratio, move_pickup_mesh, gripper_pickup_transform, still_meshes, arm_chain_other, arm_q_other, open_ratio_other, has_ft_sensor_other, buffered=True)
                assert not collision_fn_unbuffered(q_start_active)
            else:
                path_start_retract = self.plan_path_retract(q_start, collision_fn, retract_start, retract_delta)
                if path_start_retract is None:
                    if verbose:
                        self.stamp.print(f'[plan_path_with_grasp] failed to plan start retract {retract_start}')
                        self.visualize_meshes(q_start, open_ratio, move_pickup_mesh, gripper_pickup_transform, still_meshes, arm_chain_other, arm_q_other, open_ratio_other, has_ft_sensor_other, buffered=True)
                    if self.strict:
                        self.last_failure = 'start'
                        return None
                    path_start_retract = []
                else:
                    path_start_retract = [self.arm_chain.active_from_full(q) for q in path_start_retract]
                    q_start_active = path_start_retract[-1]
                    assert not collision_fn_unbuffered(q_start_active)

        path_goal_retract = []
        q_goal_active = self.arm_chain.active_from_full(q_goal)
        if collision_fn(q_goal_active):
            if retract_goal is None:
                if verbose:
                    self.stamp.print('[plan_path_with_grasp] goal is in collision')
                    self.visualize_meshes(q_goal, open_ratio, move_pickup_mesh, gripper_pickup_transform, still_meshes, arm_chain_other, arm_q_other, open_ratio_other, has_ft_sensor_other, buffered=True)
                assert not collision_fn_unbuffered(q_goal_active)
            else:
                path_goal_retract = self.plan_path_retract(q_goal, collision_fn, retract_goal, retract_delta)
                if path_goal_retract is None:
                    if verbose:
                        self.stamp.print(f'[plan_path_with_grasp] failed to plan goal retract {retract_goal}')
                        self.visualize_meshes(q_goal, open_ratio, move_pickup_mesh, gripper_pickup_transform, still_meshes, arm_chain_other, arm_q_other, open_ratio_other, has_ft_sensor_other, buffered=True)
                    if self.strict:
                        self.last_failure = 'goal'
                        return None
                    path_goal_retract = []
                else:
                    path_goal_retract = [self.arm_chain.active_from_full(q) for q in path_goal_retract]
                    q_goal_active = path_goal_retract[-1]
                    assert not collision_fn_unbuffered(q_goal_active)

        # carried part: lift straight up after the start retract and before the goal retract until it is part_clearance above the table
        path_start_lift, path_goal_lift = [], []
        use_clearance = move_pickup_mesh is not None and self.part_clearance is not None
        if use_clearance:
            too_low_fn = lambda q: self.check_part_too_low(q, open_ratio, move_pickup_mesh, gripper_pickup_transform)
            for end in ['start', 'goal']:
                q_end_active = q_start_active if end == 'start' else q_goal_active
                path_lift = self.plan_path_lift(self.arm_chain.active_to_full(q_end_active), too_low_fn, collision_fn_unbuffered)
                if path_lift is None or collision_fn_unbuffered(self.arm_chain.active_from_full(path_lift[-1])):
                    self.stamp.print(f'[plan_path_with_grasp] WARNING: cannot lift the carried part {self.part_clearance} cm above the table at the {end}')
                    if self.strict:
                        self.last_failure, self.last_block = end, ('no straight lift of the carried part', None)
                        return None
                    use_clearance = False
                    path_start_lift, path_goal_lift = [], []
                    break
                path_lift = [self.arm_chain.active_from_full(q) for q in path_lift]
                if end == 'start':
                    path_start_lift = path_lift[1:]
                else:
                    path_goal_lift = path_lift[1:]
            if len(path_start_lift) > 0: q_start_active = path_start_lift[-1]
            if len(path_goal_lift) > 0: q_goal_active = path_goal_lift[-1]
        collision_fn_clear = lambda q, buffered=None, move_ground_buffer=True: collision_fn(q, **({} if buffered is None else {'buffered': buffered}), move_ground_buffer=move_ground_buffer, part_clearance=use_clearance)

        if self.strict: # the free path keeps all clearances; its two ends must do so already (never plan without them)
            collision_fn.relax([q_start_active, q_goal_active])
            for end, q_end in [('start', q_start_active), ('goal', q_goal_active)]:
                if collision_fn_clear(q_end):
                    if verbose: self.stamp.print(f'[plan_path_with_grasp] the {end} of the free path does not keep the clearances')
                    self.last_failure, self.last_block = end, (collision_fn.hit, None)
                    return None
            collision_fn_new = collision_fn_clear
        elif collision_fn_clear(q_start_active) or collision_fn_clear(q_goal_active):
            collision_fn_new = lambda q, buffered=False, move_ground_buffer=False: collision_fn_clear(q, buffered=buffered, move_ground_buffer=move_ground_buffer) # the table clearance stays on
            assert not collision_fn_new(q_start_active), f'[plan_path_with_grasp] start in collision even without buffer'
            assert not collision_fn_new(q_goal_active), f'[plan_path_with_grasp] goal in collision even without buffer'
        else:
            collision_fn_new = collision_fn_clear
        
        # path = rrt_star(q_start_active, q_goal_active, distance_fn, sample_fn, extend_fn, collision_fn_new, radius=5.0, max_iterations=10, informed=False, early_terminate=True, verbose=verbose)
        path = rrt_connect(q_start_active, q_goal_active, distance_fn, sample_fn, extend_fn, collision_fn_new, max_iterations=1000)
        if path is None:
            if verbose:
                self.stamp.print('[plan_path_with_grasp] failed to plan path')
                self.visualize_meshes(q_start, open_ratio, move_pickup_mesh, gripper_pickup_transform, still_meshes, arm_chain_other, arm_q_other, open_ratio_other, has_ft_sensor_other, buffered=True)
                self.visualize_meshes(q_goal, open_ratio, move_pickup_mesh, gripper_pickup_transform, still_meshes, arm_chain_other, arm_q_other, open_ratio_other, has_ft_sensor_other, buffered=True)
            self.last_failure, self.last_block = 'path', None
            return None
        if verbose: self.stamp.print(f'[plan_path_with_grasp] path planned (len: {len(path)})')
        
        path = smooth_path(path, extend_fn, collision_fn_new, distance_fn, cost_fn=None, sample_fn=sample_fn, max_iterations=1000, max_time=120, tolerance=1e-5)
        path.insert(0, q_start_active)

        if verbose: self.stamp.print(f'[plan_path_with_grasp] path smoothed (len: {len(path)})')

        in_collision = False
        for q in path:
            if collision_fn_new(q):
                in_collision = True
                break
        # strict: in_collision here mostly means that the path cuts into the planner's reserve between two checked poses; whether a
        # rule is broken is decided by the exact check of the finished path (SceneChecker.check_path in run_motion_plan.py)

        if len(path_start_lift) > 0: # the lift continues the retract
            path_start_retract = (path_start_retract if len(path_start_retract) > 0 else [self.arm_chain.active_from_full(q_start)]) + path_start_lift
        if len(path_goal_lift) > 0:
            path_goal_retract = (path_goal_retract if len(path_goal_retract) > 0 else [self.arm_chain.active_from_full(q_goal)]) + path_goal_lift
        if len(path_start_retract) > 0:
            path_start_retract = interpolate_arm_path(self.arm_type, path_start_retract, distance_fn, max_lin_speed=max_speed / 3).tolist()
        if len(path_goal_retract) > 0:
            path_goal_retract = interpolate_arm_path(self.arm_type, path_goal_retract, distance_fn, max_lin_speed=max_speed / 3).tolist()
        path = interpolate_arm_path(self.arm_type, path, distance_fn, max_lin_speed=max_speed).tolist()
        path = path_start_retract + path + path_goal_retract[::-1]
        if verbose: self.stamp.print(f'[plan_path_with_grasp] path interpolated (len: {len(path)}, collision: {in_collision})')

        path = [self.arm_chain.active_to_full(q) for q in path]
        return path

    def plan_path(self, q_start, q_goal, part_meshes, open_ratio, arm_chain_other=None, arm_q_other=None, open_ratio_other=None, has_ft_sensor_other=False, retract_start=None, retract_goal=None, retract_delta=0.5, max_speed=None, verbose=False):
        '''
        Plan arm reaching/retracting with no part in held
        '''
        return self.plan_path_with_grasp(q_start, q_goal, move_pickup_mesh=None, gripper_pickup_transform=None, still_meshes=part_meshes, open_ratio=open_ratio, 
                                         arm_chain_other=arm_chain_other, arm_q_other=arm_q_other, open_ratio_other=open_ratio_other, has_ft_sensor_other=has_ft_sensor_other, 
                                         retract_start=retract_start, retract_goal=retract_goal, retract_delta=retract_delta, max_speed=max_speed, verbose=verbose)
    
    def plan_path_switch(self, q_start, q_goal, part_meshes, open_ratio, open_ratio_next, arm_chain_other=None, arm_q_other=None, open_ratio_other=None, has_ft_sensor_other=False, retract_start=None, retract_goal=None, retract_delta=0.5, max_speed=None, verbose=False):
        '''
        Plan arm switching with no part in held
        '''
        distance_fn_start, _, _, collision_fn_start = self.get_fns(None, None, part_meshes, open_ratio, arm_chain_other=arm_chain_other, arm_q_other=arm_q_other, open_ratio_other=open_ratio_other, has_ft_sensor_other=has_ft_sensor_other, verbose=False)
        distance_fn_goal, _, _, collision_fn_goal = self.get_fns(None, None, part_meshes, open_ratio_next, arm_chain_other=arm_chain_other, arm_q_other=arm_q_other, open_ratio_other=open_ratio_other, has_ft_sensor_other=has_ft_sensor_other, verbose=False)
        
        path_start_retract = self.plan_path_retract(q_start, collision_fn_start, retract_start, retract_delta)
        if path_start_retract is None:
            if verbose:
                self.stamp.print('[plan_path_switch] failed to plan start retract')
                self.visualize_meshes(q_start, open_ratio, still_meshes=part_meshes, buffered=True)
            if self.strict: # no straight way out of the release pose
                self.last_failure = 'start'
                return None, None
            path_start_retract = []
            q_start_retract = q_start
        else:
            q_start_retract = path_start_retract[-1]
        
        path_goal_retract = self.plan_path_retract(q_goal, collision_fn_goal, retract_goal, retract_delta)
        if path_goal_retract is None:
            if verbose:
                self.stamp.print('[plan_path_switch] failed to plan goal retract')
                self.visualize_meshes(q_goal, open_ratio, still_meshes=part_meshes, buffered=True)
            if self.strict: # no straight way into the grasp pose
                self.last_failure = 'goal'
                return None, None
            path_goal_retract = []
            q_goal_retract = q_goal
        else:
            q_goal_retract = path_goal_retract[-1]

        if open_ratio < open_ratio_next: # transport with the smaller open ratio
            open_ratio_transport = open_ratio
            path_goal_retract_extra = self.plan_path_retract(q_goal_retract, collision_fn_start, retract_goal, retract_delta)
            if path_goal_retract_extra is None:
                if verbose:
                    self.stamp.print('[plan_path_switch] failed to plan goal retract extra')
                    self.visualize_meshes(q_goal_retract, open_ratio, still_meshes=part_meshes, buffered=True)
                if self.strict:
                    self.last_failure = 'goal'
                    return None, None
            else:
                path_goal_retract += path_goal_retract_extra[1:]
                q_goal_retract = path_goal_retract[-1]
        else:
            open_ratio_transport = open_ratio_next
            path_start_retract_extra = self.plan_path_retract(q_start_retract, collision_fn_goal, retract_start, retract_delta)
            if path_start_retract_extra is None:
                if verbose:
                    self.stamp.print('[plan_path_switch] failed to plan start retract extra')
                    self.visualize_meshes(q_start_retract, open_ratio, still_meshes=part_meshes, buffered=True)
                if self.strict:
                    self.last_failure = 'start'
                    return None, None
            else:
                path_start_retract += path_start_retract_extra[1:]
                q_start_retract = path_start_retract[-1]

        path = self.plan_path(q_start_retract, q_goal_retract, part_meshes, open_ratio=open_ratio_transport, 
                                         arm_chain_other=arm_chain_other, arm_q_other=arm_q_other, open_ratio_other=open_ratio_other, has_ft_sensor_other=has_ft_sensor_other, 
                                         retract_start=None, retract_goal=None, retract_delta=None, max_speed=max_speed, verbose=verbose)
        if path is None:
            return None, None

        if len(path_start_retract) > 0:
            path_start_retract = [self.arm_chain.active_from_full(q) for q in path_start_retract]
            path_start_retract = interpolate_arm_path(self.arm_type, path_start_retract, distance_fn_start, max_lin_speed=max_speed / 3)
            path_start_retract = [self.arm_chain.active_to_full(q) for q in path_start_retract]

        if len(path_goal_retract) > 0:
            path_goal_retract = [self.arm_chain.active_from_full(q) for q in path_goal_retract]
            path_goal_retract = interpolate_arm_path(self.arm_type, path_goal_retract, distance_fn_goal, max_lin_speed=max_speed / 3)
            path_goal_retract = [self.arm_chain.active_to_full(q) for q in path_goal_retract]

        if open_ratio < open_ratio_next:
            return path_start_retract + path, path_goal_retract[::-1]
        else:
            return path_start_retract, path + path_goal_retract[::-1]

    def plan_path_straight(self, q_start, q_goal, open_ratio, max_speed=None, sanity_check=True, verbose=False):
        '''
        Plan straight path between start and goal (assume start and goal IK are feasible and no collision in between)
        '''
        q_start_active, q_goal_active = self.arm_chain.active_from_full(q_start), self.arm_chain.active_from_full(q_goal)
        pose_start, pose_goal = self.arm_chain.forward_kinematics_active(q_start_active, full_kinematics=False), self.arm_chain.forward_kinematics_active(q_goal_active, full_kinematics=False)
        
        # plan path
        num_steps = max(int(np.linalg.norm(pose_start[:3, 3] - pose_goal[:3, 3]) / self.step_size), 1)
        pose_path = interpolate_transformations(pose_start, pose_goal, num_steps)
        path = []
        for pose in pose_path:
            pos, ori = pose[:3, 3], pose[:3, :3]
            q = self.inverse_kinematics(pos, ori, q_init=q_start if len(path) == 0 else path[-1])
            if q is None:
                if verbose:
                    self.stamp.print('[plan_path_straight] failed to plan path')
                    self.visualize_meshes(q_start, open_ratio)
                    self.visualize_meshes(q_goal, open_ratio)
                return None 
            path.append(q)
        path = [self.arm_chain.active_from_full(q) for q in path]
        if verbose: self.stamp.print(f'[plan_path_straight] path planned (len: {len(path)})')

        # sanity check
        distance_fn, _, _, collision_fn = self.get_fns(None, None, [], open_ratio, buffered=False, verbose=False)
        in_collision = False
        if sanity_check:
            for q in path:
                if collision_fn(q):
                    in_collision = True
            if in_collision:
                if verbose:
                    self.stamp.print('[plan_path_straight] path in collision')
                    self.visualize_meshes(q_start, open_ratio)
                    self.visualize_meshes(q_goal, open_ratio)
                return None

        # interpolate path (TODO: needed?)
        path = interpolate_arm_path(self.arm_type, path, distance_fn, max_lin_speed=max_speed)
        if verbose: self.stamp.print(f'[plan_path_straight] path interpolated (len: {len(path)}, collision: {in_collision})')

        path = [self.arm_chain.active_to_full(q) for q in path]
        return path
