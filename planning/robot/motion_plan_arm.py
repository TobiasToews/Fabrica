import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(project_base_dir)

import numpy as np
import trimesh
import os
from scipy.spatial.transform import Rotation as R, Slerp

from assets.transform import get_transform_matrix_quat
from planning.path.rrt_connect import rrt_connect
from planning.path.rrt_star import rrt_star
from planning.path.meta import direct_path
from planning.path.smoothing import smooth_path
from planning.robot.util_arm import get_gripper_pos_quat_from_arm_q, get_ik_target_orientation, get_gripper_basis_directions, get_virtual_wall_margin
from planning.robot.geometry import load_arm_meshes, load_gripper_meshes, transform_arm_meshes, transform_gripper_meshes, get_arm_meshes_transforms, get_gripper_meshes_transforms, get_buffered_arm_meshes, get_buffered_gripper_meshes, get_buffered_meshes, get_gripper_grasp_base_offset
from utils.common import TimeStamp


DUAL_ARM_COL_CHECK = True
SELF_ARM_INTERSECT_CHECK = True
MAX_LIN_SPEED = 5 # cm/s
MAX_ANG_SPEED = 1 # rad/s
N_IK_RESTART = 20 # chain's default is 3
VIRTUAL_WALL_SAFETY = 1.0 # cm, extra distance to franka-interface's virtual walls (tracking error)


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

    def __init__(self, arm_chain, gripper_type, has_ft_sensor=False, arm_box=None, optimizer=None, stamp=None, virtual_walls=False):
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

    def visualize_col_managers(self, col_managers, other_meshes=[]):
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

        # move part collision manager
        if move_pickup_mesh is not None:
            move_col_manager = trimesh.collision.CollisionManager()
            move_col_manager.add_object('move', move_pickup_mesh)
        else:
            move_col_manager = None

        # still part collision manager
        still_meshes_buffered = get_buffered_meshes(still_meshes, buffer=0.5)
        part_col_manager = trimesh.collision.CollisionManager()
        for idx, mesh in enumerate(still_meshes):
            part_col_manager.add_object(f'part_still_{idx}', mesh)
        part_col_manager_buffered = trimesh.collision.CollisionManager()
        for idx, mesh in enumerate(still_meshes_buffered):
            part_col_manager_buffered.add_object(f'part_still_{idx}', mesh)

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

        def distance_fn(q1, q2, use_all_joints=True, average=True): # NOTE: use all joints or only the endeffector joint
            fk1 = self.arm_chain.forward_kinematics_active(q1, full_kinematics=True)
            fk2 = self.arm_chain.forward_kinematics_active(q2, full_kinematics=True)
            if self.arm_type == 'ur5e':
                tcp1 = fk1[-1][:3, 3] + fk1[-1][:3, 0] * tcp_offset
                tcp2 = fk2[-1][:3, 3] + fk2[-1][:3, 0] * tcp_offset
            else:
                tcp1 = fk1[-1][:3, 3] + fk1[-1][:3, 2] * tcp_offset
                tcp2 = fk2[-1][:3, 3] + fk2[-1][:3, 2] * tcp_offset
            dist = np.linalg.norm(tcp1 - tcp2)
            if use_all_joints:
                for i in range(len(fk1)):
                    dist += np.linalg.norm(fk1[i][:3, 3] - fk2[i][:3, 3])
                if average:
                    dist /= len(fk1) + 1
            return dist

        def collision_fn(q, buffered=buffered, move_ground_buffer=True):

            # the real robot stops at franka-interface's virtual walls
            if self.virtual_walls and not self.check_virtual_walls(self.arm_chain.active_to_full(q)):
                if verbose:
                    print('collision detected: virtual wall')
                return True

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
                if verbose:
                    print('collision detected: arm and ground')
                return True

            # check collision between gripper and ground
            gripper_col_manager = self.gripper_col_manager_buffered if buffered else self.gripper_col_manager
            if gripper_col_manager.in_collision_other(self.ground_col_manager):
                if verbose:
                    print('collision detected: gripper and ground')
                return True

            # check collision between move part and ground
            if move_col_manager is not None and move_col_manager.in_collision_other(self.ground_col_manager_buffered if move_ground_buffer else self.ground_col_manager):
                if verbose:
                    print('collision detected: move part and ground')
                return True

            # check collision between arm and grippers
            _, objs_in_collision = self.arm_col_manager_buffered.in_collision_other(self.gripper_col_manager_buffered, return_names=True)
            for obj_pair in objs_in_collision:
                if 'ft_sensor' in obj_pair or self.arm_chain.get_eef_link_name() in obj_pair: continue
                if verbose:
                    print('collision detected: arm and gripper')
                return True

            # check collision between arm and parts
            part_col_manager_i = part_col_manager_buffered if buffered else part_col_manager
            if self.arm_col_manager_buffered.in_collision_other(part_col_manager_i):
                if verbose:
                    print('collision detected: arm and part')
                return True
            
            # check collision between arm and move part
            if move_col_manager is not None and self.arm_col_manager_buffered.in_collision_other(move_col_manager):
                if verbose:
                    print('collision detected: arm and move part')
                return True
            
            # check collision between gripper and parts (NOTE: unbuffered gripper)
            if self.gripper_col_manager.in_collision_other(part_col_manager_i):
                if verbose:
                    print('collision detected: gripper and part')
                return True

            # check collision between move part and parts
            if move_col_manager is not None and part_col_manager_i.in_collision_other(move_col_manager):
                if verbose:
                    print('collision detected: move part and part')
                return True

            # check arm self-intersection
            if SELF_ARM_INTERSECT_CHECK:
                _, objs_in_collision = self.arm_col_manager_buffered.in_collision_internal(return_names=True)
                for obj_pair in objs_in_collision:
                    if not self.arm_chain.check_colliding_links(obj_pair[0], obj_pair[1]): continue
                    if verbose:
                        print('collision detected: arm self-intersection')
                    return True

            # check dual-arm collision
            if other_col_manager_buffered is not None:
                if other_col_manager_buffered.in_collision_other(self.arm_col_manager_buffered) or \
                    other_col_manager_buffered.in_collision_other(self.gripper_col_manager_buffered) or \
                    (move_col_manager is not None and other_col_manager_buffered.in_collision_other(move_col_manager)):
                    if verbose:
                        print('collision detected: dual-arm')
                    return True

            if verbose:
                print('no collision detected')

            return False

        def sample_fn():
            while True:
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

    def plan_path_retract(self, q, collision_fn, retract, retract_delta):
        q_retract = q.copy()
        path_retract = [q_retract]
        while collision_fn(self.arm_chain.active_from_full(q_retract), buffered=False, move_ground_buffer=False):
            fk = self.arm_chain.forward_kinematics(q_retract)
            pos, ori = fk[:3, 3], fk[:3, :3]
            q_retract = self.inverse_kinematics(pos + retract * retract_delta, ori, q_init=q_retract)
            if q_retract is None:
                return None
            path_retract.append(q_retract)
        return path_retract
    
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
                    path_goal_retract = []
                else:
                    path_goal_retract = [self.arm_chain.active_from_full(q) for q in path_goal_retract]
                    q_goal_active = path_goal_retract[-1]
                    assert not collision_fn_unbuffered(q_goal_active)

        if collision_fn(q_start_active) or collision_fn(q_goal_active):
            collision_fn_new = lambda q, buffered=False, move_ground_buffer=False: collision_fn(q, buffered=buffered, move_ground_buffer=move_ground_buffer)
            assert not collision_fn_new(q_start_active), f'[plan_path_with_grasp] start in collision even without buffer'
            assert not collision_fn_new(q_goal_active), f'[plan_path_with_grasp] goal in collision even without buffer'
        else:
            collision_fn_new = collision_fn
        
        # path = rrt_star(q_start_active, q_goal_active, distance_fn, sample_fn, extend_fn, collision_fn_new, radius=5.0, max_iterations=10, informed=False, early_terminate=True, verbose=verbose)
        path = rrt_connect(q_start_active, q_goal_active, distance_fn, sample_fn, extend_fn, collision_fn_new, max_iterations=1000)
        if path is None:
            if verbose:
                self.stamp.print('[plan_path_with_grasp] failed to plan path')
                self.visualize_meshes(q_start, open_ratio, move_pickup_mesh, gripper_pickup_transform, still_meshes, arm_chain_other, arm_q_other, open_ratio_other, has_ft_sensor_other, buffered=True)
                self.visualize_meshes(q_goal, open_ratio, move_pickup_mesh, gripper_pickup_transform, still_meshes, arm_chain_other, arm_q_other, open_ratio_other, has_ft_sensor_other, buffered=True)
            return None
        if verbose: self.stamp.print(f'[plan_path_with_grasp] path planned (len: {len(path)})')
        
        path = smooth_path(path, extend_fn, collision_fn_new, distance_fn, cost_fn=None, sample_fn=sample_fn, max_iterations=1000, max_time=120, tolerance=1e-5)
        path.insert(0, q_start_active)

        if verbose: self.stamp.print(f'[plan_path_with_grasp] path smoothed (len: {len(path)})')

        in_collision = False
        for q in path:
            if collision_fn_new(q):
                in_collision = True

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
            path_start_retract = []
            q_start_retract = q_start
        else:
            q_start_retract = path_start_retract[-1]
        
        path_goal_retract = self.plan_path_retract(q_goal, collision_fn_goal, retract_goal, retract_delta)
        if path_goal_retract is None:
            if verbose:
                self.stamp.print('[plan_path_switch] failed to plan goal retract')
                self.visualize_meshes(q_goal, open_ratio, still_meshes=part_meshes, buffered=True)
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
