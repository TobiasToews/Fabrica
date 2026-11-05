import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(project_base_dir)

import numpy as np
from scipy.spatial.transform import Rotation as R
from .ik.chain import Chain

from assets.transform import get_transform_matrix_quat, get_pos_euler_from_transform_matrix
from planning.robot.geometry import get_ft_sensor_spec, get_gripper_basis_directions
from planning.robot.workcell import get_move_arm_pos, get_move_arm_euler, get_hold_arm_pos, get_hold_arm_euler, get_panda_virtual_walls


def get_xarm7_arm_chain(base_pos, base_euler, reduced_limit=0.0):
    chain = Chain.from_urdf_file(os.path.join(project_base_dir, 'assets/xarm7/xarm7.urdf'), base_elements=['linkbase'],
        origin_translation=np.array(base_pos), origin_orientation=np.array(base_euler), scale_translation=100, reduced_limit=reduced_limit)
    chain.rest_q = [0., -0.56582579, 0., 0.35527904, 0., 0.92109843, 0.]
    return chain


def get_panda_arm_chain(base_pos, base_euler, reduced_limit=0.0):
    chain = Chain.from_urdf_file(os.path.join(project_base_dir, 'assets/panda/panda.urdf'), base_elements=['panda_link0'],
        origin_translation=np.array(base_pos), origin_orientation=np.array(base_euler), scale_translation=100, reduced_limit=reduced_limit)
    chain.rest_q = [0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4]
    return chain


def get_ur5e_arm_chain(base_pos, base_euler, reduced_limit=0.0):
    chain = Chain.from_urdf_file(os.path.join(project_base_dir, 'assets/ur5e/ur5e.urdf'), base_elements=['base_link'],
        origin_translation=np.array(base_pos), origin_orientation=np.array(base_euler), scale_translation=1, reduced_limit=reduced_limit)
    chain.rest_q = [0., -1.57079632679, 1.57079632679, -1.57079632679, -1.57079632679, 0.]
    chain.no_collision_links = [('base_link', 'upper_arm_link'), ('wrist_1_link', 'wrist_3_link')]
    return chain


def get_arm_chain(arm_type, motion_type=None, base_pos=None, base_euler=None, reduced_limit=0.0):

    # get base position and orientation
    if motion_type is not None:
        if motion_type == 'move':
            if base_pos is None: base_pos = get_move_arm_pos(arm_type)
            if base_euler is None: base_euler = get_move_arm_euler()
        elif motion_type == 'hold':
            if base_pos is None: base_pos = get_hold_arm_pos(arm_type)
            if base_euler is None: base_euler = get_hold_arm_euler()
        else:
            raise ValueError('Unknown motion type: {}'.format(motion_type))
    else:
        assert base_pos is not None and base_euler is not None
        
    # create kinematic chain
    if arm_type == 'xarm7':
        arm_chain = get_xarm7_arm_chain(base_pos, base_euler, reduced_limit=reduced_limit)
    elif arm_type == 'panda':
        arm_chain = get_panda_arm_chain(base_pos, base_euler, reduced_limit=reduced_limit)
    elif arm_type == 'ur5e':
        arm_chain = get_ur5e_arm_chain(base_pos, base_euler, reduced_limit=reduced_limit)
    else:
        raise ValueError('Unknown arm type: {}'.format(arm_type))
    arm_chain.arm_type = arm_type
    arm_chain.base_pos = base_pos
    arm_chain.base_euler = base_euler

    # set bounds for the first link to avoid unintuitive motion
    first_link = arm_chain.get_active_link(0)
    if motion_type is None:
        pass
    elif motion_type == 'move':
        first_link.bounds = (first_link.bounds[0], min(first_link.bounds[1], 0.5))
    elif motion_type == 'hold':
        first_link.bounds = (max(first_link.bounds[0], -0.5), first_link.bounds[1])
    else:
        raise ValueError('Unknown motion type: {}'.format(motion_type))
    
    return arm_chain
    

def get_virtual_wall_margin(arm_chain, arm_q):
    '''
    Smallest distance (cm) by which the arm at arm_q (full) stays inside franka-interface's virtual walls,
    negative if the real robot would stop (panda only)
    '''
    assert arm_chain.arm_type == 'panda'
    planes, thresholds = get_panda_virtual_walls()
    fk = arm_chain.forward_kinematics(arm_q, full_kinematics=True) # link0..link8, link2..link8 = frames J2..J7, flange
    base_rot = R.from_euler('xyz', arm_chain.base_euler).as_matrix()
    margin = np.inf
    for T, threshold in zip(fk[2:], thresholds):
        pos = base_rot.T @ (T[:3, 3] - arm_chain.base_pos) / 100 # world (cm) -> robot base (m)
        for axis, offset in planes:
            margin = min(margin, abs(pos[axis] - offset) - threshold)
    return margin * 100


def get_ft_pos_from_gripper_pos_quat(gripper_type, gripper_pos, gripper_quat):
    base_basis_direction, _ = get_gripper_basis_directions(gripper_type)
    ft_spec = get_ft_sensor_spec()
    gripper_rot = R.from_quat(gripper_quat[[1, 2, 3, 0]])
    ft_pos = gripper_pos + gripper_rot.apply(base_basis_direction) * ft_spec['height']
    return ft_pos


def get_gripper_pos_quat_from_arm_q(arm_chain, arm_q, gripper_type, has_ft_sensor=False):

    ef_target_matrix = arm_chain.forward_kinematics(arm_q)
    ef_init_matrix = arm_chain.forward_kinematics_active(arm_chain.rest_q)[:3, :3]
    arm_euler = arm_chain.links[0].origin_orientation
    base_init_direction, l2r_init_direction = [0, 0, 1], R.from_euler('xyz', arm_euler).apply([0, -1, 0])
    base_basis_direction, l2r_basis_direction = get_gripper_basis_directions(gripper_type)
    gripper_init_matrix = R.align_vectors([base_init_direction, l2r_init_direction], [base_basis_direction, l2r_basis_direction])[0].as_matrix()
    gripper_target_matrix = ef_target_matrix[:3, :3] @ ef_init_matrix.T @ gripper_init_matrix
    gripper_pos, gripper_quat = ef_target_matrix[:3, 3], R.from_matrix(gripper_target_matrix).as_quat()[[3, 0, 1, 2]]

    if has_ft_sensor:
        ft_spec = get_ft_sensor_spec()
        gripper_pos -= R.from_matrix(gripper_target_matrix).apply(base_basis_direction) * ft_spec['height']
    
    return gripper_pos, gripper_quat


def get_gripper_qm_from_arm_q(arm_chain, arm_q, gripper_type, has_ft_sensor=False):

    gripper_pos, gripper_quat = get_gripper_pos_quat_from_arm_q(arm_chain, arm_q, gripper_type, has_ft_sensor=has_ft_sensor)
    gripper_matrix = get_transform_matrix_quat(gripper_pos, gripper_quat)
    gripper_qm = get_pos_euler_from_transform_matrix(gripper_matrix)

    return gripper_qm


def get_gripper_path_from_arm_path(arm_chain, arm_path, gripper_type, has_ft_sensor=False):
    
    gripper_path = []
    for arm_q in arm_path:
        gripper_path.append(get_gripper_qm_from_arm_q(arm_chain, arm_q, gripper_type, has_ft_sensor=has_ft_sensor))

    return gripper_path


def get_gripper_part_qm_from_arm_q(arm_chain, arm_q, gripper_type, part_transform, has_ft_sensor=False):

    gripper_pos, gripper_quat = get_gripper_pos_quat_from_arm_q(arm_chain, arm_q, gripper_type, has_ft_sensor=has_ft_sensor)
    gripper_matrix = get_transform_matrix_quat(gripper_pos, gripper_quat)
    gripper_qm = get_pos_euler_from_transform_matrix(gripper_matrix)

    part_matrix = gripper_matrix @ part_transform
    part_qm = get_pos_euler_from_transform_matrix(part_matrix)

    return gripper_qm, part_qm


def get_gripper_part_path_from_arm_path(arm_chain, arm_path, gripper_type, part_transform, has_ft_sensor=False):
    
    gripper_path, part_path = [], []
    for arm_q in arm_path:
        gripper_qm, part_qm = get_gripper_part_qm_from_arm_q(arm_chain, arm_q, gripper_type, part_transform, has_ft_sensor=has_ft_sensor)
        gripper_path.append(gripper_qm)
        part_path.append(part_qm)

    return gripper_path, part_path


def get_ik_target_orientation(arm_chain, gripper_type, gripper_quat):
    '''
    Computes the target orientation for the end effector given the gripper orientation
    '''
    ef_init_matrix = arm_chain.forward_kinematics_active(arm_chain.rest_q)[:3, :3] # end effector initial rotation at rest pose
    base_init_direction, l2r_init_direction = [0, 0, 1], R.from_euler('xyz', arm_chain.links[0].origin_orientation).apply([0, -1, 0])
    gripper_init_matrix = R.align_vectors([base_init_direction, l2r_init_direction], [*get_gripper_basis_directions(gripper_type)])[0].as_matrix() # gripper initial rotation

    gripper_target_matrix = R.from_quat(gripper_quat[[1, 2, 3, 0]]).as_matrix()
    ef_target_matrix = gripper_target_matrix @ gripper_init_matrix.T @ ef_init_matrix

    return ef_target_matrix


def inverse_kinematics_correction(arm_chain, arm_q, gripper_type, gripper_quat): # NOTE: deprecated
    '''
    Computes the inverse kinematic on the specified target with correction on the last active joint angle
    '''
    arm_q = arm_q.copy()
    arm_euler = arm_chain.links[0].origin_orientation

    ef_init_matrix = arm_chain.forward_kinematics_active(arm_chain.rest_q)[:3, :3] # end effector initial rotation at rest pose
    base_init_direction, l2r_init_direction = [0, 0, 1], R.from_euler('xyz', arm_euler).apply([0, -1, 0])
    gripper_init_matrix = R.align_vectors([base_init_direction, l2r_init_direction], [*get_gripper_basis_directions(gripper_type)])[0].as_matrix() # gripper initial rotation

    ef_target_matrix = R.from_quat(gripper_quat[[1, 2, 3, 0]]).as_matrix() @ gripper_init_matrix.T @ ef_init_matrix # end effector target rotation for given gripper state
    ef_curr_matrix = arm_chain.forward_kinematics(arm_q) # end effector current rotation from current joint angles

    correct_rotvec = R.from_matrix(ef_curr_matrix[:3, :3].T @ ef_target_matrix).as_rotvec() # rotation vector for last joint angle correction (NOTE: ideally should be 0, 0, theta)
    
    arm_q_active = arm_chain.active_from_full(arm_q)
    arm_q_active[-1] += correct_rotvec[-1]
    if arm_q_active[-1] < -np.pi:
        arm_q_active[-1] += np.pi * 2
    elif arm_q_active[-1] > np.pi:
        arm_q_active[-1] -= np.pi * 2
    arm_q = arm_chain.active_to_full(arm_q_active)

    return arm_q
    

def check_inverse_kinematics_success(arm_chain, arm_q, gripper_type, gripper_quat, eps=1e-3, verbose=False):

    arm_q = arm_q.copy()
    arm_euler = arm_chain.links[0].origin_orientation

    ef_init_matrix = arm_chain.forward_kinematics_active(arm_chain.rest_q)[:3, :3] # end effector initial rotation at rest pose
    base_init_direction, l2r_init_direction = [0, 0, 1], R.from_euler('xyz', arm_euler).apply([0, -1, 0])
    gripper_init_matrix = R.align_vectors([base_init_direction, l2r_init_direction], [*get_gripper_basis_directions(gripper_type)])[0].as_matrix() # gripper initial rotation

    ef_target_matrix = R.from_quat(gripper_quat[[1, 2, 3, 0]]).as_matrix() @ gripper_init_matrix.T @ ef_init_matrix # end effector target rotation for given gripper state
    ef_curr_matrix = arm_chain.forward_kinematics(arm_q) # end effector current rotation from current joint angles

    correct_rotvec = R.from_matrix(ef_curr_matrix[:3, :3].T @ ef_target_matrix).as_rotvec() # rotation vector for last joint angle correction (NOTE: ideally should be 0, 0, theta)
    deviation_norm = np.linalg.norm(correct_rotvec[:2])
    
    if verbose:
        print('IK rotation deviation: {:.4f}'.format(deviation_norm) + f', Success: {deviation_norm < eps}')

    return deviation_norm < eps


def get_arm_path_from_gripper_path(gripper_path, gripper_type, arm_chain, arm_q_init, has_ft_sensor=False):
    arm_path_local = []
    arm_q = arm_q_init.copy() if arm_q_init is not None else None # full
    for qm in gripper_path:
        gripper_pos = qm[:3]
        gripper_rot = R.from_euler('xyz', qm[3:])
        gripper_quat = gripper_rot.as_quat()[[3, 0, 1, 2]]
        gripper_ori = get_ik_target_orientation(arm_chain, gripper_type, gripper_quat)
        ft_pos = get_ft_pos_from_gripper_pos_quat(gripper_type, gripper_pos, gripper_quat) if has_ft_sensor else None

        arm_q, ik_success = arm_chain.inverse_kinematics(target_position=ft_pos if has_ft_sensor else gripper_pos, target_orientation=gripper_ori, orientation_mode='all', initial_position=arm_q, optimizer='L-BFGS-B')

        if not ik_success: # IK not fully checked for every step in the path during planning
            print('inverse kinematics failed')
        arm_path_local.append(arm_q)
    return arm_path_local
