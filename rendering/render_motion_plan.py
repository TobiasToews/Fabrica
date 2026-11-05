import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
import redmax_py as redmax
import os
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import json
import pickle

from utils.renderer import SimRenderer
from assets.load import load_part_ids
from assets.transform import get_transform_matrix, get_pos_euler_from_transform_matrix, q_to_pos_quat
from planning.robot.geometry import get_gripper_finger_states, get_gripper_base_name
from planning.robot.util_arm import get_arm_chain, get_gripper_pos_quat_from_arm_q, get_gripper_path_from_arm_path
from planning.robot.sim_string import arr_to_str, get_color, get_gripper_string, get_arm_string, get_arm_joints
from planning.run_motion_plan import OPEN_RATIO_REST


def create_assembly_dualarm_fixture_xml(assembly_dir, fixture_dir, arm_chain_move, arm_chain_hold, gripper_type, has_ft_sensor, timestep=1e-3):
    arm_quat_move = R.from_euler('xyz', arm_chain_move.base_euler).as_quat()[[3, 0, 1, 2]]
    arm_quat_hold = R.from_euler('xyz', arm_chain_hold.base_euler).as_quat()[[3, 0, 1, 2]]
    rest_q_move = arm_chain_move.active_to_full(arm_chain_move.rest_q)
    rest_q_hold = arm_chain_hold.active_to_full(arm_chain_hold.rest_q)
    gripper_pos_move, gripper_quat_move = get_gripper_pos_quat_from_arm_q(arm_chain_move, rest_q_move, gripper_type, has_ft_sensor=has_ft_sensor['move'])
    gripper_pos_hold, gripper_quat_hold = get_gripper_pos_quat_from_arm_q(arm_chain_hold, rest_q_hold, gripper_type, has_ft_sensor=has_ft_sensor['hold'])
    arm_type = arm_chain_move.arm_type # NOTE: assume move and hold have the same arm type
    # board_quat = "0.7073883 0 0 0.7068252" if arm_type == 'panda' else "1 0 0 0"
    board_quat = "0.7073883 0 0 0.7068252"
    with open(os.path.join(fixture_dir, 'pickup.json'), 'r') as f:
        part_pickup_pose = json.load(f)
    string = f'''
<redmax model="dual_gripper_arm">
<option integrator="BDF1" timestep="{timestep}" gravity="0. 0. 1e-12"/>
<ground pos="0 0 -1.0" normal="0 0 1"/>
'''
    for part_id in load_part_ids(assembly_dir):
        pos, quat = q_to_pos_quat(part_pickup_pose[part_id])
        string += f'''
<robot>
    <link name="part{part_id}">
        <joint name="part{part_id}" type="free3d-exp" axis="0. 0. 0." pos="{arr_to_str(pos)}" quat="{arr_to_str(quat)}" frame="WORLD" damping="0"/>
        <body name="part{part_id}" type="mesh" filename="{assembly_dir}/{part_id}.obj" pos="0 0 0" quat="1 0 0 0" transform_type="OBJ_TO_JOINT" density="1" mu="0" rgba="{arr_to_str(get_color(part_id))}"/>
    </link>
</robot>
'''
    string += f'''
<robot>
    <link name="optical_board">
        <joint name="optical_board" type="fixed" axis="0. 0. 0." pos="0 0 0" quat="{board_quat}" frame="WORLD" damping="0"/>
        <body name="optical_board" type="mesh" filename="assets/optical_board.obj" pos="0 0 0" quat="1 0 0 0" transform_type="OBJ_TO_JOINT" density="1" mu="0" rgba="0.2 0.2 0.2 1.0"/>
    </link>
</robot>
'''
    if os.path.exists(os.path.join(fixture_dir, 'fixture.obj')): # no fixture: parts lie flat on the board (run_pickup_gen.py)
        string += f'''
<robot>
    <link name="fixture">
        <joint name="fixture" type="fixed" axis="0. 0. 0." pos="0 0 0" quat="1 0 0 0" frame="WORLD" damping="0"/>
        <body name="fixture" type="mesh" filename="{fixture_dir}/fixture.obj" pos="0 0 0" quat="1 0 0 0" transform_type="OBJ_TO_JOINT" density="1" mu="0" rgba="0.5 0.5 0.5 1.0"/>
    </link>
</robot>
'''
    string += get_gripper_string(gripper_type, gripper_pos_move, gripper_quat_move, fixed=False, has_ft_sensor=has_ft_sensor['move'], suffix='move')
    string += get_gripper_string(gripper_type, gripper_pos_hold, gripper_quat_hold, fixed=False, has_ft_sensor=has_ft_sensor['hold'], suffix='hold')
    string += get_arm_string(arm_chain_move.arm_type, arm_chain_move.base_pos, arm_quat_move, suffix='move')
    string += get_arm_string(arm_chain_hold.arm_type, arm_chain_hold.base_pos, arm_quat_hold, suffix='hold')
    string += f'''
</redmax>
    '''
    return string
    

def get_camera_option(option):
    if option == 0: # close
        camera_lookat = [-1, 1, 0]
        camera_pos = [1.25, -1.5, 1.5]
    elif option == 1: # far
        camera_lookat = [4.73023, -3.64037, 5.60194]
        camera_pos = [5.34119, -4.31921, 6.00925]
    else:
        raise NotImplementedError
    return camera_lookat, camera_pos


def render_motion_plan(assembly_dir, log_dir, record_path=None, make_gif=False, camera_lookat=None, camera_pos=None, body_color_map=None, seed=0, fps=None, speed=1.0):

    grasps_path = os.path.join(log_dir, 'grasps.pkl')
    if not os.path.exists(grasps_path):
        print(f'[render_motion_plan] {grasps_path} not found')
        return
    fixture_dir = os.path.join(log_dir, 'fixture')
    if not os.path.exists(fixture_dir):
        print(f'[render_motion_plan] {fixture_dir} not found')
        return
    motion_path = os.path.join(log_dir, 'motion.pkl')
    if not os.path.exists(motion_path):
        print(f'[render_motion_plan] {motion_path} not found')
        return
    
    with open(grasps_path, 'rb') as fp:
        grasps = pickle.load(fp)
    arm_type, gripper_type, has_ft_sensor = grasps['arm'], grasps['gripper'], grasps['ft_sensor']
    
    part_ids = load_part_ids(assembly_dir)
    num_parts = len(part_ids)
    
    with open(motion_path, 'rb') as fp:
        motion = pickle.load(fp)

    arm_chain_move = get_arm_chain(arm_type, 'move')
    arm_chain_hold = get_arm_chain(arm_type, 'hold')
    arm_chains = {'move': arm_chain_move, 'hold': arm_chain_hold}
    arm_joints = get_arm_joints(arm_type)
    num_dof_arm = len(arm_joints)

    np.random.seed(seed)
    asset_folder = os.path.join(project_base_dir, './assets')

    # initialize simulation
    xml_string = create_assembly_dualarm_fixture_xml(assembly_dir, fixture_dir, arm_chain_move, arm_chain_hold, gripper_type, has_ft_sensor)
    sim = redmax.Simulation(xml_string, asset_folder) # order: parts, gripper move, gripper hold, arm move, arm hold

    if camera_lookat is not None:
        sim.viewer_options.camera_lookat = camera_lookat
    if camera_pos is not None:
        sim.viewer_options.camera_pos = camera_pos
    if body_color_map is not None:
        sim.set_body_color_map(body_color_map)
    
    # set initial joint states
    for arm_joint_name, arm_joint_state in zip(arm_joints, arm_chain_move.rest_q):
        sim.set_joint_q_init(arm_joint_name + '_move', np.array([arm_joint_state]))
    for arm_joint_name, arm_joint_state in zip(arm_joints, arm_chain_hold.rest_q):
        sim.set_joint_q_init(arm_joint_name + '_hold', np.array([arm_joint_state]))
    finger_move_states = get_gripper_finger_states(gripper_type, OPEN_RATIO_REST, suffix='move')
    for finger_name, finger_state in finger_move_states.items():
        sim.set_joint_q_init(finger_name, np.array(finger_state))
    finger_hold_states = get_gripper_finger_states(gripper_type, OPEN_RATIO_REST, suffix='hold')
    num_dof_finger = len(finger_move_states)
    for finger_name, finger_state in finger_hold_states.items():
        sim.set_joint_q_init(finger_name, np.array(finger_state))
    sim.reset(backward_flag=False)

    # degree of freedom mapping
    assert sim.ndof_r == num_parts * 6 + 2 * (6 + num_dof_finger) + 2 * num_dof_arm
    dof_map = {'part': {}, 'move': {'gripper': None, 'arm': None}, 'hold': {'gripper': None, 'arm': {}}}
    dof_map['part'] = {part_ids[i]: i * 6 for i in range(num_parts)}
    dof_map['move']['gripper'] = num_parts * 6
    dof_map['hold']['gripper'] = dof_map['move']['gripper'] + 6 + num_dof_finger
    dof_map['move']['arm'] = dof_map['hold']['gripper'] + 6 + num_dof_finger
    dof_map['hold']['arm'] = dof_map['move']['arm'] + num_dof_arm

    q_curr = sim.get_q()
    q_his = [q_curr]

    for motion_step in motion:
        motion_type, body_type, path, active_part, description = motion_step
        # print(motion_type, body_type, path is not None, active_part, description)
        assert path is not None

        if body_type == 'arm':
            arm_path_local = path
            arm_path = [arm_chains[motion_type].active_to_full(arm_state) for arm_state in arm_path_local]
            gripper_path = get_gripper_path_from_arm_path(arm_chains[motion_type], arm_path, gripper_type, has_ft_sensor=has_ft_sensor[motion_type])
            gripper_base_name = get_gripper_base_name(gripper_type, suffix=motion_type)
            gripper_path_local = [sim.get_joint_q_from_qm(gripper_base_name, gripper_state) for gripper_state in gripper_path]
            arm_dof_start = dof_map[motion_type]['arm']
            arm_dof_end = arm_dof_start + num_dof_arm
            gripper_dof_start = dof_map[motion_type]['gripper']
            gripper_dof_end = gripper_dof_start + 6
            if active_part is None: # move arm without grasping part
                for arm_state_local, gripper_state_local in zip(arm_path_local, gripper_path_local):
                    q_curr = q_curr.copy()
                    q_curr[arm_dof_start:arm_dof_end] = arm_state_local
                    q_curr[gripper_dof_start:gripper_dof_end] = gripper_state_local
                    q_his.append(q_curr)
            else: # move arm with grasping part
                part_dof_start = dof_map['part'][active_part]
                part_dof_end = part_dof_start + 6
                part_state_curr = q_curr[part_dof_start:part_dof_end]
                gripper_state_curr = q_curr[gripper_dof_start:gripper_dof_end]
                gripper_to_part_transform = np.linalg.inv(get_transform_matrix(sim.get_joint_qm_from_q(gripper_base_name, gripper_state_curr))) @ get_transform_matrix(sim.get_joint_qm_from_q(f'part{active_part}', part_state_curr))
                for arm_state_local, gripper_state_local, gripper_state_global in zip(arm_path_local, gripper_path_local, gripper_path):
                    q_curr = q_curr.copy()
                    q_curr[arm_dof_start:arm_dof_end] = arm_state_local
                    q_curr[gripper_dof_start:gripper_dof_end] = gripper_state_local
                    q_curr[part_dof_start:part_dof_end] = sim.get_joint_q_from_qm(f'part{active_part}', get_pos_euler_from_transform_matrix(get_transform_matrix(gripper_state_global) @ gripper_to_part_transform))
                    q_his.append(q_curr)
        elif body_type == 'gripper':
            finger_dof_start = dof_map[motion_type]['gripper'] + 6
            finger_dof_end = finger_dof_start + num_dof_finger
            open_ratio = path
            q_curr = q_curr.copy()
            finger_states = []
            for finger_state in get_gripper_finger_states(gripper_type, open_ratio, suffix=motion_type).values():
                finger_states.extend(finger_state)
            q_curr[finger_dof_start:finger_dof_end] = finger_states
            q_his.append(q_curr)

    sim.set_state_his(q_his, [np.zeros_like(q_his[0]) for _ in range(len(q_his))])

    if record_path is not None:
        SimRenderer.replay(sim, record=True, record_path=record_path, make_video=not make_gif, fps=fps)
    else:
        sim.viewer_options.loop = False
        sim.viewer_options.infinite = False
        sim.viewer_options.speed = speed # in the viewer: F = faster, S = slower, space = pause
        sim.replay()

    sim_matrices = sim.export_replay_matrices()
    np.save(os.path.join(log_dir, 'traj.npy'), sim_matrices, allow_pickle=True)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--assembly-dir', type=str, required=True)
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--record-path', type=str, default=None)
    parser.add_argument('--make-gif', default=False, action='store_true')
    parser.add_argument('--camera-option', type=int, default=1)
    parser.add_argument('--camera-lookat', type=float, nargs=3, default=None)
    parser.add_argument('--camera-pos', type=float, nargs=3, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--speed', type=float, default=1.0, help='initial playback speed of the interactive viewer')
    args = parser.parse_args()
    
    camera_lookat, camera_pos = get_camera_option(args.camera_option)
    if args.camera_lookat is not None:
        camera_lookat = args.camera_lookat
    if args.camera_pos is not None:
        camera_pos = args.camera_pos
    
    render_motion_plan(args.assembly_dir, args.log_dir, 
        record_path=args.record_path, make_gif=args.make_gif, camera_lookat=camera_lookat, camera_pos=camera_pos, seed=args.seed, fps=args.fps, speed=args.speed)
