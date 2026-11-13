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
from assets.load import load_pos_quat_dict, load_part_ids
from assets.transform import get_pos_quat_from_pose, quat_to_euler
from planning.robot.geometry import get_gripper_finger_states, get_gripper_base_name
from planning.robot.util_grasp import get_gripper_path_from_part_path
from planning.robot.util_arm import get_arm_chain, get_arm_path_from_gripper_path
from planning.robot.sim_string import arr_to_str, get_color, get_gripper_string, get_arm_string
from planning.robot.workcell import get_assembly_center
from planning.run_seq_plan import SequencePlanner
from planning.run_seq_opt import SequenceOptimizer
from planning.sequence.combine_animation import combine_animation
from utils.parallel import parallel_execute


def create_assembly_dualarm_xml(assembly_dir, move_id, hold_id, still_ids,
    gripper_type=None, gripper_move_pos=None, gripper_move_quat=None, gripper_hold_pos=None, gripper_hold_quat=None,
    arm_type=None, has_ft_sensor=None, arm_move_pos=None, arm_move_euler=None, arm_hold_pos=None, arm_hold_euler=None, timestep=1e-3):
    part_ids = [move_id, hold_id] + still_ids
    pos_dict_final, quat_dict_final = load_pos_quat_dict(assembly_dir, transform='final')
    pos_dict_final = {part_id: pos_dict_final[part_id] + get_assembly_center(arm_type) for part_id in part_ids}
    arm_move_quat = R.from_euler('xyz', arm_move_euler).as_quat()[[3, 0, 1, 2]]
    arm_hold_quat = R.from_euler('xyz', arm_hold_euler).as_quat()[[3, 0, 1, 2]]
    board_quat = "0.7073883 0 0 0.7068252" if arm_type == 'panda' else "1 0 0 0"
    string = f'''
<redmax model="dual_gripper_arm">
<option integrator="BDF1" timestep="{timestep}" gravity="0. 0. 1e-12"/>
<ground pos="0 0 -1.0" normal="0 0 1"/>
'''
    for part_id in part_ids:
        joint_type = 'free3d-exp' if part_id in [move_id, hold_id] else 'fixed'
        pos, quat = get_pos_quat_from_pose(pos_dict_final[part_id], quat_dict_final[part_id], None)
        string += f'''
<robot>
    <link name="part{part_id}">
        <joint name="part{part_id}" type="{joint_type}" axis="0. 0. 0." pos="{arr_to_str(pos)}" quat="{arr_to_str(quat)}" frame="WORLD" damping="0"/>
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
    gripper_move_pos, gripper_move_quat = get_pos_quat_from_pose(gripper_move_pos, gripper_move_quat, None)
    string += get_gripper_string(gripper_type, gripper_move_pos, gripper_move_quat, fixed=False, has_ft_sensor=has_ft_sensor['move'], suffix='move')
    string += get_gripper_string(gripper_type, gripper_hold_pos, gripper_hold_quat, fixed=False, has_ft_sensor=has_ft_sensor['hold'], suffix='hold')
    string += get_arm_string(arm_type, arm_move_pos, arm_move_quat, suffix='move')
    string += get_arm_string(arm_type, arm_hold_pos, arm_hold_quat, suffix='hold')
    string += f'''
</redmax>
    '''
    return string


def render_assembly_step(asset_folder, assembly_dir, move_id, hold_id, still_ids, part_path, gripper_type, arm_type, has_ft_sensor, grasps_move, grasp_hold,
    camera_lookat=None, camera_pos=None, body_color_map=None, reverse=False, render=True, record_path=None, make_video=False, fps=None):
    if part_path is None:
        print('no path found')
        return

    grasp_move_init, grasp_move_final = grasps_move[-1], grasps_move[0]

    # initialize simulation
    xml_string = create_assembly_dualarm_xml(
        assembly_dir=assembly_dir, move_id=move_id, hold_id=hold_id, still_ids=still_ids, 
        gripper_type=gripper_type, gripper_move_pos=grasp_move_init.pos, gripper_move_quat=grasp_move_init.quat, gripper_hold_pos=grasp_hold.pos, gripper_hold_quat=grasp_hold.quat,
        arm_type=arm_type, has_ft_sensor=has_ft_sensor, arm_move_pos=grasp_move_init.arm_pos, arm_move_euler=grasp_move_init.arm_euler, arm_hold_pos=grasp_hold.arm_pos, arm_hold_euler=grasp_hold.arm_euler, timestep=1 / len(part_path))
    sim = redmax.Simulation(xml_string, asset_folder)
    if camera_lookat is not None:
        sim.viewer_options.camera_lookat = camera_lookat
    if camera_pos is not None:
        sim.viewer_options.camera_pos = camera_pos
    
    # set finger open ratio
    finger_move_states = get_gripper_finger_states(gripper_type, grasp_move_init.open_ratio, suffix='move')
    for finger_name, finger_state in finger_move_states.items():
        sim.set_joint_q_init(finger_name, np.array(finger_state))
    finger_hold_states = get_gripper_finger_states(gripper_type, grasp_hold.open_ratio, suffix='hold')
    for finger_name, finger_state in finger_hold_states.items():
        sim.set_joint_q_init(finger_name, np.array(finger_state))
    sim.reset(backward_flag=False)

    if body_color_map is not None:
        sim.set_body_color_map(body_color_map)

    # get gripper path from grasps
    gripper_move_path = get_gripper_path_from_part_path(part_path, grasp_move_final.pos, grasp_move_final.quat)
    gripper_hold_path = [np.concatenate([grasp_hold.pos, quat_to_euler(grasp_hold.quat)]) for _ in range(len(gripper_move_path))]

    # transform from global coordinate to local coordinate
    part_move_path_local = [sim.get_joint_q_from_qm(f'part{move_id}', qm) for qm in part_path]
    part_hold_path_local = [sim.get_joint_q(f'part{hold_id}') for _ in range(len(part_move_path_local))]
    gripper_move_base_name = get_gripper_base_name(gripper_type, suffix='move')
    gripper_hold_base_name = get_gripper_base_name(gripper_type, suffix='hold')
    gripper_move_path_local = [sim.get_joint_q_from_qm(gripper_move_base_name, qm) for qm in gripper_move_path]
    finger_move_path_local = [np.concatenate(list(finger_move_states.values())) for _ in range(len(gripper_move_path_local))]
    gripper_hold_path_local = [sim.get_joint_q_from_qm(gripper_hold_base_name, qm) for qm in gripper_hold_path]
    finger_hold_path_local = [np.concatenate(list(finger_hold_states.values())) for _ in range(len(gripper_hold_path_local))]

    # get arm path
    arm_move_chain = get_arm_chain(arm_type, 'move')
    arm_hold_chain = get_arm_chain(arm_type, 'hold')
    arm_move_path_local = get_arm_path_from_gripper_path(gripper_move_path, gripper_type, arm_move_chain, grasp_move_final.arm_q, has_ft_sensor=has_ft_sensor['move']) # NOTE: expensive to compute all IKs!
    arm_move_path_local = [arm_move_chain.active_from_full(arm_q) for arm_q in arm_move_path_local] # active
    arm_hold_path_local = [arm_hold_chain.active_from_full(grasp_hold.arm_q) for _ in range(len(arm_move_path_local))] # active

    # convert disassembly to assembly
    part_move_path_local, part_hold_path_local, gripper_move_path_local, finger_move_path_local, gripper_hold_path_local, finger_hold_path_local, arm_move_path_local, arm_hold_path_local = \
        part_move_path_local[::-1], part_hold_path_local[::-1], gripper_move_path_local[::-1], finger_move_path_local[::-1], gripper_hold_path_local[::-1], finger_hold_path_local[::-1], arm_move_path_local[::-1], arm_hold_path_local[::-1]
    
    # set state history
    states = [np.concatenate([part_move_state, part_hold_state, gripper_move_state, finger_move_state, gripper_hold_state, finger_hold_state, arm_move_state, arm_hold_state]) 
        for part_move_state, part_hold_state, gripper_move_state, finger_move_state, gripper_hold_state, finger_hold_state, arm_move_state, arm_hold_state 
        in zip(part_move_path_local, part_hold_path_local, gripper_move_path_local, finger_move_path_local, gripper_hold_path_local, finger_hold_path_local, arm_move_path_local, arm_hold_path_local)]
    if not reverse:
        states = states[::-1]
    sim.set_state_his(states, [np.zeros_like(states[0]) for _ in range(len(states))])

    if render:
        SimRenderer.replay(sim, record=record_path is not None, record_path=record_path, make_video=make_video, fps=fps)

    return sim.export_replay_matrices()


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


def render_seq_plan(assembly_dir, log_dir, optimized, record_dir, reverse=False, combine_record=False, make_gif=False, camera_lookat=None, camera_pos=None, num_proc=1, seed=0, fps=None):

    precedence_path = os.path.join(log_dir, 'precedence.pkl')
    if not os.path.exists(precedence_path):
        print(f'[render_seq_plan] {precedence_path} not found')
        return
    grasps_path = os.path.join(log_dir, 'grasps.pkl')
    if not os.path.exists(grasps_path):
        print(f'[render_seq_plan] {grasps_path} not found')
        return

    with open(precedence_path, 'rb') as fp:
        G_preced = pickle.load(fp)
    with open(grasps_path, 'rb') as fp:
        grasps = pickle.load(fp)

    np.random.seed(seed)
    asset_folder = os.path.join(project_base_dir, './assets')
    os.makedirs(record_dir, exist_ok=True)

    if optimized:
        with open(os.path.join(log_dir, 'tree_opt.pkl'), 'rb') as fp:
            tree = pickle.load(fp)
        seq_optimizer = SequenceOptimizer(G_preced, grasps)
        sequence, grasps_sequence = seq_optimizer.get_sequence(tree)
    else:
        with open(os.path.join(log_dir, 'tree.pkl'), 'rb') as fp:
            tree = pickle.load(fp)
        seq_planner = SequencePlanner(asset_folder, assembly_dir, G_preced, grasps, save_sdf=True, contact_eps=None)
        sequence, grasps_sequence = seq_planner.sample_sequence(tree, seed=seed)
    if sequence is None:
        print('[render_seq_plan] failed plan')
        return

    worker_args, worker_kwargs = [], []
    
    parts_assembled = sorted(load_part_ids(assembly_dir))
    for i, ((part_move, part_hold), (grasps_move, grasp_hold)) in enumerate(zip(sequence, grasps_sequence)):
        parts_rest = parts_assembled.copy()
        parts_rest.remove(part_move)
        parts_rest.remove(part_hold)

        step_name = f'{i}_{part_move}-{part_hold}'
        record_path = os.path.join(record_dir, f'{step_name}.gif' if make_gif else f'{step_name}.mp4')

        worker_args.append((asset_folder, assembly_dir, part_move, part_hold, parts_rest, G_preced.nodes[part_move]['path'], grasps['gripper'], grasps['arm'], grasps['ft_sensor'], grasps_move, grasp_hold))
        worker_kwargs.append(dict(camera_lookat=camera_lookat, camera_pos=camera_pos, body_color_map=None, reverse=reverse, render=True, record_path=record_path, make_video=not make_gif, fps=fps))

        parts_assembled.remove(part_move)

    for _ in parallel_execute(render_assembly_step, worker_args, worker_kwargs, num_proc=num_proc):
        pass
    
    if combine_record:
        output_path = os.path.join(record_dir, os.path.basename(os.path.abspath(record_dir)))
        if make_gif:
            output_path += '.gif'
        else:
            output_path += '.mp4'
        combine_animation(record_dir, output_path, reverse)


if __name__ == '__main__':
    import platform, multiprocessing
    if platform.system() == "Darwin":
        multiprocessing.set_start_method('spawn')
    
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--assembly-dir', type=str, required=True)
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--optimized', default=False, action='store_true')
    parser.add_argument('--record-dir', type=str, required=True)
    parser.add_argument('--reverse', default=False, action='store_true')
    parser.add_argument('--combine-record', default=False, action='store_true')
    parser.add_argument('--make-gif', default=False, action='store_true')
    parser.add_argument('--camera-option', type=int, default=1)
    parser.add_argument('--camera-lookat', type=float, nargs=3, default=None)
    parser.add_argument('--camera-pos', type=float, nargs=3, default=None)
    parser.add_argument('--num-proc', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()

    from planning.robot.workcell import use_log_workcell
    use_log_workcell(args.log_dir) # the place of the assembly this log is planned for
    camera_lookat, camera_pos = get_camera_option(args.camera_option)
    if args.camera_lookat is not None:
        camera_lookat = args.camera_lookat
    if args.camera_pos is not None:
        camera_pos = args.camera_pos
    
    render_seq_plan(args.assembly_dir, args.log_dir, args.optimized, args.record_dir, args.reverse, args.combine_record, args.make_gif, camera_lookat, camera_pos, args.num_proc, args.seed, args.fps)
