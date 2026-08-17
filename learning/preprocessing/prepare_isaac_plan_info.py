import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(project_base_dir)

import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R

from assets.save import sample_path
from planning.run_seq_opt import SequenceOptimizer
from planning.run_seq_plan import SequencePlanner
from planning.robot.util_arm import get_arm_chain
from planning.robot.workcell import get_assembly_center


def prepare_isaac_plan_info(log_dir, plan_info_path):
    '''
    plan_info = {
        (part_move, part_hold): {
            'arm_q_plug': [arm_q_move_preassembly, arm_q_move_assembled],
            'arm_q_socket': arm_q_hold,
            'open_ratio': open_ratio,
            'path': path,
        },
        ...
    }
    '''

    precedence_path = os.path.join(log_dir, 'precedence.pkl')
    if not os.path.exists(precedence_path):
        print(f'[prepare_isaac_grasp] {precedence_path} not found')
        return
    grasp_path = os.path.join(log_dir, 'grasps.pkl')
    if not os.path.exists(grasp_path):
        print(f'[prepare_isaac_grasp] {grasp_path} not found')
        return
    tree_path = os.path.join(log_dir, 'tree_opt.pkl')
    # tree_path = os.path.join(log_dir, 'tree.pkl')
    if not os.path.exists(tree_path):
        print(f'[prepare_isaac_grasp] {tree_path} not found')
        return
    
    os.makedirs(os.path.dirname(plan_info_path), exist_ok=True)
    
    with open(precedence_path, 'rb') as fp:
        G_preced = pickle.load(fp)
    with open(grasp_path, 'rb') as fp:
        grasps = pickle.load(fp)
    gripper_type, arm_type, has_ft_sensor = grasps['gripper'], grasps['arm'], grasps['ft_sensor']
    assert gripper_type == 'panda' and arm_type == 'panda' and not has_ft_sensor['move'] and not has_ft_sensor['hold']
    with open(tree_path, 'rb') as fp:
        tree = pickle.load(fp)
    assembly_center = get_assembly_center(arm_type)

    seq_optimizer = SequenceOptimizer(G_preced, grasps)
    sequence, grasps_sequence = seq_optimizer.get_sequence(tree)
    # asset_folder = os.path.join(project_base_dir, './assets')
    # assembly_dir = os.path.join(asset_folder, 'fabrica', os.path.basename(log_dir))
    # seq_planner = SequencePlanner(asset_folder, assembly_dir, G_preced, grasps, save_sdf=True, contact_eps=None)
    # sequence, grasps_sequence = seq_planner.sample_sequence(tree, seed=1)
    if sequence is None or grasps_sequence is None:
        print(f'[prepare_isaac_grasp] No feasible sequence found in {tree_path}')
        return
    
    plan_info = {}
    arm_chain = get_arm_chain(arm_type, motion_type='move')
    
    sequence, grasps_sequence = sequence[::-1], grasps_sequence[::-1] # reverse the sequence to be forward assembly
    for (part_move, part_hold), (grasps_move, grasp_hold) in zip(sequence, grasps_sequence):
        path = sample_path(G_preced.nodes[part_move]['path'], n_frame=10)
        path_new = []
        for state in path:
            state_new = np.concatenate([(state[:3] - assembly_center) * 0.01, R.from_euler('xyz', state[3:]).as_quat()])
            path_new.append(state_new)
        plan_info[(part_move, part_hold)] = {
            'arm_q_plug': [arm_chain.active_from_full(grasps_move[-1].arm_q), arm_chain.active_from_full(grasps_move[0].arm_q)],
            'arm_q_socket': arm_chain.active_from_full(grasp_hold.arm_q),
            'open_ratio_plug': grasps_move[0].open_ratio,
            'open_ratio_socket': grasp_hold.open_ratio,
            'path': path_new, # disassembly path
        }

    with open(plan_info_path, 'wb') as fp:
        pickle.dump(plan_info, fp)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--plan-info-path', type=str, required=True)
    args = parser.parse_args()

    prepare_isaac_plan_info(args.log_dir, args.plan_info_path)

