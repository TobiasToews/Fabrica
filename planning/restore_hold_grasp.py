import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import json
import pickle
import networkx as nx
import numpy as np

from planning.run_grasp_arm_gen import GraspArmGenerator
from planning.run_seq_opt import SequenceOptimizer
from planning.utils.copy_logs import copy_logs, get_files_to_copy, get_stats_keys_to_copy


def find_same_grasp(grasps, grasp, atol=1e-4):
    for g in grasps:
        if np.allclose(g.pos, grasp.pos, atol=atol) and np.isclose(abs(np.dot(g.quat, grasp.quat)), 1.0, atol=atol):
            return g
    return None


def restore_hold_grasp(src_dir, ref_grasps_path, ref_tree_path, dst_dir, assembly_dir, reduced_limit=0.1, seed=0, verbose=False):
    '''
    Re-plan an older grasp sequence (e.g. the one an already printed fixture was made for) on the current workcell:
    move grasps are taken from src_dir (already planned on the current workcell), the hold grasp is taken from the
    reference grasps and its arm IK / feasibility is recomputed on the current workcell
    '''
    with open(os.path.join(src_dir, 'precedence.pkl'), 'rb') as fp:
        G_preced = pickle.load(fp)
    with open(os.path.join(src_dir, 'grasps.pkl'), 'rb') as fp:
        grasps = pickle.load(fp)
    with open(ref_grasps_path, 'rb') as fp:
        grasps_ref = pickle.load(fp)
    with open(ref_tree_path, 'rb') as fp:
        tree_ref = pickle.load(fp)

    steps = [attr for _, _, attr in tree_ref.edges(data=True)] # disassembly order
    hold_ids = {(s['hold_part'], s['hold_grasp_id']) for s in steps}
    assert len(hold_ids) == 1, f'expected one hold grasp for the whole sequence, got {hold_ids}'
    hold_part, hold_grasp_id_ref = hold_ids.pop()

    # move grasps must already exist (same pose) in the current grasps
    move_ids = {}
    for s in steps:
        part, grasp_id = s['move_part'], s['move_grasp_id']
        move_ref = {g[0].grasp_id: g for g in grasps_ref['grasps'][part]['move']}[grasp_id]
        move_cur = {g[0].grasp_id: g for g in grasps['grasps'][part]['move']}.get(grasp_id)
        assert move_cur is not None and np.allclose(move_cur[0].pos, move_ref[0].pos) and np.allclose(move_cur[0].quat, move_ref[0].quat), \
            f'move grasp {grasp_id} of part {part} differs from the reference, cannot reuse it'
        move_ids[part] = grasp_id

    generator = GraspArmGenerator(os.path.join(project_base_dir, 'assets'), assembly_dir, G_preced, grasps['gripper'], grasps['arm'], grasps['ft_sensor'],
        seed=seed, ik_optimizer=grasps['settings']['optimizer'], ik_regularization=1.0, offset_delta=grasps['settings']['offset_delta'], reduced_limit=reduced_limit)

    # recompute the reference hold grasp on the current workcell
    hold_ref = {g.grasp_id: g for g in grasps_ref['grasps'][hold_part]['hold']}[hold_grasp_id_ref]
    hold_cur = find_same_grasp(grasps['grasps'][hold_part]['hold'], hold_ref)
    if hold_cur is None:
        grasp_cand = hold_ref.copy()
        grasp_cand.arm_pos = grasp_cand.arm_euler = grasp_cand.arm_q = grasp_cand.arm_q_init = None
        grasp_cand.arm_paths_pre, grasp_cand.arm_paths_post = [], []
        grasp_cand = generator.compute_contact_points(grasp_cand, hold_part, generator.n_surface_pt)
        result = generator.check_grasp_feasible(grasp_cand, hold_part, verbose=verbose)
        if result is None or result['hold'] is None:
            print(f'[restore_hold_grasp] hold grasp {hold_grasp_id_ref} of part {hold_part} is infeasible on the current workcell')
            return False
        hold_cur = result['hold']
        grasps['grasps'][hold_part]['hold'].append(hold_cur)
        print(f'[restore_hold_grasp] recomputed hold grasp {hold_cur.grasp_id} of part {hold_part}, arm_pos {hold_cur.arm_pos}')
    else:
        print(f'[restore_hold_grasp] hold grasp already in {src_dir} as id {hold_cur.grasp_id}')

    # grasp pairs of the reference sequence
    for part_move, grasp_id in move_ids.items():
        grasp_move = {g[0].grasp_id: g for g in grasps['grasps'][part_move]['move']}[grasp_id]
        _, pairs = generator.check_grasp_id_pair_feasible_batch(grasp_move, [hold_cur], verbose=verbose)
        if len(pairs) == 0:
            print(f'[restore_hold_grasp] move grasp {grasp_id} of part {part_move} collides with hold grasp {hold_cur.grasp_id}')
            return False
        key = (part_move, hold_part)
        grasps['grasp_id_pairs'][key] = list(set(grasps['grasp_id_pairs'].get(key, [])) | set(pairs))

    # reference sequence with the recomputed hold grasp
    tree_opt = nx.DiGraph()
    remap = lambda node: node if node[2] is None else (node[0], node[1], hold_cur.grasp_id if node[1] == hold_part else node[2])
    for u, v, attr in tree_ref.edges(data=True):
        tree_opt.add_edge(remap(u), remap(v), **{**attr, 'hold_grasp_id': hold_cur.grasp_id})

    os.makedirs(dst_dir, exist_ok=True)
    copy_logs(src_dir, dst_dir, get_files_to_copy('seq_plan'), get_stats_keys_to_copy('seq_plan'))
    with open(os.path.join(dst_dir, 'grasps.pkl'), 'wb') as fp:
        pickle.dump(grasps, fp)
    with open(os.path.join(dst_dir, 'tree_opt.pkl'), 'wb') as fp:
        pickle.dump(tree_opt, fp)
    SequenceOptimizer(G_preced, grasps).plot_tree(tree_opt, save_path=os.path.join(dst_dir, 'tree_opt.png'))
    with open(os.path.join(dst_dir, 'restore_info.json'), 'w') as fp:
        json.dump({'src': os.path.relpath(src_dir, project_base_dir), 'ref_grasps': ref_grasps_path, 'ref_tree': ref_tree_path,
                   'hold_part': hold_part, 'hold_grasp_id_ref': int(hold_grasp_id_ref), 'hold_grasp_id': int(hold_cur.grasp_id),
                   'steps': [{**s, 'hold_grasp_id': int(hold_cur.grasp_id)} for s in steps[::-1]]}, fp, indent=4)
    print(f'[restore_hold_grasp] saved to {os.path.relpath(dst_dir, project_base_dir)}')
    return True


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help='log dir planned on the current workcell, e.g. logs/exp_socket_set_small/socket_set_small')
    parser.add_argument('--ref-grasps', type=str, required=True, help='grasps.pkl of the reference plan')
    parser.add_argument('--ref-tree', type=str, required=True, help='tree_opt.pkl of the reference plan')
    parser.add_argument('--dst', type=str, required=True, help='new log dir')
    parser.add_argument('--assembly-dir', type=str, required=True)
    parser.add_argument('--reduced-limit', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    restore_hold_grasp(args.src, args.ref_grasps, args.ref_tree, args.dst, args.assembly_dir, args.reduced_limit, args.seed, args.verbose)
