import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import json
import pickle
import numpy as np
import networkx as nx
from scipy.spatial.transform import Rotation as R

from planning.run_seq_opt import SequenceOptimizer
from planning.utils.copy_logs import copy_logs, get_files_to_copy, get_stats_keys_to_copy


VARIANT_SEP = '__v' # pseudo-assembly name: <assembly>__v<i>


def get_base_assembly(assembly):
    return assembly.split(VARIANT_SEP)[0]


def is_move_node(node):
    # grasp tree node: (remaining_parts, part_id, grasp_id); a moved part is no longer in remaining_parts
    return node[1] is not None and node[1] not in node[0]


def get_move_grasp_rot(seq_optimizer, part, grasp_id):
    grasp = seq_optimizer.grasps[part]['move'][grasp_id][0]
    return R.from_quat(grasp.quat[[1, 2, 3, 0]])


def filter_grasp_tree(grasp_tree, seq_optimizer, disassembly_order=None, move_grasps=None, used_rots=None, min_angle=0.0):
    '''
    Remove move nodes that violate the part order, fixed grasps, or are too similar to grasps of earlier variants
    '''
    n_parts = len(seq_optimizer.parts)
    root = seq_optimizer._find_root_node(grasp_tree)
    keep = []
    for node in grasp_tree.nodes:
        if is_move_node(node):
            part, grasp_id = node[1], node[2]
            layer = n_parts - len(node[0]) - 1 # index of this move in the disassembly sequence
            if disassembly_order is not None and disassembly_order[layer] != part:
                continue
            if move_grasps is not None and part in move_grasps and move_grasps[part] != grasp_id:
                continue
            if used_rots is not None and min_angle > 0 and len(used_rots.get(part, [])) > 0:
                rot = get_move_grasp_rot(seq_optimizer, part, grasp_id)
                if min((rot.inv() * used_rot).magnitude() for used_rot in used_rots[part]) < min_angle:
                    continue
        keep.append(node)
    sub_tree = grasp_tree.subgraph(keep)
    return sub_tree.subgraph({root} | nx.descendants(sub_tree, root)) # drop branches cut off from the root


def link_variant_assets(asset_dir, assembly, variant_name):
    # the pseudo-assembly reuses the meshes of the real assembly, so all downstream scripts treat it as a normal assembly
    link_path = os.path.join(asset_dir, variant_name)
    if os.path.islink(link_path):
        return
    assert not os.path.exists(link_path), f'{link_path} exists and is not a symlink'
    os.symlink(assembly, link_path)


def select_variants(src_dir, dst_exp, n_variants=1, order=None, move_grasps=None, min_angle=np.deg2rad(30.0), asset_dir='assets/fabrica', start_index=0, verbose=False):
    assembly = get_base_assembly(os.path.basename(os.path.normpath(src_dir)))
    asset_dir = os.path.join(project_base_dir, asset_dir)
    assert os.path.isdir(os.path.join(asset_dir, assembly)), f'{os.path.join(asset_dir, assembly)} not found'

    with open(os.path.join(src_dir, 'precedence.pkl'), 'rb') as fp:
        G_preced = pickle.load(fp)
    with open(os.path.join(src_dir, 'grasps.pkl'), 'rb') as fp:
        grasps = pickle.load(fp)
    with open(os.path.join(src_dir, 'tree.pkl'), 'rb') as fp:
        tree = pickle.load(fp)

    seq_optimizer = SequenceOptimizer(G_preced, grasps)
    grasp_tree = seq_optimizer._convert_part_tree_to_grasp_tree(tree, verbose=verbose)
    if grasp_tree is None:
        print(f'[select_variant] No complete sequence in {src_dir}')
        return

    disassembly_order = order[::-1] if order is not None else None # tree is built in disassembly order
    if disassembly_order is not None:
        assert len(disassembly_order) == len(seq_optimizer.parts) - 1, f'--order needs {len(seq_optimizer.parts) - 1} parts (all except the base part)'

    used_rots = {}
    for i in range(n_variants):
        sub_tree = filter_grasp_tree(grasp_tree, seq_optimizer, disassembly_order, move_grasps, used_rots, min_angle)
        result = seq_optimizer._optimize_part_tree_from_grasp_tree(sub_tree, verbose=verbose) if sub_tree.number_of_nodes() > 0 else None
        if result is None:
            print(f'[select_variant] Only found {i} variant(s), try a smaller --min-angle or fewer constraints')
            break
        tree_opt, _ = result

        sequence, grasps_sequence = seq_optimizer.get_sequence(tree_opt)
        for (part_move, _), (grasp_move, _) in zip(sequence, grasps_sequence):
            used_rots.setdefault(part_move, []).append(get_move_grasp_rot(seq_optimizer, part_move, grasp_move[0].grasp_id))

        variant_name = f'{assembly}{VARIANT_SEP}{start_index + i}'
        link_variant_assets(asset_dir, assembly, variant_name)
        dst_dir = os.path.join(project_base_dir, 'logs', dst_exp, variant_name)
        os.makedirs(dst_dir, exist_ok=True)
        copy_logs(src_dir, dst_dir, get_files_to_copy('seq_plan'), get_stats_keys_to_copy('seq_plan'))
        with open(os.path.join(dst_dir, 'tree_opt.pkl'), 'wb') as fp:
            pickle.dump(tree_opt, fp)
        seq_optimizer.plot_tree(tree_opt, save_path=os.path.join(dst_dir, 'tree_opt.png'))

        variant_info = {
            'assembly': assembly,
            'variant': variant_name,
            'source': os.path.relpath(src_dir, project_base_dir),
            'assembly_order': [part_move for part_move, _ in sequence[::-1]],
            'steps': [{'move_part': part_move, 'hold_part': part_hold, 'move_grasp_id': int(grasp_move[0].grasp_id), 'hold_grasp_id': int(grasp_hold.grasp_id)}
                      for (part_move, part_hold), (grasp_move, grasp_hold) in zip(sequence[::-1], grasps_sequence[::-1])],
        }
        with open(os.path.join(dst_dir, 'variant.json'), 'w') as fp:
            json.dump(variant_info, fp, indent=4)
        print(f'[select_variant] {os.path.relpath(dst_dir, project_base_dir)}: order {variant_info["assembly_order"]}, move grasps {[s["move_grasp_id"] for s in variant_info["steps"]]}')


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help='log dir of the planned assembly, e.g. logs/exp_socket_set_small/socket_set_small')
    parser.add_argument('--dst-exp', type=str, required=True, help='new experiment name; variants are saved as logs/<dst-exp>/<assembly>__v<i>')
    parser.add_argument('--n-variants', type=int, default=1)
    parser.add_argument('--order', type=str, default=None, help='assembly order of the moved parts, e.g. 2,1')
    parser.add_argument('--move-grasps', type=str, default=None, help='fix move grasps per part, e.g. 1:687,2:527')
    parser.add_argument('--min-angle', type=float, default=30.0, help='min rotation [deg] between move grasps of the same part across variants')
    parser.add_argument('--asset-dir', type=str, default='assets/fabrica', help='where the assembly meshes are; variant symlinks are created here')
    parser.add_argument('--start-index', type=int, default=0, help='index of the first variant, to add variants to an existing experiment without overwriting')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    order = args.order.split(',') if args.order is not None else None
    move_grasps = {k: int(v) for k, v in (s.split(':') for s in args.move_grasps.split(','))} if args.move_grasps is not None else None
    select_variants(args.src, args.dst_exp, args.n_variants, order, move_grasps, np.deg2rad(args.min_angle), args.asset_dir, args.start_index, args.verbose)
