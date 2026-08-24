import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import pickle
import yaml

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(project_base_dir)

from planning.run_seq_opt import SequenceOptimizer


def prepare_isaac_pair_yaml(log_dir, yaml_path):
    '''
    Pairs must match the (part_move, part_hold) pairs actually produced by the optimized
    sequence (same source used by prepare_isaac_plan_info.py / generate_franka_urdf_from_plan.py).
    Enumerating precedence.pkl edges directly is NOT equivalent whenever a part has more than
    one precedence predecessor (e.g. a part whose extraction path collides with two other parts) -
    the graph then has more edges than the optimized sequence has steps, and the optimizer may
    pick a hold part other than the one implied by a given edge.
    '''

    pair_info = {}

    for subdir_name in os.listdir(log_dir):
        sublog_dir = os.path.join(log_dir, subdir_name)
        if not os.path.isdir(sublog_dir):
            continue

        precedence_path = os.path.join(sublog_dir, 'precedence.pkl')
        if not os.path.exists(precedence_path):
            print(f'[prepare_isaac_pair_yaml] {precedence_path} not found')
            continue
        grasp_path = os.path.join(sublog_dir, 'grasps.pkl')
        if not os.path.exists(grasp_path):
            print(f'[prepare_isaac_pair_yaml] {grasp_path} not found')
            continue
        tree_path = os.path.join(sublog_dir, 'tree_opt.pkl')
        if not os.path.exists(tree_path):
            print(f'[prepare_isaac_pair_yaml] {tree_path} not found')
            continue

        with open(precedence_path, 'rb') as fp:
            G_preced = pickle.load(fp)
        with open(grasp_path, 'rb') as fp:
            grasps = pickle.load(fp)
        with open(tree_path, 'rb') as fp:
            tree = pickle.load(fp)

        seq_optimizer = SequenceOptimizer(G_preced, grasps)
        sequence, _ = seq_optimizer.get_sequence(tree)
        if sequence is None:
            print(f'[prepare_isaac_pair_yaml] No feasible sequence found in {tree_path}')
            continue

        pair_info[subdir_name] = {}
        for ei, (part_move, part_hold) in enumerate(sequence):
            pair_info[subdir_name][ei] = {'plug': part_move, 'socket': part_hold}

    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, 'w') as fp:
        yaml.dump(pair_info, fp)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--yaml-path', type=str, required=True)
    args = parser.parse_args()

    prepare_isaac_pair_yaml(args.log_dir, args.yaml_path)

