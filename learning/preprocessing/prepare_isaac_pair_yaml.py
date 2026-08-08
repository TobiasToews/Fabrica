import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import pickle
import yaml

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(project_base_dir)


def prepare_isaac_pair_yaml(log_dir, yaml_path):

    pair_info = {}

    for subdir_name in os.listdir(log_dir):
        sublog_dir = os.path.join(log_dir, subdir_name)
        if not os.path.isdir(sublog_dir):
            continue

        pair_info[subdir_name] = {}

        precedence_path = os.path.join(sublog_dir, 'precedence.pkl')
        if not os.path.exists(precedence_path):
            print(f'[prepare_isaac_pair_yaml] {precedence_path} not found')
            return
        with open(precedence_path, 'rb') as fp:
            G_preced = pickle.load(fp)

        for ei, edge in enumerate(G_preced.edges):
            pair_info[subdir_name][ei] = {'plug': edge[1], 'socket': edge[0]}

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

