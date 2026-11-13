"""
Final check of a planned log before it goes to the real robot: everything the move arm executes in motion.pkl is checked
with planning/plan_checks.py (exact geometry, no buffers), plus the grasp finger overlap and that the plan was made with the
current workcell (robot height). Writes <log-dir>/verify_report.json, exit code 1 on any violation.

python planning/verify_plan.py --assembly-dir assets/fabrica/human_normal --log-dir logs/exp_human_normal_flat_v2/human_normal
"""

import os

os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)
sys.path.append(project_base_dir)

import json
import pickle
import numpy as np
import networkx as nx

from assets.load import load_part_ids
from assets.transform import get_transform_matrix
from planning.config import GRASP_MIN_FINGER_OVERLAP
from planning.plan_checks import SceneChecker, grasp_finger_overlap
from planning.robot.geometry import load_part_meshes
from planning.robot.workcell import get_panda_base_z
from planning.run_pickup_gen import load_final_part_cfg


def load_checker(assembly_dir, log_dir, grasps):
    part_meshes = load_part_meshes(assembly_dir, transform='none')
    part_meshes = {k[4:]: v for k, v in part_meshes.items()}
    with open(os.path.join(log_dir, 'fixture', 'pickup.json'), 'r') as fp:
        pickup_mats = {
            k: get_transform_matrix(v) for k, v in json.load(fp).items()
        }
    final_meshes = load_final_part_cfg(assembly_dir, grasps['arm'])['mesh']
    return SceneChecker(
        grasps['arm'],
        grasps['gripper'],
        grasps['ft_sensor']['move'],
        part_meshes,
        pickup_mats,
        final_meshes,
    )


def check_workcell(grasps):
    """the grasp arm poses must be computed with the current robot height"""
    base_z = {
        round(float(g[0].arm_pos[2]), 3)
        for part in grasps['grasps'].values()
        for g in part['move'][:1]
        if g[0].arm_pos is not None
    }
    if base_z != {round(get_panda_base_z(), 3)}:
        return [
            f'plan made with robot base z {sorted(base_z)} cm, workcell.py has {get_panda_base_z()} cm: re-plan'
        ]
    return []


def check_grasps(log_dir, grasps):
    with open(os.path.join(log_dir, 'tree_opt.pkl'), 'rb') as fp:
        tree = pickle.load(fp)
    violations, overlaps = [], {}
    used = [
        (attr['move_part'], attr['move_grasp_id'])
        for _, _, attr in tree.edges(data=True)
    ]
    flat_pickup_path = os.path.join(log_dir, 'fixture', 'flat_pickup.json')
    if os.path.exists(
        flat_pickup_path
    ):  # the first part in the fallback: the move arm brings it to the assembly spot with a move grasp of its own
        with open(flat_pickup_path, 'r') as fp:
            used += [
                (part, info['grasp_id'])
                for part, info in json.load(fp).items()
                if info['arm'] == 'place'
            ]
    for part, grasp_id in used:
        grasp = {g[0].grasp_id: g[0] for g in grasps['grasps'][part]['move']}[
            grasp_id
        ]
        overlap = grasp_finger_overlap(grasp, grasps['gripper'])
        overlaps[part] = round(overlap, 2)
        if overlap < GRASP_MIN_FINGER_OVERLAP:
            violations.append(
                f'part {part} grasp {grasp_id}: part reaches only {overlap:.2f} cm into the fingers (min {GRASP_MIN_FINGER_OVERLAP})'
            )
    return violations, overlaps


def iterate_move_paths(motion, part_ids):
    """yields (step index, path, open ratio, state, kwargs for SceneChecker.check_path) for every move-arm path, as run.py executes it"""
    state = {p: 'pickup' for p in part_ids}
    open_ratio, last_inserted = None, None
    picks = [
        (i, s[3])
        for i, s in enumerate(motion)
        if s[0] == 'move'
        and s[1] == 'arm'
        and s[4] == 'transport'
        and s[3] is not None
    ]
    for i, (arm, body, path, part, task) in enumerate(motion):
        if arm == 'hold':
            if body == 'arm' and task == 'transport' and part is not None:
                state[part] = (
                    'final'  # the human (hold arm) puts its part at the assembly
                )
            continue
        if body == 'gripper':
            open_ratio = path
            continue
        if task == 'init':
            continue
        if task == 'transport' and part is not None:
            yield i, path, open_ratio, dict(state), dict(carried=part)
            last_inserted = None
            # a carried part is inserted by the next arm path of the move arm. If it is not (the first part in the fallback: the
            # move arm stands it at the assembly spot and lets go), it is at its place after this path
            next_arm = next(
                (
                    s
                    for s in motion[i + 1 :]
                    if s[0] == 'move' and s[1] == 'arm'
                ),
                None,
            )
            if next_arm is None or not (
                next_arm[4] == 'assembly' and next_arm[3] == part
            ):
                state[part] = 'final'
                last_inserted = part
        elif task == 'assembly':
            yield (
                i,
                path,
                open_ratio,
                dict(state),
                dict(carried=part, inserting=True),
            )
            state[part] = 'final'
            last_inserted = part
        else:  # switch / transport to rest
            pick_part = next((p for j, p in picks if j > i), None)
            yield (
                i,
                path,
                open_ratio,
                dict(state),
                dict(pick_part=pick_part, released_part=last_inserted),
            )


def verify_plan(assembly_dir, log_dir, verbose=True):
    with open(os.path.join(log_dir, 'grasps.pkl'), 'rb') as fp:
        grasps = pickle.load(fp)
    with open(os.path.join(log_dir, 'motion.pkl'), 'rb') as fp:
        motion = pickle.load(fp)

    report = {'workcell': check_workcell(grasps), 'grasps': [], 'paths': {}}
    report['grasps'], report['finger_overlap'] = check_grasps(log_dir, grasps)
    checker = load_checker(assembly_dir, log_dir, grasps)
    for i, path, open_ratio, state, kwargs in iterate_move_paths(
        motion, load_part_ids(assembly_dir)
    ):
        violations = checker.check_path(path, open_ratio, state, **kwargs)
        name = f'step {i + 1} {motion[i][4]}' + (
            f' part {motion[i][3]}' if motion[i][3] is not None else ''
        )
        report['paths'][name] = violations
        if verbose:
            print(
                f'[verify_plan] {name}: '
                + ('ok' if not violations else '; '.join(violations))
            )

    n_bad = (
        len(report['workcell'])
        + len(report['grasps'])
        + sum(len(v) for v in report['paths'].values())
    )
    report['passed'] = n_bad == 0
    report['robot_base_z'] = get_panda_base_z()
    with open(os.path.join(log_dir, 'verify_report.json'), 'w') as fp:
        json.dump(report, fp, indent=4)
    for line in report['workcell'] + report['grasps']:
        print(f'[verify_plan] {line}')
    print(
        f'[verify_plan] finger overlap per move grasp (cm): {report["finger_overlap"]}'
    )
    print(
        f'[verify_plan] {"PASSED" if report["passed"] else f"FAILED ({n_bad} violations)"}: {os.path.join(log_dir, "verify_report.json")}'
    )
    return report


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--assembly-dir', type=str, required=True)
    parser.add_argument('--log-dir', type=str, required=True)
    args = parser.parse_args()
    from planning.robot.workcell import use_log_workcell

    use_log_workcell(
        args.log_dir
    )  # the place of the assembly this log is planned for
    sys.exit(
        0 if verify_plan(args.assembly_dir, args.log_dir)['passed'] else 1
    )
