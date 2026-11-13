'''
Move a planned log to a new robot base height (planning/robot/workcell.py get_panda_base_z) without re-sampling grasps:
every grasp keeps its gripper pose relative to the part (same grasp ids, sequence, pickup and insertion poses), only the arm
joint angles (arm_q, arm_q_retract) are solved again for the new base position. Grasps whose new IK fails or leaves the
virtual walls are reported and dropped (with their grasp pairs).
Run run_motion_plan.py (and run_place_poses.py) afterwards.

python planning/retarget_robot_height.py --log-dir logs/exp_human_normal_flat/human_normal --old-base-z -3.7
'''
import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import pickle
import numpy as np

from planning.robot.util_arm import get_arm_chain, get_virtual_wall_margin
from planning.robot.motion_plan_arm import VIRTUAL_WALL_SAFETY

N_IK_RESTART = 20


def retarget_q(chain_old, chain_new, q_old):
    '''joint angles (full) of chain_new reaching the end effector pose of chain_old at q_old, or None'''
    fk = chain_old.forward_kinematics(q_old)
    for q_init in [q_old] + [chain_new.sample_joint_angles() for _ in range(3)]:
        q, success = chain_new.inverse_kinematics(fk[:3, 3], fk[:3, :3], orientation_mode='all', initial_position=q_init,
                                                  n_restart=N_IK_RESTART, optimizer='least_squares', regularization_parameter=1.0)
        if success:
            fk_new = chain_new.forward_kinematics(q)
            if np.linalg.norm(fk_new[:3, 3] - fk[:3, 3]) < 1e-2 and np.allclose(fk_new[:3, :3], fk[:3, :3], atol=1e-3):
                return q
    return None


def retarget_grasp(grasp, chains_old, chains_new, motion_type, check_walls):
    chain_old, chain_new = chains_old[motion_type], chains_new[motion_type]
    q = retarget_q(chain_old, chain_new, grasp.arm_q)
    if q is None or (check_walls and get_virtual_wall_margin(chain_new, q) < VIRTUAL_WALL_SAFETY):
        return False
    q_retract = None
    if getattr(grasp, 'arm_q_retract', None) is not None:
        q_retract = retarget_q(chain_old, chain_new, grasp.arm_q_retract)
        if q_retract is None:
            return False
    grasp.arm_q, grasp.arm_pos = q, np.array(chain_new.base_pos)
    if q_retract is not None: grasp.arm_q_retract = q_retract
    return True


def retarget_grasps_file(path, old_base_z):
    with open(path, 'rb') as fp:
        grasps = pickle.load(fp)
    arm_type = grasps['arm']
    chains_new = {m: get_arm_chain(arm_type, m) for m in ['move', 'hold']}
    chains_old = {m: get_arm_chain(arm_type, m, base_pos=np.array([*chains_new[m].base_pos[:2], old_base_z])) for m in ['move', 'hold']}
    print(f'[retarget_robot_height] {os.path.relpath(path, project_base_dir)}: base z {old_base_z} -> {chains_new["move"].base_pos[2]}')

    kept = {}
    for part_id, part_grasps in grasps['grasps'].items():
        n_move, n_hold = len(part_grasps['move']), len(part_grasps['hold'])
        part_grasps['move'] = [g for g in part_grasps['move'] if all(retarget_grasp(x, chains_old, chains_new, 'move', check_walls=arm_type == 'panda') for x in g)]
        part_grasps['hold'] = [g for g in part_grasps['hold'] if retarget_grasp(g, chains_old, chains_new, 'hold', check_walls=False)] # hold arm: human in single-arm mode
        kept[part_id] = {'move': {g[0].grasp_id for g in part_grasps['move']}, 'hold': {g.grasp_id for g in part_grasps['hold']}}
        print(f'  part {part_id}: move {n_move} -> {len(part_grasps["move"])}, hold {n_hold} -> {len(part_grasps["hold"])}')
    grasps['grasp_id_pairs'] = {(pm, ph): [(m, h) for m, h in pairs if m in kept[pm]['move'] and h in kept[ph]['hold']]
                                for (pm, ph), pairs in grasps['grasp_id_pairs'].items()}
    grasps['settings']['base_z'] = float(chains_new['move'].base_pos[2])
    with open(path, 'wb') as fp:
        pickle.dump(grasps, fp)
    return kept


def check_sequence(log_dir, kept):
    with open(os.path.join(log_dir, 'tree_opt.pkl'), 'rb') as fp:
        tree = pickle.load(fp)
    ok = True
    for _, _, attr in tree.edges(data=True):
        for part, grasp_id, kind in [(attr['move_part'], attr['move_grasp_id'], 'move'), (attr['hold_part'], attr['hold_grasp_id'], 'hold')]:
            if grasp_id not in kept[part][kind]:
                print(f'[retarget_robot_height] sequence {kind} grasp {grasp_id} of part {part} is infeasible at the new height')
                ok = False
    print(f'[retarget_robot_height] planned sequence {"still feasible" if ok else "BROKEN, re-run sequence planning"}')
    return ok


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--old-base-z', type=float, required=True, help='[cm] base z the log was planned with')
    args = parser.parse_args()

    kept = retarget_grasps_file(os.path.join(args.log_dir, 'grasps.pkl'), args.old_base_z)
    if os.path.exists(os.path.join(args.log_dir, 'grasps_unfiltered.pkl')):
        retarget_grasps_file(os.path.join(args.log_dir, 'grasps_unfiltered.pkl'), args.old_base_z)
    sys.exit(0 if check_sequence(args.log_dir, kept) else 1)
