import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)


import numpy as np
import random
import json
import pickle
from tqdm import tqdm
import traceback
import trimesh
from scipy.spatial.transform import Rotation as R
from time import time

from planning.robot.geometry import load_part_meshes
from planning.robot.motion_plan_arm import ArmMotionPlanner
from planning.robot.util_arm import get_arm_chain, get_ik_target_orientation, get_ft_pos_from_gripper_pos_quat
from planning.robot.workcell import get_dual_arm_box, get_assembly_center, get_panda_base_z, get_panda_desk_z
from assets.load import load_pos_quat_dict
from assets.transform import get_transform_matrix_quat, get_transform_matrix, mat_to_pos_quat
from utils.common import TimeStamp
from planning.run_seq_plan import SequencePlanner
from planning.run_seq_opt import SequenceOptimizer
from planning.config import RETRACT_OPEN_RATIO, OPEN_RATIO_REST, RETRACT_DELTA_FAR, TRANSFER_PART_CLEARANCE, MAX_PLAN_ATTEMPTS, ASSEMBLY_ZONE_MARGIN, PLANNER_TABLE_MARGIN
from planning.plan_checks import SceneChecker, clear_failure, write_failure, CAUSE_GRASP, CAUSE_SPOT, CAUSE_INSERTION, CAUSE_HOLD_GRASP

# Arm pose with which the hold arm picks the first part where the layout
# put it, chosen and checked by run_pickup_gen.py (file in fixture/).
PICKUP_Q_HOLD_FILE = 'pickup_q_hold.json'


def get_pickup_gripper_pose(grasp, pickup_pose, final_pose, lift=0):
    gripper_final_pose = get_transform_matrix_quat(grasp.pos, grasp.quat)
    gripper_pickup_pose = pickup_pose @ np.linalg.inv(final_pose) @ gripper_final_pose
    gripper_pickup_pose[2, 3] += lift
    return gripper_pickup_pose


def get_pickup_arm_q(motion_planner, grasp, pickup_pose, final_pose, arm_q_init=None, lift=0, has_ft_sensor=False, optimizer=None, regularization=None, n_restart=None):
    if arm_q_init is None: arm_q_init = grasp.arm_q
    gripper_pickup_pose = get_pickup_gripper_pose(grasp, pickup_pose, final_pose, lift)
    gripper_pickup_pos, gripper_pickup_quat = mat_to_pos_quat(gripper_pickup_pose)
    gripper_pickup_ori = get_ik_target_orientation(motion_planner.arm_chain, motion_planner.gripper_type, gripper_pickup_quat)
    if has_ft_sensor:
        gripper_pickup_pos = get_ft_pos_from_gripper_pos_quat(motion_planner.gripper_type, gripper_pickup_pos, gripper_pickup_quat)
    arm_q_pickup = motion_planner.inverse_kinematics(gripper_pickup_pos, gripper_pickup_ori, q_init=arm_q_init, n_restart=n_restart, optimizer=optimizer, regularization_parameter=regularization)
    return arm_q_pickup


NO_PATH_REASON = { # ArmMotionPlanner.last_failure (real arm)
    'start': 'no path found: no straight move out of the start pose that ends clear of everything',
    'goal': 'no path found: no straight move into the goal pose from a pose clear of everything',
    'path': 'no path found: no free path between the two straight moves',
}


def make_move_planner(arm_chain_move, gripper_type, has_ft_sensor, arm_box_move, part_meshes_final, stamp=None):
    '''
    The motion planner of the move arm, the real robot. run_pickup_gen.py tests the straight moves at a
    pickup spot with the same planner. part_meshes_final: {part_id: mesh at its assembled pose}, they span the assembly zone.
    '''
    panda = arm_chain_move.arm_type == 'panda'
    planner = ArmMotionPlanner(arm_chain_move, gripper_type, has_ft_sensor, arm_box_move, stamp=stamp, virtual_walls=panda,
        table_z=get_panda_base_z() + get_panda_desk_z() if panda else None, part_clearance=TRANSFER_PART_CLEARANCE + PLANNER_TABLE_MARGIN if panda else None) # carried parts stay clear of the real table (planned with a reserve on the checked value)
    planner.strict = panda # plan what planning/plan_checks.py accepts: straight moves out of / into grasps, all clearances in between
    assembly_bounds = np.array([mesh.bounds for mesh in part_meshes_final.values()])
    zone_margin = ASSEMBLY_ZONE_MARGIN - PLANNER_TABLE_MARGIN # a little smaller than the checked zone
    planner.clearance_exempt_xy = (assembly_bounds[:, 0, :2].min(axis=0) - zone_margin, assembly_bounds[:, 1, :2].max(axis=0) + zone_margin) # carried parts may come low only near the assembly
    return planner


def assembly_moves_blocked(planner, grasps_move, part_mesh, meshes_before, inserted=True):
    '''
    The two straight moves at the assembly that every plan with a move grasp contains (commands of run_motion_plan below), tested
    with the planner of the move arm (make_move_planner) and its own routine, in the smallest scene that is certain at that
    moment: the parts that have to be assembled before this one (meshes_before, at their assembled poses, like part_mesh).
    grasps_move: the grasp along the insertion path ([0] at the assembled pose, [-1] at the insertion start).
      approach  with the part in the gripper, straight into the insertion start pose from a pose that is clear of everything
                (plan_path_with_grasp: goal retract along the disassembly direction). Only for a part that is inserted:
                inserted=False for the part that is held first. Its move grasps serve the fallback in which the move arm
                sets it down at its assembled pose from above; the path the precedence plan may give it (plumbers_block,
                duct: a slide along the table) is not driven.
      release   after the insertion the gripper opens (RETRACT_OPEN_RATIO) and moves straight back along the hand axis until it
                is clear of everything (plan_path / plan_path_switch: start retract)
    Returns None, or what is missing (text with ArmMotionPlanner.last_block).
    With more parts in the scene the same moves are longer or stop earlier, never easier: a grasp that fails here fails in
    every plan, and motion planning notices that only at the end of a whole planning attempt (run_grasp_arm_gen.py --flat
    drops such grasps).
    '''
    def blocked():
        what, moved = planner.last_block
        return str(what) + (f' after {moved:.2f} cm' if moved is not None else '')

    q_final, q_start = np.array(grasps_move[0].arm_q, dtype=float), np.array(grasps_move[-1].arm_q, dtype=float)
    if inserted and len(grasps_move) > 1 and not np.allclose(q_final, q_start):
        gripper_final = get_transform_matrix_quat(grasps_move[0].pos, grasps_move[0].quat) # the part sits in the gripper as at its assembled pose
        collision_fn = planner.get_fns(part_mesh, gripper_final, list(meshes_before), grasps_move[0].open_ratio, verbose=False)[3]
        if planner.plan_path_straight_out(q_start, collision_fn, get_disassembly_retract_dir(planner, q_final, q_start)) is None:
            return f'no straight move to the insertion start ({blocked()})'
    open_ratio_retract = min(grasps_move[0].open_ratio + RETRACT_OPEN_RATIO, 1.0)
    collision_fn = planner.get_fns(None, None, list(meshes_before) + [part_mesh], open_ratio_retract, verbose=False)[3]
    if planner.plan_path_straight_out(q_final, collision_fn, get_back_retract_dir(planner, q_final)) is None:
        return f'no straight move away after the release ({blocked()})'
    return None


def no_path_reason(motion_planner):
    '''why the real arm's planner returned no path, with what stopped its straight move (ArmMotionPlanner.last_block)'''
    reason = NO_PATH_REASON.get(motion_planner.last_failure, 'no path found')
    block = motion_planner.last_block if motion_planner.last_failure in ['start', 'goal'] else None
    if block is not None:
        reason += f' ({block[0]}' + (f' after {block[1]:.2f} cm' if block[1] is not None else '') + ')'
    return reason


def failure_cause(task, carrying, last_failure, blocked_by_carried_part, violations):
    '''
    What has to change after the real arm found no checked path (planning/plan_checks.py CAUSE_*):
      task: 'transport' / 'switch'; carrying: a part is in the gripper; last_failure: ArmMotionPlanner.last_failure of the last
      attempt if the planner returned no path, else None; blocked_by_carried_part: ArmMotionPlanner.blocked_by_carried_part();
      violations: the violations of all attempts (texts of SceneChecker.check_path and of no_path_reason)
    spot       the pickup side: no straight move into the grasp (switch) or out of it with the part (transport), or every
               violation is about parts lying beside the pickup: the same grasp can work where the part lies elsewhere
    insertion  no straight move to the insertion start because of the carried part itself (too close to the table or to the
               assembly on its way there): where the part is on that move does not depend on the grasp
    grasp      everything else: the arm or the gripper is what does not fit
    '''
    if last_failure == 'goal' and task == 'switch': return CAUSE_SPOT
    if last_failure == 'start' and task == 'transport' and carrying: return CAUSE_SPOT
    if last_failure == 'goal' and task == 'transport' and carrying and blocked_by_carried_part: return CAUSE_INSERTION
    if last_failure is None and len(violations) > 0 and all('lying part' in v for v in violations): return CAUSE_SPOT
    return CAUSE_GRASP


def get_back_retract_dir(motion_planner, arm_q):
    fk = motion_planner.arm_chain.forward_kinematics(arm_q)
    if motion_planner.arm_type == 'ur5e':
        retract_dir = -fk[:3, 0]
    else:
        retract_dir = -fk[:3, 2]
    retract_dir /= np.linalg.norm(retract_dir)
    return retract_dir


def get_disassembly_retract_dir(motion_planner, arm_q_assembled, arm_q_disassembled):
    fk_assembled = motion_planner.arm_chain.forward_kinematics(arm_q_assembled)
    fk_disassembled = motion_planner.arm_chain.forward_kinematics(arm_q_disassembled)
    retract_dir = fk_disassembled[:3, 3] - fk_assembled[:3, 3]
    retract_dir /= np.linalg.norm(retract_dir)
    return retract_dir


def get_transformed_part_meshes(part_meshes, part_poses):
    transformed_part_meshes = {}
    for part_id, part_mesh in part_meshes.items():
        part_pose = part_poses[part_id]
        transformed_part_meshes[part_id] = part_mesh.copy()
        transformed_part_meshes[part_id].apply_transform(part_pose)
    return transformed_part_meshes


def post_process_q(arm_chain, q, last_q=None):
    """
    Post-process joint angles to avoid unnecessary full round rotations.
    """
    if last_q is None: return np.array(q)

    q_active = arm_chain.active_from_full(q)
    last_q_active = arm_chain.active_from_full(last_q)
    bounds = arm_chain.get_active_link_bounds()
    assert len(q_active) == len(last_q_active) == len(bounds)

    for i in range(len(q_active)):
        
        # Calculate the difference between the current and previous angle
        diff = q_active[i] - last_q_active[i]
        
        # If the absolute difference is larger than pi, adjust to the shorter rotation
        if diff > np.pi:
            if q_active[i] - 2 * np.pi >= bounds[i][0]:
                q_active[i] -= 2 * np.pi
        elif diff < -np.pi:
            if q_active[i] + 2 * np.pi <= bounds[i][1]:
                q_active[i] += 2 * np.pi

        assert q_active[i] >= bounds[i][0] and q_active[i] <= bounds[i][1], f'{q_active[i]} not in {bounds[i]}'
        assert last_q_active[i] >= bounds[i][0] and last_q_active[i] <= bounds[i][1], f'{last_q_active[i]} not in {bounds[i]}'
    
    return arm_chain.active_to_full(q_active)


def post_process_motion(arm_chain, path, last_q=None):
    """
    Post-process joint angle arrays to avoid unnecessary full round rotations.
    """
    processed_path = []

    for q in path:
        if last_q is not None:
            q = post_process_q(arm_chain, q, last_q)
        processed_path.append(q)
        last_q = q  # Update previous angles for the next iteration

    return processed_path


def run_motion_plan(assembly_dir, log_dir, optimized, seed, verbose=False):

    precedence_path = os.path.join(log_dir, 'precedence.pkl')
    if not os.path.exists(precedence_path):
        print(f'[run_motion_plan] {precedence_path} not found')
        return
    grasps_path = os.path.join(log_dir, 'grasps.pkl')
    if not os.path.exists(grasps_path):
        print(f'[run_motion_plan] {grasps_path} not found')
        return
    fixture_dir = os.path.join(log_dir, 'fixture')
    pickup_path = os.path.join(fixture_dir, 'pickup.json')
    if not os.path.exists(pickup_path):
        print(f'[run_motion_plan] {pickup_path} not found')
        return
    tree_path = os.path.join(log_dir, 'tree_opt.pkl') if optimized else os.path.join(log_dir, 'tree.pkl')
    if not os.path.exists(tree_path):
        print(f'[run_motion_plan] {tree_path} not found')
        return

    with open(precedence_path, 'rb') as fp:
        G_preced = pickle.load(fp)
    with open(grasps_path, 'rb') as fp:
        grasps = pickle.load(fp)
    gripper_type, arm_type, has_ft_sensor = grasps['gripper'], grasps['arm'], grasps['ft_sensor']

    asset_folder = os.path.join(project_base_dir, './assets')
    part_pos_dict_final, part_quat_dict_final = load_pos_quat_dict(assembly_dir, transform='final')
    part_pos_dict_final = {k: v + get_assembly_center(arm_type) for k, v in part_pos_dict_final.items()}
    part_final_pose = {part_id: get_transform_matrix_quat(part_pos_dict_final[part_id], part_quat_dict_final[part_id]) for part_id in part_pos_dict_final.keys()}
    part_meshes = load_part_meshes(assembly_dir, transform='none')
    part_meshes = {k[4:]: v for k, v in part_meshes.items()} # remove part prefix
    part_ids = list(part_meshes.keys())
  
    with open(pickup_path, 'r') as f:
        part_pickup_pose = json.load(f)
    part_pickup_pose = {part_id: get_transform_matrix(part_pickup_state) for part_id, part_pickup_state in part_pickup_pose.items()}
    fixture_path = os.path.join(fixture_dir, 'fixture.obj')
    fixture_meshes = [trimesh.load_mesh(fixture_path)] if os.path.exists(fixture_path) else [] # no fixture: parts lie flat on the board (run_pickup_gen.py)

    pickup_q_path = os.path.join(fixture_dir, 'pickup_q.json')
    pickup_q_checked = json.load(open(pickup_q_path, 'r')) if os.path.exists(pickup_q_path) else {}
    pickup_q_hold_path = os.path.join(fixture_dir, PICKUP_Q_HOLD_FILE)
    pickup_q_hold_checked = {}
    if os.path.exists(pickup_q_hold_path):
        with open(pickup_q_hold_path, 'r') as fp:
            pickup_q_hold_checked = json.load(fp)
    part_meshes_pickup = get_transformed_part_meshes(part_meshes, part_pickup_pose)
    part_meshes_final = get_transformed_part_meshes(part_meshes, part_final_pose)
    part_meshes_map = {'pickup': part_meshes_pickup, 'final': part_meshes_final}
    gripper_pickup_pose = {}

    with open(tree_path, 'rb') as fp:
        tree = pickle.load(fp)
    if optimized:
        seq_optimizer = SequenceOptimizer(G_preced, grasps)
        sequence, grasps_sequence = seq_optimizer.get_sequence(tree)
    else:
        seq_planner = SequencePlanner(asset_folder, assembly_dir, G_preced, grasps, save_sdf=True, contact_eps=None)
        sequence, grasps_sequence = seq_planner.sample_sequence(tree, seed=seed)

    if sequence is None or grasps_sequence is None:
        print(f'[run_motion_plan] No feasible sequence found in {tree_path}')
        return
    
    sequence, grasps_sequence = sequence[::-1], grasps_sequence[::-1] # reverse the sequence to be forward assembly

    arm_box_move, arm_box_hold = get_dual_arm_box(arm_type)
    arm_chain_move = get_arm_chain(arm_type, 'move')
    arm_chain_hold = get_arm_chain(arm_type, 'hold')
    arm_chains = {'move': arm_chain_move, 'hold': arm_chain_hold}
    rest_q_move = arm_chain_move.active_to_full(arm_chain_move.rest_q)
    rest_q_hold = arm_chain_hold.active_to_full(arm_chain_hold.rest_q)

    stamp = TimeStamp()

    random.seed(seed)
    np.random.seed(seed)

    motion_planner_move = make_move_planner(arm_chain_move, gripper_type, has_ft_sensor['move'], arm_box_move, part_meshes_final, stamp=stamp)
    motion_planner_hold = ArmMotionPlanner(arm_chain_hold, gripper_type, has_ft_sensor['hold'], arm_box_hold, stamp=stamp)

    '''
    Plan commands
    ['move/hold', 'arm/gripper', q/open ratio, active_part, task]
    Tasks:
    - gripper: init, open, close
    - arm: init, transport, switch, assembly
    '''

    max_speed = { # speed in cm/s
        'transport': 4,
        'switch': 4,
        'assembly': 0.3,
    }
    
    # init
    commands = []
    commands.append(['move', 'arm', rest_q_move, None, 'init'])
    commands.append(['hold', 'arm', rest_q_hold, None, 'init'])
    commands.append(['move', 'gripper', OPEN_RATIO_REST, None, 'init'])
    commands.append(['hold', 'gripper', OPEN_RATIO_REST, None, 'init'])

    # For every arm command of the hold arm: the hold grasp it belongs to,
    # (part id, grasp id). If that path cannot be planned, this is the hold
    # grasp that run_planning_flat.py drops before the next attempt.
    hold_blame = {}

    def add_hold_arm(command, grasp):
        hold_blame[len(commands)] = (grasp.part_id, grasp.grasp_id)
        commands.append(command)

    # fallback for the first part (run_pickup_gen.py, 'arm': 'place'): its hold grasp cannot pick it lying flat, so the move arm
    # brings it to the assembly spot with one of its own move grasps and the hold arm takes it there
    placed_first, place_grasp = None, None
    flat_pickup_path = os.path.join(fixture_dir, 'flat_pickup.json')
    if os.path.exists(flat_pickup_path):
        with open(flat_pickup_path, 'r') as fp:
            flat_pickup = json.load(fp)
        placed_first = next((part_id for part_id, info in flat_pickup.items() if info['arm'] == 'place'), None)
        if placed_first is not None:
            assert placed_first == sequence[0][1], f'part {placed_first} is laid out for the move arm, but the sequence holds part {sequence[0][1]} first'
            moves_first = grasps['grasps'][placed_first]['move'] # a list in grasps.pkl; SequenceOptimizer turns it into {grasp id: [grasp, ...]}
            place_grasp = next(g[0] for g in (moves_first.values() if isinstance(moves_first, dict) else moves_first) if g[0].grasp_id == flat_pickup[placed_first]['grasp_id'])

    for step, ((part_move, part_hold), (grasps_move, grasp_hold)) in enumerate(zip(sequence, grasps_sequence)):

        # hold (first)
        if step == 0 and place_grasp is not None:
            up = np.array([0, 0, 1.0])
            open_ratio_retract_place = min(place_grasp.open_ratio + RETRACT_OPEN_RATIO, 1.0)
            open_ratio_retract_hold = min(grasp_hold.open_ratio + RETRACT_OPEN_RATIO, 1.0)
            gripper_pickup_pose[part_hold] = get_pickup_gripper_pose(place_grasp, part_pickup_pose[part_hold], part_final_pose[part_hold])
            # the move arm: pick the part from the board, stand it at its assembled pose on the table, let go and get out of the way
            commands.append(['move', 'arm', (np.array(pickup_q_checked[part_hold]), [None, None], open_ratio_retract_place), None, 'switch'])
            commands.append(['move', 'gripper', place_grasp.open_ratio, None, 'close'])
            commands.append(['move', 'arm', (place_grasp.arm_q, [up, up]), part_hold, 'transport'])
            commands.append(['move', 'gripper', open_ratio_retract_place, None, 'open'])
            commands.append(['move', 'arm', (rest_q_move, [None, None]), None, 'transport'])
            # the hold arm: take the part where it stands
            commands.append(['hold', 'gripper', open_ratio_retract_hold, None, 'open'])
            add_hold_arm(['hold', 'arm', (grasp_hold.arm_q, [None, None]), None, 'transport'], grasp_hold)
            commands.append(['hold', 'gripper', grasp_hold.open_ratio, None, 'close'])
        elif step == 0:
            # The hold arm picks the first part where the layout put it.
            # run_pickup_gen.py laid the part out only where the hold arm
            # has this arm pose (else the move arm brings the part, see
            # above), so it is read from there. Older logs: search it.
            if part_hold in pickup_q_hold_checked:
                pickup_q_hold = np.array(pickup_q_hold_checked[part_hold])
            else:
                pickup_q_hold = get_pickup_arm_q(motion_planner_hold, grasp_hold, part_pickup_pose[part_hold], part_final_pose[part_hold], arm_q_init=rest_q_hold, has_ft_sensor=has_ft_sensor['hold'], optimizer='least_squares', regularization=1.0)
            if pickup_q_hold is None:
                # No silent way out (until 2026-11-13 a human picked the
                # part instead): another hold grasp may reach the part.
                write_failure(log_dir, 'run_motion_plan', part_hold, grasp_hold.grasp_id, 'the hold arm has no arm pose to pick the part where it lies', CAUSE_HOLD_GRASP)
                raise Exception(f'[run_motion_plan] no arm pose of the hold arm to pick part {part_hold} ({assembly_dir})')
            gripper_pickup_pose[part_hold] = get_pickup_gripper_pose(grasp_hold, part_pickup_pose[part_hold], part_final_pose[part_hold])
            open_ratio_retract_hold = min(grasp_hold.open_ratio + RETRACT_OPEN_RATIO, 1.0)
            commands.append(['hold', 'gripper', open_ratio_retract_hold, None, 'open'])
            add_hold_arm(['hold', 'arm', (pickup_q_hold, [None, np.array([0, 0, 1.0])]), None, 'transport'], grasp_hold) # transport with goal retract
            commands.append(['hold', 'gripper', grasp_hold.open_ratio, None, 'close'])
            add_hold_arm(['hold', 'arm', (grasp_hold.arm_q, [np.array([0, 0, 1.0]), np.array([0, 0, 1.0])]), part_hold, 'transport'], grasp_hold) # transport with both retract
        
        # move (assembly)
        if part_move in pickup_q_checked: # arm pose chosen and checked by run_pickup_gen.py
            pickup_q_move = np.array(pickup_q_checked[part_move])
        else:
            pickup_q_move = get_pickup_arm_q(motion_planner_move, grasps_move[0], part_pickup_pose[part_move], part_final_pose[part_move], arm_q_init=rest_q_move, has_ft_sensor=has_ft_sensor['move'], optimizer='least_squares', regularization=1.0)
        if pickup_q_move is None:
            raise Exception(f'[run_motion_plan] Failed to solve pickup IK for move arm in step {step} ({assembly_dir})')
        gripper_pickup_pose[part_move] = get_pickup_gripper_pose(grasps_move[0], part_pickup_pose[part_move], part_final_pose[part_move])
        open_ratio_retract_move = min(grasps_move[0].open_ratio + RETRACT_OPEN_RATIO, 1.0)
        commands.append(['move', 'arm', (pickup_q_move, [None, None], open_ratio_retract_move), None, 'switch']) # transport with both retract and open gripper
        commands.append(['move', 'gripper', grasps_move[0].open_ratio, None, 'close'])
        commands.append(['move', 'arm', (grasps_move[-1].arm_q, [np.array([0, 0, 1.0]), get_disassembly_retract_dir(motion_planner_move, grasps_move[0].arm_q, grasps_move[-1].arm_q)]), part_move, 'transport']) # transport with both retract
        commands.append(['move', 'arm', grasps_move[0].arm_q, part_move, 'assembly'])

        # hold (switch/rest)
        open_ratio_retract_hold = min(grasp_hold.open_ratio + RETRACT_OPEN_RATIO, 1.0)
        if step < len(sequence) - 1:
            grasp_hold_next = grasps_sequence[step + 1][1]
            if not (grasp_hold.part_id == grasp_hold_next.part_id and grasp_hold.grasp_id == grasp_hold_next.grasp_id):
                open_ratio_retract_hold_next = min(grasp_hold_next.open_ratio + RETRACT_OPEN_RATIO, 1.0)
                commands.append(['hold', 'gripper', open_ratio_retract_hold, None, 'open'])
                add_hold_arm(['hold', 'arm', (grasp_hold_next.arm_q, [None, None], open_ratio_retract_hold_next), None, 'switch'], grasp_hold_next) # transport with both retract and open gripper
                commands.append(['hold', 'gripper', grasp_hold_next.open_ratio, None, 'close'])
        else:
            commands.append(['hold', 'gripper', open_ratio_retract_hold, None, 'open'])
            add_hold_arm(['hold', 'arm', (rest_q_hold, [None, None]), None, 'transport'], grasp_hold) # transport with start retract
            commands.append(['hold', 'gripper', OPEN_RATIO_REST, None, 'close'])
        
        # move (rest)
        commands.append(['move', 'gripper', open_ratio_retract_move, None, 'open'])
        if step == len(sequence) - 1:
            commands.append(['move', 'arm', (rest_q_move, [None, None]), None, 'transport']) # transport with start retract
            commands.append(['move', 'gripper', OPEN_RATIO_REST, None, 'close'])

    # post-process qs in commands
    last_qs = {'move': None, 'hold': None}
    for command in commands:
        motion_type, body_type, value, _, _ = command
        if body_type == 'arm':
            if type(value) == tuple:
                q = value[0]
            else:
                q = value
            q = post_process_q(arm_chains[motion_type], q, last_qs[motion_type])
            last_qs[motion_type] = q
            if type(value) == tuple:
                command[2] = (q, *value[1:])
            else:
                command[2] = q

    '''
    Plan path
    ['move/hold', 'arm/gripper', path, active_part, task]
    '''

    # exact checks of every path of the real (move) arm, re-plan with another seed if one fails (planning/plan_checks.py); the
    # paths of the hold arm are checked for their distance to the move arm
    checker = SceneChecker(arm_type, gripper_type, has_ft_sensor['move'], part_meshes, part_pickup_pose, part_meshes_final, has_ft_sensor_hold=has_ft_sensor['hold']) if arm_type == 'panda' else None
    clear_failure(log_dir)
    move_grasp_id = {part_move: grasps_move[0].grasp_id for (part_move, _), (grasps_move, _) in zip(sequence, grasps_sequence)}
    if place_grasp is not None: move_grasp_id[placed_first] = place_grasp.grasp_id # the grasp to drop if the move arm cannot bring the first part to its place
    pick_cids = [(j, c[3]) for j, c in enumerate(commands) if c[0] == 'move' and c[1] == 'arm' and c[4] == 'transport' and c[3] is not None]
    last_inserted = None
    def reseed(cid, attempt):
        random.seed(seed * 100003 + cid * 101 + attempt)
        np.random.seed(seed * 100003 + cid * 101 + attempt)
    def fail(part, reason, cause=CAUSE_GRASP):
        write_failure(log_dir, 'run_motion_plan', part, move_grasp_id.get(part), reason, cause)
        raise Exception(f'[run_motion_plan] no checked path for part {part}: {reason}')

    def fail_hold(cid, reason):
        # A path of the hold arm failed: name the hold grasp it belongs to.
        part, grasp_id = hold_blame[cid]
        write_failure(log_dir, 'run_motion_plan', part, grasp_id, reason, CAUSE_HOLD_GRASP)
        raise Exception(f'[run_motion_plan] no checked path of the hold arm with hold grasp {grasp_id} of part {part}: {reason}')

    paths = []
    current_states = {
        'parts': {part_id: 'pickup' for part_id in part_ids}, # 'pickup', 'final'
        'move': {'arm': None, 'gripper': None},
        'hold': {'arm': None, 'gripper': None},
    }
    for cid, command in enumerate(commands):
        motion_type, body_type, value, active_part, task = command
        if verbose:
            print('[run_motion_plan] motion: %s, body: %s, part: %s, task: %s' % (motion_type, body_type, active_part, task))
        assert motion_type in ['move', 'hold'] and body_type in ['arm', 'gripper']
        assert task in ['init', 'open', 'close', 'transport', 'switch', 'assembly']

        if task in ['init', 'open', 'close']:
            if task == 'init':
                path = [np.array(value)] if body_type == 'arm' else value
            else:
                path = value
            paths.append([motion_type, body_type, path, active_part, task])
            current_states[motion_type][body_type] = value

        elif task in ['transport', 'switch', 'assembly']:
            assert body_type == 'arm'

            if motion_type == 'move':
                motion_planner = motion_planner_move
            elif motion_type == 'hold':
                motion_planner = motion_planner_hold

            q_start = current_states[motion_type]['arm']
            open_ratio = current_states[motion_type]['gripper']

            part_meshes_curr = {part_id: part_meshes_map[current_states['parts'][part_id]][part_id] for part_id in part_ids}
            motion_type_other = 'hold' if motion_type == 'move' else 'move'
            # Both arms are robots: the arm that moves avoids the other one, which stands still meanwhile (until 2026-11-13
            # --single-arm switched this off for the move arm and let the hold arm jump where it had no path)
            arm_chain_other, arm_q_other, open_ratio_other = arm_chains[motion_type_other], current_states[motion_type_other]['arm'], current_states[motion_type_other]['gripper']
            other_arm = (arm_q_other, open_ratio_other) # for the exact distance check between the two arms

            checked = checker is not None and motion_type == 'move' # the real arm: every path must pass the exact checks
            hold_checked = checker is not None and motion_type == 'hold' # the hold arm: its distance to the move arm is checked
            n_attempts = MAX_PLAN_ATTEMPTS if checked or hold_checked else 1
            state_before = dict(current_states['parts'])

            if task == 'transport':
                q_goal, [retract_start, retract_goal] = value
                if retract_start is None: retract_start = get_back_retract_dir(motion_planner, q_start)
                if retract_goal is None: retract_goal = get_back_retract_dir(motion_planner, q_goal)

                blame = active_part if active_part is not None else last_inserted
                seen = [] # violations of all attempts
                for attempt in range(n_attempts):
                    if attempt > 0: reseed(cid, attempt)
                    try:
                      if active_part is not None: # transport with part (pickup -> assembly)
                        part_meshes_rest = [v for k, v in part_meshes_curr.items() if k != active_part]
                        path = motion_planner.plan_path_with_grasp(q_start, q_goal,
                            move_pickup_mesh=part_meshes_pickup[active_part], gripper_pickup_transform=gripper_pickup_pose[active_part], 
                            still_meshes=part_meshes_rest + fixture_meshes, open_ratio=open_ratio, 
                            arm_chain_other=arm_chain_other, arm_q_other=arm_q_other, open_ratio_other=open_ratio_other, has_ft_sensor_other=has_ft_sensor[motion_type_other],
                            retract_start=retract_start, retract_goal=retract_goal, retract_delta=RETRACT_DELTA_FAR,
                            max_speed=max_speed[task], verbose=verbose)
                      else: # transport without part
                        path = motion_planner.plan_path(q_start, q_goal,
                            part_meshes=list(part_meshes_curr.values()) + fixture_meshes, open_ratio=open_ratio, 
                            arm_chain_other=arm_chain_other, arm_q_other=arm_q_other, open_ratio_other=open_ratio_other, has_ft_sensor_other=has_ft_sensor[motion_type_other],
                            retract_start=retract_start, retract_goal=retract_goal, retract_delta=RETRACT_DELTA_FAR,
                            max_speed=max_speed[task], verbose=verbose)
                    except Exception as e: # AssertionError: the planner gives up on a start/goal in collision
                        if not (hold_checked or (checked and isinstance(e, AssertionError))): raise
                        print(f'[run_motion_plan] planner error: {e}')
                        path = None
                    if path is None:
                        if not (checked or hold_checked):
                            raise Exception(f'[run_motion_plan] Failed to plan path for {motion_type} {body_type} in task {task} ({assembly_dir})')
                        violations = [no_path_reason(motion_planner) if checked else 'no path found for the hold arm']
                    elif not (checked or hold_checked):
                        break
                    elif checked:
                        kwargs = dict(carried=active_part) if active_part is not None else dict(released_part=last_inserted)
                        violations = checker.check_path(path, open_ratio, state_before, other_arm=other_arm, **kwargs)
                        if not violations: break
                    else: # the hold arm, carrying the first part from where it lay or moving without a part
                        carried_at = (part_meshes_pickup[active_part], gripper_pickup_pose[active_part]) if active_part is not None else None
                        violations = checker.other_arm_violations('hold', path, open_ratio, *other_arm, carried=carried_at)
                        if not violations: break
                    seen += violations
                    print(f'[run_motion_plan] {motion_type} arm {task} part {active_part}, attempt {attempt + 1}/{MAX_PLAN_ATTEMPTS} rejected: ' + '; '.join(violations))
                else:
                    if hold_checked:
                        fail_hold(cid, f'hold arm {task} path: ' + '; '.join(violations))
                    fail(blame, f'{task} path: ' + '; '.join(violations),
                         failure_cause(task, active_part is not None, motion_planner.last_failure if path is None else None, motion_planner.blocked_by_carried_part(), seen))
                if active_part is not None:
                    current_states['parts'][active_part] = 'final'
                    # a part the move arm carries is inserted next ('assembly' sets last_inserted); the first part in the fallback
                    # stands at its place after this path and is released there
                    if motion_type == 'move': last_inserted = active_part if active_part == placed_first else None
                paths.append([motion_type, body_type, path, active_part, task])
                current_states[motion_type][body_type] = q_goal

            elif task == 'switch':
                assert active_part is None
                q_goal, [retract_start, retract_goal], open_ratio_next = value
                if retract_start is None: retract_start = get_back_retract_dir(motion_planner, q_start)
                if retract_goal is None: retract_goal = get_back_retract_dir(motion_planner, q_goal)

                pick_part = next((p for j, p in pick_cids if j > cid), None) if motion_type == 'move' else None
                seen = [] # violations of all attempts
                for attempt in range(n_attempts):
                    if attempt > 0: reseed(cid, attempt)
                    try:
                        path1, path2 = motion_planner.plan_path_switch(q_start, q_goal,
                            part_meshes=list(part_meshes_curr.values()) + fixture_meshes, open_ratio=open_ratio, open_ratio_next=open_ratio_next,
                            arm_chain_other=arm_chain_other, arm_q_other=arm_q_other, open_ratio_other=open_ratio_other, has_ft_sensor_other=has_ft_sensor[motion_type_other],
                            retract_start=retract_start, retract_goal=retract_goal, retract_delta=RETRACT_DELTA_FAR,
                            max_speed=max_speed[task], verbose=verbose)
                    except Exception as e:
                        if not (hold_checked or (checked and isinstance(e, AssertionError))): raise
                        print(f'[run_motion_plan] planner error: {e}')
                        path1, path2 = None, None
                    if None in [path1, path2]:
                        if not (checked or hold_checked):
                            raise Exception(f'[run_motion_plan] Failed to plan path for {motion_type} {body_type} in task {task} ({assembly_dir})')
                        violations = [no_path_reason(motion_planner) if checked else 'no path found for the hold arm']
                    elif not (checked or hold_checked):
                        break
                    elif checked:
                        kwargs = dict(pick_part=pick_part, released_part=last_inserted, other_arm=other_arm)
                        violations = (checker.check_path(path1, open_ratio, state_before, label='before opening: ', **kwargs) if len(path1) > 0 else []) + \
                                     (checker.check_path(path2, open_ratio_next, state_before, label='after opening: ', **kwargs) if len(path2) > 0 else [])
                        if not violations: break
                    else: # the hold arm changes its grasp
                        violations = (checker.other_arm_violations('hold', path1, open_ratio, *other_arm, label='before opening: ') if len(path1) > 0 else []) + \
                                     (checker.other_arm_violations('hold', path2, open_ratio_next, *other_arm, label='after opening: ') if len(path2) > 0 else [])
                        if not violations: break
                    seen += violations
                    print(f'[run_motion_plan] {motion_type} arm {task} to part {pick_part}, attempt {attempt + 1}/{MAX_PLAN_ATTEMPTS} rejected: ' + '; '.join(violations))
                else:
                    if hold_checked:
                        fail_hold(cid, f'hold arm {task} path: ' + '; '.join(violations))
                    # the way out of the release pose belongs to the grasp of the part just released, the rest to the next part: to the
                    # spot where it lies if the straight move into its grasp is what fails (or its neighbours are too close), else to its grasp
                    no_path = None in [path1, path2]
                    leaving = (no_path and motion_planner.last_failure == 'start') or any('(just released)' in v for v in violations)
                    if pick_part is None or (leaving and last_inserted is not None):
                        fail(last_inserted, f'path away from the released part: ' + '; '.join(violations))
                    fail(pick_part, f'approach path: ' + '; '.join(violations), failure_cause(task, False, motion_planner.last_failure if no_path else None, False, seen))
                if open_ratio == open_ratio_next:
                    paths.append([motion_type, 'arm', path1 + path2, None, task])
                else:
                    if len(path1) > 0:
                        paths.append([motion_type, 'arm', path1, None, task])
                    paths.append([motion_type, 'gripper', open_ratio_next, None, 'open'])
                    if len(path2) > 0:
                        paths.append([motion_type, 'arm', path2, None, task])
                current_states[motion_type]['arm'] = q_goal
                current_states[motion_type]['gripper'] = open_ratio_next

            elif task == 'assembly':
                q_goal = value
                path = motion_planner.plan_path_straight(q_start, q_goal, open_ratio, max_speed=max_speed[task], sanity_check=task != 'assembly', verbose=verbose) # straight line path, assume no collision
            
                if path is None:
                    if checked: fail(active_part, 'no insertion path')
                    raise Exception(f'[run_motion_plan] Failed to plan path for {motion_type} {body_type} in task {task} ({assembly_dir})')
                if checked:
                    violations = checker.check_path(path, open_ratio, state_before, carried=active_part, inserting=True, other_arm=other_arm)
                    if violations: fail(active_part, 'insertion path: ' + '; '.join(violations)) # fixed by the grasp, no re-plan possible
                    last_inserted = active_part
                paths.append([motion_type, body_type, path, active_part, task])
                current_states[motion_type][body_type] = q_goal

            else:
                raise NotImplementedError
            
        else:
            raise NotImplementedError

    # post-process motions in paths
    last_qs = {'move': None, 'hold': None}
    for i in range(len(paths)):
        motion_type, body_type, path, _, _ = paths[i]
        if body_type == 'arm':
            path = post_process_motion(arm_chains[motion_type], path, last_qs[motion_type])
            path = [q.tolist() for q in path]
            paths[i][2] = path
            last_qs[motion_type] = path[-1]

    # convert commands and motion from full to active
    for i in range(len(commands)):
        motion_type, body_type, value, _, _ = commands[i]
        if body_type == 'arm':
            if type(value) == tuple:
                q = value[0]
            else:
                q = value
            q_active = arm_chains[motion_type].active_from_full(q).tolist()
            if type(value) == tuple:
                commands[i][2] = (q_active, *value[1:])
            else:
                commands[i][2] = q_active
    for i in range(len(paths)):
        motion_type, body_type, path, _, _ = paths[i]
        if body_type == 'arm':
            paths[i][2] = [arm_chains[motion_type].active_from_full(q).tolist() for q in path]

    with open(os.path.join(log_dir, 'commands.pkl'), 'wb') as fp:
        pickle.dump(commands, fp)
    with open(os.path.join(log_dir, 'motion.pkl'), 'wb') as fp:
        pickle.dump(paths, fp)

    stats_path = os.path.join(log_dir, 'stats.json')
    with open(stats_path, 'r') as fp:
        stats = json.load(fp)
    stats['motion_plan'] = {'time': round(time() - stamp.start_time, 2)}
    with open(stats_path, 'w') as fp:
        json.dump(stats, fp)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--assembly-dir', type=str, required=True)
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--optimized', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', default=True, action='store_true')
    parser.add_argument('--no-viz', default=False, action='store_true', help='no debug viewers (for unattended runs)')
    # --single-arm (until 2026-11-13: the move arm did not avoid the hold arm) is gone: both arms are always planned as robots.
    # real_robot/run.py --single-arm still lets a human do the steps of the hold arm.
    args = parser.parse_args()

    from planning.robot.workcell import use_log_workcell
    use_log_workcell(args.log_dir) # the place of the assembly this log is planned for
    if args.no_viz: ArmMotionPlanner.VISUALIZE = False
    try:
        run_motion_plan(args.assembly_dir, args.log_dir, args.optimized, args.seed, args.verbose)
    except Exception as e:
        print(e)
        sys.exit(2 if os.path.exists(os.path.join(args.log_dir, 'planning_failure.json')) else 1)
