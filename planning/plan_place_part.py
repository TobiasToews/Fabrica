'''
Let the real (move) arm take one part from its pickup spot and stand it at its final pose at the assembly spot (e.g. the
first part, which the human otherwise brings there in single-arm mode). Uses a move grasp of the part that can pick it in
the resting pose of the layout, plans with the same planner as run_motion_plan.py and checks every path with
planning/plan_checks.py (re-plans with another seed, then tries the next grasp).

  home -> pickup (gripper opens) -> grasp -> lift, carry, lower onto the table at the final pose -> release -> home
By default only this part lies on the table (move it before laying out the others); --others-on-table checks against all of them.

Writes <log-dir>/place_part<id>/motion.pkl (+ workcell.json), run it with
  python real_robot/run.py --single-arm --fn <log-dir>/place_part<id>/motion.pkl

python planning/plan_place_part.py --assembly-dir assets/fabrica/human_normal --log-dir logs/exp_human_normal_flat_v3/human_normal --part 0
'''
import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import json
import pickle
import random
import numpy as np

from assets.transform import get_transform_matrix
from planning.config import RETRACT_OPEN_RATIO, RETRACT_DELTA_FAR, OPEN_RATIO_REST, TRANSFER_PART_CLEARANCE, MAX_PLAN_ATTEMPTS, GRASP_MIN_FINGER_OVERLAP, ASSEMBLY_ZONE_MARGIN
from planning.plan_checks import SceneChecker, grasp_finger_overlap
from planning.robot.geometry import load_part_meshes
from planning.robot.motion_plan_arm import ArmMotionPlanner
from planning.robot.util_arm import get_arm_chain
from planning.robot.workcell import get_panda_base_z, get_panda_desk_z
from planning.run_motion_plan import get_pickup_gripper_pose, get_back_retract_dir, post_process_motion
from planning.run_pickup_gen import load_final_part_cfg, pickup_ik_solutions, load_ik_cache, save_ik_cache

UP = np.array([0, 0, 1.0])


def search_spot(part, usable, raw, pickup_mats, final_meshes, final_mat, gripper_type, pickable, step=1.25, max_tries=200):
    '''nearest free spot (and resting pose) to the part's current spot where pickable() finds a clean pickup'''
    from shapely.geometry import MultiPoint, box
    from shapely.affinity import translate
    from planning.robot.geometry import load_gripper_meshes
    from planning.run_pickup_gen import get_flat_rots, get_flat_pickup_candidates, BOARD_HALF_SIZE, PART_GAP
    gripper_meshes = load_gripper_meshes(gripper_type, os.path.join(project_base_dir, 'assets'))
    mesh_final = final_meshes[part]
    rots = get_flat_rots(mesh_final)
    candidates = []
    for grasp in usable:
        for c in get_flat_pickup_candidates(grasp, mesh_final, rots, gripper_type, gripper_meshes):
            if not any(np.allclose(c['rot'], o) for o in candidates): candidates.append(c['rot'])
    others = [MultiPoint(raw[k].copy().apply_transform(pickup_mats[k]).vertices[:, :2]).convex_hull for k in raw if k != part]
    bounds = np.array([m.bounds for m in final_meshes.values()])
    zone = box(*(bounds[:, 0, :2].min(axis=0) - ASSEMBLY_ZONE_MARGIN), *(bounds[:, 1, :2].max(axis=0) + ASSEMBLY_ZONE_MARGIN))
    board = box(-BOARD_HALF_SIZE[0] + 1, -BOARD_HALF_SIZE[1] + 1, BOARD_HALF_SIZE[0] - 1, BOARD_HALF_SIZE[1] - 1)
    current = raw[part].copy().apply_transform(pickup_mats[part]).bounds[:, :2].mean(axis=0)
    grid = np.array([[x, y] for x in np.arange(-BOARD_HALF_SIZE[0], BOARD_HALF_SIZE[0], step) for y in np.arange(-BOARD_HALF_SIZE[1], BOARD_HALF_SIZE[1], step)])
    grid = grid[np.argsort(np.linalg.norm(grid - current, axis=1))]
    tries = 0
    for xy in grid:
        for rot in candidates:
            rot4 = np.block([[rot, np.zeros((3, 1))], [np.zeros((1, 3)), 1]])
            m = mesh_final.copy().apply_transform(rot4)
            center = m.bounds[:, :2].mean(axis=0)
            fp = translate(MultiPoint(m.convex_hull.vertices[:, :2] - center).convex_hull, xoff=xy[0], yoff=xy[1])
            if not board.contains(fp) or fp.intersects(zone) or any(fp.distance(o) < PART_GAP for o in others): continue
            T = np.eye(4); T[:3, 3] = [xy[0] - center[0], xy[1] - center[1], -m.bounds[0, 2]]
            mat = T @ rot4 @ final_mat
            tries += 1
            best = pickable(mat)
            if best is not None: return best, mat
            if tries >= max_tries: return None, None
    return None, None


def plan_place_part(assembly_dir, log_dir, part, seed=0, others_on_table=False, verbose=False):
    ArmMotionPlanner.VISUALIZE = False
    with open(os.path.join(log_dir, 'grasps.pkl'), 'rb') as fp:
        grasps = pickle.load(fp)
    assert grasps['arm'] == 'panda'
    has_ft = grasps['ft_sensor']['move']
    with open(os.path.join(log_dir, 'fixture', 'pickup.json'), 'r') as fp:
        pickup_mats = {k: get_transform_matrix(v) for k, v in json.load(fp).items()}
    part_cfg = load_final_part_cfg(assembly_dir, grasps['arm'])
    final_meshes = part_cfg['mesh']
    from assets.transform import get_transform_matrix_quat
    final_mat = get_transform_matrix_quat(part_cfg['pos'][part], part_cfg['quat'][part])
    raw = {k[4:]: v for k, v in load_part_meshes(assembly_dir, transform='none').items()}
    pickup_meshes = {k: raw[k].copy().apply_transform(pickup_mats[k]) for k in raw}

    chain = get_arm_chain('panda', 'move')
    planner = ArmMotionPlanner(chain, grasps['gripper'], has_ft, virtual_walls=True,
                               table_z=get_panda_base_z() + get_panda_desk_z(), part_clearance=TRANSFER_PART_CLEARANCE)
    bounds = np.array([m.bounds for m in final_meshes.values()])
    planner.clearance_exempt_xy = (bounds[:, 0, :2].min(axis=0) - ASSEMBLY_ZONE_MARGIN, bounds[:, 1, :2].max(axis=0) + ASSEMBLY_ZONE_MARGIN)
    checker = SceneChecker('panda', grasps['gripper'], has_ft, raw, pickup_mats, final_meshes)
    rest = chain.active_to_full(chain.rest_q)
    # default: the part is moved before the other parts are laid out (only it lies on the table); --others-on-table: all lie there
    on_table = list(raw) if others_on_table else [part]
    lying = {k: 'pickup' for k in on_table}
    after = {**lying, part: 'final'}
    others_lying = [pickup_meshes[k] for k in on_table if k != part]

    ik_cache = load_ik_cache(log_dir)
    usable = []
    for g in grasps['grasps'][part]['move']:
        grasp = g[0]
        if grasp_finger_overlap(grasp, grasps['gripper']) < GRASP_MIN_FINGER_OVERLAP: continue
        if checker.check_path([grasp.arm_q], grasp.open_ratio, {k: v for k, v in lying.items() if k != part}, carried=part, inserting=True):
            continue # the arm pose at the final pose (part standing at the assembly spot) is not clean
        usable.append(grasp)

    def pickable(mat):
        '''(score, grasp, q) of the best grasp that picks the part lying at mat, or None'''
        checker.set_pickups(raw, {**pickup_mats, part: mat})
        best = None
        for grasp in usable:
            for score, q_pick in pickup_ik_solutions(ik_cache, planner, grasp, part, mat, final_mat, has_ft):
                if not checker.check_pickup(q_pick, min(grasp.open_ratio + RETRACT_OPEN_RATIO, 1.0), part, lying, closed_ratio=grasp.open_ratio):
                    if best is None or score > best[0]: best = (score, grasp, q_pick)
                    break
        return best

    best = pickable(pickup_mats[part])
    moved = False
    if best is None: # the spot of the layout is not reachable for the arm (e.g. it was chosen for the human): search a free spot nearby
        print(f'[plan_place_part] part {part} cannot be picked at its spot of the layout, searching another spot')
        best, new_mat = search_spot(part, usable, raw, pickup_mats, final_meshes, final_mat, grasps['gripper'], pickable)
        if best is not None:
            pickup_mats = {**pickup_mats, part: new_mat}
            pickup_meshes[part] = raw[part].copy().apply_transform(new_mat)
            moved = True
    save_ik_cache(log_dir, ik_cache)
    checker.set_pickups(raw, pickup_mats)
    candidates = [best] if best is not None else []
    print(f'[plan_place_part] part {part}: ' + (f'picked with move grasp {best[1].grasp_id}' if best else 'no reachable pickup'))

    for score, grasp, q_pick in candidates:
        open_retract = min(grasp.open_ratio + RETRACT_OPEN_RATIO, 1.0)
        q_place = grasp.arm_q
        gripper_pickup = get_pickup_gripper_pose(grasp, pickup_mats[part], final_mat)
        for attempt in range(MAX_PLAN_ATTEMPTS):
            random.seed(seed * 1000 + attempt); np.random.seed(seed * 1000 + attempt)
            try:
                # home -> pickup, gripper opens on the way
                p1, p2 = planner.plan_path_switch(rest, q_pick, part_meshes=[pickup_meshes[k] for k in on_table], open_ratio=OPEN_RATIO_REST, open_ratio_next=open_retract,
                                                  retract_start=get_back_retract_dir(planner, rest), retract_goal=UP, retract_delta=RETRACT_DELTA_FAR, max_speed=4)
                # carry: lift off the table, lower onto the table at the final pose
                carry = planner.plan_path_with_grasp(q_pick, q_place, move_pickup_mesh=pickup_meshes[part], gripper_pickup_transform=gripper_pickup,
                                                     still_meshes=others_lying, open_ratio=grasp.open_ratio, retract_start=UP, retract_goal=UP,
                                                     retract_delta=RETRACT_DELTA_FAR, max_speed=4)
                # release and go home, straight up first
                back = planner.plan_path(q_place, rest, part_meshes=others_lying + [final_meshes[part]], open_ratio=open_retract,
                                         retract_start=UP, retract_goal=get_back_retract_dir(planner, rest), retract_delta=RETRACT_DELTA_FAR, max_speed=4)
            except AssertionError as e:
                print(f'[plan_place_part] grasp {grasp.grasp_id}, attempt {attempt + 1}: planner error {e}')
                continue
            if None in [p1, p2, carry, back]:
                print(f'[plan_place_part] grasp {grasp.grasp_id}, attempt {attempt + 1}: no path')
                continue
            violations = (checker.check_path(p1, OPEN_RATIO_REST, lying, pick_part=part, label='approach: ') if len(p1) else []) + \
                         (checker.check_path(p2, open_retract, lying, pick_part=part, label='approach: ') if len(p2) else []) + \
                         checker.check_path(carry, grasp.open_ratio, lying, carried=part, label='carry: ') + \
                         checker.check_path(back, open_retract, after, released_part=part, label='back: ')
            if violations:
                print(f'[plan_place_part] grasp {grasp.grasp_id}, attempt {attempt + 1} rejected: ' + '; '.join(violations))
                continue

            # run.py format: ['move/hold', 'arm/gripper', path (7 active joints) / open ratio, part, task]
            def active(path, last=None):
                return [chain.active_from_full(q).tolist() for q in post_process_motion(chain, path, last)]
            approach = active(p1 + p2 if len(p1) and len(p2) else (p1 or p2))
            carry_a = active(carry, chain.active_to_full(np.array(approach[-1])))
            back_a = active(back, chain.active_to_full(np.array(carry_a[-1])))
            motion = [
                ['move', 'arm', [chain.active_from_full(rest).tolist()], None, 'init'],
                ['move', 'gripper', OPEN_RATIO_REST, None, 'init'],
                ['move', 'gripper', open_retract, None, 'open'],
                ['move', 'arm', approach, None, 'switch'],
                ['move', 'gripper', grasp.open_ratio, None, 'close'],
                ['move', 'arm', carry_a, part, 'transport'],
                ['move', 'gripper', open_retract, None, 'open'],
                ['move', 'arm', back_a, None, 'transport'],
                ['move', 'gripper', OPEN_RATIO_REST, None, 'close'],
            ]
            out_dir = os.path.join(log_dir, f'place_part{part}')
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, 'motion.pkl'), 'wb') as fp:
                pickle.dump(motion, fp)
            workcell_path = os.path.join(log_dir, 'workcell.json')
            workcell = json.load(open(workcell_path)) if os.path.exists(workcell_path) else {'robot_base_z_cm': get_panda_base_z()}
            workcell.update({'robot_base_z_cm': get_panda_base_z(), 'verified': True, 'note': f'place part {part} with the move arm (planning/plan_place_part.py), grasp {grasp.grasp_id}'})
            with open(os.path.join(out_dir, 'workcell.json'), 'w') as fp:
                json.dump(workcell, fp, indent=4)
            if moved:
                from assets.transform import get_pos_euler_from_transform_matrix
                from planning.run_pickup_gen import save_pickup_template
                with open(os.path.join(out_dir, 'pickup.json'), 'w') as fp:
                    json.dump({k: get_pos_euler_from_transform_matrix(m).tolist() for k, m in pickup_mats.items()}, fp)
                info = json.load(open(os.path.join(log_dir, 'fixture', 'flat_pickup.json')))
                info[part] = {**info[part], 'arm': 'move'}
                save_pickup_template(pickup_meshes, info, os.path.join(out_dir, 'pickup_template_a4.pdf'), page='a4')
                b = pickup_meshes[part].bounds
                print(f'[plan_place_part] NEW spot of part {part}: board x {b[0, 0]:.1f}..{b[1, 0]:.1f}, y {b[0, 1]:.1f}..{b[1, 1]:.1f} cm -> {out_dir}/pickup_template_a4.pdf')
            print(f'[plan_place_part] part {part} with move grasp {grasp.grasp_id}: all paths pass the checks -> {out_dir}/motion.pkl')
            return True
    print(f'[plan_place_part] no clean plan for part {part}')
    return False


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--assembly-dir', type=str, required=True)
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--part', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--others-on-table', default=False, action='store_true', help='the other parts already lie on the table (default: only this part)')
    args = parser.parse_args()
    from planning.robot.workcell import use_log_workcell
    use_log_workcell(args.log_dir) # the place of the assembly this log is planned for
    sys.exit(0 if plan_place_part(args.assembly_dir, args.log_dir, args.part, args.seed, args.others_on_table) else 1)
