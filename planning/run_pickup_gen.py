import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import json
import pickle
import random
import shutil
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from time import time

from assets.load import load_pos_quat_dict
from assets.transform import get_transform_matrix, get_transform_matrix_quat, mat_to_pos_quat, get_pos_euler_from_transform_matrix
from planning.config import RETRACT_OPEN_RATIO, RETRACT_DELTA_FAR
from planning.robot.geometry import load_part_meshes, load_gripper_meshes, transform_gripper_meshes
from planning.robot.motion_plan_arm import ArmMotionPlanner, VIRTUAL_WALL_SAFETY
from planning.robot.util_arm import get_arm_chain, get_virtual_wall_margin
from planning.robot.workcell import get_assembly_center, get_fixture_min_y, get_dual_arm_box
from planning.run_fixture_gen import run_bin_packing, generate_pickup_meshes, check_part_gripper_collision, DELTA_BUFFER_SIZE, DX
from planning.run_motion_plan import get_pickup_arm_q
from planning.run_seq_opt import SequenceOptimizer


'''
Fixture-free pickup: every part lies flat on the board in one of its stable resting poses, and is picked from (nearly) above.
Replaces run_fixture_gen.py: writes fixture/pickup.json and no fixture.obj.
'''

MAX_TILT = 30.0 # max angle [deg] between gripper approach direction and straight down
MIN_TIP_ANGLE = 20.0 # min tilt [deg] of the board/part before the part tips over (support margin vs. height of the center of mass)
GROUND_CLEARANCE = 0.5 # min height [cm] of the (opened) gripper above the board
YAW_OFFSETS = [0, 180, 90, -90] # rotations [deg] of a resting part about the vertical axis to try for the arm IK
PICKUP_SHIFT_Y = -20.0 # cm, pickup area moved this far from get_fixture_min_y towards the assembly (the real desk ends beside the robot base)
MAX_LAYOUT_SIZE = [12 * DX, 14 * DX] # max size [cm] of the pickup area, x between the arms, y from get_fixture_min_y (board edge at y = 46.25)


def get_flat_rots(mesh, min_tip_angle=MIN_TIP_ANGLE):
    '''
    Stable resting rotations of a mesh lying on the board (about the origin), with the support margin and tipping angle
    '''
    transforms, probs = trimesh.poses.compute_stable_poses(mesh)
    flat_rots = []
    for transform, prob in zip(transforms, probs):
        mesh_rest = mesh.copy().apply_transform(transform)
        support = mesh_rest.vertices[mesh_rest.vertices[:, 2] < mesh_rest.bounds[0, 2] + 1e-3][:, :2]
        if len(support) < 3: continue
        try:
            hull = ConvexHull(support)
        except Exception: # degenerate (line) support
            continue
        margin = -np.max(hull.equations[:, :2] @ mesh_rest.center_mass[:2] + hull.equations[:, 2])
        tip_angle = np.degrees(np.arctan2(margin, mesh_rest.center_mass[2] - mesh_rest.bounds[0, 2]))
        if tip_angle < min_tip_angle: continue
        flat_rots.append({'rot': transform[:3, :3], 'prob': float(prob), 'margin': float(margin), 'tip_angle': float(tip_angle)})
    return flat_rots


def get_flat_pickup_candidates(grasp, part_mesh, flat_rots, gripper_type, gripper_meshes, max_tilt=MAX_TILT, clearance=GROUND_CLEARANCE):
    '''
    For a grasp (defined at the final pose), the resting rotations where the gripper approaches from (nearly) above
    and stays above the board; the yaw first aligns the gripper fingers with the x axis (as in the fixture), then is
    rotated by YAW_OFFSETS as alternatives for the arm IK
    '''
    grasp_rot = R.from_quat(grasp.quat[[1, 2, 3, 0]]).as_matrix()
    candidates = []
    for flat_rot in flat_rots:
        approach = flat_rot['rot'] @ grasp_rot @ np.array([0, 0, 1])
        tilt = np.degrees(np.arccos(np.clip(-approach[2], -1, 1)))
        if tilt > max_tilt: continue

        l2r = flat_rot['rot'] @ grasp_rot @ np.array([0, -1, 0])
        yaw = -np.arctan2(l2r[1], l2r[0])
        rot = R.from_euler('z', yaw).as_matrix() @ flat_rot['rot']

        # gripper height does not depend on the yaw
        pickup_mat = np.eye(4)
        pickup_mat[:3, :3] = rot
        pickup_mat[2, 3] = -(part_mesh.vertices @ rot.T)[:, 2].min() # rest on the board
        gripper_pos, gripper_quat = mat_to_pos_quat(pickup_mat @ get_transform_matrix_quat(grasp.pos, grasp.quat))
        min_z = np.inf
        for open_ratio in [grasp.open_ratio, min(grasp.open_ratio + RETRACT_OPEN_RATIO, 1.0)]:
            meshes = transform_gripper_meshes(gripper_type, gripper_meshes, gripper_pos, gripper_quat, np.eye(4), open_ratio)
            min_z = min(min_z, min(mesh.bounds[0, 2] for mesh in meshes.values()))
        if min_z < clearance: continue

        for i, yaw_offset in enumerate(YAW_OFFSETS):
            candidates.append({'rot': R.from_euler('z', yaw_offset, degrees=True).as_matrix() @ rot, 'tilt': float(tilt), 'tip_angle': flat_rot['tip_angle'], 'margin': flat_rot['margin'],
                               'prob': flat_rot['prob'], 'gripper_min_z': float(min_z), 'yaw_offset': yaw_offset, 'yaw_rank': i})
    candidates.sort(key=lambda c: (round(c['tilt'], 1), -c['tip_angle'], c['yaw_rank']))
    return candidates


PAGE_SIZES = {'a4': (21.0, 29.7), 'a3': (29.7, 42.0)} # portrait [cm]
PAGE_MARGIN = 0.5 # cm, unprintable border of most printers


def save_pickup_template(part_meshes_pickup, pickup_info, save_path, dx=2.5, page=None):
    '''
    1:1 printable placement template (board frame, cm): part outlines, the face lying on the board, and the board holes to align it.
    page=None: page fitted around the parts plus one board hole; page='a4'/'a3': cropped to the parts on that paper (portrait or landscape)
    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union

    bounds = np.array([mesh.bounds for mesh in part_meshes_pickup.values()])
    parts_lower, parts_upper = bounds[:, 0, :2].min(axis=0), bounds[:, 1, :2].max(axis=0)
    if page is None:
        lower = np.floor(parts_lower / dx - 0.5) * dx - 0.5 * dx # extend to one board hole beyond the parts
        upper = np.ceil(parts_upper / dx + 0.5) * dx + 0.5 * dx
        page_size = upper - lower
        offset = np.zeros(2)
    else:
        parts_size = parts_upper - parts_lower
        page_size = None
        for size in [np.array(PAGE_SIZES[page]), np.array(PAGE_SIZES[page])[::-1]]: # portrait, then landscape
            if np.all(parts_size + 2 * PAGE_MARGIN <= size):
                page_size = size
                break
        if page_size is None:
            print(f'[save_pickup_template] parts span {parts_size[0]:.1f} x {parts_size[1]:.1f} cm, too large for {page.upper()} with {PAGE_MARGIN} cm margins')
            return None
        center = (parts_lower + parts_upper) / 2
        lower = center - (page_size - 2 * PAGE_MARGIN) / 2 # drawing area: the page minus the margins, centered on the parts
        upper = center + (page_size - 2 * PAGE_MARGIN) / 2
        offset = np.full(2, PAGE_MARGIN)
    size = upper - lower

    fig = plt.figure(figsize=(page_size[0] / 2.54, page_size[1] / 2.54))
    ax = fig.add_axes([offset[0] / page_size[0], offset[1] / page_size[1], size[0] / page_size[0], size[1] / page_size[1]]) # 1 cm = 1 cm on paper
    ax.set_xlim(lower[0], upper[0]); ax.set_ylim(lower[1], upper[1]); ax.set_aspect('equal'); ax.axis('off')
    if page is not None:
        ax.add_patch(plt.Rectangle(lower, *size, fill=False, lw=0.3, color='lightgray')) # cut line

    for x in np.arange(np.ceil(lower[0] / dx - 0.5) * dx + 0.5 * dx, upper[0], dx): # board holes at (k + 0.5) * dx
        for y in np.arange(np.ceil(lower[1] / dx - 0.5) * dx + 0.5 * dx, upper[1], dx):
            ax.add_patch(plt.Circle((x, y), 0.25, fill=False, lw=0.5, color='gray'))
            ax.plot([x - 0.4, x + 0.4], [y, y], lw=0.3, color='gray'); ax.plot([x, x], [y - 0.4, y + 0.4], lw=0.3, color='gray')

    outlines = []
    for part_id, mesh in part_meshes_pickup.items():
        triangles = [Polygon(tri) for tri in mesh.triangles[:, :, :2] if Polygon(tri).area > 1e-8]
        outline = unary_union(triangles).buffer(0.01).buffer(-0.01) # silhouette seen from above
        outlines.append(outline)
        for polygon in getattr(outline, 'geoms', [outline]):
            ax.plot(*polygon.exterior.xy, lw=1.0, color='black')
        contact = unary_union([Polygon(tri[:, :2]) for tri in mesh.triangles if np.all(tri[:, 2] < mesh.bounds[0, 2] + 1e-3) and Polygon(tri[:, :2]).area > 1e-8])
        for polygon in getattr(contact, 'geoms', [contact]): # faces lying on the board
            if not polygon.is_empty: ax.fill(*polygon.exterior.xy, color='tab:blue', alpha=0.25, lw=0)
        support = mesh.vertices[mesh.vertices[:, 2] < mesh.bounds[0, 2] + 1e-3][:, :2]
        hull = support[ConvexHull(support).vertices]
        ax.plot(*np.vstack([hull, hull[:1]]).T, lw=0.8, ls='--', color='tab:blue') # support polygon (also for parts resting on edges/corners)
        ax.plot(support[:, 0], support[:, 1], 'o', ms=2, color='tab:blue')
        info = pickup_info[part_id]
        ax.text(*mesh.center_mass[:2], f'{part_id}', ha='center', va='center', fontsize=14, weight='bold')
        ax.text(mesh.center_mass[0], mesh.center_mass[1] - 0.6, f'{info["arm"]}', ha='center', va='top', fontsize=6)

    # scale bar and notes in a free spot of the page (fixed corner for the fitted page)
    legend_size = np.array([10.6, 2.3])
    occupied = unary_union(outlines).buffer(0.3)
    corner = lower + 0.2
    for y in np.arange(lower[1] + 0.2, upper[1] - legend_size[1], 0.25):
        free = [x for x in np.arange(lower[0] + 0.2, upper[0] - legend_size[0], 0.25) if not box(x, y, x + legend_size[0], y + legend_size[1]).intersects(occupied)]
        if free:
            corner = np.array([free[0], y])
            break
    else:
        print('[save_pickup_template] no free spot for the scale bar, it overlaps a part')
    ax.plot([corner[0] + 0.3, corner[0] + 10.3], [corner[1] + 0.3] * 2, lw=1.5, color='red') # print scale check
    ax.text(corner[0] + 5.3, corner[1] + 0.45, '10 cm (print at 100%, no "fit to page")', ha='center', va='bottom', fontsize=6, color='red')
    ax.text(corner[0] + 0.3, corner[1] + legend_size[1] - 0.1,
            f'move robot: +x ->   assembly spot: -y\nboard x [{lower[0]:.2f}, {upper[0]:.2f}] cm, y [{lower[1]:.2f}, {upper[1]:.2f}] cm\n'
            f'blue: on the board (dashed: support area, dots: contact corners)', ha='left', va='top', fontsize=6)
    fig.savefig(save_path)
    plt.close(fig)
    return lower, upper


def save_pickup_templates(log_dir, pages=(None, 'a4')):
    '''(Re)draw the placement templates of a finished layout: fixture/pickup_template.pdf (fitted) and fixture/pickup_template_<page>.pdf'''
    import trimesh as _trimesh
    fixture_dir = os.path.join(log_dir, 'fixture')
    with open(os.path.join(fixture_dir, 'flat_pickup.json'), 'r') as fp:
        pickup_info = json.load(fp)
    scene = _trimesh.load(os.path.join(fixture_dir, 'pickup_layout.glb'), force='scene')
    part_meshes_pickup = {}
    for node in scene.graph.nodes_geometry:
        if not node.endswith('.obj'): continue # the board box of the layout
        transform, geometry_name = scene.graph[node]
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        part_meshes_pickup[node[:-len('.obj')]] = mesh
    for page in pages:
        path = os.path.join(fixture_dir, 'pickup_template.pdf' if page is None else f'pickup_template_{page}.pdf')
        result = save_pickup_template(part_meshes_pickup, pickup_info, path, page=page)
        if result is not None:
            print(f'[save_pickup_templates] {path}' + ('' if page is None else f' ({page.upper()})'))


def generate_flat_pickup_pose(pose_info, min_y, max_size=MAX_LAYOUT_SIZE):
    '''
    Smallest 2d layout of the resting parts on the board (generate_pickup_pose of run_fixture_gen.py without the fixture bottom and size limits)
    '''
    best = None
    for bin_x in np.arange(4 * DX, max_size[0] + 0.5 * DX, DX):
        for bin_y in np.arange(4 * DX, max_size[1] + 0.5 * DX, DX):
            if best is not None and (bin_x * bin_y, bin_y) >= (best[1][0] * best[1][1], best[1][1]): continue
            packer = run_bin_packing(pose_info, [bin_x, bin_y])
            if packer is not None:
                best = (packer, [bin_x, bin_y])
    if best is None:
        return None, None
    packer, bin_size = best

    pickup_pose = {}
    for rect in packer[0]:
        part_id = rect.rid
        part_transform = np.eye(4)
        part_transform[:3, :3] = pose_info[part_id]['rot_mat']
        part_transform[:3, 3] = np.array([
            rect.x + rect.width / 2 - pose_info[part_id]['center_x'] - bin_size[0] / 2,
            rect.y + rect.height / 2 - pose_info[part_id]['center_y'] + min_y,
            -pose_info[part_id]['min_z']])
        pickup_pose[part_id] = get_pos_euler_from_transform_matrix(part_transform).tolist()
    return pickup_pose, bin_size


def check_lift(motion_planner, q, lift=RETRACT_DELTA_FAR):
    '''
    A part lying on the board is inside the ground safety margin, so the motion planner first lifts it straight up by RETRACT_DELTA_FAR
    '''
    fk = motion_planner.arm_chain.forward_kinematics(q)
    q_lift = motion_planner.inverse_kinematics(fk[:3, 3] + np.array([0, 0, lift]), fk[:3, :3], q_init=q)
    return q_lift is not None and motion_planner.check_virtual_walls(q_lift)


def load_final_part_cfg(assembly_dir, arm_type):
    part_meshes_final = load_part_meshes(assembly_dir, transform='final')
    part_meshes_final = {k.replace('part', ''): v for k, v in part_meshes_final.items()}
    for part_mesh in part_meshes_final.values():
        part_mesh.apply_translation(get_assembly_center(arm_type))
    part_pos_dict_final, part_quat_dict_final = load_pos_quat_dict(assembly_dir, transform='final')
    part_pos_dict_final = {part_id: part_pos_dict_final[part_id] + get_assembly_center(arm_type) for part_id in part_meshes_final.keys()}
    return {'mesh': part_meshes_final, 'pos': part_pos_dict_final, 'quat': part_quat_dict_final}


def filter_flat_grasps(assembly_dir, log_dir, max_tilt=MAX_TILT, clearance=GROUND_CLEARANCE, min_tip_angle=MIN_TIP_ANGLE, filter_hold=False):
    '''
    Keep only grasps that can pick their part lying flat on the board (run before sequence planning);
    hold grasps are only used on the board for the first held part, so they are kept by default
    '''
    grasps_path = os.path.join(log_dir, 'grasps.pkl')
    grasps_all_path = os.path.join(log_dir, 'grasps_unfiltered.pkl')
    if not os.path.exists(grasps_all_path):
        shutil.copy(grasps_path, grasps_all_path)
    with open(grasps_all_path, 'rb') as fp:
        grasps = pickle.load(fp)

    part_cfg_final = load_final_part_cfg(assembly_dir, grasps['arm'])
    gripper_meshes = load_gripper_meshes(grasps['gripper'], os.path.join(project_base_dir, 'assets'))
    if grasps['arm'] == 'panda': # the move arm is the real robot: its assembly/disassembly poses must stay inside franka-interface's virtual walls
        arm_chain_move = get_arm_chain(grasps['arm'], 'move')
        in_walls = lambda grasps_move: all(get_virtual_wall_margin(arm_chain_move, g.arm_q) >= VIRTUAL_WALL_SAFETY for g in grasps_move)
    else:
        in_walls = lambda grasps_move: True

    kept_ids = {}
    for part_id, part_grasps in grasps['grasps'].items():
        part_mesh = part_cfg_final['mesh'][part_id]
        flat_rots = get_flat_rots(part_mesh, min_tip_angle)
        is_flat = lambda g: len(get_flat_pickup_candidates(g, part_mesh, flat_rots, grasps['gripper'], gripper_meshes, max_tilt, clearance)) > 0
        n_move, n_hold = len(part_grasps['move']), len(part_grasps['hold'])
        part_grasps['move'] = [g for g in part_grasps['move'] if in_walls(g) and is_flat(g[0])]
        if filter_hold:
            part_grasps['hold'] = [g for g in part_grasps['hold'] if is_flat(g)]
        kept_ids[part_id] = {'move': {g[0].grasp_id for g in part_grasps['move']}, 'hold': {g.grasp_id for g in part_grasps['hold']}}
        print(f'[filter_flat_grasps] part {part_id}: {len(flat_rots)} stable poses, move {n_move} -> {len(part_grasps["move"])}, hold {n_hold} -> {len(part_grasps["hold"])}')

    grasps['grasp_id_pairs'] = {(part_move, part_hold): [(m, h) for m, h in pairs if m in kept_ids[part_move]['move'] and h in kept_ids[part_hold]['hold']]
                                for (part_move, part_hold), pairs in grasps['grasp_id_pairs'].items()}
    grasps['settings']['flat_pickup'] = {'max_tilt': max_tilt, 'clearance': clearance, 'min_tip_angle': min_tip_angle, 'filter_hold': filter_hold}
    with open(grasps_path, 'wb') as fp:
        pickle.dump(grasps, fp)


def run_flat_seq_opt(assembly_dir, log_dir, max_tilt=MAX_TILT, clearance=GROUND_CLEARANCE, min_tip_angle=MIN_TIP_ANGLE, plot=False, verbose=False):
    '''
    run_seq_opt.py, but the first hold grasp (the only one that picks its part from the board) must pick the part lying flat
    '''
    with open(os.path.join(log_dir, 'precedence.pkl'), 'rb') as fp:
        G_preced = pickle.load(fp)
    with open(os.path.join(log_dir, 'grasps.pkl'), 'rb') as fp:
        grasps = pickle.load(fp)
    with open(os.path.join(log_dir, 'tree.pkl'), 'rb') as fp:
        tree = pickle.load(fp)

    t_start = time()
    seq_optimizer = SequenceOptimizer(G_preced, grasps)
    grasp_tree = seq_optimizer._convert_part_tree_to_grasp_tree(tree, verbose=verbose)
    if grasp_tree is None:
        print(f'[run_flat_seq_opt] No complete sequence in {log_dir}')
        return False

    part_cfg_final = load_final_part_cfg(assembly_dir, grasps['arm'])
    gripper_meshes = load_gripper_meshes(grasps['gripper'], os.path.join(project_base_dir, 'assets'))
    flat_rots = {}
    first_holds = [node for node in grasp_tree.nodes if len(node[0]) == 1 and node[1] in node[0]] # only the first held part remains (disassembly order)
    infeasible = []
    for node in first_holds:
        part_id, grasp_id = node[1], node[2]
        if part_id not in flat_rots: flat_rots[part_id] = get_flat_rots(part_cfg_final['mesh'][part_id], min_tip_angle)
        grasp = seq_optimizer.grasps[part_id]['hold'][grasp_id]
        if len(get_flat_pickup_candidates(grasp, part_cfg_final['mesh'][part_id], flat_rots[part_id], grasps['gripper'], gripper_meshes, max_tilt, clearance)) == 0:
            infeasible.append(node)
    print(f'[run_flat_seq_opt] {len(first_holds) - len(infeasible)}/{len(first_holds)} first hold grasps can pick their part lying flat')
    grasp_tree = grasp_tree.copy()
    grasp_tree.remove_nodes_from(infeasible)

    result = seq_optimizer._optimize_part_tree_from_grasp_tree(grasp_tree, verbose=verbose)
    if result is None:
        print(f'[run_flat_seq_opt] Failed sequence planning in {log_dir}')
        return False
    tree_opt, _ = result

    with open(os.path.join(log_dir, 'tree_opt.pkl'), 'wb') as fp:
        pickle.dump(tree_opt, fp)
    stats_path = os.path.join(log_dir, 'stats.json')
    with open(stats_path, 'r') as fp:
        stats = json.load(fp)
    stats['seq_opt'] = {'time': round(time() - t_start, 2), 'flat': True}
    with open(stats_path, 'w') as fp:
        json.dump(stats, fp)
    if plot:
        seq_optimizer.plot_tree(tree_opt, save_path=os.path.join(log_dir, 'tree_opt.png'))
    return True


def run_pickup_gen(assembly_dir, log_dir, max_tilt=MAX_TILT, clearance=GROUND_CLEARANCE, min_tip_angle=MIN_TIP_ANGLE, seed=0, pickup_shift_y=PICKUP_SHIFT_Y, verbose=False):
    '''
    Choose a flat resting pose for every part of the optimized sequence and lay the parts out on the board
    '''
    with open(os.path.join(log_dir, 'precedence.pkl'), 'rb') as fp:
        G_preced = pickle.load(fp)
    with open(os.path.join(log_dir, 'grasps.pkl'), 'rb') as fp:
        grasps = pickle.load(fp)
    with open(os.path.join(log_dir, 'tree_opt.pkl'), 'rb') as fp:
        tree = pickle.load(fp)
    gripper_type, arm_type, has_ft_sensor = grasps['gripper'], grasps['arm'], grasps['ft_sensor']

    sequence, grasps_sequence = SequenceOptimizer(G_preced, grasps).get_sequence(tree)
    if sequence is None:
        print(f'[run_pickup_gen] No feasible sequence found in {log_dir}')
        return False
    sequence_forward, grasps_sequence_forward = sequence[::-1], grasps_sequence[::-1]

    # grasp used to pick up each part: hold grasp for the first held part, move grasp for the others
    pickup_grasps = {} # part_id: (grasp, arm)
    for i, ((part_move, part_hold), (grasps_move, grasp_hold)) in enumerate(zip(sequence_forward, grasps_sequence_forward)):
        if i == 0: pickup_grasps[part_hold] = (grasp_hold, 'hold')
        pickup_grasps[part_move] = (grasps_move[0], 'move')

    part_cfg_final = load_final_part_cfg(assembly_dir, arm_type)
    gripper_meshes = load_gripper_meshes(gripper_type, os.path.join(project_base_dir, 'assets'))

    candidates = {}
    for part_id, (grasp, _) in pickup_grasps.items():
        part_mesh = part_cfg_final['mesh'][part_id]
        candidates[part_id] = get_flat_pickup_candidates(grasp, part_mesh, get_flat_rots(part_mesh, min_tip_angle), gripper_type, gripper_meshes, max_tilt, clearance)
        if len(candidates[part_id]) == 0:
            print(f'[run_pickup_gen] part {part_id} cannot be picked lying flat with grasp {grasp.grasp_id}, run with --filter first')
            return False

    arm_box_move, arm_box_hold = get_dual_arm_box(arm_type)
    motion_planners = {'move': ArmMotionPlanner(get_arm_chain(arm_type, 'move'), gripper_type, has_ft_sensor['move'], arm_box_move, virtual_walls=arm_type == 'panda'),
                       'hold': ArmMotionPlanner(get_arm_chain(arm_type, 'hold'), gripper_type, has_ft_sensor['hold'], arm_box_hold)}
    final_mats = {part_id: get_transform_matrix_quat(part_cfg_final['pos'][part_id], part_cfg_final['quat'][part_id]) for part_id in pickup_grasps}
    min_fixture_y = get_fixture_min_y(arm_type) + pickup_shift_y

    t_start = time()
    random.seed(seed)
    np.random.seed(seed) # IK restarts are random
    choice = {part_id: 0 for part_id in pickup_grasps}
    while True:
        pose_info = {}
        for part_id in pickup_grasps:
            rot = candidates[part_id][choice[part_id]]['rot']
            mesh = part_cfg_final['mesh'][part_id].copy().apply_transform(np.block([[rot, np.zeros((3, 1))], [np.zeros((1, 3)), 1]]))
            pose_info[part_id] = {
                'extent_x': mesh.extents[0], 'extent_y': mesh.extents[1],
                'center_x': mesh.bounds[0, 0] + mesh.extents[0] / 2, 'center_y': mesh.bounds[0, 1] + mesh.extents[1] / 2,
                'min_z': mesh.bounds[0, 2],
                'rot_mat': rot,
            }

        # 2d layout, buffer parts until no gripper hits a neighbouring part
        while True:
            pose_pickup, bin_size = generate_flat_pickup_pose(pose_info, min_fixture_y)
            if bin_size is None:
                print('[run_pickup_gen] parts do not fit on the pickup area')
                return False
            part_meshes_pickup, gripper_meshes_pickup = generate_pickup_meshes(part_cfg_final, sequence, grasps_sequence, gripper_type, pose_pickup)
            parts_to_buffer = check_part_gripper_collision(part_meshes_pickup, gripper_meshes_pickup, sequence)
            if len(parts_to_buffer) == 0: break
            for part_id in parts_to_buffer:
                pose_info[part_id]['extent_x'] += DELTA_BUFFER_SIZE

        # pickup IK, try the next resting pose of a part that cannot be reached
        pose_pickup_global = {part_id: get_transform_matrix(pose) @ final_mats[part_id] for part_id, pose in pose_pickup.items()}
        unreachable = []
        for part_id, (grasp, arm) in pickup_grasps.items():
            planner = motion_planners[arm]
            for q_init in [planner.arm_chain.active_to_full(planner.arm_chain.rest_q), grasp.arm_q]: # run_motion_plan.py starts the pickup IK from the rest pose
                q = get_pickup_arm_q(planner, grasp, pose_pickup_global[part_id], final_mats[part_id], arm_q_init=q_init, has_ft_sensor=has_ft_sensor[arm], optimizer='least_squares', regularization=1.0)
                if q is None or not planner.check_virtual_walls(q) or not check_lift(planner, q):
                    unreachable.append(part_id)
                    break
        if len(unreachable) == 0: break
        for part_id in unreachable:
            if verbose: print(f'[run_pickup_gen] part {part_id}: pickup/lift IK failed for candidate {choice[part_id]}')
            choice[part_id] += 1
            if choice[part_id] >= len(candidates[part_id]):
                print(f'[run_pickup_gen] part {part_id} is unreachable in every flat resting pose')
                return False

    fixture_dir = os.path.join(log_dir, 'fixture')
    os.makedirs(fixture_dir, exist_ok=True)
    with open(os.path.join(fixture_dir, 'pickup.json'), 'w') as fp:
        json.dump({part_id: get_pos_euler_from_transform_matrix(mat).tolist() for part_id, mat in pose_pickup_global.items()}, fp)
    if os.path.exists(os.path.join(fixture_dir, 'fixture.obj')):
        os.remove(os.path.join(fixture_dir, 'fixture.obj')) # no fixture: parts lie on the board
    pickup_info = {part_id: {'arm': pickup_grasps[part_id][1], 'grasp_id': int(pickup_grasps[part_id][0].grasp_id),
                             **{k: v for k, v in candidates[part_id][choice[part_id]].items() if k != 'rot'}} for part_id in pickup_grasps}
    with open(os.path.join(fixture_dir, 'flat_pickup.json'), 'w') as fp:
        json.dump(pickup_info, fp, indent=4)
    trimesh.Scene([trimesh.creation.box(bounds=[[-bin_size[0] / 2, min_fixture_y, -0.2], [bin_size[0] / 2, min_fixture_y + bin_size[1], 0]])] +
                  list(part_meshes_pickup.values())).export(os.path.join(fixture_dir, 'pickup_layout.glb'))
    lower, upper = save_pickup_template(part_meshes_pickup, pickup_info, os.path.join(fixture_dir, 'pickup_template.pdf'))
    print(f'[run_pickup_gen] placement template: {os.path.join(fixture_dir, "pickup_template.pdf")} ({upper[0] - lower[0]:.1f} x {upper[1] - lower[1]:.1f} cm)')
    if save_pickup_template(part_meshes_pickup, pickup_info, os.path.join(fixture_dir, 'pickup_template_a4.pdf'), page='a4') is not None:
        print(f'[run_pickup_gen] placement template on A4: {os.path.join(fixture_dir, "pickup_template_a4.pdf")}')
    for part_id, info in pickup_info.items():
        print(f'[run_pickup_gen] part {part_id} ({info["arm"]} grasp {info["grasp_id"]}): tilt {info["tilt"]:.1f} deg, tips over at {info["tip_angle"]:.0f} deg, yaw offset {info["yaw_offset"]} deg, gripper {info["gripper_min_z"]:.2f} cm above board')

    stats_path = os.path.join(log_dir, 'stats.json')
    with open(stats_path, 'r') as fp:
        stats = json.load(fp)
    stats['fixture_gen'] = {'time': round(time() - t_start, 2), 'flat': True}
    with open(stats_path, 'w') as fp:
        json.dump(stats, fp)
    return True


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--assembly-dir', type=str, required=True)
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--filter', action='store_true', default=False, help='step 1: filter grasps.pkl to move grasps that can pick a flat part (run before run_seq_plan.py)')
    parser.add_argument('--seq-opt', action='store_true', default=False, help='step 2: replaces run_seq_opt.py, the first hold grasp must pick a flat part')
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--filter-hold', action='store_true', default=False, help='also filter hold grasps (too strict: only the first hold grasp picks from the board)')
    parser.add_argument('--pickup-shift-y', type=float, default=PICKUP_SHIFT_Y, help='[cm] shift of the pickup area towards the assembly (negative y)')
    parser.add_argument('--template', type=str, default=None, choices=['fit', 'a4', 'a3'], help='only (re)draw the placement template of the existing layout')
    parser.add_argument('--max-tilt', type=float, default=MAX_TILT, help='max angle [deg] between gripper approach and straight down')
    parser.add_argument('--clearance', type=float, default=GROUND_CLEARANCE, help='min height [cm] of the gripper above the board')
    parser.add_argument('--min-tip-angle', type=float, default=MIN_TIP_ANGLE, help='min tipping angle [deg] of a resting pose')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    if args.template is not None:
        save_pickup_templates(args.log_dir, pages=[None if args.template == 'fit' else args.template])
        sys.exit(0)

    if args.seq_opt:
        success = run_flat_seq_opt(args.assembly_dir, args.log_dir, args.max_tilt, args.clearance, args.min_tip_angle, args.plot, args.verbose)
    elif args.filter:
        success = filter_flat_grasps(args.assembly_dir, args.log_dir, args.max_tilt, args.clearance, args.min_tip_angle, filter_hold=args.filter_hold) is not False
    else:
        success = run_pickup_gen(args.assembly_dir, args.log_dir, args.max_tilt, args.clearance, args.min_tip_angle, args.seed, args.pickup_shift_y, args.verbose)
    sys.exit(0 if success else 1)
