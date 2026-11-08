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
from planning.config import RETRACT_OPEN_RATIO as _RETRACT_OPEN_RATIO, GRASP_MIN_FINGER_OVERLAP, MIN_SIGMA, MIN_JOINT_LIMIT_MARGIN


'''
Fixture-free pickup: every part lies flat on the board in one of its stable resting poses, and is picked from (nearly) above.
Replaces run_fixture_gen.py: writes fixture/pickup.json and no fixture.obj.
'''

MAX_TILT = 30.0 # max angle [deg] between gripper approach direction and straight down
MIN_TIP_ANGLE_FALLBACK = 10.0 # deg, resting poses allowed for a part that has none tipping only above MIN_TIP_ANGLE (marked on the template)
MIN_TIP_ANGLE = 20.0 # min tilt [deg] of the board/part before the part tips over (support margin vs. height of the center of mass)
GROUND_CLEARANCE = 0.5 # min height [cm] of the (opened) gripper above the board
YAW_OFFSETS = [0, 180, 90, -90] # rotations [deg] of a resting part about the vertical axis to try for the arm IK
PICKUP_SHIFT_Y = -15.0 # cm, pickup area moved this far from get_fixture_min_y towards the assembly (the real desk ends beside the robot base)
MAX_LAYOUT_SIZE = [12 * DX, 14 * DX]
PLACEMENT_STEP = 1.25 # cm, grid of spots tried for each part (half the board hole spacing)
PART_GAP = 0.5 # cm, between the footprints of two lying parts
MAX_PLACEMENT_TRIES = 150 # spots checked with the robot per part before giving up
LIFT_CHECK_HEIGHTS = [2.5, 5.0, 7.5, 10.0] # cm above the pickup pose, checked for singularities and joint limits
N_PICKUP_IK_SAMPLES = 12 # random IK starts per pickup (besides the rest pose and the grasp's assembly pose)
PICKUP_AREA_OFFSETS = [(dx, dy) for dx in (0.0, -5.0, 5.0, -10.0, 10.0) for dy in (0.0, -5.0, 5.0, -10.0, 10.0)] # cm, tried around the default pickup area
BOARD_HALF_SIZE = (31.25, 46.25) # cm, optical board (x between the arms, y along them) # max size [cm] of the pickup area, x between the arms, y from get_fixture_min_y (board edge at y = 46.25)


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
        flat_rots.append({'rot': transform[:3, :3], 'prob': float(prob), 'margin': float(margin), 'tip_angle': float(tip_angle)})
    stable = [r for r in flat_rots if r['tip_angle'] >= min_tip_angle]
    if len(stable) == 0 and min_tip_angle > MIN_TIP_ANGLE_FALLBACK: # e.g. a small cylinder: rolls lying, but stands upright
        stable = [r for r in flat_rots if r['tip_angle'] >= MIN_TIP_ANGLE_FALLBACK]
        for r in stable: r['fallback'] = True
    return stable


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
            candidates.append({'rot': R.from_euler('z', yaw_offset, degrees=True).as_matrix() @ rot, 'tilt': float(tilt), 'tip_angle': flat_rot['tip_angle'], 'margin': flat_rot['margin'], 'fallback': bool(flat_rot.get('fallback', False)),
                               'prob': flat_rot['prob'], 'gripper_min_z': float(min_z), 'yaw_offset': yaw_offset, 'yaw_rank': i})
    candidates.sort(key=lambda c: (round(c['tilt'], 1), -c['tip_angle'], c['yaw_rank']))
    return candidates


PAGE_SIZES = {'a4': (21.0, 29.7), 'a3': (29.7, 42.0)} # portrait [cm]
PAGE_MARGIN = 0.5 # cm, unprintable border of most printers


TILE_OVERLAP = 1.0 # cm, overlap of neighbouring sheets of a tiled template


def save_pickup_template(part_meshes_pickup, pickup_info, save_path, dx=2.5, page=None):
    '''
    1:1 printable placement template (board frame, cm): part outlines, the face lying on the board, and the board holes to align it.
    page=None: one page fitted around the parts plus one board hole; page='a4': the parts on A4 sheets (portrait or landscape,
    several sheets with TILE_OVERLAP if they do not fit on one), all in one PDF
    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union

    bounds = np.array([mesh.bounds for mesh in part_meshes_pickup.values()])
    parts_lower, parts_upper = bounds[:, 0, :2].min(axis=0) - 0.5, bounds[:, 1, :2].max(axis=0) + 0.5

    # what is drawn (board frame)
    outlines, drawing = [], []
    for part_id, mesh in part_meshes_pickup.items():
        triangles = [Polygon(tri) for tri in mesh.triangles[:, :, :2] if Polygon(tri).area > 1e-8]
        outline = unary_union(triangles).buffer(0.01).buffer(-0.01) # silhouette seen from above
        outlines.append(outline)
        contact = unary_union([Polygon(tri[:, :2]) for tri in mesh.triangles if np.all(tri[:, 2] < mesh.bounds[0, 2] + 1e-3) and Polygon(tri[:, :2]).area > 1e-8])
        support = mesh.vertices[mesh.vertices[:, 2] < mesh.bounds[0, 2] + 1e-3][:, :2]
        drawing.append((part_id, mesh, outline, contact, support, support[ConvexHull(support).vertices]))
    occupied = unary_union(outlines).buffer(0.3)

    def draw(ax, lower, upper, sheet_note=None, overlap_edges=()):
        size = upper - lower
        ax.set_xlim(lower[0], upper[0]); ax.set_ylim(lower[1], upper[1]); ax.set_aspect('equal'); ax.axis('off')
        ax.add_patch(plt.Rectangle(lower, *size, fill=False, lw=0.3, color='lightgray')) # cut line
        for x in np.arange(np.ceil(lower[0] / dx - 0.5) * dx + 0.5 * dx, upper[0], dx): # board holes at (k + 0.5) * dx
            for y in np.arange(np.ceil(lower[1] / dx - 0.5) * dx + 0.5 * dx, upper[1], dx):
                ax.add_patch(plt.Circle((x, y), 0.25, fill=False, lw=0.5, color='gray'))
                ax.plot([x - 0.4, x + 0.4], [y, y], lw=0.3, color='gray'); ax.plot([x, x], [y - 0.4, y + 0.4], lw=0.3, color='gray')
        for edge in overlap_edges: # dashed: the part of this sheet covered by the neighbouring sheet
            ax.plot(*edge, lw=0.6, ls='--', color='tab:green')
        for part_id, mesh, outline, contact, support, hull in drawing:
            for polygon in getattr(outline, 'geoms', [outline]):
                ax.plot(*polygon.exterior.xy, lw=1.0, color='black')
            for polygon in getattr(contact, 'geoms', [contact]): # faces lying on the board
                if not polygon.is_empty: ax.fill(*polygon.exterior.xy, color='tab:blue', alpha=0.25, lw=0)
            ax.plot(*np.vstack([hull, hull[:1]]).T, lw=0.8, ls='--', color='tab:blue') # support polygon (also for parts resting on edges/corners)
            ax.plot(support[:, 0], support[:, 1], 'o', ms=2, color='tab:blue')
            ax.text(*mesh.center_mass[:2], f'{part_id}', ha='center', va='center', fontsize=14, weight='bold', clip_on=True)
            ax.text(mesh.center_mass[0], mesh.center_mass[1] - 0.6, f'{pickup_info[part_id]["arm"]}', ha='center', va='top', fontsize=6, clip_on=True)
            if pickup_info[part_id].get('fallback'):
                ax.text(mesh.center_mass[0], mesh.center_mass[1] - 1.2, f'tips at {pickup_info[part_id]["tip_angle"]:.0f} deg: place carefully', ha='center', va='top', fontsize=6, color='tab:red', clip_on=True)
        # scale bar and notes in a free spot of this sheet
        legend_size = np.array([10.6, 2.3 + (0.6 if sheet_note else 0)])
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
        note = (sheet_note + '\n' if sheet_note else '') + f'move robot: +x ->   assembly spot: -y\nboard x [{lower[0]:.2f}, {upper[0]:.2f}] cm, y [{lower[1]:.2f}, {upper[1]:.2f}] cm\n' \
               f'blue: on the board (dashed: support area, dots: contact corners)'
        ax.text(corner[0] + 0.3, corner[1] + legend_size[1] - 0.1, note, ha='left', va='top', fontsize=6)

    if page is None:
        lower = np.floor(parts_lower / dx - 0.5) * dx - 0.5 * dx # extend to one board hole beyond the parts
        upper = np.ceil(parts_upper / dx + 0.5) * dx + 0.5 * dx
        fig = plt.figure(figsize=((upper - lower)[0] / 2.54, (upper - lower)[1] / 2.54))
        draw(fig.add_axes([0, 0, 1, 1]), lower, upper)
        fig.savefig(save_path)
        plt.close(fig)
        return lower, upper

    # tiles: the orientation with the fewest sheets
    parts_size = parts_upper - parts_lower
    best = None
    for page_size in [np.array(PAGE_SIZES[page]), np.array(PAGE_SIZES[page])[::-1]]: # portrait, landscape
        area = page_size - 2 * PAGE_MARGIN
        n = np.maximum(1, np.ceil((parts_size - TILE_OVERLAP) / (area - TILE_OVERLAP))).astype(int)
        if best is None or n.prod() < best[1].prod(): best = (page_size, n)
    page_size, n = best
    area = page_size - 2 * PAGE_MARGIN
    span = n * area - (n - 1) * TILE_OVERLAP
    start = (parts_lower + parts_upper) / 2 - span / 2
    n_sheets = int(n.prod())
    with PdfPages(save_path) as pdf:
        for row in range(n[1] - 1, -1, -1): # top row first
            for col in range(n[0]):
                lower = start + np.array([col, row]) * (area - TILE_OVERLAP)
                upper = lower + area
                sheet = (n[1] - 1 - row) * n[0] + col + 1
                edges = []
                if col > 0: edges.append(([lower[0] + TILE_OVERLAP] * 2, [lower[1], upper[1]]))
                if col < n[0] - 1: edges.append(([upper[0] - TILE_OVERLAP] * 2, [lower[1], upper[1]]))
                if row > 0: edges.append(([lower[0], upper[0]], [lower[1] + TILE_OVERLAP] * 2))
                if row < n[1] - 1: edges.append(([lower[0], upper[0]], [upper[1] - TILE_OVERLAP] * 2))
                note = None if n_sheets == 1 else f'SHEET {sheet}/{n_sheets} (column {col + 1}/{n[0]} from the left, row {n[1] - row}/{n[1]} from the top); ' \
                                                    f'green dashed: {TILE_OVERLAP:.0f} cm overlap, lay the neighbouring sheet on it, crosses on crosses'
                fig = plt.figure(figsize=(page_size[0] / 2.54, page_size[1] / 2.54))
                draw(fig.add_axes([PAGE_MARGIN / page_size[0], PAGE_MARGIN / page_size[1], area[0] / page_size[0], area[1] / page_size[1]]), lower, upper, note, edges)
                pdf.savefig(fig)
                plt.close(fig)
    if n_sheets > 1:
        print(f'[save_pickup_template] {n_sheets} {page.upper()} sheets ({n[0]} x {n[1]}, {"portrait" if page_size[0] < page_size[1] else "landscape"}) with {TILE_OVERLAP:.0f} cm overlap')
    return start, start + span


def save_assembly_template(assembly_dir, log_dir, dx=2.5, page='a4'):
    '''
    1:1 template of the assembly spot (board frame, cm): where the parts that touch the board in the assembled state
    (the part held by the human) have to stand, plus the outline of the finished assembly. Writes fixture/assembly_template_<page>.pdf
    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    with open(os.path.join(log_dir, 'grasps.pkl'), 'rb') as fp:
        arm_type = pickle.load(fp)['arm']
    meshes = load_final_part_cfg(assembly_dir, arm_type)['mesh']
    footprint = lambda mesh: unary_union([Polygon(tri) for tri in mesh.triangles[:, :, :2] if Polygon(tri).area > 1e-8]).buffer(0.01).buffer(-0.01)
    on_board = [part_id for part_id, mesh in meshes.items() if mesh.bounds[0, 2] < 0.05]
    bounds = np.array([mesh.bounds for mesh in meshes.values()])
    center = (bounds[:, 0, :2].min(axis=0) + bounds[:, 1, :2].max(axis=0)) / 2
    page_size = np.array(PAGE_SIZES[page])
    lower, upper = center - (page_size - 2 * PAGE_MARGIN) / 2, center + (page_size - 2 * PAGE_MARGIN) / 2
    size = upper - lower

    fig = plt.figure(figsize=(page_size[0] / 2.54, page_size[1] / 2.54))
    ax = fig.add_axes([PAGE_MARGIN / page_size[0], PAGE_MARGIN / page_size[1], size[0] / page_size[0], size[1] / page_size[1]])
    ax.set_xlim(lower[0], upper[0]); ax.set_ylim(lower[1], upper[1]); ax.set_aspect('equal'); ax.axis('off')
    ax.add_patch(plt.Rectangle(lower, *size, fill=False, lw=0.3, color='lightgray'))
    for x in np.arange(np.ceil(lower[0] / dx - 0.5) * dx + 0.5 * dx, upper[0], dx):
        for y in np.arange(np.ceil(lower[1] / dx - 0.5) * dx + 0.5 * dx, upper[1], dx):
            ax.add_patch(plt.Circle((x, y), 0.25, fill=False, lw=0.5, color='gray'))
            ax.plot([x - 0.4, x + 0.4], [y, y], lw=0.3, color='gray'); ax.plot([x, x], [y - 0.4, y + 0.4], lw=0.3, color='gray')
    outline = unary_union([footprint(mesh) for mesh in meshes.values()])
    for polygon in getattr(outline, 'geoms', [outline]):
        ax.plot(*polygon.exterior.xy, lw=0.8, ls='--', color='gray') # finished assembly seen from above
    for part_id in on_board:
        mesh = meshes[part_id]
        for polygon in getattr(footprint(mesh), 'geoms', [footprint(mesh)]):
            ax.fill(*polygon.exterior.xy, color='tab:orange', alpha=0.3, lw=0); ax.plot(*polygon.exterior.xy, lw=1.2, color='black')
        ax.text(*mesh.bounds[:, :2].mean(axis=0), f'{part_id}', ha='center', va='center', fontsize=16, weight='bold')
    ax.plot([lower[0] + 0.3, lower[0] + 10.3], [lower[1] + 0.3] * 2, lw=1.5, color='red')
    ax.text(lower[0] + 5.3, lower[1] + 0.45, '10 cm (print at 100%, no "fit to page")', ha='center', va='bottom', fontsize=6, color='red')
    ax.text(lower[0] + 0.3, upper[1] - 0.3, f'ASSEMBLY SPOT: hold part(s) {", ".join(on_board)} exactly on the orange area (dashed: finished assembly from above)\n'
            f'move robot: +x ->   pickup area: +y\nboard x [{lower[0]:.2f}, {upper[0]:.2f}] cm, y [{lower[1]:.2f}, {upper[1]:.2f}] cm', ha='left', va='top', fontsize=6)
    path = os.path.join(log_dir, 'fixture', f'assembly_template_{page}.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'[save_assembly_template] {path}')
    return path


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


IK_CACHE_FILE = 'pickup_ik_cache.pkl'


def load_ik_cache(log_dir):
    path = os.path.join(log_dir, IK_CACHE_FILE)
    if os.path.exists(path):
        with open(path, 'rb') as fp:
            return pickle.load(fp)
    return {}


def save_ik_cache(log_dir, cache):
    with open(os.path.join(log_dir, IK_CACHE_FILE), 'wb') as fp:
        pickle.dump(cache, fp)


def pickup_ik_solutions(cache, planner, grasp, part_id, pickup_mat, final_mat, has_ft_sensor):
    '''
    Arm poses of the real arm for one pickup that pass every check not depending on the other parts (IK, virtual walls,
    5 cm lift, singularity, joint limits), best conditioned first; score = min(sigma / MIN_SIGMA, joint margin / MIN_JOINT_LIMIT_MARGIN).
    Cached per robot height, part, grasp and pickup pose (in the log, reused by later runs).
    '''
    from planning.plan_checks import joint_condition
    from planning.robot.workcell import get_panda_base_z
    key = ('lift', round(get_panda_base_z(), 3), part_id, int(grasp.grasp_id), tuple(np.round(pickup_mat[:3, :4], 3).flatten()))
    if key not in cache:
        rest = planner.arm_chain.active_to_full(planner.arm_chain.rest_q)
        starts = [(rest, None), (grasp.arm_q, None)] + [(planner.arm_chain.sample_joint_angles(), 1) for _ in range(N_PICKUP_IK_SAMPLES)]
        solutions = []
        for q_init, n_restart in starts:
            q = get_pickup_arm_q(planner, grasp, pickup_mat, final_mat, arm_q_init=q_init, has_ft_sensor=has_ft_sensor, optimizer='least_squares', regularization=1.0, n_restart=n_restart)
            if q is None or not planner.check_virtual_walls(q):
                continue
            # the pickup pose and the poses while lifting the part straight up (the arm folds further when lifting near the robot)
            score, q_prev = np.inf, q
            fk = planner.arm_chain.forward_kinematics(q)
            for lift in [0.0] + LIFT_CHECK_HEIGHTS:
                q_lift = q if lift == 0 else planner.inverse_kinematics(fk[:3, 3] + np.array([0, 0, lift]), fk[:3, :3], q_init=q_prev, n_restart=1)
                if q_lift is None or not planner.check_virtual_walls(q_lift) or np.abs(np.asarray(q_lift) - np.asarray(q_prev)).max() > 0.5:
                    score = 0
                    break
                sigma, margin, _ = joint_condition(planner.arm_chain.active_from_full(q_lift))
                score = min(score, sigma / MIN_SIGMA, margin / MIN_JOINT_LIMIT_MARGIN)
                q_prev = q_lift
            if score < 1:
                continue
            solutions.append((score, np.asarray(q)))
        cache[key] = sorted(solutions, key=lambda x: -x[0])
    return cache[key]


def area_allowed(center_x, min_y, bin_size, final_meshes, margin=1.0):
    '''the pickup area lies on the board and away from the assembly zone (assembly + TRANSFER_PART_CLEARANCE)'''
    from planning.config import TRANSFER_PART_CLEARANCE
    lower, upper = np.array([center_x - bin_size[0] / 2, min_y]), np.array([center_x + bin_size[0] / 2, min_y + bin_size[1]])
    if np.any(lower < -np.array(BOARD_HALF_SIZE) + margin) or np.any(upper > np.array(BOARD_HALF_SIZE) - margin):
        return False
    bounds = np.array([mesh.bounds for mesh in final_meshes.values()])
    zone_lower, zone_upper = bounds[:, 0, :2].min(axis=0) - TRANSFER_PART_CLEARANCE, bounds[:, 1, :2].max(axis=0) + TRANSFER_PART_CLEARANCE
    return bool(np.any(upper <= zone_lower) or np.any(lower >= zone_upper))


def generate_flat_pickup_pose(pose_info, min_y, max_size=MAX_LAYOUT_SIZE, center_x=0.0):
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
            rect.x + rect.width / 2 - pose_info[part_id]['center_x'] - bin_size[0] / 2 + center_x,
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
        from planning.plan_checks import grasp_finger_overlap
        deep = lambda g: grasp_finger_overlap(g, grasps['gripper'], gripper_meshes) >= GRASP_MIN_FINGER_OVERLAP # the part reaches far enough into the fingers
        from planning.plan_checks import joint_condition
        from planning.config import MIN_SIGMA_INSERTION
        def conditioned(grasps_move): # insertion poses away from singularities and joint limits
            if grasps['arm'] != 'panda': return True
            conditions = [joint_condition(arm_chain_move.active_from_full(x.arm_q)) for x in grasps_move]
            return all(sigma >= MIN_SIGMA_INSERTION and margin >= MIN_JOINT_LIMIT_MARGIN for sigma, margin, _ in conditions)
        part_grasps['move'] = [g for g in part_grasps['move'] if in_walls(g) and conditioned(g) and deep(g[0]) and is_flat(g[0])]
        if filter_hold:
            part_grasps['hold'] = [g for g in part_grasps['hold'] if is_flat(g)]
        kept_ids[part_id] = {'move': {g[0].grasp_id for g in part_grasps['move']}, 'hold': {g.grasp_id for g in part_grasps['hold']}}
        print(f'[filter_flat_grasps] part {part_id}: {len(flat_rots)} stable poses, move {n_move} -> {len(part_grasps["move"])}, hold {n_hold} -> {len(part_grasps["hold"])}')

    grasps['grasp_id_pairs'] = {(part_move, part_hold): [(m, h) for m, h in pairs if m in kept_ids[part_move]['move'] and h in kept_ids[part_hold]['hold']]
                                for (part_move, part_hold), pairs in grasps['grasp_id_pairs'].items()}
    grasps['settings']['flat_pickup'] = {'max_tilt': max_tilt, 'clearance': clearance, 'min_tip_angle': min_tip_angle, 'filter_hold': filter_hold, 'min_finger_overlap': GRASP_MIN_FINGER_OVERLAP}
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
    Lay the parts out on the table one by one, in reverse pick order, each at the free spot nearest to the pickup area
    (centered between the arms, pickup_shift_y from the default fixture place) where its pickup passes every real-robot check
    '''
    return _run_pickup_gen(assembly_dir, log_dir, max_tilt, clearance, min_tip_angle, seed, pickup_shift_y, 0.0, verbose)


def _run_pickup_gen(assembly_dir, log_dir, max_tilt, clearance, min_tip_angle, seed, pickup_shift_y, pickup_center_x, verbose=False):
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

    # clearance of the real (move) arm at each pickup: to the parts still lying and the parts already assembled at that moment
    from planning.plan_checks import SceneChecker, clear_failure, write_failure, joint_condition
    from planning.robot.geometry import load_part_meshes
    clear_failure(log_dir)
    part_meshes_raw = {k[4:]: v for k, v in load_part_meshes(assembly_dir, transform='none').items()}
    checker = SceneChecker(arm_type, gripper_type, has_ft_sensor['move'], part_meshes_raw, {k: np.eye(4) for k in part_meshes_raw}, part_cfg_final['mesh']) if arm_type == 'panda' else None
    state_at_pick = {}
    assembled = {sequence_forward[0][1]} # first held part (human / hold arm)
    for part_move, _ in sequence_forward:
        state_at_pick[part_move] = {k: ('final' if k in assembled else 'pickup') for k in part_meshes_raw}
        assembled.add(part_move)

    ik_cache = load_ik_cache(log_dir)
    t_start = time()
    random.seed(seed)
    np.random.seed(seed) # IK restarts are random
    from shapely.geometry import MultiPoint, box as shapely_box
    from shapely.affinity import translate as _translate
    shapely_translate = lambda shape, xy: _translate(shape, xoff=xy[0], yoff=xy[1])
    anchor = np.array([pickup_center_x, min_fixture_y + 10.0]) # preferred place of the parts
    final_bounds = np.array([mesh.bounds for mesh in part_cfg_final['mesh'].values()])
    from planning.config import TRANSFER_PART_CLEARANCE
    zone = shapely_box(*(final_bounds[:, 0, :2].min(axis=0) - TRANSFER_PART_CLEARANCE), *(final_bounds[:, 1, :2].max(axis=0) + TRANSFER_PART_CLEARANCE))
    board = shapely_box(-BOARD_HALF_SIZE[0] + 1.0, -BOARD_HALF_SIZE[1] + 1.0, BOARD_HALF_SIZE[0] - 1.0, BOARD_HALF_SIZE[1] - 1.0)
    grid = np.array([[x, y] for x in np.arange(-BOARD_HALF_SIZE[0], BOARD_HALF_SIZE[0], PLACEMENT_STEP) for y in np.arange(-BOARD_HALF_SIZE[1], BOARD_HALF_SIZE[1], PLACEMENT_STEP)])
    grid = grid[np.argsort(np.linalg.norm(grid - anchor, axis=1))]

    # resting pose shape of every candidate: footprint (centered) and the transform that puts it on the table at (x, y)
    shapes = {}
    for part_id in pickup_grasps:
        shapes[part_id] = []
        for candidate in candidates[part_id]:
            rot4 = np.block([[candidate['rot'], np.zeros((3, 1))], [np.zeros((1, 3)), 1]])
            mesh = part_cfg_final['mesh'][part_id].copy().apply_transform(rot4)
            center = mesh.bounds[:, :2].mean(axis=0)
            footprint = MultiPoint(mesh.convex_hull.vertices[:, :2] - center).convex_hull
            shapes[part_id].append((footprint, rot4, center, mesh.bounds[0, 2]))

    def place(part_id, k, xy):
        footprint, rot4, center, min_z = shapes[part_id][k]
        transform = np.eye(4)
        transform[:3, 3] = [xy[0] - center[0], xy[1] - center[1], -min_z]
        return transform @ rot4 # applied to the part in its final pose (as pose_pickup of the fixture pipeline)

    order = [part_move for part_move, _ in sequence_forward][::-1] + [sequence_forward[0][1]] # last picked first; the human's part last
    placed, footprints, choice, pickup_q = {}, {}, {}, {}
    for part_id in order:
        grasp, arm = pickup_grasps[part_id]
        planner = motion_planners[arm]
        found, tried, violations_seen = False, 0, {}
        for xy in grid:
            for k, (footprint, _, _, _) in enumerate(shapes[part_id]):
                shape = shapely_translate(footprint, xy)
                if not board.contains(shape) or shape.intersects(zone) or any(shape.distance(other) < PART_GAP for other in footprints.values()):
                    continue
                pose = place(part_id, k, xy)
                mat = pose @ final_mats[part_id]
                if arm == 'hold' or checker is None: # the human takes it / no real-robot checks: free spot and reachable
                    q = get_pickup_arm_q(planner, grasp, mat, final_mats[part_id], arm_q_init=planner.arm_chain.active_to_full(planner.arm_chain.rest_q), has_ft_sensor=has_ft_sensor[arm], optimizer='least_squares', regularization=1.0)
                    if arm == 'move' and (q is None or not planner.check_virtual_walls(q) or not check_lift(planner, q)): continue
                    found = True
                else:
                    tried += 1
                    checker.set_pickups(part_meshes_raw, {**placed, part_id: mat})
                    for score, q in pickup_ik_solutions(ik_cache, planner, grasp, part_id, mat, final_mats[part_id], has_ft_sensor[arm]):
                        violations = checker.check_pickup(q, min(grasp.open_ratio + _RETRACT_OPEN_RATIO, 1.0), part_id, state_at_pick[part_id])
                        if not violations:
                            pickup_q[part_id] = q
                            found = True
                            break
                        for v in violations: violations_seen[v.split(' (poses')[0]] = violations_seen.get(v.split(' (poses')[0], 0) + 1
                if found:
                    placed[part_id], footprints[part_id], choice[part_id] = mat, shape, k
                    break
            if found or tried >= MAX_PLACEMENT_TRIES: break
        if not found:
            reasons = '; '.join(f'{v} ({n}x)' for v, n in sorted(violations_seen.items(), key=lambda x: -x[1])[:3])
            write_failure(log_dir, 'run_pickup_gen', part_id, grasp.grasp_id if arm == 'move' else None, f'no spot within {tried} tries where the pickup passes the checks: {reasons}')
            save_ik_cache(log_dir, ik_cache)
            return False
        print(f'[run_pickup_gen] part {part_id} placed at ({footprints[part_id].centroid.x:.1f}, {footprints[part_id].centroid.y:.1f}), resting pose candidate {choice[part_id]}' + (f', {tried} spots tried' if tried else ''))
        if verbose and violations_seen: print('    rejected because: ' + '; '.join(f'{v} ({n}x)' for v, n in violations_seen.items()))

    pose_pickup_global = placed
    part_meshes_pickup = {part_id: part_meshes_raw[part_id].copy().apply_transform(mat) for part_id, mat in placed.items()}
    layout_bounds = np.array([mesh.bounds for mesh in part_meshes_pickup.values()])

    save_ik_cache(log_dir, ik_cache)
    fixture_dir = os.path.join(log_dir, 'fixture')
    os.makedirs(fixture_dir, exist_ok=True)
    with open(os.path.join(fixture_dir, 'pickup_q.json'), 'w') as fp: # checked arm pose of every move-arm pickup, used by run_motion_plan.py
        json.dump({part_id: np.asarray(q).tolist() for part_id, q in pickup_q.items()}, fp, indent=4)
    with open(os.path.join(fixture_dir, 'pickup.json'), 'w') as fp:
        json.dump({part_id: get_pos_euler_from_transform_matrix(mat).tolist() for part_id, mat in pose_pickup_global.items()}, fp)
    if os.path.exists(os.path.join(fixture_dir, 'fixture.obj')):
        os.remove(os.path.join(fixture_dir, 'fixture.obj')) # no fixture: parts lie on the board
    pickup_info = {part_id: {'arm': pickup_grasps[part_id][1], 'grasp_id': int(pickup_grasps[part_id][0].grasp_id),
                             **{k: v for k, v in candidates[part_id][choice[part_id]].items() if k != 'rot'}} for part_id in pickup_grasps}
    with open(os.path.join(fixture_dir, 'flat_pickup.json'), 'w') as fp:
        json.dump(pickup_info, fp, indent=4)
    trimesh.Scene([trimesh.creation.box(bounds=[[*(layout_bounds[:, 0, :2].min(axis=0) - 1.0), -0.2], [*(layout_bounds[:, 1, :2].max(axis=0) + 1.0), 0]])] +
                  [mesh.copy() for mesh in part_meshes_pickup.values()]).export(os.path.join(fixture_dir, 'pickup_layout.glb'))
    lower, upper = save_pickup_template(part_meshes_pickup, pickup_info, os.path.join(fixture_dir, 'pickup_template.pdf'))
    print(f'[run_pickup_gen] placement template: {os.path.join(fixture_dir, "pickup_template.pdf")} ({upper[0] - lower[0]:.1f} x {upper[1] - lower[1]:.1f} cm)')
    save_pickup_template(part_meshes_pickup, pickup_info, os.path.join(fixture_dir, 'pickup_template_a4.pdf'), page='a4') # A4 sheets, tiled if needed
    print(f'[run_pickup_gen] placement template on A4: {os.path.join(fixture_dir, "pickup_template_a4.pdf")}')
    save_assembly_template(assembly_dir, log_dir)
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
    parser.add_argument('--template', type=str, default=None, choices=['fit', 'a4', 'assembly'], help='only (re)draw the placement template of the existing layout (assembly: the A4 template of the assembly spot)')
    parser.add_argument('--max-tilt', type=float, default=MAX_TILT, help='max angle [deg] between gripper approach and straight down')
    parser.add_argument('--clearance', type=float, default=GROUND_CLEARANCE, help='min height [cm] of the gripper above the board')
    parser.add_argument('--min-tip-angle', type=float, default=MIN_TIP_ANGLE, help='min tipping angle [deg] of a resting pose')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    if args.template == 'assembly':
        save_assembly_template(args.assembly_dir, args.log_dir)
        sys.exit(0)
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
