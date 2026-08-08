import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
from scipy.spatial import Delaunay
from rectpack import newPacker
import trimesh
import json
from time import time

from assets.load import load_pos_quat_dict
from assets.transform import get_transform_matrix, get_transform_matrix_quat, mat_to_pos_quat, get_pos_euler_from_transform_matrix
from planning.robot.geometry import load_part_meshes, load_gripper_meshes, transform_gripper_meshes, get_buffered_meshes
from planning.robot.workcell import get_assembly_center, get_board_dx, get_fixture_min_y
from planning.run_seq_plan import SequencePlanner
from planning.run_seq_opt import SequenceOptimizer
from planning.utils.fixture_countersunk import generate_countersunk_pad


# fixture board parameters
DX = get_board_dx()
BOTTOM_THICKNESS = 0.5 # bottom thickness of the fixture without mold
EDGE_THICKNESS = 3.0 # thickness of the fixture edge
MIN_MOLD_DEPTH = 1.0 # minimum depth of the mold
MOLD_EDGE_OFFSET_PART = [0.05, 0.05, 0.0] # offset from part edge to mold edge
MOLD_EDGE_OFFSET_GRIPPER = [0.8, 0.8, 0.4] # offset from gripper edge to mold edge
PART_BOUNDARY_OFFSET = 0.2 # offset from part boundary to part edge
PART_GAP = 2.0 # gap between parts
MAX_BIN_SIZE_SINGLE = [8 * DX, 10 * DX] # maximum size of bin for rect pack (one print)
MAX_BIN_SIZE_DOUBLE = [8 * DX, 20 * DX] # maximum size of bin for rect pack (two prints)
MAX_BIN_SIZE_BLOCKING = [12 * DX, 20 * DX] # maximum size of bin for rect pack (blocking collision check)
DELTA_BIN_SIZE = 1 * DX # delta size of bin for rect pack
DELTA_BUFFER_SIZE = 2.0 # delta size of buffer for part-gripper collision


def generate_individual_pose_info(part_cfg_final, sequence, grasps_sequence):
    
    part_meshes_final = part_cfg_final['mesh']
    pose_info = {}
    sequence_forward = sequence[::-1]
    grasps_sequence_forward = grasps_sequence[::-1]

    for i, ((part_move, part_hold), (grasps_move, grasp_hold)) in enumerate(zip(sequence_forward, grasps_sequence_forward)):
        grasp_move_final = grasps_move[0]

        if i == 0: # first step, both arm pick up
            gripper_l2r_dir = R.from_quat(grasp_hold.quat[[1, 2, 3, 0]]).apply([0, -1, 0])
            gripper_b2f_dir = R.from_quat(grasp_hold.quat[[1, 2, 3, 0]]).apply([0, 0, 1])
            target_l2r_dir = np.array([1, 0, 0])
            target_b2f_dir = np.array([0, 0, -1])
            pickup_rot_mat = R.align_vectors([target_l2r_dir, target_b2f_dir], [gripper_l2r_dir, gripper_b2f_dir])[0].as_matrix()

            hold_mesh = part_meshes_final[part_hold].copy()
            pickup_transform_mat = np.eye(4)
            pickup_transform_mat[:3, :3] = pickup_rot_mat
            hold_mesh.apply_transform(pickup_transform_mat)

            pose_info[part_hold] = {
                'extent_x': hold_mesh.extents[0], 
                'extent_y': hold_mesh.extents[1], 
                'center_x': np.min(hold_mesh.vertices[:, 0]) + hold_mesh.extents[0] / 2,
                'center_y': np.min(hold_mesh.vertices[:, 1]) + hold_mesh.extents[1] / 2,
                'min_z': np.min(hold_mesh.vertices[:, 2]),
                'rot_mat': pickup_rot_mat,
            }
        
        # move arm pick up
        gripper_l2r_dir = R.from_quat(grasp_move_final.quat[[1, 2, 3, 0]]).apply([0, -1, 0])
        gripper_b2f_dir = R.from_quat(grasp_move_final.quat[[1, 2, 3, 0]]).apply([0, 0, 1])
        target_l2r_dir = np.array([1, 0, 0])
        target_b2f_dir = np.array([0, 0, -1])
        pickup_rot_mat = R.align_vectors([target_l2r_dir, target_b2f_dir], [gripper_l2r_dir, gripper_b2f_dir])[0].as_matrix()

        move_mesh = part_meshes_final[part_move].copy()
        pickup_transform_mat = np.eye(4)
        pickup_transform_mat[:3, :3] = pickup_rot_mat
        move_mesh.apply_transform(pickup_transform_mat)

        pose_info[part_move] = {
            'extent_x': move_mesh.extents[0], 
            'extent_y': move_mesh.extents[1], 
            'center_x': np.min(move_mesh.vertices[:, 0]) + move_mesh.extents[0] / 2,
            'center_y': np.min(move_mesh.vertices[:, 1]) + move_mesh.extents[1] / 2,
            'min_z': np.min(move_mesh.vertices[:, 2]),
            'rot_mat': pickup_rot_mat,
        }

    return pose_info


def plot_packing(packer):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    for index, abin in enumerate(packer):
        bw, bh  = abin.width, abin.height

        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        for rect in abin:
            x, y, w, h = rect.x, rect.y, rect.width, rect.height
            plt.axis([0,bw,0,bh])
            # print('rectangle', w,h)
            patch = patches.Rectangle(
                                        (x, y),  # (x,y)
                                        w,          # width
                                        h,          # height
                                        facecolor="#00ffff",
                                        edgecolor="black",
                                        linewidth=3
                                    )
            ax.add_patch(patch)
            rx, ry = patch.get_xy()
            cx = rx + patch.get_width()/2.0
            cy = ry + patch.get_height()/2.0

            ax.annotate(f'w:{w}\nh:{h}', (cx, cy), color='b', weight='bold', 
                        fontsize=4, ha='center', va='center')
        
        plt.show()


def run_bin_packing(pose_info, bin_size):
    packer = newPacker(rotation=False)

    for part_id, part_pose_info in pose_info.items():
        packer.add_rect(part_pose_info['extent_x'] + PART_GAP, part_pose_info['extent_y'] + PART_GAP, part_id)
    packer.add_bin(bin_size[0], bin_size[1])

    packer.pack()
    all_rects = packer.rect_list()
    if len(all_rects) == len(pose_info):
        return packer
    else:
        return None


def generate_pickup_pose(pose_info, min_fixture_y, render=False):

    packer = run_bin_packing(pose_info, MAX_BIN_SIZE_DOUBLE) # try big bin size
    if packer is None:
        return None, None # no feasible bin size
    
    packer = run_bin_packing(pose_info, MAX_BIN_SIZE_SINGLE) # try small bin size
    if packer is not None:
        max_bin_size = MAX_BIN_SIZE_SINGLE
    else:
        max_bin_size = MAX_BIN_SIZE_DOUBLE

    best_packer = packer
    best_bin_size = max_bin_size
    best_bin_area = np.prod(max_bin_size)

    delta_bin_size = DELTA_BIN_SIZE
    min_bin_area = np.sum([(pose_info[part_id]['extent_x'] + PART_GAP) * (pose_info[part_id]['extent_y'] + PART_GAP) for part_id in pose_info.keys()])
    min_bin_x = max(np.ceil(min_bin_area / max_bin_size[1] / delta_bin_size), 4) * delta_bin_size
    min_bin_y = max(np.ceil(min_bin_area / max_bin_size[0] / delta_bin_size), 4) * delta_bin_size

    for bin_x in np.arange(min_bin_x, max_bin_size[0] + 0.5 * delta_bin_size, delta_bin_size):
        for bin_y in np.arange(min_bin_y, max_bin_size[1] + 0.5 * delta_bin_size, delta_bin_size):
            current_area = bin_x * bin_y
            if current_area >= best_bin_area:
                continue

            packer = run_bin_packing(pose_info, [bin_x, bin_y])
            if packer is not None:
                best_packer = packer
                best_bin_size = [bin_x, bin_y]
                best_bin_area = current_area

    packer, bin_size = best_packer, best_bin_size

    if render:
        plot_packing(packer)

    pickup_pose = {}
    for rect in packer[0]:
        part_id = rect.rid
        part_transform = np.eye(4)
        part_transform[:3, :3] = pose_info[part_id]['rot_mat']
        part_transform[:3, 3] = np.array([
            rect.x + rect.width / 2 - pose_info[part_id]['center_x'] - bin_size[0] / 2,
            rect.y + rect.height / 2 - pose_info[part_id]['center_y'] + min_fixture_y,
            BOTTOM_THICKNESS - pose_info[part_id]['min_z']])
        pickup_pose[part_id] = get_pos_euler_from_transform_matrix(part_transform).tolist()

    return pickup_pose, bin_size


def get_swept_mesh(mesh_start, mesh_end):
    mesh_swept = trimesh.points.PointCloud(mesh_start.vertices.tolist() + mesh_end.vertices.tolist())
    mesh_swept = trimesh.convex.convex_hull(np.unique(mesh_swept.vertices.round(decimals=6), axis=0), qhull_options='Qx Qs Qt')
    return mesh_swept


def generate_pickup_meshes(part_cfg_final, sequence, grasps_sequence, gripper_type, pickup_pose):

    part_meshes_final = part_cfg_final['mesh']
    asset_folder = os.path.join(project_base_dir, 'assets')
    gripper_meshes = load_gripper_meshes(gripper_type, asset_folder)

    # pickup part meshes
    part_meshes_pickup = {k: v.copy() for k, v in part_meshes_final.items()}
    for part_id, part_transform in pickup_pose.items():
        part_meshes_pickup[part_id].apply_transform(get_transform_matrix(part_transform))
    
    # pickup gripper meshes
    sequence_forward = sequence[::-1]
    grasps_sequence_forward = grasps_sequence[::-1]
    gripper_meshes_pickup = {}

    for i, ((part_move, part_hold), (grasps_move_t, grasp_hold)) in enumerate(zip(sequence_forward, grasps_sequence_forward)):
        grasp_move_final = grasps_move_t[0]

        if i == 0:
            gripper_final_mat_hold = get_transform_matrix_quat(grasp_hold.pos, grasp_hold.quat)
            part_pickup_mat_hold = get_transform_matrix(pickup_pose[part_hold])
            gripper_pickup_mat_hold = part_pickup_mat_hold @ gripper_final_mat_hold
            gripper_hold_pickup_pos, gripper_hold_pickup_quat = mat_to_pos_quat(gripper_pickup_mat_hold)
            gripper_meshes_hold_tight = transform_gripper_meshes(gripper_type, gripper_meshes, gripper_hold_pickup_pos, gripper_hold_pickup_quat, np.eye(4), grasp_hold.open_ratio - 0.05)
            gripper_meshes_hold_loose = transform_gripper_meshes(gripper_type, gripper_meshes, gripper_hold_pickup_pos, gripper_hold_pickup_quat, np.eye(4), grasp_hold.open_ratio + 0.15)
            gripper_meshes_pickup[part_hold] = trimesh.boolean.union([get_swept_mesh(gripper_meshes_hold_tight[gripper_part], gripper_meshes_hold_loose[gripper_part]) for gripper_part in gripper_meshes_hold_tight.keys()])

        gripper_final_mat_move = get_transform_matrix_quat(grasp_move_final.pos, grasp_move_final.quat)
        part_pickup_mat_move = get_transform_matrix(pickup_pose[part_move])
        gripper_pickup_mat_move = part_pickup_mat_move @ gripper_final_mat_move
        gripper_move_pickup_pos, gripper_move_pickup_quat = mat_to_pos_quat(gripper_pickup_mat_move)
        gripper_meshes_move_tight = transform_gripper_meshes(gripper_type, gripper_meshes, gripper_move_pickup_pos, gripper_move_pickup_quat, np.eye(4), grasp_move_final.open_ratio - 0.05)
        gripper_meshes_move_loose = transform_gripper_meshes(gripper_type, gripper_meshes, gripper_move_pickup_pos, gripper_move_pickup_quat, np.eye(4), grasp_move_final.open_ratio + 0.15)
        gripper_meshes_pickup[part_move] = trimesh.boolean.union([get_swept_mesh(gripper_meshes_move_tight[gripper_part], gripper_meshes_move_loose[gripper_part]) for gripper_part in gripper_meshes_move_tight.keys()])

    return part_meshes_pickup, gripper_meshes_pickup


def generate_fixture(part_meshes_pickup, gripper_meshes_pickup, bin_size, min_fixture_y):

    # determine fixture height and part positions
    board_height_max = 0.0

    for part_id in part_meshes_pickup.keys():

        part_mesh = part_meshes_pickup[part_id]
        part_com = part_mesh.center_mass

        board_height = BOTTOM_THICKNESS + MIN_MOLD_DEPTH
        while True:
            part_sliced = part_mesh.slice_plane([0, 0, board_height], [0, 0, -1], cap=True)
            hole_hull = Delaunay(part_sliced.vertices[:, :2])
            if hole_hull.find_simplex(part_com[:2]) >= 0: # com inside hole hull
                break
            board_height += 1.0

        if board_height > board_height_max:
            board_height_max = board_height

    # verify bin size
    part_meshes_concat = trimesh.util.concatenate(list(part_meshes_pickup.values()))
    part_meshes_vertices_in_fixture = part_meshes_concat.vertices[part_meshes_concat.vertices[:, 2] < board_height_max]
    vertices_min, vertices_max = part_meshes_vertices_in_fixture.min(axis=0), part_meshes_vertices_in_fixture.max(axis=0)
    part_extent = vertices_max - vertices_min
    edge_gap = (np.array(bin_size) - part_extent[:2]) / 2
    assert np.all(edge_gap >= 0), 'Bin size is too small'

    # generate compact fixture mesh
    box_min, box_max = np.zeros(3), np.zeros(3)
    box_units = np.floor(part_extent[:2] / DX) + 1
    box_units = np.ceil(box_units / 2) * 2 # make it even
    box_extent = box_units * DX
    box_min = np.array([-box_extent[0] / 2, min_fixture_y, 0])
    box_max = np.array([box_extent[0] / 2, min_fixture_y + box_extent[1], board_height_max])
    board_mesh = trimesh.creation.box(bounds=[box_min, box_max])
    board_mesh_bottom = trimesh.creation.box(bounds=[box_min, [box_max[0], box_max[1], BOTTOM_THICKNESS]])

    # part translation
    box_center = (box_min + box_max) / 2.0
    part_center = (vertices_min + vertices_max) / 2.0
    part_translation = box_center - part_center
    part_translation[2] = 0.0

    # create convex hull for each part with swept volume
    part_meshes_swept = {}
    for part_id, part_mesh in part_meshes_pickup.items():
        part_mesh_swept_low = part_mesh.slice_plane([0, 0, board_height_max + 0.01], [0, 0, -1], cap=True)
        part_mesh_swept_high = part_mesh_swept_low.copy()
        part_mesh_swept_high.apply_translation([0, 0, board_height_max - BOTTOM_THICKNESS + 0.01])
        part_meshes_swept[part_id] = get_swept_mesh(part_mesh_swept_low, part_mesh_swept_high)

    # subtract parts from board, only keep part area
    part_boxes = []
    for part_id, part_mesh in part_meshes_swept.items():
        part_mesh_buffered = get_buffered_meshes(part_mesh, np.array(MOLD_EDGE_OFFSET_PART) / 2)
        part_mesh_buffered.apply_translation(part_translation)
        board_mesh = trimesh.boolean.difference([board_mesh, part_mesh_buffered])

        part_vertices = part_mesh_buffered.vertices
        part_min, part_max = part_vertices.min(axis=0), part_vertices.max(axis=0)
        part_min -= PART_BOUNDARY_OFFSET
        part_max += PART_BOUNDARY_OFFSET
        part_min[2] = 0.0 - 1e-2
        part_max[2] = part_meshes_pickup[part_id].center_mass[2] + 0.5
        part_box = trimesh.creation.box(bounds=[part_min, part_max])
        part_boxes.append(part_box)

        if gripper_meshes_pickup[part_id].vertices.min(axis=0)[2] < board_height_max:
            gripper_hull_pickup = gripper_meshes_pickup[part_id].slice_plane([0, 0, board_height_max + 0.01], [0, 0, -1], cap=True).convex_hull
            gripper_hull_pickup_buffered = get_buffered_meshes(gripper_hull_pickup, np.array(MOLD_EDGE_OFFSET_GRIPPER) / 2)
            gripper_hull_pickup_buffered.apply_translation(part_translation)
            board_mesh = trimesh.boolean.difference([board_mesh, gripper_hull_pickup_buffered])

    part_boxes = trimesh.boolean.union(part_boxes)
    board_mesh = trimesh.boolean.intersection([board_mesh, part_boxes])
    board_mesh = trimesh.boolean.union([board_mesh, board_mesh_bottom])

    return board_mesh, part_translation


def check_part_gripper_collision(part_meshes_pickup, gripper_meshes_pickup, sequence):

    part_disassembly_sequence = [part_move for part_move, _ in sequence] + [sequence[-1][1]]
    parts_to_buffer = []
    col_manager = trimesh.collision.CollisionManager()
    for part_id in part_disassembly_sequence:
        if col_manager.in_collision_single(gripper_meshes_pickup[part_id]):
            parts_to_buffer.append(part_id)
        col_manager.add_object(part_id, part_meshes_pickup[part_id])

    return parts_to_buffer


def add_countersunk_pads_to_fixture(fixture_mesh, min_fixture_y):
    bin_size = fixture_mesh.extents[:2]
    pad_lower_x, pad_upper_x = -bin_size[0] / 2 - DX / 2, bin_size[0] / 2 + DX / 2
    pad_lower_y, pad_upper_y = min_fixture_y + DX / 2, min(min_fixture_y + bin_size[1] - DX / 2, min_fixture_y + MAX_BIN_SIZE_DOUBLE[1] // 2 + DX / 2)
    pad_centers = [(pad_lower_x, pad_lower_y), (pad_upper_x, pad_lower_y), (pad_lower_x, pad_upper_y), (pad_upper_x, pad_upper_y)]
    pad_meshes = []
    for pad_center in pad_centers:
        pad_mesh = generate_countersunk_pad()
        pad_mesh.apply_translation([pad_center[0], pad_center[1], 0.0])
        pad_meshes.append(pad_mesh)
    fixture_mesh = trimesh.boolean.union([fixture_mesh] + pad_meshes)
    return fixture_mesh


def run_fixture_gen(assembly_dir, log_dir, optimized, seed, render=False):
    import pyglet
    # Only force pyglet's headless (EGL) backend when there's truly no X
    # display to fall back to. On WSL2 with WSLg, DISPLAY is set and the
    # regular Xlib backend renders fine (GPU-accelerated via WSLg), whereas
    # pyglet's headless/EGL backend fails to create a context in that setup.
    pyglet.options["headless"] = (not render) and not os.environ.get("DISPLAY")

    precedence_path = os.path.join(log_dir, 'precedence.pkl')
    if not os.path.exists(precedence_path):
        print(f'[run_fixture_gen] {precedence_path} not found')
        return
    grasps_path = os.path.join(log_dir, 'grasps.pkl')
    if not os.path.exists(grasps_path):
        print(f'[run_fixture_gen] {grasps_path} not found')
        return

    with open(precedence_path, 'rb') as fp:
        G_preced = pickle.load(fp)
    with open(grasps_path, 'rb') as fp:
        grasps = pickle.load(fp)
    arm_type = grasps['arm']

    tree_path = os.path.join(log_dir, 'tree_opt.pkl') if optimized else os.path.join(log_dir, 'tree.pkl')
    if not os.path.exists(tree_path):
        print(f'[run_fixture_gen] {tree_path} not found')
        return
    with open(tree_path, 'rb') as fp:
        tree = pickle.load(fp)

    asset_folder = os.path.join(project_base_dir, './assets')
    if optimized:
        seq_optimizer = SequenceOptimizer(G_preced, grasps)
        sequence, grasps_sequence = seq_optimizer.get_sequence(tree)
    else:
        seq_planner = SequencePlanner(asset_folder, assembly_dir, G_preced, grasps, save_sdf=True, contact_eps=None)
        sequence, grasps_sequence = seq_planner.sample_sequence(tree, seed=seed)

    if sequence is None or grasps_sequence is None:
        print(f'[run_fixture_gen] No feasible sequence found in {tree_path}')
        return

    # get part meshes
    part_meshes_final = load_part_meshes(assembly_dir, transform='final')
    part_meshes_final = {k.replace('part', ''): v for k, v in part_meshes_final.items()}
    for part_id, part_mesh in part_meshes_final.items():
        part_mesh.apply_translation(get_assembly_center(arm_type))
    part_pos_dict_final, part_quat_dict_final = load_pos_quat_dict(assembly_dir, transform='final')
    part_pos_dict_final = {part_id: part_pos_dict_final[part_id] + get_assembly_center(arm_type) for part_id in part_meshes_final.keys()}
    part_cfg_final = {'mesh': part_meshes_final, 'pos': part_pos_dict_final, 'quat': part_quat_dict_final}

    min_fixture_y = get_fixture_min_y(arm_type)

    t_start = time()

    # get part orientation from grasps
    pose_info_individual = generate_individual_pose_info(part_cfg_final, sequence, grasps_sequence)

    bin_size = None
    while True: # 2d packing and make sure collision free with gripper

        # get pickup pose (relative to final pose)
        pose_pickup, bin_size = generate_pickup_pose(pose_info_individual, min_fixture_y, render=render)
        if bin_size is None:
            print(f'[run_fixture_gen] Bin size exceeds maximum size')
            return

        # get pickup meshes
        part_meshes_pickup, gripper_meshes_pickup = generate_pickup_meshes(part_cfg_final, sequence, grasps_sequence, grasps['gripper'], pose_pickup)

        # check part-gripper collision
        parts_to_buffer = check_part_gripper_collision(part_meshes_pickup, gripper_meshes_pickup, sequence)
        if len(parts_to_buffer) == 0:
            break
        else: # if collision, buffer parts
            for part_id in parts_to_buffer:
                pose_info_individual[part_id]['extent_x'] += DELTA_BUFFER_SIZE

    # generate fixture by subtracting part and gripper meshes
    fixture_mesh, part_translation = generate_fixture(part_meshes_pickup, gripper_meshes_pickup, bin_size, min_fixture_y)
    for part_id, part_mesh in part_meshes_pickup.items():
        part_mesh.apply_translation(part_translation)

    # add countersunk pads to fixture
    fixture_mesh = add_countersunk_pads_to_fixture(fixture_mesh, min_fixture_y)
    fixture_size = fixture_mesh.vertices.max(axis=0)[:2] - fixture_mesh.vertices.min(axis=0)[:2]
    # print(f'assembly_dir: {assembly_dir}, fixture size: {fixture_size}, fixture area: {np.prod(fixture_size)}')

    scene = trimesh.Scene([fixture_mesh] + list(part_meshes_pickup.values()))
    if render:
        scene.show()

    # transform pickup pose from relative-to-final to global
    pose_pickup_global = {}
    for part_id, part_pose in pose_pickup.items():
        part_pose_global = get_transform_matrix(part_pose) @ get_transform_matrix_quat(part_cfg_final['pos'][part_id], part_cfg_final['quat'][part_id])
        part_pose_global[:3, 3] += part_translation
        pose_pickup_global[part_id] = get_pos_euler_from_transform_matrix(part_pose_global).tolist()

    fixture_dir = os.path.join(log_dir, 'fixture')
    os.makedirs(fixture_dir, exist_ok=True)
    with open(os.path.join(fixture_dir, 'pickup.json'), 'w') as fp:
        json.dump(pose_pickup_global, fp)
    fixture_mesh.export(os.path.join(fixture_dir, 'fixture.obj'))
    with open(os.path.join(fixture_dir, 'fixture.png'), 'wb') as fp:
        fp.write(scene.save_image(visible=False))
    
    stats_path = os.path.join(log_dir, 'stats.json')
    with open(stats_path, 'r') as fp:
        stats = json.load(fp)
    stats['fixture_gen'] = {'time': round(time() - t_start, 2)}
    with open(stats_path, 'w') as fp:
        json.dump(stats, fp)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--assembly-dir', type=str, required=True)
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--optimized', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--render', default=False, action='store_true')
    args = parser.parse_args()

    run_fixture_gen(args.assembly_dir, args.log_dir, args.optimized, args.seed, render=args.render)
