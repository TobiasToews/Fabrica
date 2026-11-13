import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(project_base_dir)

import copy
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from argparse import ArgumentParser

from assets.transform import get_transform_matrix_quat, get_transform_matrix_euler, get_pos_euler_from_transform_matrix
from planning.robot.geometry import get_gripper_basis_directions, get_gripper_grasp_base_offset


class Grasp:

    def __init__(self, part_id, grasp_id, pos, quat, open_ratio, arm_pos=None, arm_euler=None, arm_q=None, arm_q_init=None, arm_paths_pre=[], arm_paths_post=[]):
        self.part_id = part_id
        self.grasp_id = grasp_id
        self.pos = pos
        self.quat = quat
        self.open_ratio = open_ratio
        self.arm_pos = arm_pos
        self.arm_euler = arm_euler
        self.arm_q = arm_q
        self.arm_q_init = arm_q_init
        self.arm_paths_pre = arm_paths_pre # [(arm_path, in_grasp), ...]
        self.arm_paths_post = arm_paths_post # [(arm_path, in_grasp), ...]
        self.parts_in_collision = [] # for single arm
        self.parts_in_collision_move = [] # for move grasp in dual arm
        self.parts_in_collision_hold = {'move': [], 'fix': []} # for hold grasp in dual arm
        self.pos_retract = None
        self.arm_q_retract = None
        self.contact_points = None

    def copy(self):
        return copy.deepcopy(self)
    

def filter_sparse_pairs(pairs, min_distance):
    # Split the pairs into two sets of points
    points_1 = pairs[:, 0]  # First points of all pairs (p1)
    points_2 = pairs[:, 1]  # Second points of all pairs (p2)
    
    # Build two KD-Trees: one for each set of points
    tree_1 = cKDTree(points_1)
    tree_2 = cKDTree(points_2)
    
    # Initialize mask to keep selected pairs
    keep_mask = np.ones(len(pairs), dtype=bool)
    
    for i in range(len(pairs)):
        if not keep_mask[i]:
            continue  # Skip already filtered pairs
        
        # Query neighbors for the first and second points
        neighbors_1 = tree_1.query_ball_point(points_1[i], r=min_distance)
        neighbors_2 = tree_2.query_ball_point(points_2[i], r=min_distance)
        
        # Combine neighbors
        neighbors = set(neighbors_1).intersection(neighbors_2)
        
        # Mark close neighbors (excluding self) as filtered
        for j in neighbors:
            if i != j:
                keep_mask[j] = False
    
    # Return filtered pairs
    return pairs[keep_mask]


def compute_antipodal_pairs(mesh, sample_budget=200, antipodal_thres=0.95, clearance=1.0, sparsity=0.05, collision_meshes=[], ground_z=0.0, visualize=False, verbose=False, max_rounds=None):
    '''
    Compute pairs of antipodal points for a given mesh through ray casting and surface sampling
    max_rounds: give up after this many sampling rounds (None: sample until sample_budget pairs are found, as in the original;
    a mesh without such pairs then never returns). Returns an array (n, 2, 3), n = 0 if none was found.
    '''
    mesh_col_manager = trimesh.collision.CollisionManager()
    mesh_col_manager.add_object('mesh', mesh)
    for i, collision_mesh in enumerate(collision_meshes):
        mesh_col_manager.add_object(f'collision_mesh_{i}', collision_mesh)

    antipodal_pairs_all = []
    n_rounds = 0

    while len(antipodal_pairs_all) < sample_budget:
        if max_rounds is not None and n_rounds >= max_rounds:
            break
        n_rounds += 1

        # randomly sample surface points
        sample_points, sample_face_idx = mesh.sample(sample_budget, return_index=True)
        sample_normals = mesh.face_normals[sample_face_idx]

        # ray casting for computing antipodal points
        init_offset = 0.05 # move ray origins slightly inside the surface
        ray_caster = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        intersect_points, ray_idx, intersect_face_idx = ray_caster.intersects_location(sample_points - init_offset * sample_normals, -sample_normals)
        intersect_normals = mesh.face_normals[intersect_face_idx]
        antipodal_idx = np.einsum('ij,ij->i', intersect_normals, -sample_normals[ray_idx]) > antipodal_thres
        antipodal_pairs = np.stack([sample_points[ray_idx][antipodal_idx], intersect_points[antipodal_idx]], axis=1)
        # TODO: what if cannot find antipodal pairs?

        # check ground clearance
        feasible_idx = np.all(antipodal_pairs[:, :, 2] > ground_z + clearance, axis=1)
        antipodal_pairs = antipodal_pairs[feasible_idx]
        if len(antipodal_pairs) == 0: continue

        # check reachability clearance
        feasible_idx = []
        for idx, antipodal_pair in enumerate(antipodal_pairs):
            ray_direction = antipodal_pair[1] - antipodal_pair[0]
            ray_direction /= np.linalg.norm(ray_direction)
            sphere0 = trimesh.primitives.Sphere(radius=clearance, center=antipodal_pair[0] - (clearance + 0.01) * ray_direction)
            sphere1 = trimesh.primitives.Sphere(radius=clearance, center=antipodal_pair[1] + (clearance + 0.01) * ray_direction)
            if mesh_col_manager.in_collision_single(sphere0) or mesh_col_manager.in_collision_single(sphere1):
                continue
            feasible_idx.append(idx)
        antipodal_pairs = antipodal_pairs[feasible_idx]

        if len(antipodal_pairs_all) == 0:
            antipodal_pairs_all = antipodal_pairs
        else:
            antipodal_pairs_all = np.concatenate([antipodal_pairs_all, antipodal_pairs], axis=0)

    if len(antipodal_pairs_all) == 0: # only with max_rounds
        return np.zeros((0, 2, 3))
    antipodal_pairs = filter_sparse_pairs(antipodal_pairs_all, min_distance=sparsity)
    antipodal_pairs = sort_antipodal_pairs(antipodal_pairs, mesh)

    if visualize:
        pts = [antipodal_pairs[i][0] for i in range(len(antipodal_pairs))]
        pc = trimesh.PointCloud(pts, colors=[255, 0, 0, 255])
        trimesh.Scene([mesh, pc]).show()

    if verbose:
        print(f'Found {len(antipodal_pairs)} pairs of antipodal points')

    return antipodal_pairs


def sort_antipodal_pairs(antipodal_pairs, mesh):
    '''
    Sort antipodal pairs based on the distance from center of mesh to center of antipodal points
    '''
    if len(antipodal_pairs) == 0: return antipodal_pairs
    antipodal_center = np.mean(antipodal_pairs, axis=1)
    dist = np.linalg.norm(antipodal_center - mesh.centroid, axis=1)
    sorted_idx = np.argsort(dist)
    return antipodal_pairs[sorted_idx]


def reorder_antipodal_pairs(antipodal_pairs, arm_pos):
    '''
    Reorder points in each antipodal pair based on the distance to arm position for consistent gripper orientation
    '''
    assert len(arm_pos) == 3
    dist_l = np.linalg.norm(antipodal_pairs[:, 0] - arm_pos, axis=1)
    dist_r = np.linalg.norm(antipodal_pairs[:, 1] - arm_pos, axis=1)
    reorder_idx = (dist_l > dist_r)
    antipodal_pairs[reorder_idx] = np.stack([antipodal_pairs[reorder_idx, 1], antipodal_pairs[reorder_idx, 0]], axis=1)
    return antipodal_pairs


def get_gripper_pos_quat(gripper_type, grasp_center, base_direction, l2r_direction, open_ratio, offset_delta=0.0):
    offset = get_gripper_grasp_base_offset(gripper_type, open_ratio, delta=offset_delta)
    pos = grasp_center + base_direction * offset
    rotation = R.align_vectors([base_direction, l2r_direction], [*get_gripper_basis_directions(gripper_type)])[0]
    quat = rotation.as_quat()[[3, 0, 1, 2]]
    return pos, quat


def get_grasp_info_from_gripper_state(gripper_type, pos, quat, open_ratio, offset_delta=0.0):
    rotation = R.from_quat(quat[[1, 2, 3, 0]])
    basis_base_direction, basis_l2r_direction = get_gripper_basis_directions(gripper_type)
    base_direction = rotation.apply(basis_base_direction)
    base_direction /= np.linalg.norm(base_direction)
    l2r_direction = rotation.apply(basis_l2r_direction)
    l2r_direction /= np.linalg.norm(l2r_direction)
    offset = get_gripper_grasp_base_offset(gripper_type, open_ratio, delta=offset_delta)
    grasp_center = pos - base_direction * offset
    return {
        'grasp_center': grasp_center, 'base_direction': base_direction, 'l2r_direction': l2r_direction
    }


def get_reverse_grasp(gripper_type, grasp):
    grasp_info = get_grasp_info_from_gripper_state(gripper_type, grasp.pos, grasp.quat, grasp.open_ratio)
    pos, quat_reverse = get_gripper_pos_quat(gripper_type, grasp_info['grasp_center'], grasp_info['base_direction'], -grasp_info['l2r_direction'], grasp.open_ratio)
    grasp_reverse = grasp.copy()
    grasp_reverse.pos = pos
    grasp_reverse.quat = quat_reverse
    return grasp_reverse


def get_antipodal_aligned_grasp(gripper_type, grasp):
    grasp_info = get_grasp_info_from_gripper_state(gripper_type, grasp.pos, grasp.quat, grasp.open_ratio)
    rest_l2r_direction = np.array([-1.0, 0, 0])
    if np.arccos(np.clip(np.dot(grasp_info['l2r_direction'], rest_l2r_direction), -1.0, 1.0)) > np.arccos(np.clip(np.dot(grasp_info['l2r_direction'], -rest_l2r_direction), -1.0, 1.0)):
        grasp = get_reverse_grasp(gripper_type, grasp)
    return grasp


def generate_gripper_states(gripper_type, antipodal_points, open_ratio, sample_budget=10, disassembly_direction=None, offset_delta=0.0):
    antipodal_points = np.array(antipodal_points, dtype=float)
    grasp_center = np.mean(antipodal_points, axis=0)
    l2r_direction = antipodal_points[1] - antipodal_points[0]
    l2r_direction /= np.linalg.norm(l2r_direction)

    # get one base direction
    if disassembly_direction is None:
        random_direction = np.random.rand(3)
        random_direction /= np.linalg.norm(random_direction)
        base_direction_basis = np.cross(l2r_direction, random_direction)
    else:
        disassembly_direction = np.array(disassembly_direction, dtype=float)
        disassembly_direction /= np.linalg.norm(disassembly_direction)
        if np.dot(disassembly_direction, l2r_direction) in [1, -1]: # disassembly direction is parallel to l2r direction
            random_direction = np.random.rand(3)
            random_direction /= np.linalg.norm(random_direction)
            base_direction_basis = np.cross(disassembly_direction, random_direction)
        else:
            base_direction_basis = np.cross(disassembly_direction, l2r_direction)
    base_direction_basis /= np.linalg.norm(base_direction_basis)
    
    base_direction_list = []
    # generate base directions by sampling angles
    for angle in np.linspace(0, 2 * np.pi, sample_budget + 1)[:-1]:
        base_rotation = R.from_rotvec(angle * l2r_direction)
        base_direction = base_rotation.apply(base_direction_basis)
        base_direction /= np.linalg.norm(base_direction)
        if np.dot(base_direction, np.array([0, 0, 1])) < 0: # ignore grasps pointing upward
            continue
        if disassembly_direction is not None and np.dot(base_direction, disassembly_direction) < 0:
            continue
        base_direction_list.append(base_direction)
    
    # sort base directions by angle with disassembly direction
    if disassembly_direction is not None:
        base_direction_list.sort(key=lambda x: np.dot(x, disassembly_direction), reverse=True)

    # generate state candidates from base directions
    pos_list, quat_list = [], []
    for base_direction in base_direction_list:
        pos, quat = get_gripper_pos_quat(gripper_type, grasp_center, base_direction, l2r_direction, open_ratio, offset_delta=offset_delta)
        pos_list.append(pos)
        quat_list.append(quat)
    return pos_list, quat_list


def get_gripper_path_from_part_path(part_path, gripper_pos, gripper_quat):
    T_g_0 = get_transform_matrix_quat(gripper_pos, gripper_quat) # get initial transform matrix of gripper
    T_p = [get_transform_matrix_euler(p[:3], p[3:]) for p in part_path] # get transform matrix of part path
    T_p_0 = T_p[0] # initial transform matrix of part
    T_p_g = np.linalg.inv(T_p_0) @ T_g_0 # get transform matrix from path to gripper
    T_g = [T_p_i @ T_p_g for T_p_i in T_p] # get transform matrix of gripper in global coordinate
    gripper_path = [get_pos_euler_from_transform_matrix(T_g_i) for T_g_i in T_g] # get gripper path in global coordinate
    return gripper_path


def sample_points_with_normal_alignment(mesh, direction, N, batch_size=1000):
    """
    Repeatedly sample points from the mesh surface until N points with 
    normals aligned to the given direction (angle < 90 degrees) are obtained.

    Parameters:
    - mesh: trimesh.Trimesh object, the mesh to sample from.
    - direction: numpy array (3,), the desired normal alignment direction.
    - N: int, the required number of aligned points.
    - batch_size: int, number of points to sample per iteration.

    Returns:
    - sampled_points: numpy array (N, 3), the sampled points.
    - sampled_normals: numpy array (N, 3), the corresponding normals.
    """
    # Normalize the input direction vector
    direction = direction / np.linalg.norm(direction)

    sampled_points = []
    sampled_normals = []

    while len(sampled_points) < N:
        # Sample points from the mesh surface
        candidate_points, face_indices = trimesh.sample.sample_surface(mesh, batch_size)

        # Get the corresponding normals
        candidate_normals = mesh.face_normals[face_indices]

        # Compute the dot product to check alignment
        dot_products = np.dot(candidate_normals, direction)

        # Select points where the normal aligns with the direction (dot > 0 means angle < 90 degrees)
        aligned_indices = dot_products > 0.5

        aligned_points = candidate_points[aligned_indices]
        aligned_normals = candidate_normals[aligned_indices]

        # Append aligned points and normals
        sampled_points.extend(aligned_points.tolist())
        sampled_normals.extend(aligned_normals.tolist())

    # Select exactly N points
    sampled_points = np.array(sampled_points[:N])
    sampled_normals = np.array(sampled_normals[:N])

    return sampled_points, sampled_normals


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--obj-path', type=str, required=True)
    parser.add_argument('--sample-budget', type=int, default=100)
    parser.add_argument('--antipodal-thres', type=float, default=0.95)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    mesh = trimesh.load_mesh(args.obj_path)
    compute_antipodal_pairs(mesh, args.sample_budget, args.antipodal_thres, visualize=args.visualize)
