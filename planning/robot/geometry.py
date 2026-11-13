import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(project_base_dir)

import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh
from assets.load import load_assembly
from assets.transform import get_scale_matrix, get_translate_matrix, get_revolute_matrix, get_transform_matrix_quat, get_pos_quat_from_pose


'''
Utils
'''

def apply_transform(matrix_base, matrix_transform): # apply transform to base matrix in-place
    return np.matmul(matrix_transform, matrix_base, out=matrix_base)


def save_meshes(meshes, folder, include_color=False):
    for name, mesh in meshes.items():
        mesh.export(os.path.join(folder, f'{name}.obj'), header=None, include_color=include_color)


def add_buffer_to_mesh(mesh, buffer):
    vertex_normals = mesh.vertex_normals
    new_vertices = mesh.vertices + buffer * vertex_normals
    return trimesh.Trimesh(new_vertices, mesh.faces, vertex_normals=vertex_normals)


def offset_mesh(mesh, buffer):
    '''
    Mesh grown by at least buffer everywhere: every face moves out by buffer along its own normal (add_buffer_to_mesh moves a
    vertex along its averaged normal, which moves the faces of a 90 degree edge only 0.71 x buffer and of a box corner 0.58 x).
    Edges and corners stick out further than buffer, so "no contact with the offset mesh" implies "distance >= buffer".
    '''
    mesh = mesh.copy()
    mesh.merge_vertices(merge_tex=True, merge_norm=True) # faces must share their vertices, or the offset faces come apart
    normals = mesh.face_normals
    vertices = mesh.vertices.copy()
    for v, faces in enumerate(mesh.vertex_faces):
        n = np.unique(np.round(normals[faces[faces >= 0]], 4), axis=0) # the different face directions that meet in this vertex
        if len(n) == 0: continue
        # shortest shift that moves every one of them out by buffer (least squares if they cannot all be met, e.g. on a round part)
        vertices[v] += np.linalg.lstsq(n, np.full(len(n), float(buffer)), rcond=None)[0]
    return trimesh.Trimesh(vertices, mesh.faces, process=False)


def get_buffered_meshes(meshes, buffer):
    if isinstance(meshes, trimesh.Trimesh):
        return add_buffer_to_mesh(meshes, buffer=buffer)
    elif isinstance(meshes, dict):
        meshes = {k: v.copy() for k, v in meshes.items()}
        for name, mesh in meshes.items():
            meshes[name] = add_buffer_to_mesh(mesh, buffer=buffer)
    elif isinstance(meshes, list):
        meshes = [mesh.copy() for mesh in meshes]
        for i, mesh in enumerate(meshes):
            meshes[i] = add_buffer_to_mesh(mesh, buffer=buffer)
    else:
        raise NotImplementedError
    return meshes


def get_buffered_gripper_meshes(gripper_type, gripper_meshes):
    finger_buffer = 0.25
    hand_knuckle_buffer = 1.0
    gripper_finger_meshes = {}
    gripper_hand_knuckle_meshes = {}
    for name, mesh in gripper_meshes.items():
        if name in get_gripper_finger_names(gripper_type):
            gripper_finger_meshes[name] = mesh.copy()
        else:
            gripper_hand_knuckle_meshes[name] = mesh.copy()
    gripper_finger_meshes = get_buffered_meshes(gripper_finger_meshes, buffer=finger_buffer)
    gripper_hand_knuckle_meshes = get_buffered_meshes(gripper_hand_knuckle_meshes, buffer=hand_knuckle_buffer)
    return {**gripper_finger_meshes, **gripper_hand_knuckle_meshes}


def get_buffered_arm_meshes(arm_meshes):
    return get_buffered_meshes(arm_meshes, buffer=1.0)


def get_combined_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        meshes = [scene_or_mesh.geometry[name] for name in scene_or_mesh.geometry]
        combined_mesh = trimesh.util.concatenate(meshes)
        return combined_mesh
    elif isinstance(scene_or_mesh, trimesh.Trimesh):
        return scene_or_mesh
    else:
        raise ValueError("Input is neither a trimesh.Trimesh nor a trimesh.Scene.")
    

def get_combined_meshes(meshes):
    if isinstance(meshes, dict):
        return {key: get_combined_mesh(mesh) for key, mesh in meshes.items()}
    elif isinstance(meshes, list):
        return [get_combined_mesh(mesh) for mesh in meshes]
    else:
        raise ValueError("Input is neither a dict nor a list.")


'''
Parts
'''

def load_part_meshes(assembly_dir, transform='none', rename=True, combined=True, **kwargs):
    assembly = load_assembly(assembly_dir, transform=transform, **kwargs)
    if rename:
        part_meshes = {f'part{part_id}': assembly[part_id]['mesh'] for part_id in assembly}
    else:
        part_meshes = {part_id: assembly[part_id]['mesh'] for part_id in assembly}
    if combined:
        part_meshes = get_combined_meshes(part_meshes)
    return part_meshes


def get_part_mesh_transform(pos, quat, pose=None):
    if pose is not None:
        pos, quat = get_pos_quat_from_pose(pos, quat, pose)
    return get_transform_matrix_quat(pos, quat)


def transform_part_mesh(mesh, pos, quat, pose=None):
    mesh = mesh.copy()
    mesh.apply_transform(get_part_mesh_transform(pos, quat, pose))
    return mesh


def get_part_meshes_transforms(meshes, pos_dict, quat_dict, pose=None):
    return {k: get_part_mesh_transform(pos_dict[k], quat_dict[k], pose) for k in meshes.keys()}


def transform_part_meshes(meshes, pos_dict, quat_dict, pose=None):
    transforms = get_part_meshes_transforms(meshes, pos_dict, quat_dict, pose)
    meshes = {k: v.copy() for k, v in meshes.items()}
    for name, mesh in meshes.items():
        mesh.apply_transform(transforms[name])
    return meshes


'''
Grippers
'''

def get_panda_grasp_base_offset():
    return 10.34 + 0.9 # NOTE: extra 0.9 for finger tip


def get_robotiq_85_grasp_base_offset(open_ratio):
    return 3.5 + np.sin(0.9208 + (1 - open_ratio) * 0.8757) * 5.715 + 6.93075 + 0.9 # NOTE: extra 0.9 for finger tip


def get_robotiq_140_grasp_base_offset(open_ratio):
    return 1.5 + 3.8 + np.sin(0.8680 + (1 - open_ratio) * 0.8757) * 10 + 5.4905 + 1.0 # NOTE: extra 1.0 for finger tip


def get_gripper_grasp_base_offset(gripper_type, open_ratio, delta=0.0):
    if gripper_type == 'panda':
        return get_panda_grasp_base_offset() + delta
    elif gripper_type == 'robotiq-85':
        return get_robotiq_85_grasp_base_offset(open_ratio) + delta
    elif gripper_type == 'robotiq-140':
        return get_robotiq_140_grasp_base_offset(open_ratio) + delta
    else:
        raise NotImplementedError


def get_panda_basis_directions():
    return [0, 0, -1], [0, 1, 0]


def get_robotiq_85_basis_directions():
    return [0, 0, -1], [-1, 0, 0]


def get_robotiq_140_basis_directions():
    return [0, 0, -1], [0, 1, 0]


def get_gripper_basis_directions(gripper_type):
    if gripper_type == 'panda':
        return get_panda_basis_directions()
    elif gripper_type == 'robotiq-85':
        return get_robotiq_85_basis_directions()
    elif gripper_type == 'robotiq-140':
        return get_robotiq_140_basis_directions()
    else:
        raise NotImplementedError


def get_panda_open_ratio(antipodal_points):
    antipodal_points = np.array(antipodal_points, dtype=float)
    antipodal_width = np.linalg.norm(antipodal_points[1] - antipodal_points[0])
    open_ratio = antipodal_width / 8
    if open_ratio > 1:
        return None
    else:
        return open_ratio


def get_robotiq_85_open_ratio(antipodal_points):
    antipodal_points = np.array(antipodal_points, dtype=float)
    antipodal_width = np.linalg.norm(antipodal_points[1] - antipodal_points[0])
    if antipodal_width > 3.92853109 * 2:
        return None
    else:
        return 1.0 - (np.arccos((antipodal_width / 2 + 0.8 - 1.27) / 5.715) - 0.9208) / 0.8757


def get_robotiq_140_open_ratio(antipodal_points):
    antipodal_points = np.array(antipodal_points, dtype=float)
    antipodal_width = np.linalg.norm(antipodal_points[1] - antipodal_points[0])
    if antipodal_width > 7.22376574 * 2:
        return None
    else:
        return 1.0 - (np.arccos((antipodal_width / 2 + 0.325 + 2.3 - 1.7901 - 1.27) / 10) - 0.8680) / 0.8757


def get_gripper_open_ratio(gripper_type, antipodal_points):
    if gripper_type == 'panda':
        return get_panda_open_ratio(antipodal_points)
    elif gripper_type == 'robotiq-85':
        return get_robotiq_85_open_ratio(antipodal_points)
    elif gripper_type == 'robotiq-140':
        return get_robotiq_140_open_ratio(antipodal_points)
    else:
        raise NotImplementedError
    

def get_panda_finger_states(open_ratio):
    finger_open_extent = 4 * open_ratio
    return {
        'panda_leftfinger': [finger_open_extent],
        'panda_rightfinger': [finger_open_extent],
    }


def get_robotiq_85_finger_states(open_ratio):
    finger_states = {}
    for side_i in ['left', 'right']:
        for side_j in ['outer', 'inner']:
            for link in ['knuckle', 'finger']:
                name = f'{side_i}_{side_j}_{link}'
                if name in ['left_outer_finger', 'right_outer_finger']: continue
                sign = -1 if name in ['left_inner_finger', 'right_inner_knuckle'] else 1
                finger_states[f'robotiq_{name}'] = [sign * (1 - open_ratio) * 0.8757]
    return finger_states


def get_robotiq_140_finger_states(open_ratio):
    finger_states = {}
    for side in ['left', 'right']:
        for link in ['outer_knuckle', 'inner_finger', 'inner_knuckle']:
            name = f'{side}_{link}'
            sign = 1 if link == 'inner_finger' or name == 'left_outer_knuckle' else -1
            finger_states[f'robotiq_{name}'] = [sign * (1 - open_ratio) * 0.8757]
    return finger_states


def get_gripper_finger_states(gripper_type, open_ratio, suffix=None):
    if gripper_type == 'panda':
        finger_states = get_panda_finger_states(open_ratio)
    elif gripper_type == 'robotiq-85':
        finger_states = get_robotiq_85_finger_states(open_ratio)
    elif gripper_type == 'robotiq-140':
        finger_states = get_robotiq_140_finger_states(open_ratio)
    else:
        raise NotImplementedError
    if suffix is not None:
        finger_states = {f'{key}_{suffix}': value for key, value in finger_states.items()}
    return finger_states


def get_gripper_base_name(gripper_type, suffix=None):
    if gripper_type == 'panda':
        name = 'panda_hand'
    elif gripper_type in ['robotiq-85', 'robotiq-140']:
        name = 'robotiq_base'
    else:
        raise NotImplementedError
    if suffix is not None:
        name = f'{name}_{suffix}'
    return name


def get_gripper_hand_names(gripper_type, suffix=None):
    if gripper_type == 'panda':
        names = ['panda_hand']
    elif gripper_type == 'robotiq-85':
        names = ['robotiq_base']
        for side_i in ['left', 'right']:
            for side_j in ['outer', 'inner']:
                for link in ['knuckle', 'finger']:
                    name = f'{side_i}_{side_j}_{link}'
                    if name not in ['left_inner_finger', 'right_inner_finger']:
                        names.append(f'robotiq_{name}')
    elif gripper_type == 'robotiq-140':
        names = ['robotiq_base']
        for side_i in ['left', 'right']:
            for side_j in ['outer', 'inner']:
                for link in ['knuckle', 'finger']:
                    name = f'{side_i}_{side_j}_{link}'
                    names.append(f'robotiq_{name}')
    else:
        raise NotImplementedError
    if suffix is not None:
        names = [f'{name}_{suffix}' for name in names]
    return names


def get_gripper_knuckle_names(gripper_type, suffix=None): # NOTE: for avoiding grasping parts inside knuckles
    if gripper_type == 'panda':
        names = []
    elif gripper_type == 'robotiq-85' or gripper_type == 'robotiq-140':
        names = []
        for side_i in ['left', 'right']:
            for side_j in ['outer', 'inner']:
                name = f'{side_i}_{side_j}_knuckle'
                names.append(f'robotiq_{name}')
    else:
        raise NotImplementedError
    if suffix is not None:
        names = [f'{name}_{suffix}' for name in names]
    return names


def get_gripper_finger_names(gripper_type, suffix=None):
    if gripper_type == 'panda':
        names = ['panda_leftfinger', 'panda_rightfinger']
    elif gripper_type == 'robotiq-85':
        names = ['robotiq_left_inner_finger', 'robotiq_right_inner_finger']
    elif gripper_type == 'robotiq-140':
        names = ['robotiq_left_pad', 'robotiq_right_pad']
    else:
        raise NotImplementedError
    if suffix is not None:
        names = [f'{name}_{suffix}' for name in names]
    return names


def load_panda_meshes(asset_folder, visual=False):
    meshes = {}
    dir_name = 'visual' if visual else 'collision'
    meshes['panda_hand'] = trimesh.load(os.path.join(asset_folder, 'panda', dir_name, 'hand.obj'))
    meshes['panda_leftfinger'] = trimesh.load(os.path.join(asset_folder, 'panda', dir_name, 'finger.obj'))
    meshes['panda_rightfinger'] = trimesh.load(os.path.join(asset_folder, 'panda', dir_name, 'finger.obj'))
    return meshes


def load_robotiq_85_meshes(asset_folder, visual=False):
    meshes = {}
    dir_name = 'visual' if visual else 'collision'
    postfix = 'fine' if visual else 'coarse'
    meshes['robotiq_base'] = trimesh.load(os.path.join(asset_folder, 'robotiq_85', dir_name, f'robotiq_base_{postfix}.obj'))
    for side_i in ['left', 'right']:
        for side_j in ['outer', 'inner']:
            for link in ['knuckle', 'finger']:
                meshes[f'robotiq_{side_i}_{side_j}_{link}'] = trimesh.load(os.path.join(asset_folder, 'robotiq_85', dir_name, f'{side_j}_{link}_{postfix}.obj'))
    return meshes


def load_robotiq_140_meshes(asset_folder, visual=False):
    meshes = {}
    dir_name = 'visual' if visual else 'collision'
    postfix = 'fine' if visual else 'coarse'
    meshes['robotiq_base'] = trimesh.load(os.path.join(asset_folder, 'robotiq_140', dir_name, f'robotiq_base_{postfix}.obj'))
    for side in ['left', 'right']:
        for link in ['outer_knuckle', 'outer_finger', 'inner_finger']:
            meshes[f'robotiq_{side}_{link}'] = trimesh.load(os.path.join(asset_folder, 'robotiq_140', dir_name, f'{link}_{postfix}.obj'))
        meshes[f'robotiq_{side}_pad'] = trimesh.load(os.path.join(asset_folder, 'robotiq_140', dir_name, f'pad_{postfix}.obj'))
        meshes[f'robotiq_{side}_inner_knuckle'] = trimesh.load(os.path.join(asset_folder, 'robotiq_140', dir_name, f'inner_knuckle_{postfix}.obj'))
    return meshes


def get_ft_sensor_spec():
    return {'radius': 3.75, 'height': 3.0} # NOTE: hardcoded values for robotiq sensor


def load_ft_sensor_mesh():
    spec = get_ft_sensor_spec()
    return trimesh.creation.cylinder(radius=spec['radius'], height=spec['height'])


def load_gripper_meshes(gripper_type, asset_folder, has_ft_sensor=False, visual=False, combined=True):
    if gripper_type == 'panda':
        gripper_meshes = load_panda_meshes(asset_folder, visual=visual)
    elif gripper_type == 'robotiq-85':
        gripper_meshes =  load_robotiq_85_meshes(asset_folder, visual=visual)
    elif gripper_type == 'robotiq-140':
        gripper_meshes =  load_robotiq_140_meshes(asset_folder, visual=visual)
    else:
        raise NotImplementedError
    if has_ft_sensor:
        gripper_meshes['ft_sensor'] = load_ft_sensor_mesh()
    if combined:
        gripper_meshes = get_combined_meshes(gripper_meshes)
    return gripper_meshes


def get_panda_meshes_transforms(meshes, open_ratio):
    transforms = {k: np.eye(4) for k in meshes.keys()}

    transforms['panda_leftfinger'] = get_translate_matrix([0, 4 * open_ratio, 5.84])
    transforms['panda_rightfinger'] = get_translate_matrix([0, -4 * open_ratio, 5.84]) @ get_scale_matrix([1, -1, 1])
    
    return transforms


def get_robotiq_85_meshes_transforms(meshes, open_ratio):
    transforms = {k: np.eye(4) for k in meshes.keys()}

    close_extent = 0.8757 * (1 - open_ratio)

    transforms['robotiq_left_outer_knuckle'] = get_translate_matrix([3.06011444260539, 0.0, 6.27920162695395]) @ get_revolute_matrix('Y', -close_extent)
    transforms['robotiq_left_outer_finger'] = transforms['robotiq_left_outer_knuckle'] @ get_translate_matrix([3.16910442266543, 0.0, -0.193396375724605])
    transforms['robotiq_left_inner_knuckle'] = get_translate_matrix([1.27000000001501, 0.0, 6.93074999999639]) @ get_revolute_matrix('Y', -close_extent)
    transforms['robotiq_left_inner_finger'] = transforms['robotiq_left_inner_knuckle'] @ get_translate_matrix([3.4585310861294003, 0.0, 4.5497019381797505]) @ get_revolute_matrix('Y', close_extent)

    transforms['robotiq_right_outer_knuckle'] = get_transform_matrix_quat([-3.06011444260539, 0.0, 6.27920162695395], [0, 0, 0, 1]) @ get_revolute_matrix('Y', -close_extent)
    transforms['robotiq_right_outer_finger'] = transforms['robotiq_right_outer_knuckle'] @ get_translate_matrix([3.16910442266543, 0.0, -0.193396375724605])
    transforms['robotiq_right_inner_knuckle'] = get_transform_matrix_quat([-1.27000000001501, 0.0, 6.93074999999639], [0, 0, 0, 1]) @ get_revolute_matrix('Y', -close_extent)
    transforms['robotiq_right_inner_finger'] = transforms['robotiq_right_inner_knuckle'] @ get_translate_matrix([3.4585310861294003, 0.0, 4.5497019381797505]) @ get_revolute_matrix('Y', close_extent)

    return transforms


def get_robotiq_140_meshes_transforms(meshes, open_ratio):
    transforms = {k: np.eye(4) for k in meshes.keys()}

    close_extent = 0.8757 * (1 - open_ratio)

    transforms['robotiq_left_outer_knuckle'] = get_transform_matrix_quat([0.0, -3.0601, 5.4905], [0.41040502, 0.91190335, 0.0, 0.0]) @ get_revolute_matrix('X', -close_extent)
    transforms['robotiq_left_outer_finger'] = transforms['robotiq_left_outer_knuckle'] @ get_translate_matrix([0.0, 1.821998610742, 2.60018192872234])
    transforms['robotiq_left_inner_finger'] = transforms['robotiq_left_outer_finger'] @ get_transform_matrix_quat([0.0, 8.17554015893473, -2.82203446692936], [0.93501321, -0.35461287, 0.0, 0.0]) @ get_revolute_matrix('X', close_extent)
    transforms['robotiq_left_pad'] = transforms['robotiq_left_inner_finger'] @ get_transform_matrix_quat([0.0, 3.8, -2.3], [0, 0, 0.70710678, 0.70710678])
    transforms['robotiq_left_inner_knuckle'] = get_transform_matrix_quat([0.0, -1.27, 6.142], [0.41040502, 0.91190335, 0.0, 0.0]) @ get_revolute_matrix('X', -close_extent)

    transforms['robotiq_right_outer_knuckle'] = get_transform_matrix_quat([0.0, 3.0601, 5.4905], [0.0, 0.0, 0.91190335, 0.41040502]) @ get_revolute_matrix('X', -close_extent)
    transforms['robotiq_right_outer_finger'] = transforms['robotiq_right_outer_knuckle'] @ get_translate_matrix([0.0, 1.821998610742, 2.60018192872234])
    transforms['robotiq_right_inner_finger'] = transforms['robotiq_right_outer_finger'] @ get_transform_matrix_quat([0.0, 8.17554015893473, -2.82203446692936], [0.93501321, -0.35461287, 0.0, 0.0]) @ get_revolute_matrix('X', close_extent)
    transforms['robotiq_right_pad'] = transforms['robotiq_right_inner_finger'] @ get_transform_matrix_quat([0.0, 3.8, -2.3], [0, 0, 0.70710678, 0.70710678])
    transforms['robotiq_right_inner_knuckle'] = get_transform_matrix_quat([0.0, 1.27, 6.142], [0.0, 0.0, -0.91190335, -0.41040502]) @ get_revolute_matrix('X', -close_extent)

    return transforms


def get_ft_sensor_mesh_transform(gripper_type):
    spec = get_ft_sensor_spec()

    # compute transformation matrix to align cylinder to gripper's base direction
    base_basis_direction, _ = get_gripper_basis_directions(gripper_type)
    base_basis_direction /= np.linalg.norm(base_basis_direction)
    default_axis = np.array([0, 0, 1])
    if np.allclose(base_basis_direction, default_axis):
        rotation_matrix = np.eye(3)
    else:
        rot_vec = np.cross(default_axis, base_basis_direction)
        angle = np.arccos(np.dot(default_axis, base_basis_direction))
        rotation_matrix = R.from_rotvec(rot_vec * angle).as_matrix()
    
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = (spec['height'] / 2) * base_basis_direction
    return transform


def get_gripper_meshes_transforms(gripper_type, meshes, pos, quat, pose, open_ratio):
    if gripper_type == 'panda':
        transforms = get_panda_meshes_transforms(meshes, open_ratio)
    elif gripper_type == 'robotiq-85':
        transforms = get_robotiq_85_meshes_transforms(meshes, open_ratio)
    elif gripper_type == 'robotiq-140':
        transforms = get_robotiq_140_meshes_transforms(meshes, open_ratio)
    else:
        raise NotImplementedError
    
    if 'ft_sensor' in meshes:
        transforms['ft_sensor'] = get_ft_sensor_mesh_transform(gripper_type)
    
    pos, quat = get_pos_quat_from_pose(pos, quat, pose)
    for name, transform in transforms.items():
        apply_transform(transform, get_transform_matrix_quat(pos, quat))
    return transforms


def transform_gripper_meshes(gripper_type, meshes, pos, quat, pose, open_ratio):
    transforms = get_gripper_meshes_transforms(gripper_type, meshes, pos, quat, pose, open_ratio)
    meshes = {k: v.copy() for k, v in meshes.items()}
    for name, mesh in meshes.items():
        mesh.apply_transform(transforms[name])
    return meshes


'''
Arms
'''

def load_arm_meshes(arm_type, asset_folder, visual=False, convex=True, combined=True): # NOTE: mapping between link names and mesh files
    meshes = {}
    if arm_type == 'xarm7':
        if visual:
            meshes['linkbase'] = trimesh.load(os.path.join(asset_folder, 'xarm7', 'visual', 'linkbase_smooth.obj'))
            for i in range(1, 8):
                meshes[f'link{i}'] = trimesh.load(os.path.join(asset_folder, 'xarm7', 'visual', f'link{i}_smooth.obj'))
        else:
            meshes['linkbase'] = trimesh.load(os.path.join(asset_folder, 'xarm7', 'collision', 'linkbase_vhacd.obj'))
            for i in range(1, 8):
                meshes[f'link{i}'] = trimesh.load(os.path.join(asset_folder, 'xarm7', 'collision', f'link{i}_vhacd.obj'))
    elif arm_type == 'panda':
        if visual:
            for i in range(0, 8):
                meshes[f'panda_link{i}'] = trimesh.load(os.path.join(asset_folder, 'panda', 'visual', f'link{i}.obj'))
        else:
            for i in range(0, 8):
                meshes[f'panda_link{i}'] = trimesh.load(os.path.join(asset_folder, 'panda', 'collision', f'link{i}.obj'))
    elif arm_type == 'ur5e':
        linknames = ['base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link']
        if visual:
            for name in linknames:
                filename = name.replace('_link', '').replace('_', '')
                meshes[name] = trimesh.load(os.path.join(asset_folder, 'ur5e', 'visual', f'{filename}.obj'))
        else:
            for name in linknames:
                filename = name.replace('_link', '').replace('_', '')
                meshes[name] = trimesh.load(os.path.join(asset_folder, 'ur5e', 'collision', f'{filename}.obj'))
    else:
        raise NotImplementedError
    if not visual and convex:
        meshes = {k: v.convex_hull for k, v in meshes.items()}
    if combined:
        meshes = get_combined_meshes(meshes)
    return meshes


def get_arm_meshes_transforms(meshes, chain, q):
    transforms = {k: np.eye(4) for k in meshes.keys()}
    matrices = chain.forward_kinematics(q, full_kinematics=True)
    for name, matrix in zip(transforms.keys(), matrices):
        transforms[name] = matrix
    return transforms


def transform_arm_meshes(meshes, chain, q):
    transforms = get_arm_meshes_transforms(meshes, chain, q)
    meshes = {k: v.copy() for k, v in meshes.items()}
    for name, mesh in meshes.items():
        mesh.apply_transform(transforms[name])
    return meshes
