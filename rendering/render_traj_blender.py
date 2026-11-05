import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import shutil
import math
import numpy as np
import pickle
import re
import trimesh
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import bbox

from assets.save import sample_path
from planning.robot.geometry import load_arm_meshes, load_gripper_meshes, load_part_meshes
from rendering.render_utils import images_to_video, SuppressStdout, get_color


def get_mat_map():
    return {
        'limb': bbox.mat.PlasticMat(bbox.utils.Color.from_rgb(1.0, 1.0, 1.0, 1.0, value=5.0)),
        'joint': bbox.mat.PlasticMat(bbox.utils.Color.from_rgb(0.5, 0.5, 0.5, 1.0, value=2.0)),
        'hand': bbox.mat.PlasticMat(bbox.utils.Color.from_rgb(0.05, 0.05, 0.05, 1.0, value=2.0)),
        'fixture': bbox.mat.PlasticMat(bbox.utils.Color.from_rgb(0.93, 0.91, 0.85, 1.0, value=1.5)),
        'optical_board': bbox.mat.PlasticMat(bbox.utils.Color.from_rgb(0.2, 0.2, 0.2, 1.0, value=1.5)),
    }


def get_arm_shapes(arm_type, arm_meshes, mat_map):
    arm_shapes = {}
    for motion_type in ['move', 'hold']:
        for name, mesh in arm_meshes.items():
            shape_list = []

            if isinstance(mesh, trimesh.Trimesh):
                submeshes = [mesh]
            elif isinstance(mesh, trimesh.Scene):
                submeshes = [geometry for geometry in mesh.geometry.values()]
            else:
                raise Exception(f'Unknown mesh type {type(mesh)}')
            for component_id, submesh in enumerate(sorted(submeshes, key=lambda x: len(x.faces), reverse=True)):
                mat = None
                if arm_type == 'xarm7':
                    if re.match(r'linkbase', name):
                        if component_id == 7:
                            mat = mat_map['limb']
                        else:
                            mat = mat_map['joint']
                    elif re.match(r'link7', name):
                        mat = mat_map['joint']
                    else:
                        if component_id == 0:
                            mat = mat_map['limb']
                        else:
                            mat = mat_map['joint']
                elif arm_type == 'panda' or arm_type == 'ur5e':
                    color = submesh.visual.material.diffuse / 255
                    color = color**3
                    mat = bbox.mat.PlasticMat(bbox.utils.Color.from_rgb(*color, saturation=1.0, value=4.0))
                else:
                    raise NotImplementedError

                shape = bbox.shape.MeshShape.from_trimesh(submesh)
                shape.set_mat(mat)
                shape.set_normal_mode('vertex', use_auto_smooth=True)
                shape_list.append(shape)
            arm_shapes[f'{name}_{motion_type}'] = shape_list
    return arm_shapes


def get_gripper_shapes(gripper_type, gripper_meshes, has_ft_sensor, mat_map):
    gripper_shapes = {}
    for motion_type in ['move', 'hold']:
        for name, mesh in gripper_meshes.items():
            shape_list = []

            if re.match(f'ft_sensor', name) and not has_ft_sensor[motion_type]:
                continue
            
            if isinstance(mesh, trimesh.Trimesh):
                submeshes = [mesh]
            elif isinstance(mesh, trimesh.Scene):
                submeshes = [geometry for geometry in mesh.geometry.values()]
            else:
                raise Exception(f'Unknown mesh type {type(mesh)}')
            for component_id, submesh in enumerate(sorted(submeshes, key=lambda x: len(x.faces), reverse=True)):
                mat = None
                if re.match(f'ft_sensor', name):
                    mat = mat_map['hand']
                else:
                    if gripper_type == 'robotiq-85' or gripper_type == 'robotiq-140':
                        if re.match(r'robotiq_(left|right)_outer_knuckle', name):
                            mat = mat_map['joint']
                        else:
                            mat = mat_map['hand']
                    elif gripper_type == 'panda':
                        color = submesh.visual.material.diffuse / 255
                        color = color**3
                        mat = bbox.mat.PlasticMat(bbox.utils.Color.from_rgb(*color, saturation=1.0, value=4.0))
                    else:
                        raise NotImplementedError

                shape = bbox.shape.MeshShape.from_trimesh(submesh)
                shape.set_mat(mat)
                shape.set_normal_mode('vertex', use_auto_smooth=True)
                shape_list.append(shape)
            gripper_shapes[f'{name}_{motion_type}'] = shape_list
    return gripper_shapes


def get_part_shapes(part_meshes):
    part_colors = get_color(part_meshes.keys())
    part_shapes = {}
    for name, mesh in part_meshes.items():
        mat = bbox.mat.PlasticMat(bbox.utils.Color.from_rgb(*part_colors[name], value=3.0))
        shape = bbox.shape.MeshShape.from_trimesh(mesh)
        shape.set_mat(mat)
        shape.set_normal_mode('vertex', use_auto_smooth=True)
        shape_list = [shape]
        part_shapes[name] = shape_list
    return part_shapes


def get_environment_shapes(environment_meshes, mat_map):
    environment_shapes = {}
    for name, mesh in environment_meshes.items():
        mat = mat_map[name]
        shape = bbox.shape.MeshShape.from_trimesh(mesh)
        shape.set_mat(mat)
        shape.set_normal_mode('vertex', use_auto_smooth=True)
        environment_shapes[name] = [shape]
    return environment_shapes


def get_all_shapes(arm_type, gripper_type, has_ft_sensor, arm_meshes, gripper_meshes, part_meshes, environment_meshes):
    mat_map = get_mat_map()
    arm_shapes = get_arm_shapes(arm_type, arm_meshes, mat_map)
    gripper_shapes = get_gripper_shapes(gripper_type, gripper_meshes, has_ft_sensor, mat_map)
    part_shapes = get_part_shapes(part_meshes)
    environment_shapes = get_environment_shapes(environment_meshes, mat_map)
    all_shapes = {**arm_shapes, **gripper_shapes, **part_shapes, **environment_shapes}
    return all_shapes


def render_traj(assembly_dir, log_dir, record_path, interval, keep_img, verbose=False):

    grasps_path = os.path.join(log_dir, 'grasps.pkl')
    if not os.path.exists(grasps_path):
        print(f'[render_traj] {grasps_path} not found')
        return
    fixture_dir = os.path.join(log_dir, 'fixture')
    fixture_path = os.path.join(fixture_dir, 'fixture.obj')
    has_fixture = os.path.exists(fixture_path) # no fixture: parts lie flat on the board (run_pickup_gen.py)

    with open(grasps_path, 'rb') as fp:
        grasps = pickle.load(fp)
    arm_type, gripper_type, has_ft_sensor = grasps['arm'], grasps['gripper'], grasps['ft_sensor']

    asset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets')
    arm_meshes = load_arm_meshes(arm_type, asset_dir, visual=True, convex=False, combined=False)
    gripper_meshes = load_gripper_meshes(gripper_type, asset_dir, has_ft_sensor=True, visual=True, combined=False)
    part_meshes = load_part_meshes(assembly_dir, transform='none', combined=False)
    environment_meshes = {
        'optical_board': trimesh.load_mesh(os.path.join(asset_dir, 'optical_board.obj'))
    }
    if has_fixture:
        environment_meshes['fixture'] = trimesh.load_mesh(fixture_path)

    if verbose:
        print('[render_traj] meshes loaded')

    width = 1024
    height = 768
    resolution = 50
    # width = 512
    # height = 512
    # resolution = 100
    spp = 32
    scale = 0.1
    fps = 30
    time = 30

    blender = bbox.Blender(width=width, height=height, resolution_percentage=resolution, samples=spp, use_gpu=True, use_denoise=True, max_bounces=4)

    if verbose:
        print('[render_traj] blender initialized')

    ground_mat = bbox.mat.InvisibleMat(0.9)
    ground_shape = bbox.shape.PlaneShape(size=1000 * scale, location=(0, 0, -1 * scale), rotation=bbox.utils.lookat((0, 0, 0), (0, 0, 1), (1, 0, 0)))
    ground_shape.set_shadow()
    ground_shape.set_mat(ground_mat)
    blender.set_shadow_threshold(0.01)

    ambient_color = bbox.utils.Color.from_rgb(0.3, 0.3, 0.3, 1.0)
    blender.set_ambient_light(ambient_color)

    # Far view
    origin = np.array([5.0954, -5.99603, 5.59304])
    target = np.array([4.56643, -5.26702, 5.15853])
    camera = bbox.cam.PerspectiveCam(origin=origin, target=target, up=(0, 0, 1), fov=75)
    
    # Close view
    # origin = np.array([1.79691, -3.39051, 3.03326])
    # target = np.array([1.41078, -2.72671, 2.39264])
    # camera = bbox.cam.PerspectiveCam(origin=origin, target=target, up=(0, 0, 1), fov=60)

    if verbose:
        print('[render_traj] scene initialized')

    traj_path = os.path.join(log_dir, 'traj.npy')
    trajs = np.load(traj_path, allow_pickle=True)
    trajs = sample_path(trajs, n_frame=time * fps)

    record_tmp_dir = record_path + '_tmp'
    os.makedirs(record_tmp_dir, exist_ok=True)

    all_shapes = get_all_shapes(arm_type, gripper_type, has_ft_sensor, arm_meshes, gripper_meshes, part_meshes, environment_meshes)

    if verbose:
        print('[render_traj] shapes initialized')

    imgs = []
    for i, body_matrices in tqdm(enumerate(trajs), desc='Rendering', total=len(trajs)):

        if i % interval != 0: continue

        img_file = os.path.join(record_tmp_dir, f'{i:05d}.png')

        for shape_name, shape_list in all_shapes.items():

            body_matrix = body_matrices[shape_name]
            R = body_matrix[:3, :3]
            t = body_matrix[:3, 3] * scale
            for shape in shape_list:
                shape.set_rotation(Rotation.from_matrix(R).as_euler('xyz'))
                shape.set_location(t)
                shape.set_scale(scale)
                shape.show()

        with SuppressStdout():
            blender.render(img_file, cam=camera)
        imgs.append(img_file)

    images_to_video(imgs, record_path, fps=fps)
    if not keep_img:
        shutil.rmtree(record_tmp_dir)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--assembly-dir', type=str, required=True)
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--record-path', type=str, required=True)
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--keep-img', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    render_traj(args.assembly_dir, args.log_dir, args.record_path, args.interval, args.keep_img, args.verbose)
