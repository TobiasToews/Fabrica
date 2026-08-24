import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
import os
from argparse import ArgumentParser
from tqdm import tqdm
import trimesh
from time import time
from scipy.spatial.transform import Rotation as R
import pickle

from assets.load import load_assembly_all_transformed, load_pos_quat_dict
from assets.transform import get_pos_quat_from_pose, get_transform_from_path
from planning.robot.util_grasp import Grasp, compute_antipodal_pairs, generate_gripper_states
from planning.robot.geometry import load_gripper_meshes, transform_gripper_meshes, transform_part_mesh, transform_part_meshes, get_gripper_meshes_transforms, get_part_meshes_transforms, get_buffered_gripper_meshes, get_gripper_open_ratio, get_gripper_finger_names
from planning.robot.workcell import get_assembly_center, get_fixture_min_y
from planning.config import RETRACT_OPEN_RATIO, RETRACT_DELTA_NEAR
from planning.run_fixture_gen import MAX_BIN_SIZE_BLOCKING
from utils.seed import set_seed
from utils.parallel import parallel_execute


class GraspGenerator:

    def __init__(self, asset_folder, assembly_dir, preced_graph, gripper_type=None, arm_type=None, has_ft_sensor=False, seed=0, n_surface_pt=100, n_angle=10, antipodal_thres=0.95, offset_delta=0.0):
        set_seed(seed)

        # assembly
        self.assembly = load_assembly_all_transformed(assembly_dir)
        self.part_ids = list(self.assembly.keys())
        self.part_pos_dict, self.part_quat_dict = load_pos_quat_dict(assembly_dir, transform='final')
        self.part_pos_dict = {part_id: self.part_pos_dict[part_id] + get_assembly_center(arm_type) for part_id in self.part_ids}
        self.part_meshes = {part_id: self.assembly[part_id]['mesh'] for part_id in self.part_ids}
        self.part_meshes_convex = {part_id: trimesh.convex.convex_hull(mesh) for part_id, mesh in self.part_meshes.items()}
        self.part_col_manager = trimesh.collision.CollisionManager()
        for name, mesh in self.part_meshes.items():
            self.part_col_manager.add_object(name, mesh)
        self.part_final_transforms = get_part_meshes_transforms(self.part_meshes, self.part_pos_dict, self.part_quat_dict, np.eye(4))
        self.G_preced = preced_graph

        # gripper
        self.gripper_type = gripper_type
        self.gripper_meshes = load_gripper_meshes(self.gripper_type, asset_folder, has_ft_sensor=has_ft_sensor)
        self.gripper_meshes_visual = load_gripper_meshes(self.gripper_type, asset_folder, has_ft_sensor=has_ft_sensor, visual=True)
        self.gripper_col_manager = trimesh.collision.CollisionManager()
        for name, mesh in self.gripper_meshes.items():
            self.gripper_col_manager.add_object(name, mesh)
        self.gripper_meshes_buffered = get_buffered_gripper_meshes(self.gripper_type, self.gripper_meshes)
        self.gripper_col_manager_buffered = trimesh.collision.CollisionManager()
        for name, mesh in self.gripper_meshes_buffered.items():
            self.gripper_col_manager_buffered.add_object(name, mesh)
        self.gripper_finger_meshes = {name: mesh.copy() for name, mesh in self.gripper_meshes.items() if name in get_gripper_finger_names(self.gripper_type)}
        
        # ground and fixture
        self.ground_col_manager = trimesh.collision.CollisionManager()
        ground_mesh = trimesh.creation.box((1000, 1000, 0.4)) # NOTE: 0.2cm safety
        self.ground_col_manager.add_object('ground', ground_mesh)
        fixture_box = trimesh.creation.box((MAX_BIN_SIZE_BLOCKING[0], MAX_BIN_SIZE_BLOCKING[1], 100.0), transform=trimesh.transformations.translation_matrix([0, MAX_BIN_SIZE_BLOCKING[1] / 2 + get_fixture_min_y(arm_type), 50.0]))
        self.ground_col_manager.add_object('fixture', fixture_box)

        # sampling budget
        self.n_surface_pt = n_surface_pt
        self.n_angle = n_angle
        self.antipodal_thres = antipodal_thres
        self.offset_delta = offset_delta

    def get_final_mesh(self, part_id, pose=np.eye(4), convex=False):
        part_meshes = self.part_meshes_convex if convex else self.part_meshes
        return transform_part_mesh(part_meshes[part_id], self.part_pos_dict[part_id], self.part_quat_dict[part_id], pose)

    def get_final_meshes(self, part_ids, pose=np.eye(4), convex=False):
        part_meshes = self.part_meshes_convex if convex else self.part_meshes
        return list(transform_part_meshes({part_id: part_meshes[part_id] for part_id in part_ids}, self.part_pos_dict, self.part_quat_dict, pose).values())
    
    def transform_grasp(self, grasp, pose):
        grasp = grasp.copy()
        grasp.pos, grasp.quat = get_pos_quat_from_pose(grasp.pos, grasp.quat, pose)
        return grasp

    def apply_transforms_to_col_manager(self, col_manager, transforms):
        for name, transform in transforms.items():
            col_manager.set_transform(name, transform)

    def check_grasp_feasible(self, grasp, part_id, verbose=False):
        n_timestep = 2
        parts_after = self.G_preced.nodes[part_id]['parts_after']

        if self.G_preced.nodes[part_id]['path'] is not None:
            path = self.G_preced.nodes[part_id]['path']
            part_transforms = get_transform_from_path(path, n_sample=n_timestep)
        else:
            part_transforms = [np.eye(4)]

        parts_in_collision = set()
        for timestep, part_transform in enumerate(part_transforms):
            part_rel_transform = part_transform @ np.linalg.inv(part_transforms[0])
            grasp_t = self.transform_grasp(grasp, part_rel_transform)
            gripper_pos, gripper_quat, grasp_open_ratio = grasp_t.pos, grasp_t.quat, grasp_t.open_ratio

            if timestep == 0: # check collision for retract state
                retract_open_ratio = min(grasp_open_ratio + RETRACT_OPEN_RATIO, 1.0)
                open_ratios = [grasp_open_ratio, retract_open_ratio]
            else:
                open_ratios = [grasp_open_ratio]

            for open_ratio in open_ratios:
            
                # gripper collision manager
                gripper_transforms = get_gripper_meshes_transforms(self.gripper_type, self.gripper_meshes, gripper_pos, gripper_quat, np.eye(4), min(open_ratio + 0.05, 1.0)) # NOTE: 0.05 for numerical stability in collision check
                self.apply_transforms_to_col_manager(self.gripper_col_manager, gripper_transforms)
                self.apply_transforms_to_col_manager(self.gripper_col_manager_buffered, gripper_transforms)

                # check gripper-ground collision
                if self.gripper_col_manager_buffered.in_collision_other(self.ground_col_manager):
                    if verbose: print('[check_grasp_feasible] gripper-ground collision')
                    return None
                
                # check gripper-part collision
                part_final_transforms = self.part_final_transforms.copy()
                part_final_transforms[part_id] = part_transform
                self.apply_transforms_to_col_manager(self.part_col_manager, part_final_transforms)
                _, contact_data = self.gripper_col_manager.in_collision_other(self.part_col_manager, return_data=True)
                for cdata in contact_data:
                    if part_id in cdata.names:
                        if verbose: print('[check_grasp_feasible] gripper-grasping part collision')
                        return None
                _, contact_data = self.gripper_col_manager_buffered.in_collision_other(self.part_col_manager, return_data=True)
                for cdata in contact_data:
                    for part_id_i in self.part_ids:
                        if part_id_i == part_id: continue
                        if part_id_i in cdata.names:
                            if part_id_i in parts_after:
                                if verbose: print('[check_grasp_feasible] gripper-after parts collision')
                                return None
                            parts_in_collision.add(part_id_i)
        
        grasp.parts_in_collision = list(parts_in_collision)
        return grasp

    def compute_retract_grasp(self, grasp):
        gripper_b2f_dir = R.from_quat(grasp.quat[[1, 2, 3, 0]]).apply([0, 0, 1])
        gripper_pos, gripper_quat, open_ratio = grasp.pos, grasp.quat, grasp.open_ratio
        retract_open_ratio = min(open_ratio + RETRACT_OPEN_RATIO, 1.0)
        gripper_transforms = get_gripper_meshes_transforms(self.gripper_type, self.gripper_meshes, gripper_pos, gripper_quat, np.eye(4), retract_open_ratio)
        num_retract = 0
        while True:
            self.apply_transforms_to_col_manager(self.gripper_col_manager_buffered, gripper_transforms)
            if self.gripper_col_manager_buffered.in_collision_other(self.part_col_manager):
                for transform in gripper_transforms.values():
                    transform[:3, 3] -= RETRACT_DELTA_NEAR * gripper_b2f_dir
                num_retract += 1
            else:
                break
        grasp.pos_retract = gripper_pos - num_retract * RETRACT_DELTA_NEAR * gripper_b2f_dir
        return grasp
    
    def visualize_grasp(self, grasp):
        render_meshes = [self.part_meshes[grasp.part_id].copy().apply_transform(self.part_final_transforms[grasp.part_id])]
        gripper_meshes_i = transform_gripper_meshes(self.gripper_type, self.gripper_meshes, grasp.pos, grasp.quat, np.eye(4), grasp.open_ratio)
        render_meshes += list(gripper_meshes_i.values())
        trimesh.Scene(render_meshes).show()

    def visualize_grasps(self, grasps):
        render_meshes = [self.part_meshes[grasp.part_id].copy().apply_transform(self.part_final_transforms[grasp.part_id]) for grasp in grasps]
        for grasp in grasps:
            gripper_meshes_i = transform_gripper_meshes(self.gripper_type, self.gripper_meshes, grasp.pos, grasp.quat, np.eye(4), grasp.open_ratio)
            render_meshes += list(gripper_meshes_i.values())
        trimesh.Scene(render_meshes).show()

    def generate_grasps(self, part_id, max_n_grasp=None, n_proc=1, verbose=False):
        grasps_cand = []
        grasp_id = 0

        part_mesh = self.part_meshes[part_id].copy()
        part_mesh.apply_transform(self.part_final_transforms[part_id])

        # compute antipodal points
        antipodal_pairs = compute_antipodal_pairs(part_mesh, sample_budget=self.n_surface_pt, antipodal_thres=self.antipodal_thres)
        for antipodal_points in antipodal_pairs:
            open_ratio = get_gripper_open_ratio(self.gripper_type, antipodal_points)
            if open_ratio is None or open_ratio > 0.95: continue

            # compute grasps
            gripper_pos_list, gripper_quat_list = generate_gripper_states(self.gripper_type, antipodal_points, open_ratio, self.n_angle, offset_delta=self.offset_delta)
            for gripper_pos, gripper_quat in zip(gripper_pos_list, gripper_quat_list):
                grasp = Grasp(part_id, grasp_id, gripper_pos, gripper_quat, open_ratio)
                grasp_id += 1
                grasps_cand.append(grasp)

        # check grasp feasibility
        grasps = []
        args = [(grasp, part_id, False if n_proc > 1 else verbose) for grasp in grasps_cand]
        for grasp in parallel_execute(self.check_grasp_feasible, args, num_proc=n_proc, show_progress=verbose, desc='grasp generation'):
            if grasp is not None:
                grasps.append(grasp)

        # limit number of grasps
        if max_n_grasp is not None:
            grasps = np.random.choice(grasps, min(max_n_grasp, len(grasps)), replace=False).tolist()

        if verbose:
            print(f'[generate_grasps] {len(grasps)} grasps generated for part {part_id}')

        return grasps
    
    def generate_grasps_all(self, max_n_grasp=None, n_proc=1, verbose=False):
        
        grasps = {part_id: [] for part_id in self.part_ids}
        args = [(part_id, max_n_grasp, max(n_proc // len(self.part_ids), 1), False) for part_id in self.part_ids]
        for grasps_i, ret_arg in parallel_execute(self.generate_grasps, args, num_proc=min(n_proc, len(self.part_ids)), return_args=True, show_progress=verbose, desc='grasp generation'):
            part_id = ret_arg[0]
            grasps[part_id] = grasps_i
        
        if verbose:
            for part_id, grasps_i in grasps.items():
                print(f'[generate_grasps_all] {len(grasps_i)} grasps generated for part {part_id}')

        return grasps
    

def run_grasp_gen(assembly_dir, log_dir, gripper, ft_sensor, seed, n_surface_pt, n_angle, antipodal_thres, offset_delta, max_n_grasp, num_proc, verbose):
    asset_folder = os.path.join(project_base_dir, './assets')

    precedence_path = os.path.join(log_dir, 'precedence.pkl')
    if not os.path.exists(precedence_path):
        print(f'[run_grasp_gen] {precedence_path} not found')
        return
    
    with open(precedence_path, 'rb') as fp:
        G_preced = pickle.load(fp)

    grasp_generator = GraspGenerator(asset_folder, assembly_dir, G_preced, gripper, None, ft_sensor, seed, n_surface_pt, n_angle, antipodal_thres, offset_delta)

    grasps_all = grasp_generator.generate_grasps_all(max_n_grasp=max_n_grasp, n_proc=num_proc, verbose=verbose)
    
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, 'grasps.pkl'), 'wb') as fp:
            pickle.dump(
                {
                    'gripper': gripper,
                    'ft_sensor': ft_sensor,
                    'grasps': grasps_all, 
                    'settings': {
                        'n_surface_pt': n_surface_pt,
                        'n_angle': n_angle,
                        'antipodal_thres': antipodal_thres,
                        'offset_delta': offset_delta
                    }
                }, fp
            )
        with open(os.path.join(log_dir, 'grasp_stats.txt'), 'w') as f:
            for part_id, grasps in grasps_all.items():
                f.write(f'part {part_id}: {len(grasps)}\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--assembly-dir', type=str, required=True, help='directory of assembly')
    parser.add_argument('--log-dir', type=str, required=True, help='directory to load precedence and save generated grasps')
    parser.add_argument('--gripper', type=str, default='panda', choices=['panda', 'robotiq-85', 'robotiq-140'], help='gripper type')
    parser.add_argument('--ft-sensor', default=False, action='store_true', help='force torque sensor installed')
    parser.add_argument('--max-n-grasp', type=int, default=None, help='maximum number of grasps per part')
    parser.add_argument('--n-surface-pt', type=int, default=200, help='number of surface point samples for generating antipodal pairs')
    parser.add_argument('--n-angle', type=int, default=10, help='number of grasp angle samples')
    parser.add_argument('--antipodal-thres', type=float, default=0.95)
    parser.add_argument('--offset-delta', type=float, default=0.0)
    parser.add_argument('--num-proc', type=int, default=1, help='number of processes')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--verbose', action='store_true', default=True, help='verbose')
    args = parser.parse_args()

    run_grasp_gen(args.assembly_dir, args.log_dir, args.gripper, args.ft_sensor, args.seed, args.n_surface_pt, args.n_angle, args.antipodal_thres, args.offset_delta, args.max_n_grasp, args.num_proc, args.verbose)
