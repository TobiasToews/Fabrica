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
import json

from assets.transform import get_transform_from_path
from planning.robot.util_grasp import compute_antipodal_pairs, generate_gripper_states, get_antipodal_aligned_grasp, sample_points_with_normal_alignment, get_grasp_info_from_gripper_state, get_reverse_grasp
from planning.robot.util_arm import get_arm_chain, get_ik_target_orientation, get_gripper_pos_quat_from_arm_q, get_ft_pos_from_gripper_pos_quat
from planning.robot.workcell import get_dual_arm_box
from planning.robot.geometry import load_arm_meshes, transform_gripper_meshes, transform_arm_meshes, get_arm_meshes_transforms, get_gripper_meshes_transforms, get_buffered_arm_meshes, \
    get_gripper_base_name, get_gripper_open_ratio, get_gripper_finger_names, get_gripper_knuckle_names
from planning.config import RETRACT_OPEN_RATIO, CHECK_GRIPPERS_INTERLOCK
from planning.run_grasp_gen import GraspGenerator, Grasp
from utils.parallel import parallel_execute


class GraspArmGenerator(GraspGenerator):

    def __init__(self, asset_folder, assembly_dir, preced_graph, gripper_type=None, arm_type=None, has_ft_sensor=None, seed=0, n_surface_pt=100, n_angle=10, antipodal_thres=0.95, ik_optimizer=None, ik_regularization=None, offset_delta=0.0, reduced_limit=0.0, flat=False):
        GraspGenerator.__init__(self, asset_folder, assembly_dir, preced_graph, gripper_type=gripper_type, arm_type=arm_type, has_ft_sensor=has_ft_sensor['move'] or has_ft_sensor['hold'], seed=seed, n_surface_pt=n_surface_pt, n_angle=n_angle, antipodal_thres=antipodal_thres, offset_delta=offset_delta)

        # flat: grasps for the fixture-free pipeline (planning/flat_grasp.py). The contact is evaluated on a grid of the finger
        # pads, and the move grasps that are kept can pick their part lying flat on the table (generate_grasps_flat).
        self.flat = flat
        self.pads = None
        self.part_faces = {} # part id -> PartFaces of the part at its final pose
        if flat:
            from planning.flat_grasp import finger_pads
            self.pads = finger_pads(self.gripper_type, self.gripper_meshes_visual)
            # the original pipeline keeps gripper and arm out of a box where its fixture stands (run_grasp_gen.py: 30 x 50 x 100 cm,
            # from y = get_fixture_min_y on). Without a fixture nothing stands there: the parts lie where the layout puts them,
            # and the assembly may stand in that area (an assembly placed there got no grasp at all, position test of 2026-11-12)
            self.ground_col_manager.remove_object('fixture')

        # arm
        self.arm_type = arm_type
        self.arm_meshes = load_arm_meshes(arm_type, asset_folder)
        self.arm_meshes_buffered = get_buffered_arm_meshes(self.arm_meshes)
        self.arm_col_manager_buffered = trimesh.collision.CollisionManager()
        for name, mesh in self.arm_meshes_buffered.items():
            self.arm_col_manager_buffered.add_object(name, mesh)
        self.ik_optimizer = ik_optimizer
        self.ik_regularization = ik_regularization
        self.has_ft_sensor = has_ft_sensor

        self.col_manager_move_buffered, self.col_manager_hold_buffered = trimesh.collision.CollisionManager(), trimesh.collision.CollisionManager()
        self.col_manager_hold = trimesh.collision.CollisionManager()  # unbuffered hold manager
        for name, mesh in self.gripper_meshes_buffered.items():
            self.col_manager_move_buffered.add_object(name, mesh)
            self.col_manager_move_buffered.add_object(name + '_open', mesh) # gripper fingers opened
            self.col_manager_hold_buffered.add_object(name, mesh)
            self.col_manager_hold_buffered.add_object(name + '_open', mesh) # gripper fingers opened
        for name, mesh in self.gripper_meshes.items():  # unbuffered gripper meshes for hold
            self.col_manager_hold.add_object(name, mesh)
            self.col_manager_hold.add_object(name + '_open', mesh)
        for name, mesh in self.arm_meshes_buffered.items():
            self.col_manager_move_buffered.add_object(name, mesh)
            self.col_manager_hold_buffered.add_object(name, mesh)
        for name, mesh in self.arm_meshes.items():  # unbuffered arm meshes for hold
            self.col_manager_hold.add_object(name, mesh)

        self.arm_chains = {'move': get_arm_chain(arm_type, 'move', reduced_limit=reduced_limit), 'hold': get_arm_chain(arm_type, 'hold', reduced_limit=reduced_limit)}

        # gripper knuckle
        self.gripper_knuckle_names = get_gripper_knuckle_names(self.gripper_type)
        self.gripper_knuckle_meshes = {name: self.gripper_meshes[name] for name in self.gripper_knuckle_names}

        # safety box
        self.box_col_manager = {}
        for motion_type, (box_lower, box_upper) in zip(['move', 'hold'], get_dual_arm_box(arm_type)):
            self.box_col_manager[motion_type] = trimesh.collision.CollisionManager()
            box_inner_mesh = trimesh.creation.box(bounds=np.vstack([box_lower - 0.1, box_upper]))
            box_outer_mesh = trimesh.creation.box(bounds=np.vstack([box_lower - 1.0, box_upper + 1.0]))
            box_mesh = box_outer_mesh.difference(box_inner_mesh)
            self.box_col_manager[motion_type].add_object('box', box_mesh)

    def visualize_col_managers(self, col_managers, other_meshes=[]):
        meshes = {}
        meshes.update(self.arm_meshes_buffered)
        meshes.update(self.gripper_meshes_buffered)
        meshes.update({name + '_open': mesh for name, mesh in self.gripper_meshes_buffered.items()})
        meshes.update(self.part_meshes)
        meshes_viz = []
        for col_manager in col_managers:
            transforms = {}
            for name in col_manager._objs:
                o = col_manager._objs[name]["obj"]
                transform_fcl = o.getTransform()
                transforms[name] = np.eye(4)
                transforms[name][:3, 3] = transform_fcl.getTranslation()
                transforms[name][:3, :3] = transform_fcl.getRotation()
            meshes_viz_i = {name: meshes[name].copy().apply_transform(transforms[name]) for name in transforms}
            meshes_viz.extend(list(meshes_viz_i.values()))
        trimesh.Scene(meshes_viz + list(other_meshes)).show()

    def check_grasp_feasible(self, grasp, part_id, verbose=False):
        n_timestep = 3
        parts_after = self.G_preced.nodes[part_id]['parts_after']

        path = self.G_preced.nodes[part_id]['path']
        if path is not None:
            part_transforms = get_transform_from_path(path, n_sample=n_timestep)
        else:
            part_transforms = [np.eye(4)]

        # grasp = get_antipodal_aligned_grasp(self.gripper_type, grasp) # NOTE: seems unstable for control
        grasps = {'move': [grasp.copy() for _ in range(len(part_transforms))], 'hold': grasp.copy()}

        parts_in_collision_move = set()
        parts_in_collision_hold = {'move': set(), 'fix': set()}

        '''
        Check move grasps - fix parts and hold grasp - fix parts collision
        '''
        for timestep, part_transform in enumerate(part_transforms):
            part_rel_transform = part_transform @ np.linalg.inv(part_transforms[0])
            grasp_t = self.transform_grasp(grasp, part_rel_transform)
            if grasps['move'] is not None:
                grasps['move'][timestep] = grasp_t
            gripper_pos, gripper_quat, grasp_open_ratio = grasp_t.pos, grasp_t.quat, grasp_t.open_ratio
            ft_pos = get_ft_pos_from_gripper_pos_quat(self.gripper_type, gripper_pos, gripper_quat)

            if timestep == 0: # check collision for retract state
                retract_open_ratio = min(grasp_open_ratio + RETRACT_OPEN_RATIO, 1.0)
                open_ratios = [grasp_open_ratio, retract_open_ratio]
            else:
                open_ratios = [grasp_open_ratio]

            part_final_transforms = self.part_final_transforms.copy()
            part_final_transforms[part_id] = part_transform
            self.apply_transforms_to_col_manager(self.part_col_manager, part_final_transforms)

            for open_ratio in open_ratios:

                # gripper collision manager
                gripper_transforms = get_gripper_meshes_transforms(self.gripper_type, self.gripper_meshes, gripper_pos, gripper_quat, np.eye(4), min(open_ratio + 0.05, 1.0)) # NOTE: 0.05 for numerical stability in collision check
                self.apply_transforms_to_col_manager(self.gripper_col_manager, gripper_transforms)
                self.apply_transforms_to_col_manager(self.gripper_col_manager_buffered, gripper_transforms)

                # check gripper-ground collision
                if self.gripper_col_manager_buffered.in_collision_other(self.ground_col_manager):
                    if verbose: print('[check_grasp_feasible] gripper-ground collision')
                    if timestep == 0: # move and hold grasp both infeasible
                        return None
                    else: # move grasp infeasible
                        grasps['move'] = None
                        break
                
                # check gripper-part collision
                _, contact_data = self.gripper_col_manager.in_collision_other(self.part_col_manager, return_data=True)
                for cdata in contact_data:
                    if part_id in cdata.names:
                        if timestep == 0: # move and hold grasp both infeasible
                            return None
                        else: # move grasp infeasible
                            grasps['move'] = None
                            break
                if grasps['move'] is None: break
                _, contact_data = self.gripper_col_manager_buffered.in_collision_other(self.part_col_manager, return_data=True)
                for cdata in contact_data:
                    for part_id_i in self.part_ids:
                        if part_id_i == part_id: continue
                        if part_id_i in cdata.names:
                            if part_id_i in parts_after:
                                if verbose: print('[check_grasp_feasible] gripper-after parts collision')
                                if timestep == 0:
                                    return None
                                else:
                                    grasps['move'] = None
                                    break
                            parts_in_collision_move.add(part_id_i)
                            if timestep == 0:
                                parts_in_collision_hold['fix'].add(part_id_i)
                                parts_in_collision_hold['move'].add(part_id_i)
                    if grasps['move'] is None: break
                if grasps['move'] is None: break

                # check gripper knuckle-part collision (part can only be in between gripper fingers)
                if timestep == 0 and open_ratio == open_ratios[0] and len(self.gripper_knuckle_names) > 0:
                    gripper_knuckle_meshes = {name: mesh.copy().apply_transform(gripper_transforms[name]) for name, mesh in self.gripper_knuckle_meshes.items()}
                    knuckle_mesh = trimesh.util.concatenate(list(gripper_knuckle_meshes.values())).convex_hull
                    _, contact_data = self.part_col_manager.in_collision_single(knuckle_mesh, return_data=True)
                    for cdata in contact_data:
                        for part_id_i in self.part_ids:
                            if part_id_i == part_id: continue
                            if part_id_i in cdata.names:
                                if part_id_i in parts_after:
                                    if verbose: print('[check_grasp_feasible] gripper-knuckle-after parts collision')
                                    return None
                                parts_in_collision_move.add(part_id_i)
                                parts_in_collision_hold['fix'].add(part_id_i)
                                parts_in_collision_hold['move'].add(part_id_i)

            for arm_type, arm_chain in self.arm_chains.items():
                if timestep > 0 and (arm_type == 'hold' or grasps['move'] is None): continue

                # check IK
                gripper_ori = get_ik_target_orientation(arm_chain, self.gripper_type, gripper_quat)
                if timestep == 0:
                    arm_q_default = arm_chain.active_to_full(arm_chain.rest_q)
                    optimizer = self.ik_optimizer
                    regularization_parameter = self.ik_regularization
                else:
                    arm_q_default = grasps['move'][timestep - 1].arm_q
                    optimizer = 'L-BFGS-B'
                    regularization_parameter = 0.0
                arm_q, ik_success = arm_chain.inverse_kinematics_above_ground(target_position=ft_pos if self.has_ft_sensor[arm_type] else gripper_pos, target_orientation=gripper_ori, orientation_mode='all', initial_position=arm_q_default, optimizer=optimizer, regularization_parameter=regularization_parameter)
                if not ik_success:
                    if verbose: print(f'[check_grasp_feasible] IK failed for {arm_type}')
                    grasps[arm_type] = None
                    continue
                debug_gripper_pos, debug_gripper_quat = get_gripper_pos_quat_from_arm_q(arm_chain, arm_q, self.gripper_type, has_ft_sensor=self.has_ft_sensor[arm_type])
                if not (np.allclose(gripper_pos, debug_gripper_pos, atol=1e-4) and np.allclose(gripper_quat, debug_gripper_quat, atol=1e-4)):
                    if verbose: print(f'[check_grasp_feasible] IK failed for {arm_type}')
                    grasps[arm_type] = None
                    continue
            
                # arm collision manager
                arm_transforms = get_arm_meshes_transforms(self.arm_meshes_buffered, arm_chain, arm_q)
                self.apply_transforms_to_col_manager(self.arm_col_manager_buffered, arm_transforms)

                # check arm-ground/box collision
                _, objs_in_collision_ground = self.arm_col_manager_buffered.in_collision_other(self.ground_col_manager, return_names=True)
                _, objs_in_collision_box = self.arm_col_manager_buffered.in_collision_other(self.box_col_manager[arm_type], return_names=True)
                objs_in_collision = list(objs_in_collision_ground) + list(objs_in_collision_box)
                for obj_pair in objs_in_collision:
                    if arm_chain.get_base_link_name() in obj_pair: continue
                    if verbose: print('[check_grasp_feasible] arm-ground collision')
                    grasps[arm_type] = None
                    break
                if grasps[arm_type] is None: continue

                # check arm-part collision
                _, objs_in_collision = self.arm_col_manager_buffered.in_collision_other(self.part_col_manager, return_names=True)
                for obj_pair in objs_in_collision:
                    if part_id in obj_pair:
                        if verbose: print('[check_grasp_feasible] arm-grasping part collision')
                        grasps[arm_type] = None
                        break
                    for part_id_i in self.part_ids:
                        if part_id_i in obj_pair:
                            if part_id_i in parts_after:
                                if verbose: print('[check_grasp_feasible] arm-after parts collision')
                                grasps[arm_type] = None
                                break
                            if arm_type == 'move':
                                parts_in_collision_move.add(part_id_i)
                            elif arm_type == 'hold':
                                parts_in_collision_hold['fix'].add(part_id_i)
                                parts_in_collision_hold['move'].add(part_id_i)
                    if grasps[arm_type] is None: break
                if grasps[arm_type] is None: continue
            
                # check arm self-collision
                _, collision_names = self.arm_col_manager_buffered.in_collision_internal(return_names=True)
                for (col_arm_name1, col_arm_name2) in collision_names:
                    if arm_chain.check_colliding_links(col_arm_name1, col_arm_name2):
                        if verbose: print('[check_grasp_feasible] arm self-collision')
                        grasps[arm_type] = None
                        break
                if grasps[arm_type] is None: continue
            
                # check arm-gripper collision
                _, collision_names = self.gripper_col_manager.in_collision_other(self.arm_col_manager_buffered, return_names=True)
                for (col_gripper_name, col_arm_name) in collision_names:
                    if col_gripper_name != 'ft_sensor' and col_arm_name != arm_chain.get_eef_link_name():
                        if verbose: print('[check_grasp_feasible] arm-gripper collision')
                        grasps[arm_type] = None
                        break
                if grasps[arm_type] is None: continue

                if arm_type == 'move':
                    grasps[arm_type][timestep].arm_pos = arm_chain.base_pos
                    grasps[arm_type][timestep].arm_euler = arm_chain.base_euler
                    grasps[arm_type][timestep].arm_q = arm_q
                elif arm_type == 'hold':
                    grasps[arm_type].arm_pos = arm_chain.base_pos
                    grasps[arm_type].arm_euler = arm_chain.base_euler
                    grasps[arm_type].arm_q = arm_q
                else:
                    raise NotImplementedError

            if all(grasp is None for grasp in grasps.values()):
                return None
            
        '''
        Check retract grasp reachability
        '''
        if grasps['move'] is not None:
            grasps['move'][0] = self.compute_retract_grasp(grasps['move'][0], 'move')
            if grasps['move'][0].arm_q_retract is None:
                grasps['move'] = None
            else:
                for i in range(1, len(grasps['move'])):
                    grasps['move'][i].pos_retract = grasps['move'][0].pos_retract
                    grasps['move'][i].arm_q_retract = grasps['move'][0].arm_q_retract
        if grasps['hold'] is not None:
            grasps['hold'] = self.compute_retract_grasp(grasps['hold'], 'hold')
            if grasps['hold'].arm_q_retract is None:
                grasps['hold'] = None
        if all(grasp is None for grasp in grasps.values()):
            return None
        
        '''
        Check hold grasps - move parts collision
        '''
        if grasps['hold'] is not None:

            # gripper collision manager
            gripper_pos, gripper_quat, open_ratio = grasps['hold'].pos, grasps['hold'].quat, grasps['hold'].open_ratio
            gripper_transforms = get_gripper_meshes_transforms(self.gripper_type, self.gripper_meshes, gripper_pos, gripper_quat, np.eye(4), min(open_ratio + 0.05, 1.0)) # NOTE: 0.05 for numerical stability in collision check
            self.apply_transforms_to_col_manager(self.gripper_col_manager, gripper_transforms)
            self.apply_transforms_to_col_manager(self.gripper_col_manager_buffered, gripper_transforms)

            # arm collision manager
            arm_chain = self.arm_chains['hold']
            arm_q = grasps['hold'].arm_q
            arm_transforms = get_arm_meshes_transforms(self.arm_meshes_buffered, arm_chain, arm_q)
            self.apply_transforms_to_col_manager(self.arm_col_manager_buffered, arm_transforms)

            for part_id_i in self.part_ids:
                path = self.G_preced.nodes[part_id_i]['path']
                if path is not None:
                    part_transforms = get_transform_from_path(path, n_sample=n_timestep)
                else:
                    continue

                for part_transform in part_transforms[1:]:

                    part_final_transforms = self.part_final_transforms.copy()
                    part_final_transforms[part_id_i] = part_transform
                    self.apply_transforms_to_col_manager(self.part_col_manager, part_final_transforms)

                    # check gripper-part collision
                    _, objs_in_collision = self.gripper_col_manager_buffered.in_collision_other(self.part_col_manager, return_names=True)
                    for obj_pair in objs_in_collision:
                        if part_id_i in obj_pair:
                            parts_in_collision_hold['move'].add(part_id_i)
                            break
                    if part_id_i in parts_in_collision_hold['move']:
                        break
                    
                    # check arm-part collision
                    _, objs_in_collision = self.arm_col_manager_buffered.in_collision_other(self.part_col_manager, return_names=True)
                    for obj_pair in objs_in_collision:
                        if part_id_i in obj_pair:
                            parts_in_collision_hold['move'].add(part_id_i)
                            break
                    if part_id_i in parts_in_collision_hold['move']:
                        break
        
        if all(grasp is None for grasp in grasps.values()):
            return None
        
        '''
        Assign results
        '''
        if grasps['move'] is not None:
            for grasp in grasps['move']:
                grasp.parts_in_collision_move = list(parts_in_collision_move)
        if grasps['hold'] is not None:
            grasps['hold'].parts_in_collision_hold['move'] = list(parts_in_collision_hold['move'])
            grasps['hold'].parts_in_collision_hold['fix'] = list(parts_in_collision_hold['fix'])
            
        return grasps
    
    def compute_contact_points(self, grasp, part_id, n_sample=1000):
        if self.pads is not None: # flat pipeline: fixed grid on the gripping pads instead of n_sample random points on the fingers
            from planning.flat_grasp import PartFaces, set_pad_contact
            if part_id not in self.part_faces:
                self.part_faces[part_id] = PartFaces(self.part_meshes[part_id].copy().apply_transform(self.part_final_transforms[part_id]))
            return set_pad_contact(grasp, self.pads, self.gripper_type, self.gripper_meshes_visual, self.part_faces[part_id])

        part_mesh = self.part_meshes[part_id].copy()
        part_mesh.apply_transform(self.part_final_transforms[part_id])

        gripper_pos, gripper_quat, open_ratio = grasp.pos, grasp.quat, grasp.open_ratio
        gripper_finger_meshes = {name: mesh.copy() for name, mesh in self.gripper_meshes_visual.items() if name in get_gripper_finger_names(self.gripper_type)}
        gripper_finger_meshes = transform_gripper_meshes(self.gripper_type, gripper_finger_meshes, gripper_pos, gripper_quat, np.eye(4), open_ratio - 0.01) # NOTE: 0.01 for slight penetration

        grasp_info = get_grasp_info_from_gripper_state(self.gripper_type, grasp.pos, grasp.quat, grasp.open_ratio)

        contact_points = []
        for name, mesh in gripper_finger_meshes.items():
            assert 'left' in name or 'right' in name
            l2r_direction = grasp_info['l2r_direction'] if 'left' in name else -grasp_info['l2r_direction']
            if self.arm_type == 'panda': l2r_direction = -l2r_direction # TODO: fix bug for panda!
            gripper_surface_points, _ = sample_points_with_normal_alignment(mesh, l2r_direction, n_sample)
            # trimesh.Scene([mesh, trimesh.PointCloud(gripper_surface_points, color=[255,0,0])]).show()
            # gripper_surface_points = mesh.sample(n_sample)
            contact_points.extend(gripper_surface_points[part_mesh.contains(gripper_surface_points)])

        grasp.contact_points = contact_points
        return grasp

    def compute_contact_points_chunk(self, grasps, part_id):
        return [self.compute_contact_points(grasp, part_id) for grasp in grasps]

    def assembly_scene(self):
        '''
        flat: the planner of the move arm and the parts at their assembled poses for assembly_moves_chunk, with the inflated
        versions of the part meshes that the planner tests against already computed. Built once per assembly, and in the parent
        process before the worker processes start: they get it by fork and share its memory. (Built in every worker instead,
        50 planners with the inflated meshes of duct needed more memory than the machine has, and the kernel killed a worker.)
        '''
        if getattr(self, '_assembly_scene', None) is None:
            from planning.robot.motion_plan_arm import ArmMotionPlanner
            from planning.run_motion_plan import make_move_planner
            ArmMotionPlanner.VISUALIZE = False
            final = {k: self.part_meshes[k].copy().apply_transform(self.part_final_transforms[k]) for k in self.part_meshes}
            planner = make_move_planner(get_arm_chain(self.arm_type, 'move'), self.gripper_type, self.has_ft_sensor['move'], get_dual_arm_box(self.arm_type)[0], final)
            for buffer in planner.part_buffers():
                planner.part_manager(list(final.values()), buffer) # computes and keeps the inflated mesh of every part
            self._assembly_scene = (planner, final)
        return self._assembly_scene

    def assembly_moves_chunk(self, moves, part_id):
        '''
        flat: for each move grasp of the chunk (the grasp along the insertion path) None, or which straight move at the assembly
        the real arm cannot do with it (run_motion_plan.assembly_moves_blocked), with the planner of assembly_scene. The random
        numbers (IK restarts) are set per grasp, so that the result does not depend on how the grasps are split into chunks.
        '''
        import gc
        import random
        from planning.run_motion_plan import assembly_moves_blocked
        planner, final = self.assembly_scene()
        before = [final[k] for k in sorted(self.G_preced.nodes[part_id]['parts_after'])] # the parts that are assembled before this one
        inserted = len(before) > 0 # nothing before it: the part can be the one that is held first, which is not inserted along its path
        blocked = []
        for move in moves:
            random.seed(move[0].grasp_id)
            np.random.seed(move[0].grasp_id)
            blocked.append(assembly_moves_blocked(planner, move, final[part_id], before, inserted=inserted))
            # the collision structures the planner builds for a grasp are only released by the garbage collector (they are held
            # in reference cycles). Left to itself it runs too late: with the two large parts of duct (28000 and 38000 faces)
            # a process grew by 60 MB per grasp, 280 -> 1744 MB after 24 grasps; collected after each grasp it stays constant
            gc.collect()
        return blocked

    def compute_retract_grasp(self, grasp, arm_type):
        grasp = super().compute_retract_grasp(grasp)
        gripper_pos_retract, gripper_ori_retract = grasp.pos_retract, get_ik_target_orientation(self.arm_chains[arm_type], self.gripper_type, grasp.quat)
        grasp.arm_q_retract, ik_success = self.arm_chains[arm_type].inverse_kinematics(target_position=gripper_pos_retract, target_orientation=gripper_ori_retract, orientation_mode='all', initial_position=grasp.arm_q, optimizer=self.ik_optimizer, regularization_parameter=self.ik_regularization)
        if not ik_success: grasp.arm_q_retract = None
        return grasp
    
    def antipodal_candidates(self, part_id, part_mesh, sample_budget, grasp_id):
        '''
        Grasp candidates from antipodal point pairs on the part surface, as in generate_grasps; their ids count on from grasp_id.
        Every call draws new surface points (the random stream goes on), so a second call does not repeat the pairs of the first.
        '''
        grasps_cand = []
        antipodal_pairs = compute_antipodal_pairs(part_mesh, sample_budget=sample_budget, antipodal_thres=self.antipodal_thres, collision_meshes=[])
        for antipodal_points in antipodal_pairs:
            open_ratio = get_gripper_open_ratio(self.gripper_type, antipodal_points)
            if open_ratio is None or open_ratio > 0.95: continue
            gripper_pos_list, gripper_quat_list = generate_gripper_states(self.gripper_type, antipodal_points, open_ratio, self.n_angle, offset_delta=self.offset_delta)
            for gripper_pos, gripper_quat in zip(gripper_pos_list, gripper_quat_list):
                grasps_cand.append(Grasp(part_id, grasp_id, gripper_pos, gripper_quat, open_ratio))
                grasp_id += 1
        return grasps_cand

    def lying_candidates(self, part_id, part_mesh, flat_rots, sample_budget, grasp_id):
        '''
        Grasp candidates made for the part lying on the board in the resting poses flat_rots (flat_grasp.lying_grasp_candidates);
        their ids count on from grasp_id. Marked with source = 'lying' (the candidates of antipodal_candidates have no mark).
        '''
        from planning.config import FLAT_LYING_ANGLES
        from planning.flat_grasp import lying_grasp_candidates
        grasps_cand = []
        for pos, quat, open_ratio, _ in lying_grasp_candidates(part_mesh, flat_rots, self.gripper_type, self.gripper_meshes, sample_budget, FLAT_LYING_ANGLES, self.antipodal_thres):
            grasp = Grasp(part_id, grasp_id, pos, quat, open_ratio)
            grasp.source = 'lying'
            grasps_cand.append(grasp)
            grasp_id += 1
        return grasps_cand

    def generate_grasps_flat(self, part_id, max_n_grasp=None, n_proc=1, verbose=False):
        '''
        Grasps of a part for the flat pipeline: a move grasp must pick the part lying flat on the table and insert it, so the
        flat-pickup test and the other criteria of the grasp filter (flat_grasp.move_grasp_criteria) are applied BEFORE the number
        of grasps is cut to max_n_grasp, and the grasps that are kept are the best rated ones (flat_grasp.rate_flat_grasp), spread
        over the approach directions, instead of a random choice.
          1. the matching between the assembled and the lying pose, from both sides: candidates from n_surface_pt antipodal
             pairs at the assembled pose, as in generate_grasps (all are checked; hold grasps as before), and candidates made
             on the lying part in its stable resting poses (lying_candidates, FLAT_LYING_PT pairs), which are then checked at
             the assembled pose like the others. All of them are rated together.
          2. fewer than FLAT_MIN_GRASPS usable move grasps: FLAT_RESAMPLE_PT further pairs from new surface points; of those only
             the candidates that can pick the part lying flat are checked
          3. no usable move grasp in the stable resting poses: the less stable resting poses are admitted for this part
             (get_flat_rots with_fallback; the grasp filter decides the same from the grasps that are kept), and candidates
             are made on the part lying in those poses as well
        Hold grasps are chosen at random as before. The part that is held first has to be taken from the table too: it gets, in
        addition, the best rated hold grasps that can pick it lying flat (up to max_n_grasp / 2). And for every part that can
        be inserted while this part is held, hold grasps whose gripper leaves room for that insertion are kept if the random
        choice has fewer than FLAT_HOLD_ROOM of them (flat_grasp.holds_with_room).
        The results do not depend on the order in which the worker processes finish (they are sorted by the grasp id).
        '''
        from planning.config import FLAT_FALLBACK_POSES, FLAT_MIN_GRASPS, FLAT_RESAMPLE_PT, FLAT_HOLD_ROOM, FLAT_LYING_PT # read at the call: the values can be set per run
        from planning.flat_grasp import PartFaces, get_flat_rots, get_flat_pickup_candidates, move_grasp_criteria, rate_flat_grasp, \
            select_grasps, approach_direction, part_center_of_mass, insertion_score, insertion_ranks, holds_with_room

        part_mesh = self.part_meshes[part_id].copy()
        part_mesh.apply_transform(self.part_final_transforms[part_id])
        self.part_faces[part_id] = PartFaces(part_mesh) # before the worker processes start
        rots_stable = get_flat_rots(part_mesh)
        rots_all = get_flat_rots(part_mesh, with_fallback=True) if FLAT_FALLBACK_POSES else rots_stable
        arm_chain = self.arm_chains['move'] if self.arm_type == 'panda' else None
        first_held = len(self.G_preced.nodes[part_id]['parts_after']) == 0
        is_flat = lambda grasp, rots: len(get_flat_pickup_candidates(grasp, part_mesh, rots, self.gripper_type, self.gripper_meshes)) > 0

        def with_contact(grasps_cand): # a grasp whose pads do not touch the part is none
            chunks = [grasps_cand[i::n_proc] for i in range(max(n_proc, 1))] # one process per chunk: a contact takes a few ms, less than starting a process
            args = [(chunk, part_id) for chunk in chunks if len(chunk) > 0]
            grasps = [grasp for chunk in parallel_execute(self.compute_contact_points_chunk, args, num_proc=n_proc, show_progress=False) for grasp in chunk if len(grasp.contact_points) > 0]
            return sorted(grasps, key=lambda grasp: grasp.grasp_id)

        def feasible(grasps_cand): # reachable and collision free at the assembled pose, as a move and / or as a hold grasp
            args = [(grasp, part_id, False) for grasp in grasps_cand]
            results = [result for result in parallel_execute(self.check_grasp_feasible, args, num_proc=n_proc, show_progress=False) if result is not None]
            return sorted(results, key=lambda r: (r['move'][0] if r['move'] is not None else r['hold']).grasp_id)

        criteria = {} # grasp id -> criteria of the move grasp, with 'flat' for all resting poses and 'flat_stable' for the stable ones
        def usable(results, stable_only):
            moves = []
            for result in results:
                if result['move'] is None: continue
                grasp_id = result['move'][0].grasp_id
                if grasp_id not in criteria:
                    criteria[grasp_id] = move_grasp_criteria(result['move'], part_mesh, rots_all, self.gripper_type, self.gripper_meshes, arm_chain)
                    criteria[grasp_id]['flat_stable'] = criteria[grasp_id]['flat'] and is_flat(result['move'][0], rots_stable)
                c = criteria[grasp_id]
                if c['walls'] and c['conditioned'] and c['overlap'] and (c['flat_stable'] if stable_only else c['flat']):
                    moves.append(result['move'])
            # the straight moves of the real arm at the assembly (to the insertion start with the part, away after the release),
            # for the grasps that pass everything else: a planner routine per grasp, so it runs in the worker processes.
            # criteria[...]['assembly']: None, or which of the two moves is missing
            if arm_chain is not None:
                untested = [move for move in moves if 'assembly' not in criteria[move[0].grasp_id]]
                if len(untested) > 0: self.assembly_scene() # built here, in the parent: the worker processes share it
                # a planner routine with collision tests per grasp, which needs memory (duct: 200 MB per process): over all parts,
                # which are processed at the same time, no more processes than cores
                n_chunks = max(1, min(n_proc, (os.cpu_count() or 1) // max(len(self.part_meshes), 1)))
                chunks = [chunk for chunk in (untested[i::n_chunks] for i in range(n_chunks)) if len(chunk) > 0]
                for blocked, (chunk, _) in parallel_execute(self.assembly_moves_chunk, [(chunk, part_id) for chunk in chunks], num_proc=n_chunks, show_progress=False, return_args=True):
                    for move, why in zip(chunk, blocked): criteria[move[0].grasp_id]['assembly'] = why # the results come in the order the processes finish
                moves = [move for move in moves if criteria[move[0].grasp_id]['assembly'] is None]
            return moves

        # 1. the matching of the two poses a grasp has to serve. Candidates made at the assembled pose, as in generate_grasps
        #    (tested on the lying part by usable() below), and candidates made on the lying part in its stable resting poses
        #    (tested at the assembled pose by feasible(), like all others)
        grasps_cand = self.antipodal_candidates(part_id, part_mesh, self.n_surface_pt, 0)
        n_cand, n_cand_more = len(grasps_cand), 0
        grasps_lying = self.lying_candidates(part_id, part_mesh, rots_stable, FLAT_LYING_PT, n_cand) if FLAT_LYING_PT > 0 else []
        n_lying = len(grasps_lying)
        results = feasible(with_contact(grasps_cand)) + feasible([grasp for grasp in with_contact(grasps_lying) if is_flat(grasp, rots_stable)])

        # 2. more surface points for a part with (nearly) no usable move grasp
        if len(usable(results, stable_only=True)) < FLAT_MIN_GRASPS:
            grasps_more = self.antipodal_candidates(part_id, part_mesh, FLAT_RESAMPLE_PT, n_cand + n_lying)
            n_cand_more = len(grasps_more)
            results += feasible([grasp for grasp in with_contact(grasps_more) if is_flat(grasp, rots_all)])

        # 3. the lying poses of this part: the stable ones, or also the less stable ones if no grasp fits a stable one
        #    (then candidates are made for the part lying in those poses as well)
        moves = usable(results, stable_only=True)
        fallback = False
        if len(moves) == 0 and len(rots_all) > len(rots_stable):
            if FLAT_LYING_PT > 0:
                grasps_less = self.lying_candidates(part_id, part_mesh, [r for r in rots_all if r.get('fallback')], FLAT_LYING_PT, n_cand + n_lying + n_cand_more)
                n_lying += len(grasps_less)
                results += feasible([grasp for grasp in with_contact(grasps_less) if is_flat(grasp, rots_all)])
            moves = usable(results, stable_only=False)
            fallback = len(moves) > 0
        rots = rots_all if fallback else rots_stable

        # the move grasps that are kept: best rated, spread over the approach directions
        center_of_mass = part_center_of_mass(part_mesh)
        def rated(grasps, with_insertion):
            ranks = insertion_ranks([insertion_score(grasp, self.G_preced) for grasp in grasps]) if with_insertion else np.zeros(len(grasps))
            return [rate_flat_grasp(grasp, get_flat_pickup_candidates(grasp, part_mesh, rots, self.gripper_type, self.gripper_meshes)[0], center_of_mass, self.gripper_type, rank)['rating']
                    for grasp, rank in zip(grasps, ranks)]
        ratings = rated([move[0] for move in moves], with_insertion=True)
        for move, rating in zip(moves, ratings):
            for grasp in move: grasp.flat_rating = rating
        chosen = select_grasps(ratings, [approach_direction(move[0]) for move in moves], max_n_grasp)
        grasps_move = [moves[i] for i in chosen]

        # hold grasps: a random choice as in generate_grasps; the part held first also gets the best ones that can pick it lying flat;
        # and every part keeps hold grasps that leave room for each part that can be inserted while it is held (holds_with_room)
        holds = [result['hold'] for result in results if result['hold'] is not None]
        n_hold_flat, n_hold_room = 0, 0
        if max_n_grasp is None or len(holds) <= max_n_grasp:
            grasps_hold = holds
        else:
            grasps_hold = [holds[i] for i in np.random.choice(len(holds), max_n_grasp, replace=False)]
            if first_held:
                chosen_ids = {hold.grasp_id for hold in grasps_hold}
                flat_holds = [hold for hold in holds if hold.grasp_id not in chosen_ids and is_flat(hold, rots)]
                flat_ratings = rated(flat_holds, with_insertion=False)
                extra = [flat_holds[i] for i in select_grasps(flat_ratings, [approach_direction(hold) for hold in flat_holds], max_n_grasp // 2)]
                n_hold_flat = len(extra)
                grasps_hold = grasps_hold + extra
            # the parts that are not yet there when this one is (the others cannot be inserted while it is held)
            later_parts = sorted(p for p in self.G_preced.nodes if p != part_id and p not in self.G_preced.nodes[part_id]['parts_after'])
            room = holds_with_room(holds, grasps_hold, later_parts, FLAT_HOLD_ROOM)
            n_hold_room = len(room)
            grasps_hold = grasps_hold + room

        n_move = sum(result['move'] is not None for result in results)
        no_assembly = [c['assembly'] for c in criteria.values() if c.get('assembly') is not None] # why, for each grasp dropped for the moves at the assembly
        from_lying =lambda grasps: sum(getattr(grasp, 'source', None) == 'lying' for grasp in grasps)
        n_usable_lying, n_kept_lying = from_lying([move[0] for move in moves]), from_lying([move[0] for move in grasps_move])
        print(f'[generate_grasps] part {part_id}: {n_cand} candidates' + (f' + {n_lying} made for the lying part' if n_lying > 0 else '') +
              (f' + {n_cand_more} from {FLAT_RESAMPLE_PT} further surface point pairs' if n_cand_more > 0 else '') +
              f', {n_move} feasible move grasps, {len(moves)} usable lying flat' + (f' ({n_usable_lying} of them made for the lying part, {n_kept_lying} of those kept)' if n_lying > 0 else '') +
              (f', {len(no_assembly)} more dropped (no straight move of the arm to the insertion start or away after the release)' if len(no_assembly) > 0 else '') +
              (' in a less stable resting pose (no grasp fits a stable one)' if fallback else '') +
              f' -> kept {len(grasps_move)} move + {len(grasps_hold)} hold' + (f' ({n_hold_flat} of the hold grasps added to pick the part lying flat)' if first_held and n_hold_flat > 0 else '') +
              (f' ({n_hold_room} of the hold grasps added to leave room for the insertion of other parts)' if n_hold_room > 0 else ''), flush=True)
        if len(moves) == 0 and n_move > 0:
            failed = {name: sum(not c[name] for c in criteria.values()) for name in ['walls', 'conditioned', 'overlap', 'flat']}
            print(f'[generate_grasps] part {part_id}: no usable move grasp; of the {len(criteria)} feasible move grasps fail: walls {failed["walls"]}, singularity / joint limits {failed["conditioned"]}, '
                  f'finger overlap {failed["overlap"]}, flat pickup {failed["flat"]} (stable poses {len(rots_stable)}, with the less stable ones {len(rots_all)})' +
                  (f'; the {len(no_assembly)} that pass these fail at the assembly, e.g. {no_assembly[0]}' if len(no_assembly) > 0 else ''), flush=True)

        return {'move': grasps_move, 'hold': grasps_hold}

    def generate_grasps(self, part_id, max_n_grasp=None, n_proc=1, verbose=False):
        if self.flat:
            return self.generate_grasps_flat(part_id, max_n_grasp=max_n_grasp, n_proc=n_proc, verbose=verbose)

        grasps_cand = []
        grasp_id = 0

        part_mesh = self.part_meshes[part_id].copy()
        part_mesh.apply_transform(self.part_final_transforms[part_id])

        collision_meshes = []
        # for part_after in self.G_preced.nodes[part_id]['parts_after']: # NOTE: disabled due to slowing down
        #     collision_meshes.append(self.part_meshes[part_after].copy().apply_transform(self.part_final_transforms[part_after]))

        # compute antipodal points
        antipodal_pairs = compute_antipodal_pairs(part_mesh, sample_budget=self.n_surface_pt, antipodal_thres=self.antipodal_thres, collision_meshes=collision_meshes)
        for antipodal_points in antipodal_pairs:
            open_ratio = get_gripper_open_ratio(self.gripper_type, antipodal_points)
            if open_ratio is None or open_ratio > 0.95: continue

            # compute grasps
            for antipodal_points_i in [antipodal_points]:
            # for antipodal_points_i in [antipodal_points, antipodal_points[::-1]]: # NOTE: more grasps but slower
                gripper_pos_list, gripper_quat_list = generate_gripper_states(self.gripper_type, antipodal_points_i, open_ratio, self.n_angle, offset_delta=self.offset_delta)
                for gripper_pos, gripper_quat in zip(gripper_pos_list, gripper_quat_list):
                    grasp = Grasp(part_id, grasp_id, gripper_pos, gripper_quat, open_ratio)
                    grasp_id += 1
                    grasps_cand.append(grasp)

        # calculate gripper-part contact area
        grasps_cand_new = []
        args = [(grasp, part_id, self.n_surface_pt) for grasp in grasps_cand]
        for grasp in parallel_execute(self.compute_contact_points, args, num_proc=n_proc, show_progress=verbose, desc='contact area computation'):
            if len(grasp.contact_points) == 0:
                continue
            grasps_cand_new.append(grasp)
        grasps_cand = grasps_cand_new

        # check grasp feasibility
        grasps = []
        args = [(grasp, part_id, False if n_proc > 1 else verbose) for grasp in grasps_cand]
        for grasp in parallel_execute(self.check_grasp_feasible, args, num_proc=n_proc, show_progress=verbose, desc='grasp generation'):
            if grasp is not None:
                grasps.append(grasp)

        # transform from [{'move': grasp_move, 'hold': grasp_hold}, ...] to {'move': [grasp_move, ...], 'hold': [grasp_hold, ...]}
        grasps_new = {'move': [], 'hold': []}
        for grasp in grasps:
            if grasp['move'] is not None:
                grasps_new['move'].append(grasp['move'])
            if grasp['hold'] is not None:
                grasps_new['hold'].append(grasp['hold'])
        grasps = grasps_new

        # limit number of grasps
        if max_n_grasp is not None:
            random_move_indices = np.random.choice(len(grasps['move']), min(max_n_grasp, len(grasps['move'])), replace=False)
            grasps['move'] = [grasps['move'][i] for i in random_move_indices]
            grasps['hold'] = np.random.choice(grasps['hold'], min(max_n_grasp, len(grasps['hold'])), replace=False).tolist()

        if verbose:
            print(f'[generate_grasps] {len(grasps["move"])} move grasps and {len(grasps["hold"])} hold grasps generated for part {part_id}')
        
        return grasps
    
    def generate_grasps_all(self, max_n_grasp=None, n_proc=1, verbose=False):
        grasps = {part_id: [] for part_id in self.part_ids}
        if self.flat and self.arm_type == 'panda':
            self.assembly_scene() # before the parts are handed to their processes: all of them and their workers share it
        args = [(part_id, max_n_grasp, max(n_proc // len(self.part_ids), 1), False) for part_id in self.part_ids]
        for grasps_i, ret_arg in parallel_execute(self.generate_grasps, args, num_proc=min(n_proc, len(self.part_ids)), return_args=True, show_progress=verbose, desc='grasp generation'):
            part_id = ret_arg[0]
            grasps[part_id] = grasps_i
            # self.visualize_grasps([g[0] for g in grasps_i['move']] + grasps_i['hold'])
        
        if verbose:
            for part_id, grasps_i in grasps.items():
                print(f'[generate_grasps_all] {len(grasps_i["move"])} move grasps and {len(grasps_i["hold"])} hold grasps generated for part {part_id}')

        return grasps
    
    def check_grasp_id_pair_feasible_batch(self, grasp_move, grasps_hold, verbose=False):
        grasp_id_pairs = []
        part_move, part_hold = grasp_move[0].part_id, grasps_hold[0].part_id

        interlock_col_manager = trimesh.collision.CollisionManager()
        finger_meshes = {name: mesh.copy() for name, mesh in self.gripper_meshes.items() if name in get_gripper_finger_names(self.gripper_type)}

        transforms_move = []
        for i, grasp_move_i in enumerate(grasp_move):
            if part_hold in grasp_move_i.parts_in_collision_move: return (part_move, part_hold), []

            gripper_pos_move, gripper_quat_move, open_ratio_move = grasp_move_i.pos, grasp_move_i.quat, grasp_move_i.open_ratio
            gripper_transforms_move = get_gripper_meshes_transforms(self.gripper_type, self.gripper_meshes, gripper_pos_move, gripper_quat_move, np.eye(4), open_ratio_move)
            gripper_transforms_move_open = get_gripper_meshes_transforms(self.gripper_type, self.gripper_meshes, gripper_pos_move, gripper_quat_move, np.eye(4), min(open_ratio_move + RETRACT_OPEN_RATIO, 1.0))
            gripper_transforms_move_open = {k + '_open': v for k, v in gripper_transforms_move_open.items()}
            arm_transforms_move = get_arm_meshes_transforms(self.arm_meshes_buffered, self.arm_chains['move'], grasp_move_i.arm_q)
            transforms_move.append({**gripper_transforms_move, **gripper_transforms_move_open, **arm_transforms_move})

            if i == 0:
                finger_meshes_move = {name: mesh.copy().apply_transform(gripper_transforms_move[name]) for name, mesh in finger_meshes.items()}
                if CHECK_GRIPPERS_INTERLOCK:
                    interlock_col_manager.add_object('gripper_move', trimesh.util.concatenate(list(finger_meshes_move.values())).convex_hull)

        for i, grasp_hold in enumerate(grasps_hold):
            if part_move in grasp_hold.parts_in_collision_hold['move']: continue

            gripper_pos_hold, gripper_quat_hold, open_ratio_hold = grasp_hold.pos, grasp_hold.quat, grasp_hold.open_ratio
            gripper_transforms_hold = get_gripper_meshes_transforms(self.gripper_type, self.gripper_meshes, gripper_pos_hold, gripper_quat_hold, np.eye(4), open_ratio_hold)
            gripper_transforms_hold_open = get_gripper_meshes_transforms(self.gripper_type, self.gripper_meshes, gripper_pos_hold, gripper_quat_hold, np.eye(4), min(open_ratio_hold + RETRACT_OPEN_RATIO, 1.0))
            gripper_transforms_hold_open = {k + '_open': v for k, v in gripper_transforms_hold_open.items()}
            arm_transforms_hold = get_arm_meshes_transforms(self.arm_meshes, self.arm_chains['hold'], grasp_hold.arm_q)  # use unbuffered arm meshes
            self.apply_transforms_to_col_manager(self.col_manager_hold_buffered, {**gripper_transforms_hold, **gripper_transforms_hold_open, **get_arm_meshes_transforms(self.arm_meshes_buffered, self.arm_chains['hold'], grasp_hold.arm_q)})
            self.apply_transforms_to_col_manager(self.col_manager_hold, {**gripper_transforms_hold, **gripper_transforms_hold_open, **arm_transforms_hold})  # unbuffered

            # check move-hold collision (buffered move vs unbuffered hold for less conservative check)
            in_collision = False
            for transform_move in transforms_move:
                self.apply_transforms_to_col_manager(self.col_manager_move_buffered, transform_move)
                if self.col_manager_move_buffered.in_collision_other(self.col_manager_hold):
                    if verbose: print('[check_grasp_id_pair_feasible_batch] move-hold collision')
                    in_collision = True
                    break
            if in_collision: continue

            # check if gripper_move and gripper_hold form an interlock by checking convex hull collision
            if CHECK_GRIPPERS_INTERLOCK:
                if 'gripper_hold' in interlock_col_manager._objs: interlock_col_manager.remove_object('gripper_hold')
                finger_meshes_hold = {name: mesh.copy().apply_transform(gripper_transforms_hold[name]) for name, mesh in finger_meshes.items()}
                interlock_col_manager.add_object('gripper_hold', trimesh.util.concatenate(list(finger_meshes_hold.values())).convex_hull)
                if interlock_col_manager.in_collision_internal():
                    if verbose: print('[check_grasp_id_pair_feasible_batch] interlock')
                    continue

            grasp_id_pairs.append((grasp_move[0].grasp_id, grasp_hold.grasp_id))
        
        return (part_move, part_hold), grasp_id_pairs
    
    def filter_grasp_id_pairs_all(self, grasps_all, n_proc=1, verbose=False):
        part_ids = list(grasps_all.keys())
        grasp_id_pairs_all = {}
        args = []
        for part_move in part_ids:
            for part_hold in part_ids:
                if part_move in self.G_preced.nodes[part_hold]['parts_after']: continue
                grasp_id_pairs_all[(part_move, part_hold)] = []
                if len(grasps_all[part_hold]['hold']) > 0:
                    args.extend([(grasp_move, grasps_all[part_hold]['hold'], False if n_proc > 1 else verbose) for grasp_move in grasps_all[part_move]['move']])
        for (part_move, part_hold), grasp_id_pairs in parallel_execute(self.check_grasp_id_pair_feasible_batch, args, num_proc=n_proc, show_progress=verbose, desc='grasp pair filtering'):
            grasp_id_pairs_all[(part_move, part_hold)].extend(grasp_id_pairs)
        if verbose:
            for part_move, part_hold in grasp_id_pairs_all.keys():
                print(f'[filter_grasp_id_pairs_all] {part_move}-{part_hold}: {len(grasp_id_pairs_all[(part_move, part_hold)])} grasp pairs')
        return grasp_id_pairs_all
    

def run_grasp_arm_gen(assembly_dir, log_dir, gripper, arm, ft_sensor, seed, n_surface_pt, n_angle, antipodal_thres, ik_optimizer, ik_regularization, offset_delta, reduced_limit, max_n_grasp, num_proc, verbose, flat=False):
    asset_folder = os.path.join(project_base_dir, './assets')

    precedence_path = os.path.join(log_dir, 'precedence.pkl')
    if not os.path.exists(precedence_path):
        print(f'[run_grasp_arm_gen] {precedence_path} not found')
        return
    
    with open(precedence_path, 'rb') as fp:
        G_preced = pickle.load(fp)

    has_ft_sensor = {'move': False, 'hold': False}
    if ft_sensor in ['all', 'move']: has_ft_sensor['move'] = True
    if ft_sensor in ['all', 'hold']: has_ft_sensor['hold'] = True

    t_start = time()
    grasp_generator = GraspArmGenerator(asset_folder, assembly_dir, G_preced, gripper, arm, has_ft_sensor,
        seed, n_surface_pt, n_angle, antipodal_thres, ik_optimizer, ik_regularization, offset_delta, reduced_limit, flat=flat)

    grasps_all = grasp_generator.generate_grasps_all(max_n_grasp=max_n_grasp, n_proc=num_proc, verbose=verbose)
    grasp_id_pairs_all = grasp_generator.filter_grasp_id_pairs_all(grasps_all, n_proc=num_proc, verbose=verbose)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, 'grasps.pkl'), 'wb') as fp:
            pickle.dump(
                {
                    'gripper': gripper,
                    'arm': arm,
                    'ft_sensor': has_ft_sensor,
                    'grasps': grasps_all, 
                    'grasp_id_pairs': grasp_id_pairs_all,
                    'settings': {
                        'n_surface_pt': n_surface_pt,
                        'n_angle': n_angle,
                        'antipodal_thres': antipodal_thres,
                        'optimizer': ik_optimizer,
                        'offset_delta': offset_delta,
                        'max_n_grasp': max_n_grasp,
                        'flat': flat, # grasps chosen for the flat pipeline, contact points on the pad grid (generate_grasps_flat)
                    }
                }, fp
            )

        success = True
        success_joint = {part_id: False for part_id in grasps_all.keys()}
        with open(os.path.join(log_dir, 'grasp_stats.txt'), 'w') as fp:
            fp.write('--- grasp stats ---\n')
            for part_id, grasps in grasps_all.items():
                fp.write(f'part {part_id}: {len(grasps["move"])} move + {len(grasps["hold"])} hold\n')
                if len(grasps["move"]) == 0 and len(grasps["hold"]) == 0:
                    success = False
            fp.write('--- grasp pair stats ---\n')
            for (part_move, part_hold), grasp_id_pairs in grasp_id_pairs_all.items():
                fp.write(f'part pair {part_move}-{part_hold}: {len(grasp_id_pairs)}\n')
                if len(grasp_id_pairs) > 0:
                    success_joint[part_move] = True
                    success_joint[part_hold] = True
        success = success and all(list(success_joint.values()))
        
        stats_path = os.path.join(log_dir, 'stats.json')
        with open(stats_path, 'r') as fp:
            stats = json.load(fp)
        stats['grasp_gen'] = {'success': success, 'time': round(time() - t_start, 2)}  
        with open(stats_path, 'w') as fp:
            json.dump(stats, fp)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--assembly-dir', type=str, required=True, help='directory of assembly')
    parser.add_argument('--log-dir', type=str, required=True, help='directory to load precedence and save generated grasps')
    parser.add_argument('--gripper', type=str, default='panda', choices=['panda', 'robotiq-85', 'robotiq-140'], help='gripper type')
    parser.add_argument('--arm', type=str, default='panda')
    parser.add_argument('--ft-sensor', type=str, default='none', choices=['none', 'all', 'move', 'hold'], help='force torque sensor installed')
    parser.add_argument('--max-n-grasp', type=int, default=None, help='maximum number of grasps per part')
    parser.add_argument('--n-surface-pt', type=int, default=200, help='number of surface point samples for generating antipodal pairs')
    parser.add_argument('--n-angle', type=int, default=10, help='number of grasp angle samples')
    parser.add_argument('--antipodal-thres', type=float, default=0.95)
    parser.add_argument('--ik-optimizer', type=str, default='least_squares', help='IK optimizer')
    parser.add_argument('--ik-regularization', type=float, default=1.0, help='IK regularization')
    parser.add_argument('--offset-delta', type=float, default=0.0)
    parser.add_argument('--reduced-limit', type=float, default=0.1, help='reduced joint limit in percentage')
    parser.add_argument('--num-proc', type=int, default=1, help='number of processes')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--verbose', action='store_true', default=True, help='verbose')
    parser.add_argument('--flat', action='store_true', default=False, help='grasps for the fixture-free pipeline: move grasps that can pick their part lying flat, chosen by rating (run_planning_flat.py)')
    args = parser.parse_args()

    from planning.robot.workcell import use_log_workcell
    use_log_workcell(args.log_dir) # the place of the assembly this log is planned for
    run_grasp_arm_gen(args.assembly_dir, args.log_dir, args.gripper, args.arm, args.ft_sensor, args.seed, args.n_surface_pt, args.n_angle, args.antipodal_thres, args.ik_optimizer, args.ik_regularization, args.offset_delta, args.reduced_limit, args.max_n_grasp, args.num_proc, args.verbose, flat=args.flat)
