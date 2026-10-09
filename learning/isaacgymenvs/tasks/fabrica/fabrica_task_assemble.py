"""Fabrica: class for specialist (part-specific) assembly policy training.

Inherits Fabrica environment class and Factory abstract task class (not enforced).

Can be executed with python train.py task=FabricaTaskAssemble.

NOTE: to train a policy for a certain assembly, must collect disassembly paths 
for this assembly before training since the disassembly paths will be used to 
calculate reward during RL training.

"""


import hydra
import numpy as np
import omegaconf
import os
import torch
import warp as wp
import json
import h5py
import pickle

from isaacgym import gymapi, gymtorch, torch_utils
from isaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
from isaacgymenvs.tasks.factory.factory_schema_config_task import (
    FactorySchemaConfigTask,
)
from isaacgymenvs.tasks.fabrica.fabrica_env import FabricaEnv
from isaacgymenvs.tasks.fabrica.fabrica_algo_utils import do_pos_path_transform, undo_pos_path_transform, do_deltapos_path_transform, undo_deltapos_path_transform, sample_random_points_on_trajectory, get_curriculum_difficulty
from isaacgymenvs.utils import torch_jit_utils


ATTACH_PLUG_TO_GRIPPER = True


class FabricaTaskAssemble(FabricaEnv, FactoryABCTask):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        """Initialize instance variables. Initialize task superclass."""

        self.cfg = cfg
        self._get_task_yaml_params()

        super().__init__(
            cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )

        self.arm_dof_pos_preassembly, self.arm_dof_pos_assembled, self.disassembly_path = self._load_assembly_plan_info()
        self.path_scale = torch.linalg.norm(self.disassembly_path[:, 0, :3] - self.disassembly_path[:, -1, :3], dim=-1) / 0.02

        self._acquire_task_tensors()
        self.parse_controller_spec()
        self.cfg_ctrl['default_dof_pos_tensor'] = self.arm_dof_pos_assembled[:, :7].clone().detach()

        if self.viewer != None:
            self._set_viewer_params()

    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name="factory_schema_config_task", node=FactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
        self.max_episode_length = (
            self.cfg_task.rl.max_episode_length
        )  # required instance var for VecTask

        ppo_path = os.path.join(
            "train/FabricaTaskAssemblePPO.yaml"
        )  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo["train"]  # strip superfluous nesting

    def _load_assembly_plan_info(self):
        """Load assembly plan info for plugs and sockets in each environment."""
        """
        plan_info = {
            (part_plug, part_socket): {
                'arm_q_plug': [arm_q_plug_preassembly, arm_q_plug_assembled],
                'arm_q_socket': arm_q_socket,
                'open_ratio_plug': open_ratio_plug,
                'open_ratio_socket': open_ratio_socket,
                'path': path,
            },
            ...
        }
        """
        assemblies = self.cfg_env.env.assemblies
        plan_infos = {}
        for assembly in assemblies:
            plan_info_path = os.path.join(os.getcwd(), self.cfg_task.env.data_dir, self.cfg_env.env.plan_info_dir, f'{assembly}.pkl')
            with open(plan_info_path, 'rb') as f:
                plan_info = pickle.load(f)
            plan_infos[assembly] = plan_info

        arm_dof_pos_preassembly, arm_dof_pos_assembled = [], []
        disassembly_path = []
        for i in range(self.num_envs):
            j = i % len(self.part_names['assembly'])
            assembly = self.part_names['assembly'][j]
            part_plug, part_socket = self.part_names['plug'][j], self.part_names['socket'][j]
            if (part_plug, part_socket) not in plan_infos[assembly]:
                raise ValueError(f'No assembly plan info found for {part_plug} and {part_socket} in assembly {assembly} (available: {plan_infos[assembly].keys()})')
            part_plan_info = plan_infos[assembly][(part_plug, part_socket)]
            gripper_dof_pos = [self.asset_info_franka_table.franka_gripper_width_max * part_plan_info['open_ratio_plug']] * 2
            arm_dof_pos_preassembly.append(np.concatenate([part_plan_info['arm_q_plug'][0], gripper_dof_pos]))
            arm_dof_pos_assembled.append(np.concatenate([part_plan_info['arm_q_plug'][1], gripper_dof_pos]))
            disassembly_path.append(part_plan_info['path'])
        disassembly_path = torch.as_tensor(np.array(disassembly_path), dtype=torch.float32).to(self.device)
        disassembly_path[:, :, :3] += torch.tensor(self.cfg_base.env.assembly_center, device=self.device) + torch.tensor([0.0, 0.0, self.cfg_base.env.table_height], device=self.device)

        return torch.as_tensor(arm_dof_pos_preassembly, dtype=torch.float32).to(self.device), \
            torch.as_tensor(arm_dof_pos_assembled, dtype=torch.float32).to(self.device), \
            disassembly_path

    def _acquire_task_tensors(self):
        """Acquire tensors."""
        
        # Grasp pose tensors
        self.palm_to_finger_center = torch.tensor([0.0, 0.0, self.cfg_task.env.palm_to_finger_dist], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.robot_to_gripper_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        self.socket_pos_observed = self.socket_pos.clone().detach()
        self.socket_pos_noise = torch.zeros((self.num_envs, 3), device=self.device)
        self.socket_pos_real = self.socket_pos.clone().detach()
        self.insertion_successes = torch.zeros((self.num_envs, ), dtype=torch.bool, device=self.device)
        self.deviations = torch.zeros((self.num_envs, ), dtype=torch.bool, device=self.device)
        self.identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.curriculum_difficulty = None

        self.ctrl_target_fingertip_centered_quat_true = torch.zeros(
            (self.num_envs, 4), device=self.device
        )

        self.prev_delta_pos_observed = torch.zeros((self.num_envs, 3), device=self.device)

    def _refresh_task_tensors(self):
        """Refresh tensors."""
        pass

    def pre_physics_step(self, actions):
        """Reset environments. Apply actions from policy as position/rotation targets, force/torque targets, and/or PD gains."""

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self._actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]
        
        self._actions = self._preprocess_actions(self._actions)
        self._apply_actions_as_ctrl_targets(actions=self._actions,
                                            ctrl_target_gripper_dof_pos=0.0,
                                            do_scale=True)

    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward."""

        self.progress_buf[:] += 1

        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()

        # Ensure the plug remains attached to the gripper
        if ATTACH_PLUG_TO_GRIPPER:
            self._attach_plug_to_gripper(torch.arange(self.num_envs, device=self.device))

        self.compute_observations()
        self.compute_reward()

    def compute_observations(self):
        """Compute observations."""

        delta_pos_observed = self.socket_pos_observed - self.plug_pos
        delta_pos_real = self.socket_pos_real - self.plug_pos
        pos_noise = self.socket_pos_noise
        delta_quat = torch_utils.quat_mul(self.plug_quat, torch_utils.quat_conjugate(self.socket_quat))
        last_delta_pos_observed = self.obs_buf[:, :3].clone().detach()
        last_delta_quat = self.obs_buf[:, 3:7].clone().detach()
        self.prev_delta_pos_observed = last_delta_pos_observed * 0.2 + self.prev_delta_pos_observed * 0.8

        if self.cfg_task.env.path_transform:
            delta_pos_observed = do_deltapos_path_transform(delta_pos_observed, self.disassembly_path[:, -1, :3], self.disassembly_path[:, 0, :3])
            delta_pos_real = do_deltapos_path_transform(delta_pos_real, self.disassembly_path[:, -1, :3], self.disassembly_path[:, 0, :3])
            pos_noise = do_deltapos_path_transform(pos_noise, self.disassembly_path[:, -1, :3], self.disassembly_path[:, 0, :3])

            delta_pos_observed[:, 2] /= self.path_scale # scale z-axis observation by length of disassembly path
            delta_pos_real[:, 2] /= self.path_scale # scale z-axis observation by length of disassembly path

        obs_tensors = [
            delta_pos_observed, # 3
            # delta_quat, # 4
            # last_delta_pos_observed, # 3
            # last_delta_quat, # 4
            # self.prev_delta_pos_observed, # 3
        ]

        self.obs_buf = torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations)

        state_tensors = [
            delta_pos_observed,  # 3
            delta_pos_real,  # 3
            # pos_noise,  # 3
            # delta_quat,  # 4
            # last_delta_pos_observed,  # 3
            # last_delta_quat,  # 4
            # self.prev_delta_pos_observed,  # 3
        ]

        self.states_buf = torch.cat(state_tensors, dim=-1)

        return self.obs_buf  # shape = (num_envs, num_observations)

    def compute_reward(self):
        """Detect successes and failures. Update reward and reset buffers."""

        self._update_rew_buf()
        self._update_reset_buf()

    def _update_rew_buf(self):
        """Compute reward at current timestep."""
        
        self.prev_rew_buf = self.rew_buf.clone()

        # Keypoint reward
        keypoint_delta_pos = self.plug_pos - self.socket_pos_real
        if self.cfg_task.env.path_transform:
            keypoint_delta_pos = do_deltapos_path_transform(keypoint_delta_pos, self.disassembly_path[:, -1, :3], self.disassembly_path[:, 0, :3])
            keypoint_delta_pos[:, 2] /= self.path_scale # scale z-axis observation by length of disassembly path

        keypoint_dist = torch.linalg.norm(keypoint_delta_pos, dim=-1)
        # keypoint_rwd = 1.0 / torch.clamp(keypoint_dist, min=0.0005) # rev-linear
        keypoint_rwd = -torch.clamp(keypoint_dist, max=0.03) # neg-linear
        # keypoint_rwd = -torch.clamp(keypoint_dist, max=0.03) ** 2 # neg-quad
        # keypoint_rwd = -torch.log(torch.clamp(keypoint_dist, min=0.0005)) # neg-log
        self.rew_buf[:] = self.cfg_task.rl.keypoint_reward_scale * keypoint_rwd

        # Insertion success
        is_plug_inserted_in_socket = torch.where(
            keypoint_dist < self.cfg_task.rl.close_error_thresh,
            torch.ones_like(self.progress_buf),
            torch.zeros_like(self.progress_buf),
        )
        self.insertion_successes = torch.logical_or(self.insertion_successes, is_plug_inserted_in_socket)

        # Deviation
        deviation_dist = torch.linalg.norm(self.plug_pos.unsqueeze(1) - self.disassembly_path[:, :, :3], dim=-1).min(dim=-1).values
        self.deviations = torch.logical_or(self.deviations, deviation_dist > self.cfg_task.rl.deviation_thresh)
        # print("Deviation:", self.deviations.shape)

        is_last_step = self.progress_buf[0] == self.max_episode_length - 1
        if is_last_step:
            self.extras["keypoint_reward"] = torch.mean(self.cfg_task.rl.keypoint_reward_scale * keypoint_rwd)
            self.extras["insertion_successes"] = torch.mean(self.insertion_successes.float())
            self.extras["deviation"] = torch.mean(self.deviations.float())
            print("Insertion Success:", self.extras["insertion_successes"].item(), "| Keypoint Reward:", self.extras["keypoint_reward"].item(), "| Keypoint Dist:", keypoint_dist.mean().item(), "| Deviation:", self.extras["deviation"].item())

    def _update_reset_buf(self):
        """Assign environments for reset if successful or failed."""
        self.reset_buf[:] = torch.where(self.progress_buf >= self.cfg_task.rl.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

    def reset_idx(self, env_ids):
        """Reset specified environments."""

        self._reset_franka(env_ids)

        if not ATTACH_PLUG_TO_GRIPPER:
            self.disable_gravity()

        self._reset_object(env_ids)

        if ATTACH_PLUG_TO_GRIPPER:
            self.plug_to_gripper_rel_pos = None
            self.plug_to_gripper_rel_quat = None
            self._attach_plug_to_gripper(env_ids)
        else:
            self.close_gripper(20)
            self.enable_gravity()

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        self.insertion_successes[env_ids] = torch.zeros((len(env_ids),), dtype=torch.bool, device=self.device)
        self.deviations[env_ids] = torch.zeros((len(env_ids),), dtype=torch.bool, device=self.device)

        self.prev_delta_pos_observed[env_ids] = torch.zeros((len(env_ids), 3), device=self.device)

    def _reset_franka(self, env_ids):
        """Reset DOF states and DOF targets of Franka."""

        # shape of dof_pos = (num_envs, num_dofs)
        # shape of dof_vel = (num_envs, num_dofs)

        # Initialize Franka 
        self.dof_pos[env_ids] = self.arm_dof_pos_preassembly[env_ids].clone().detach()
        self.dof_vel[env_ids] = 0.0  # shape = (num_envs, num_dofs)
        self.ctrl_target_dof_pos[env_ids] = self.arm_dof_pos_preassembly[env_ids].clone().detach()

        multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # Set DOF torque
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(torch.zeros_like(self.dof_torque)),
                                                gymtorch.unwrap_tensor(self.franka_actor_ids_sim),
                                                len(self.franka_actor_ids_sim))

        self.simulate_and_refresh()

    def _reset_object(self, env_ids):
        """Reset plug and socket."""

        self._reset_socket(env_ids)
        self._reset_plug(env_ids)

    def _reset_socket(self, env_ids):
        """Reset root state of socket."""

        # Randomize socket position 
        socket_pos_noise = 2 * (torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        self.socket_pos_noise[env_ids] = socket_pos_noise @ torch.diag(torch.tensor(self.cfg_task.randomize.socket_pos_noise, device=self.device))
        
        # Disable path-guided reset
        self.socket_pos_real[env_ids] = self.socket_pos_observed[env_ids] + self.socket_pos_noise[env_ids]

        # Enable path-guided reset
        # if self.curriculum_difficulty is None:
        #     self.curriculum_difficulty = 0.0
        # else:
        #     self.curriculum_difficulty = get_curriculum_difficulty(self.extras["insertion_successes"], self.curriculum_difficulty, difficulty_delta=0.1)
        # # self.curriculum_difficulty = 1.0
        # self.extras["curriculum_difficulty"] = self.curriculum_difficulty
        # sample_upper_bound = min(max(int((1.0 - self.curriculum_difficulty) * self.disassembly_path.shape[1]), 1), self.disassembly_path.shape[1])
        # self.socket_pos_observed[env_ids] = sample_random_points_on_trajectory(self.disassembly_path[env_ids, :sample_upper_bound, :3])
        # socket_plug_dist = torch.linalg.norm(self.socket_pos_observed[env_ids] - self.plug_pos[env_ids], dim=-1)
        # apply_noise_idx = socket_plug_dist > 0.005
        # self.socket_pos_real[env_ids] = self.socket_pos_observed[env_ids].clone()
        # self.socket_pos_real[env_ids[apply_noise_idx]] += self.socket_pos_noise[env_ids[apply_noise_idx]]
        
        self.root_pos[env_ids, self.socket_actor_id_env] = self.socket_pos_real[env_ids]

        # Set socket orientation to be upright
        self.root_quat[env_ids, self.socket_actor_id_env] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)

        # Set socket velocities to be zero
        self.root_linvel[env_ids, self.socket_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.socket_actor_id_env] = 0.0

        socket_actor_ids_sim = self.socket_actor_ids_sim[env_ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state),
                                                     gymtorch.unwrap_tensor(socket_actor_ids_sim),
                                                     len(socket_actor_ids_sim))
        self.simulate_and_refresh()

    def _reset_plug(self, env_ids):
        """Reset root state of plug."""

        # Set plug position and orientation
        self.root_pos[env_ids, self.plug_actor_id_env] = self.disassembly_path[env_ids, -1, :3]
        self.root_quat[env_ids, self.plug_actor_id_env] = self.disassembly_path[env_ids, -1, 3:]

        # Set plug velocities to be zero
        self.root_linvel[env_ids, self.plug_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.plug_actor_id_env] = 0.0

        plug_actor_ids_sim = self.plug_actor_ids_sim[env_ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state),
                                                     gymtorch.unwrap_tensor(plug_actor_ids_sim),
                                                     len(plug_actor_ids_sim))

        self.simulate_and_refresh()

    def _attach_plug_to_gripper(self, env_ids):
        """Attach plug to gripper by synchronizing their poses."""

        # Compute the relative pose of the plug to the gripper
        if self.plug_to_gripper_rel_pos is None or self.plug_to_gripper_rel_quat is None:
            self.plug_to_gripper_rel_quat, self.plug_to_gripper_rel_pos = torch_jit_utils.tf_inverse(self.fingertip_centered_quat, self.fingertip_centered_pos)
            self.plug_to_gripper_rel_quat, self.plug_to_gripper_rel_pos = torch_jit_utils.tf_combine(self.plug_to_gripper_rel_quat, self.plug_to_gripper_rel_pos, self.plug_quat, self.plug_pos)

        # Synchronize plug pose with gripper
        self.plug_quat[env_ids], self.plug_pos[env_ids] = torch_jit_utils.tf_combine(
            self.fingertip_centered_quat[env_ids], self.fingertip_centered_pos[env_ids], self.plug_to_gripper_rel_quat[env_ids], self.plug_to_gripper_rel_pos[env_ids]
        )

        multi_env_ids_int32 = self.plug_actor_ids_sim[env_ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )

    def _reset_buffers(self, env_ids):
        """Reset buffers. """

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    # TODO: Continue on a viewer that changes the camera to the environment that is failing (aka deviation gets too big)
    # def _set_viewer_params(self, cam_pos= gymapi.Vec3(-1.0, -1.0, 1.0), cam_target= gymapi.Vec3(0.0, 0.0, 0.5)):
    #     """Set viewer parameters."""


    #     nx = 2
    #     ny = 1
    #     n_vec = gymapi.Vec3(nx, ny, 0.0)

    #     cam_pos_standard = gymapi.Vec3(-1.0, -1.0, 1.0)
    #     cam_target_standard = gymapi.Vec3(0.0, 0.0, 0.5)

    #     # table_depth: 1.28  # depth of table
    #     # table_width: 0.91  # width of table
    #     # table_height: 1.04  # height of table


    #     cam_pos_table_dim = gymapi.Vec3(2.0, 1.5, 0.0)
    #     cam_pos_table_dim = gymapi.Vec3(1.28, 0.91, 0.0)
    #     cam_target_table_dim = gymapi.Vec3(2.0, 1.5, 0.0)

    #     cam_pos = cam_pos_standard + gymapi.Vec3(n_vec.x * cam_pos_table_dim.x, n_vec.y * cam_pos_table_dim.y, cam_pos_table_dim.z)
    #     #cam_target = cam_target_standard + gymapi.Vec3(1,0,0)
    #     # cam_target= cam_pos + gymapi.Vec3(n_vec.x * cam_target_table_dim.x, n_vec.y * cam_target_table_dim.y, cam_target_table_dim.z)
    #     self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)


    def _set_viewer_params(self):
        """Set viewer parameters."""

        cam_pos = gymapi.Vec3(-1.0, -1.0, 1.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _preprocess_actions(self, actions):
        """Preprocess actions from policy."""

        # Untransform actions from path-aligned coordinates
        if self.cfg_task.env.path_transform:
            actions[:, 2] *= self.path_scale # scale z-axis action by length of disassembly path
            actions = undo_deltapos_path_transform(actions, self.disassembly_path[:, -1, :3], self.disassembly_path[:, 0, :3])

        # Residual actions
        assert not (self.cfg_task.env.residual_action and self.cfg_task.env.openloop), "Cannot have both residual action and open-loop"
        if self.cfg_task.env.residual_action or self.cfg_task.env.openloop:
            residual_actions = self.disassembly_path[:, 0, :3] - self.plug_pos
            if self.cfg_task.env.path_transform:
                residual_actions_transformed = do_deltapos_path_transform(residual_actions, self.disassembly_path[:, -1, :3], self.disassembly_path[:, 0, :3])
                residual_actions_transformed[:, 2] /= self.path_scale # scale z-axis residual action by length of disassembly path
                residual_actions_norm = torch.linalg.norm(residual_actions_transformed, dim=-1, keepdim=True) # calculate norm of residual actions based on rescaled residual actions
            else:
                residual_actions_norm = torch.linalg.norm(residual_actions, dim=-1, keepdim=True)
            # residual_actions /= torch.clamp(residual_actions_norm, min=self.cfg_task.rl.close_error_thresh)
            residual_actions /= residual_actions_norm
            if self.cfg_task.env.residual_action:
                actions += residual_actions
            elif self.cfg_task.env.openloop:
                actions = residual_actions
            else:
                raise ValueError("Invalid action mode")

        # Open-loop random baseline
        # actions = torch.randn_like(actions, dtype=torch.float32, device=self.device) + residual_actions

        # Open-loop take over when close to success
        # success_env_ids = torch.where(self.insertion_successes)[0]
        # actions[success_env_ids] = residual_actions[success_env_ids].clone()

        return actions

    def _apply_actions_as_ctrl_targets(self, actions, ctrl_target_gripper_dof_pos, do_scale):
        """Apply actions from policy as position/rotation targets."""

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]

        if do_scale:
            pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device))
        self.ctrl_target_fingertip_centered_pos = self.fingertip_centered_pos + pos_actions

        self.ctrl_target_fingertip_centered_quat = self.fingertip_centered_quat

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

        self.generate_ctrl_signals()

    def _pose_world_to_robot_base(self, pos, quat):
        """Convert pose from world frame to robot base frame."""

        robot_base_transform_inv = torch_utils.tf_inverse(self.robot_base_quat, self.robot_base_pos)
        quat_in_robot_base, pos_in_robot_base = torch_utils.tf_combine(robot_base_transform_inv[0],
                                                                       robot_base_transform_inv[1],
                                                                       quat,
                                                                       pos)
        return pos_in_robot_base, quat_in_robot_base


    # TODO: Maybe continue here aswell
    # def _record_deviating_environments(self, env_ids):
    #     """Record environments that have deviated from the disassembly path."""
    # pass



    
    def step(self, actions: torch.Tensor):
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

        # randomize actions
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        self.pre_physics_step(action_tensor)

        # step physics and render each frame
        for i in range(self.control_freq_inv):
            if self.force_render:
                self.render()
            self.gym.simulate(self.sim)

            if ATTACH_PLUG_TO_GRIPPER and self.control_freq_inv > 1: # NOTE: Added by yunsheng for attaching plug to gripper
                self.refresh_base_tensors()
                self.refresh_env_tensors()
                self._refresh_task_tensors()
                self._attach_plug_to_gripper(torch.arange(self.num_envs, device=self.device))

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        self.control_steps += 1

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)

        # randomize observations
        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras