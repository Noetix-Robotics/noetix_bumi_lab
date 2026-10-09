# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

from __future__ import annotations

import gymnasium as gym
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply
from NoetixBumi.managers import MyObservationManager, MyActionManager

from rsl_rl.utils.motion_loader import *

class ManagerBasedRLAmpEnv(ManagerBasedRLEnv, gym.Env):
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        self.motion_loader = None
        # initialize the base class to setup the scene.
        super().__init__(cfg=cfg, render_mode=render_mode)

        self.cfg = cfg
        self._full_dof = True

        self.observation_manager = MyObservationManager(self.cfg.observations, self)
        self.action_manager = MyActionManager(self.cfg.actions, self)
        self.robot: Articulation = self.scene["robot"]
        self.feet_both_contact_time = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.commanded_translation_distance = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.commanded_translation_time = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.prev_commanded_translation_distance = torch.zeros_like(self.commanded_translation_distance)
        self.prev_commanded_translation_time = torch.zeros_like(self.commanded_translation_time)
        # track how far each base actually walked inside an episode and persist last episode stats
        self.walked_distance = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.prev_walked_distance = torch.zeros_like(self.walked_distance)
        # ---------------------- Initial Motion Loader ----------------------
        robot_joint_names = list(self.robot.joint_names)
        robot_body_names = list(self.robot.body_names)
        self.leg_joint_subset = list(range(len(robot_joint_names)))
        self.lower_body_subset = list(range(len(robot_body_names)))
        self.motion_loader_class = eval(self.cfg.motion_loader_name)
        self.motion_loader = self.motion_loader_class(
            device=self.device,
            time_between_frames=self.step_dt,
            reference_observation_horizon=self.cfg.reference_observation_horizon,
            num_preload_transitions=self.cfg.num_preload_transitions,
            motion_files=self.cfg.motion_files,
            sim_joint_names=robot_joint_names,
            sim_key_pos_names=robot_body_names,
        )
        self.scale_arm = 0.0
        self.alpha = 0.8
        self._arm_target_full_ids = []
        self._arm_sub_idxs = []
        self.arm_phase = torch.zeros((self.num_envs, 0), device=self.device)
        self.arm_freq = torch.zeros((self.num_envs, 0), device=self.device)
        self.arm_sign = torch.zeros((1, 0), device=self.device)
        self._prev_arm_target = None

    """
    Properties
    """

    def visualize_motion(self, time, traj_idx):
        """
        Update the robot simulation state based on the AMP motion capture data at a given time.

        This function sets the joint positions and velocities, root position and orientation,
        and linear/angular velocities according to the AMP motion frame at the specified time,
        then steps the simulation and updates the scene.

        Args:
            time (float): The time (in seconds) at which to fetch the AMP motion frame.

        Returns:
            None
        """
        assert self.num_envs == 1

        visual_motion_frame = self.motion_loader.get_full_frame_at_time(traj_idx, time)
        motion_loader_cls = type(self.motion_loader)
        device = self.device

        dof_pos = torch.zeros((len(self.robot.joint_names)), dtype=torch.float, device=device)
        dof_vel = torch.zeros((len(self.robot.joint_names)), dtype=torch.float, device=device)

        if self._full_dof:
            dof_pos[:] = motion_loader_cls.get_joint_pose(visual_motion_frame)[..., self.motion_loader.joint_mapping["motion2sim"]]
            dof_vel[:] = motion_loader_cls.get_joint_vel(visual_motion_frame)[..., self.motion_loader.joint_mapping["motion2sim"]]
        self.robot.write_joint_position_to_sim(dof_pos)
        self.robot.write_joint_velocity_to_sim(dof_vel)

        env_ids = torch.arange(self.num_envs, device=device)

        root_pos = motion_loader_cls.get_root_pos(visual_motion_frame)

        quat_xyzw = motion_loader_cls.get_root_rot(visual_motion_frame)
        quat_wxyz = torch.tensor(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=torch.float, device=device
        )

        base_lin_vel = motion_loader_cls.get_linear_vel(visual_motion_frame)
        base_ang_vel = motion_loader_cls.get_angular_vel(visual_motion_frame)

        lin_vel = quat_apply(quat_wxyz, base_lin_vel)
        ang_vel = quat_apply(quat_wxyz, base_ang_vel)

        # root state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        root_state = torch.zeros((self.num_envs, 13), device=device)
        root_state[:, 0:3] = torch.tile(root_pos.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 3:7] = torch.tile(quat_wxyz.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 7:10] = torch.tile(lin_vel.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 10:13] = torch.tile(ang_vel.unsqueeze(0), (self.num_envs, 1))

        self.robot.write_root_state_to_sim(root_state, env_ids)
        self.sim.render()
        self.scene.update(dt=self.step_dt)

        key_pos = motion_loader_cls.get_key_pos_local(visual_motion_frame)
        key_pos = key_pos.view(-1, 3)[self.motion_loader.key_pos_mapping["motion2sim"]] + root_pos
        return key_pos

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # track how much translation-centric command was issued in the current episode
        command = self.command_manager.get_command("base_velocity")
        command_xy_norm = torch.norm(command[:, :2], dim=1)
        self.commanded_translation_distance += command_xy_norm * self.step_dt
        self.commanded_translation_time += (command_xy_norm > 0.1).float() * self.step_dt

        # The 21DOF policy directly controls every joint.
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for sub_i in range(self.cfg.decimation):
            self._sim_step_counter += 1
            prev_root_pos = self.robot.data.root_pos_w[:, :2].clone()
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
            curr_root_pos = self.robot.data.root_pos_w[:, :2]
            self.walked_distance += torch.norm(curr_root_pos - prev_root_pos, dim=1)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        terminal_env_ids = reset_env_ids.clone()
        terminal_actor_obs = self.obs_buf["policy"][terminal_env_ids].clone()
        terminal_critic_obs = self.obs_buf["critic"][terminal_env_ids].clone()
        terminal_amp_obs = self.obs_buf["discriminator"][terminal_env_ids].clone()

        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # reset buffer
            self.feet_both_contact_time[reset_env_ids] = 0.0
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

            # stash commanded translation stats for curriculum and reset accumulators
            self.prev_commanded_translation_distance[reset_env_ids] = self.commanded_translation_distance[reset_env_ids]
            self.prev_commanded_translation_time[reset_env_ids] = self.commanded_translation_time[reset_env_ids]
            self.commanded_translation_distance[reset_env_ids] = 0.0
            self.commanded_translation_time[reset_env_ids] = 0.0

            # stash walked distance stats alongside commanded stats
            self.prev_walked_distance[reset_env_ids] = self.walked_distance[reset_env_ids]
            self.walked_distance[reset_env_ids] = 0.0

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)

        self.extras["terminal_observations"] = {
            "env_ids": terminal_env_ids,
            "policy": terminal_actor_obs,
            "critic": terminal_critic_obs,
            "amp": terminal_amp_obs,
        }

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
