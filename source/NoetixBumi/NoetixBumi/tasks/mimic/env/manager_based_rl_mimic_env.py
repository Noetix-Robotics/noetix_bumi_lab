# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

from __future__ import annotations

import gymnasium as gym
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from isaaclab.utils.math import quat_apply
from NoetixBumi.managers import MyObservationManager

from NoetixBumi.tasks.mimic.mdp import MotionLoader
from NoetixBumi.tasks.mimic.mdp.terminations import (
    bad_anchor_pos_z_only,
    bad_anchor_ori,
    bad_motion_body_pos_z_only,
)
from isaaclab.envs.mdp.terminations import time_out
import numpy as np

class ManagerBasedRLMimicEnv(ManagerBasedRLEnv, gym.Env):
    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode)
        self.cfg: ManagerBasedRLEnvCfg
        self.observation_manager = MyObservationManager(self.cfg.observations, self)
        self.robot: Articulation = self.scene["robot"]
        # ---------------------- Initial Motion Loader ----------------------
        self.motion = MotionLoader(
            self.cfg.commands.motion.motion_file,
            torch.tensor([0], dtype=torch.long, device=self.sim.device),
            self.sim.device,
        )

        self.last_episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    
    def visualize_motion(self, time_steps):
        assert self.num_envs == 1

        root_states = self.robot.data.default_root_state.clone()
        root_states[:, :3] = self.motion.body_pos_w[time_steps][:, 0] + self.scene.env_origins[:, None, :]
        root_states[:, 3:7] = self.motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = self.motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = self.motion.body_ang_vel_w[time_steps][:, 0]

        self.robot.write_root_state_to_sim(root_states)
        self.robot.write_joint_state_to_sim(self.motion.joint_pos[time_steps], self.motion.joint_vel[time_steps])
        self.scene.write_data_to_sim()
        self.sim.render()  # We don't want physic (sim.step())
        self.scene.update(self.step_dt)

        key_pos = self.motion._body_pos_w[time_steps]
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

        self.last_episode_length_buf = self.episode_length_buf.clone()
        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

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

        # print(self.last_episode_length_buf[reset_env_ids])
        terminal_env_ids = reset_env_ids.clone()
        terminal_actor_obs = self.obs_buf["policy"][terminal_env_ids].clone()
        terminal_critic_obs = self.obs_buf["critic"][terminal_env_ids].clone()

        # Record termination reasons
        termination_reasons = {}
        for env_id in reset_env_ids:
            reasons = []
            try:
                if bad_anchor_pos_z_only(self, "motion", self.cfg.terminations.anchor_pos.params["threshold"])[env_id.item()]:
                    reasons.append("anchor_pos")
            except:
                pass
            try:
                if bad_anchor_ori(self, self.cfg.terminations.anchor_ori.params["asset_cfg"], "motion", self.cfg.terminations.anchor_ori.params["threshold"])[env_id.item()]:
                    reasons.append("anchor_ori")
            except:
                pass
            try:
                if bad_motion_body_pos_z_only(self, "motion", self.cfg.terminations.ee_body_pos.params["threshold"], self.cfg.terminations.ee_body_pos.params["body_names"])[env_id.item()]:
                    reasons.append("ee_body_pos")
            except:
                pass
            try:
                if time_out(self, time_out=True)[env_id.item()]:
                    reasons.append("time_out")
            except:
                pass
            termination_reasons[env_id.item()] = reasons

        if len(reset_env_ids) > 0:

            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            # reset buffer
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)
            
            current_average_episode_length = torch.mean(self.last_episode_length_buf[terminal_env_ids], dtype=torch.float)
            # self.curriculum_manager.cfg.curriculum_termination_threshold.func._update_adaptive_threshold(current_average_episode_length, len(terminal_env_ids))
            # self.curriculum_manager.cfg.curriculum_regulization_weight.func._update_adaptive_threshold(current_average_episode_length, len(terminal_env_ids))

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
        }

        self.extras["termination_reasons"] = termination_reasons

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
