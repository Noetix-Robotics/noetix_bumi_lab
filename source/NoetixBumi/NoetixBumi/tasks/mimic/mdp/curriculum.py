# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import torch
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import numpy as np


class modify_reward_sigma(ManagerTermBase):
    """Curriculum that modifies the reward weight based on a step-wise schedule."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.enable_adaptive_reward_sigma = True
        self.reward_sigma_alpha = 0.001
        self.reward_sigma_scale = 1.0
        self.reward_sigma_type = "origin"
        self.reward_sigma = {
            "motion_global_anchor_pos": 0.3,
            "motion_global_anchor_ori": 0.4,
            "motion_body_pos": 0.3,
            "motion_body_ori": 0.3,
            "motion_body_lin_vel": 1.0,
            "motion_body_ang_vel": 15,
            "motion_joint_pos": 0.3,
        }

        self._reward_error_ema = dict()
        for key, value in self.reward_sigma.items():
            self._reward_error_ema[key] = value

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
    ):
        sigma = self.reward_sigma
        # self._update_reward_adaptive_sigma()
        return sigma

    def _update_reward_adaptive_sigma(self, term: str, error: torch.Tensor):

        if self.enable_adaptive_reward_sigma:
            self._reward_error_ema[term] = (
                self._reward_error_ema[term] * (1 - self.reward_sigma_alpha)
                + error.mean().item() * self.reward_sigma_alpha
            )
            if self.reward_sigma_type == "scale":
                self.reward_sigma[term] = min(
                    self._reward_error_ema[term] * self.reward_sigma_scale, self.reward_sigma[term]
                )
            elif self.reward_sigma_type == "mean":
                self.reward_sigma[term] = (
                    min(self._reward_error_ema[term], self.reward_sigma[term]) + self._reward_error_ema[term]
                ) / 2
            elif self.reward_sigma_type == "origin":
                self.reward_sigma[term] = min(self._reward_error_ema[term], self.reward_sigma[term])


class modify_regulization_weight(ManagerTermBase):
    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.initial_penalty_scale = 0.1
        self.degree = 1.0e-06
        self.level_down_threshold = 40
        self.level_up_threshold = 42
        self.average_episode_length = 0.0
        self.num_compute_average_epl = 10000
        self.sacling = 1.0
        self.sacling_min = 0.9
        self.sacling_max = 1.1
        self.term_info = self.cfg.params.get("term_info", [])
        self.term_num = len(self.term_info)
        print(self.term_info)
        self._term_cfg_list = [env.reward_manager.get_term_cfg(term["term_name"]) for term in self.term_info]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_info: list,
    ):

        for i in range(self.term_num):
            self._term_cfg_list[i].weight = self.term_info[i]["weight"] * self.sacling

            env.reward_manager.set_term_cfg(self.term_info[i]["term_name"], self._term_cfg_list[i])

        return self.sacling

    def _update_adaptive_threshold(self, current_average_episode_length, num):

        self.average_episode_length = self.average_episode_length * (
            1 - num / self.num_compute_average_epl
        ) + current_average_episode_length * (num / self.num_compute_average_epl)

        if self.average_episode_length < self.level_down_threshold:
            for i in range(self.term_num):
                self.sacling *= 1 - self.degree
                self.sacling = np.clip(self.sacling, self.sacling_min, self.sacling_max)
        elif self.average_episode_length > self.level_up_threshold:
            for i in range(self.term_num):
                self.sacling *= 1 + self.degree
                self.sacling = np.clip(self.sacling, self.sacling_min, self.sacling_max)


class modify_termination_threshold(ManagerTermBase):
    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.enable_termination_threshold = True

        self.initial_threshold = 1.5
        self.degree = 2.5e-05
        self.level_down_threshold = 40
        self.level_up_threshold = 42
        self.average_episode_length = 0.0
        self.num_compute_average_epl = 10000

        self.term_info = self.cfg.params.get("term_info", [])
        print(self.term_info)
        self.term_num = len(self.term_info)

        self._term_cfg_list = [env.termination_manager.get_term_cfg(term["term_name"]) for term in self.term_info]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_info: list,
    ):

        for i in range(self.term_num):
            self._term_cfg_list[i].params["threshold"] = self.term_info[i]["threshold"]

            env.termination_manager.set_term_cfg(self.term_info[i]["term_name"], self._term_cfg_list[i])

        return self.average_episode_length

    def _update_adaptive_threshold(self, current_average_episode_length, num):

        self.average_episode_length = self.average_episode_length * (
            1 - num / self.num_compute_average_epl
        ) + current_average_episode_length * (num / self.num_compute_average_epl)

        if self.average_episode_length < self.level_down_threshold:
            for i in range(self.term_num):
                self.term_info[i]["threshold"] *= 1 + self.degree
                self.term_info[i]["threshold"] = np.clip(
                    self.term_info[i]["threshold"], self.term_info[i]["min"], self.term_info[i]["max"]
                )
        elif self.average_episode_length > self.level_up_threshold:
            for i in range(self.term_num):
                self.term_info[i]["threshold"] *= 1 - self.degree
                self.term_info[i]["threshold"] = np.clip(
                    self.term_info[i]["threshold"], self.term_info[i]["min"], self.term_info[i]["max"]
                )
