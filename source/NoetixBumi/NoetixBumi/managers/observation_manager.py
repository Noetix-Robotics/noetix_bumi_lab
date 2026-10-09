# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""Observation manager for computing observation signals for a given world."""

from __future__ import annotations

import numpy as np
import torch
from isaaclab.managers.observation_manager import ObservationManager


class MyObservationManager(ObservationManager):
    def compute_group(self, group_name: str, update_history: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        group_obs = super().compute_group(group_name, update_history)

        if (
            not isinstance(group_obs, torch.Tensor)
            or not self._group_obs_concatenate[group_name]
            or group_name == "discriminator"
        ):
            return group_obs

        first_term_cfg = self._group_obs_term_cfgs[group_name][0]
        history_length = first_term_cfg.history_length if hasattr(first_term_cfg, "history_length") else 0

        if history_length == 0:
            return group_obs

        group_obs = group_obs.flatten(1, 2)

        return group_obs
