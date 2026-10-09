# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="NoHim-21DOF-WalkRun-AMP-v0",
    entry_point="NoetixBumi.tasks.walk_rand_arms.noetix_bumi.env:ManagerBasedRLAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nohim_21dof_amp_cfg:NoHim21DofAmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_ppo_cfg:NoHim21DofRunnerCfg",
    },
)
