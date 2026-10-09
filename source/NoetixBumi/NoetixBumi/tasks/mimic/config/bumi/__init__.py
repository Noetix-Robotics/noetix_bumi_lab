import gymnasium as gym

from . import agents, bumi_flat_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Mimic-Bumi-v0",
    entry_point="NoetixBumi.tasks.mimic.env:ManagerBasedRLMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": bumi_flat_env_cfg.BumiFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BumiPpoRunnerCfg",
    },
)

