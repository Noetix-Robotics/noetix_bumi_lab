# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class RslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name = "PPO"
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1.0e-3
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0
   
   
@configclass
class PpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyRunner"
    seed = 30
    num_steps_per_env = 24
    max_iterations = 40000
    save_interval = 1000
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=False,
        actor_hidden_dims=[1024, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg()
    clip_actions = 18.0

    experiment_name = "bumi_mimic"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "bumi_mimic"
    wandb_project = "bumi_mimic"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"


@configclass
class BumiPpoRunnerCfg(PpoRunnerCfg):
    experiment_name = "bumi_mimic"
