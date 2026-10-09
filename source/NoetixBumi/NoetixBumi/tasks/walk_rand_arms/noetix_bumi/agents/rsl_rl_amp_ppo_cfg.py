# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg
from NoetixBumi.assets.robots.bumi.bumi import BUMI_21DOF_NAMES

@configclass
class NoHim21DofActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = "ActorCritic"
    init_noise_std = 1.0
    actor_obs_normalization = True
    critic_obs_normalization = True
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"


@configclass
class NoHim21DofAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name = "AMPPPO"
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.005
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 5.0e-4
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0
    discriminator_learning_rate = 5.0e-6
    discriminator_gradient_penalty_coef = 5
    discriminator_num_mini_batches = 80
    discriminator_loss_function = "WassersteinLoss"
    amp_replay_buffer_size = 200000
    normalize_advantage_per_mini_batch = False
    symmetry_cfg = RslRlSymmetryCfg(
        use_data_augmentation=True,
        use_mirror_loss=True,
        data_augmentation_func=(
            "NoetixBumi.tasks.walk_rand_arms.noetix_bumi.mdp.symmetry.bumi_21dof:"
            "data_augmentation_func_bumi"
        ),
        mirror_loss_coeff=1.0,
    )
    rnd_cfg = None


@configclass
class NoHim21DofRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "AmpOnPolicyRunner"
    seed = 42
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    policy = NoHim21DofActorCriticCfg()
    algorithm = NoHim21DofAlgorithmCfg()
    clip_actions = 18.0
    experiment_name = "nohim_21dof_amp"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "walk_rand_arm"
    wandb_project = "walk_rand_arm"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
    normalize_style_reward = False
    amp_reward_coef = 1.0
    amp_reward_lerp = 0.5
    style_reward_function = "wasserstein_mapping"
    discriminator_shape = [1024, 512]
    joint_names = BUMI_21DOF_NAMES
    discriminator_mask_joint_names = [
        "l_ankle_pitch_joint",
        "r_ankle_pitch_joint",
        "l_ankle_roll_joint",
        "r_ankle_roll_joint",
    ]
    discriminator_mask_dims = None
