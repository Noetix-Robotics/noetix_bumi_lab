# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""Symmetry transforms for the full BUMI 21-DOF policy."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from ...env import ManagerBasedRLAmpEnv


@torch.no_grad()
def data_augmentation_func_bumi(
    env: ManagerBasedRLEnv | ManagerBasedRLAmpEnv, obs: TensorDict | None, actions: torch.Tensor | None
):
    """The symmetry data augmentation function."""
    if obs is None:
        obs_aug = None
    else:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)

        if len(env.observation_space["policy"].shape) == 3:
            num_frames = env.observation_space["policy"].shape[1]
            single_actor_obs_dim = env.observation_space["policy"].shape[2]
        else:
            num_frames = 1
            single_actor_obs_dim = env.observation_space["policy"].shape[-1]
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        policy_obs = obs["policy"].clone()
        if num_frames == 1:
            obs_aug["policy"][batch_size : 2 * batch_size] = flip_bumi_actor_obs(policy_obs, env)
        else:
            frames = torch.split(policy_obs, single_actor_obs_dim, dim=-1)
            flipped_frames = torch.cat([flip_bumi_actor_obs(frame, env) for frame in frames], dim=-1)
            obs_aug["policy"][batch_size : 2 * batch_size] = flipped_frames

        if len(env.observation_space["critic"].shape) == 3:
            num_frames = env.observation_space["critic"].shape[1]
            single_critic_obs_dim = env.observation_space["critic"].shape[2]
        else:
            num_frames = 1
            single_critic_obs_dim = env.observation_space["critic"].shape[-1]
        obs_aug["critic"][:batch_size] = obs["critic"][:]
        critic_obs = obs["critic"].clone()
        if num_frames == 1:
            obs_aug["critic"][batch_size : 2 * batch_size] = flip_bumi_critic_obs(critic_obs, single_actor_obs_dim, env)
        else:
            frames = torch.split(critic_obs, single_critic_obs_dim, dim=-1)
            flipped_frames = [flip_bumi_critic_obs(frame, single_actor_obs_dim, env) for frame in frames]
            obs_aug["critic"][batch_size : 2 * batch_size] = torch.cat(flipped_frames, dim=-1)

        if "next_obs" in obs:
            obs_aug["next_obs"][:batch_size] = obs["next_obs"][:]
            next_obs = obs["next_obs"].clone()
            if num_frames == 1:
                obs_aug["next_obs"][batch_size : 2 * batch_size] = flip_bumi_actor_obs(next_obs, env)
            else:
                frames = torch.split(next_obs, single_actor_obs_dim, dim=-1)
                flipped_frames = [flip_bumi_actor_obs(frame, env) for frame in frames]
                obs_aug["next_obs"][batch_size : 2 * batch_size] = torch.cat(flipped_frames, dim=-1)

    if actions is None:
        actions_aug = None
    else:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size : 2 * batch_size] = flip_bumi_actions(actions)
    return obs_aug, actions_aug


def flip_bumi_actor_obs(one_step_obs: torch.Tensor, env: ManagerBasedRLEnv | ManagerBasedRLAmpEnv):
    if one_step_obs is None:
        return one_step_obs
    flipped_one_step_obs = torch.zeros_like(one_step_obs)
    flipped_one_step_obs[..., :3] = one_step_obs[..., 0:3]
    flipped_one_step_obs[..., 1] = -flipped_one_step_obs[..., 1]
    flipped_one_step_obs[..., 2] = -flipped_one_step_obs[..., 2]
    flipped_one_step_obs[..., 3:6] = one_step_obs[..., 3:6]
    flipped_one_step_obs[..., 3] = -flipped_one_step_obs[..., 3]
    flipped_one_step_obs[..., 5] = -flipped_one_step_obs[..., 5]
    flipped_one_step_obs[..., 6:8] = one_step_obs[..., 6:8] * torch.tensor(
        [-1.0, 1.0], dtype=one_step_obs.dtype, device=one_step_obs.device
    )
    flipped_one_step_obs[..., 8 : 8 + env.num_actions] = flip_bumi_dof(one_step_obs[..., 8 : 8 + env.num_actions])
    flipped_one_step_obs[..., 8 + env.num_actions : 8 + 2 * env.num_actions] = flip_bumi_dof(
        one_step_obs[..., 8 + env.num_actions : 8 + 2 * env.num_actions]
    )
    flipped_one_step_obs[..., 8 + 2 * env.num_actions : 8 + 3 * env.num_actions] = flip_bumi_dof(
        one_step_obs[..., 8 + 2 * env.num_actions : 8 + 3 * env.num_actions]
    )
    return flipped_one_step_obs


def flip_bumi_critic_obs(obs: torch.Tensor, single_actor_obs_dim: int, env: ManagerBasedRLEnv | ManagerBasedRLAmpEnv):
    if obs is None:
        return obs
    actor_obs = obs[..., :single_actor_obs_dim]
    num_actor_obs = single_actor_obs_dim
    flipped_critic_obs = torch.zeros_like(obs)
    flipped_critic_obs[..., :num_actor_obs] = flip_bumi_actor_obs(actor_obs, env)
    flipped_critic_obs[..., num_actor_obs : num_actor_obs + 3] = obs[..., num_actor_obs : num_actor_obs + 3]
    flipped_critic_obs[..., num_actor_obs + 1] = -flipped_critic_obs[..., num_actor_obs + 1]
    flipped_critic_obs[..., num_actor_obs + 3 : num_actor_obs + 4] = obs[..., num_actor_obs + 4 : num_actor_obs + 5]
    flipped_critic_obs[..., num_actor_obs + 4 : num_actor_obs + 5] = obs[..., num_actor_obs + 3 : num_actor_obs + 4]
    cursor = num_actor_obs + 5
    remaining = obs.shape[-1] - cursor
    if remaining >= 7 + 3 * env.num_actions:
        # payload is invariant; swap left/right friction values.
        flipped_critic_obs[..., cursor] = obs[..., cursor]
        cursor += 1
        flipped_critic_obs[..., cursor : cursor + 3] = obs[..., cursor + 3 : cursor + 6]
        flipped_critic_obs[..., cursor + 3 : cursor + 6] = obs[..., cursor : cursor + 3]
        cursor += 6
        # Kp, Kd and effort limits are positive physical parameters:
        # mirror them by left/right permutation without changing signs.
        for _ in range(3):
            flipped_critic_obs[..., cursor : cursor + env.num_actions] = swap_bumi_dof(
                obs[..., cursor : cursor + env.num_actions]
            )
            cursor += env.num_actions
    if env.cfg.scene.height_scanner:
        resolution = 1.0 / env.cfg.scene.height_scanner.pattern_cfg.resolution
        num_x, num_y = env.cfg.scene.height_scanner.pattern_cfg.size
        num_x_size, num_y_size = int((num_x * resolution) + 1), int((num_y * resolution) + 1)
        base_idx = torch.arange(num_x_size * num_y_size, dtype=torch.int, device=flipped_critic_obs.device).view(
            int(num_x_size), int(num_y_size)
        )
        flip_indices = base_idx.flip(dims=[1]).reshape(-1)
        flipped_critic_obs[..., cursor:] = obs[..., cursor:][..., flip_indices]
    elif cursor < obs.shape[-1]:
        flipped_critic_obs[..., cursor:] = obs[..., cursor:]
    return flipped_critic_obs


def flip_bumi_actions(actions: torch.Tensor | None):
    if actions is None:
        return None
    return flip_bumi_dof(actions)


def swap_bumi_dof(dof: torch.Tensor):
    """Swap left/right BUMI DOFs without applying coordinate sign changes."""
    swapped_dof = torch.zeros_like(dof)
    swapped_dof[..., 0] = dof[..., 1]
    swapped_dof[..., 1] = dof[..., 0]
    swapped_dof[..., 2] = dof[..., 2]
    swapped_dof[..., 3] = dof[..., 4]
    swapped_dof[..., 4] = dof[..., 3]
    swapped_dof[..., 5] = dof[..., 6]
    swapped_dof[..., 6] = dof[..., 5]
    swapped_dof[..., 7] = dof[..., 8]
    swapped_dof[..., 8] = dof[..., 7]
    swapped_dof[..., 9] = dof[..., 10]
    swapped_dof[..., 10] = dof[..., 9]
    swapped_dof[..., 11] = dof[..., 12]
    swapped_dof[..., 12] = dof[..., 11]
    swapped_dof[..., 13] = dof[..., 14]
    swapped_dof[..., 14] = dof[..., 13]
    swapped_dof[..., 15] = dof[..., 16]
    swapped_dof[..., 16] = dof[..., 15]
    swapped_dof[..., 17] = dof[..., 18]
    swapped_dof[..., 18] = dof[..., 17]
    swapped_dof[..., 19] = dof[..., 20]
    swapped_dof[..., 20] = dof[..., 19]
    return swapped_dof


def flip_bumi_dof(dof: torch.Tensor):
    """Mirror BUMI DOFs in the Isaac Sim articulation order.

    The BUMI URDF uses +Y axes for pitch joints and +X/+Z axes for roll/yaw
    joints. Under a left-right reflection, Y-axis joint coordinates keep their
    sign while X- and Z-axis coordinates change sign.
    """
    flipped_dof = torch.zeros_like(dof)
    # [0, 1] l/r leg pitch: +Y axis, same sign.
    flipped_dof[..., 0] = dof[..., 1]
    flipped_dof[..., 1] = dof[..., 0]
    # [2] waist yaw: +Z axis, sign changes.
    flipped_dof[..., 2] = -dof[..., 2]
    # [3, 4] l/r leg roll: +X axis, swap and change sign.
    flipped_dof[..., 3] = -dof[..., 4]
    flipped_dof[..., 4] = -dof[..., 3]
    # [5, 6] l/r arm pitch: +Y axis, same sign.
    flipped_dof[..., 5] = dof[..., 6]
    flipped_dof[..., 6] = dof[..., 5]
    # [7, 8] l/r leg yaw: +Z axis, swap and change sign.
    flipped_dof[..., 7] = -dof[..., 8]
    flipped_dof[..., 8] = -dof[..., 7]
    # [9, 10] l/r arm roll: +X axis, swap and change sign.
    flipped_dof[..., 9] = -dof[..., 10]
    flipped_dof[..., 10] = -dof[..., 9]
    # [11, 12] l/r knee pitch: +Y axis, same sign.
    flipped_dof[..., 11] = dof[..., 12]
    flipped_dof[..., 12] = dof[..., 11]
    # [13, 14] l/r arm yaw: +Z axis, swap and change sign.
    flipped_dof[..., 13] = -dof[..., 14]
    flipped_dof[..., 14] = -dof[..., 13]
    # [15, 16] l/r ankle pitch: +Y axis, same sign.
    flipped_dof[..., 15] = dof[..., 16]
    flipped_dof[..., 16] = dof[..., 15]
    # [17, 18] l/r elbow pitch: +Y axis, same sign.
    flipped_dof[..., 17] = dof[..., 18]
    flipped_dof[..., 18] = dof[..., 17]
    # [19, 20] l/r ankle roll: +X axis, swap and change sign.
    flipped_dof[..., 19] = -dof[..., 20]
    flipped_dof[..., 20] = -dof[..., 19]
    return flipped_dof
