# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
import torch
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def randomize_actuator_effort_limit(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    effort_limit_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"] = "scale",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize actuator effort limits and write implicit-actuator values to simulation."""
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    for actuator in asset.actuators.values():
        if isinstance(asset_cfg.joint_ids, slice):
            actuator_indices = slice(None)
            if not isinstance(actuator.joint_indices, (slice, torch.Tensor)):
                raise TypeError("Actuator joint indices must be a slice or tensor.")
        elif isinstance(actuator.joint_indices, slice):
            actuator_indices = torch.tensor(asset_cfg.joint_ids, device=asset.device)
        else:
            actuator_joint_indices = actuator.joint_indices
            asset_joint_ids = torch.tensor(asset_cfg.joint_ids, device=asset.device)
            actuator_indices = torch.nonzero(torch.isin(actuator_joint_indices, asset_joint_ids)).view(-1)
            if len(actuator_indices) == 0:
                continue

        effort_limit = actuator.effort_limit_sim[env_ids].clone()
        effort_limit = _randomize_prop_by_op(
            effort_limit,
            effort_limit_distribution_params,
            dim_0_ids=None,
            dim_1_ids=actuator_indices,
            operation=operation,
            distribution=distribution,
        )
        actuator.effort_limit_sim[env_ids] = effort_limit

        if isinstance(actuator, ImplicitActuator):
            asset.write_joint_effort_limit_to_sim(
                effort_limit,
                joint_ids=actuator.joint_indices,
                env_ids=env_ids,
            )


def reset_root_state_amp(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    root_pos_range: dict[str, tuple[float, float]],
    root_vel_range: dict[str, tuple[float, float]],
    joint_pos_range: tuple[float, float],
    joint_vel_range: tuple[float, float],
    reference_state_initialization: bool = False,
    prob_rsi: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # cast env_ids to allow broadcasting
    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    if (
        env.unwrapped.motion_loader is not None
        and reference_state_initialization
        and prob_rsi > torch.rand(1, dtype=torch.float32, device=env.device)
    ):
        # get frames of amp motions
        frames = env.motion_loader.get_full_frame_batch(len(env_ids))

        # get root state of amp motions
        positions = env.unwrapped.motion_loader_class.get_root_pos_batch(frames) + env.scene.env_origins[env_ids]
        positions[:, 2] += 0.05
        quat_xyzw = env.unwrapped.motion_loader_class.get_root_rot_batch(frames)  # xyzw
        orientations = torch.cat((quat_xyzw[:, -1].unsqueeze(1), quat_xyzw[:, :-1]), dim=1)  # xyzw -> wxyz

        # base root velocities
        base_lin_vel = env.unwrapped.motion_loader_class.get_linear_vel_batch(frames)
        base_ang_vel = env.unwrapped.motion_loader_class.get_angular_vel_batch(frames)
        # world root velocities
        lin_vel = math_utils.quat_apply(orientations, base_lin_vel)
        ang_vel = math_utils.quat_apply(orientations, base_ang_vel)
        velocities = torch.cat([lin_vel, ang_vel], dim=-1)

        # get joint state of amp motions
        joint_pos = env.unwrapped.motion_loader_class.get_joint_pose_batch(frames)[
            ..., env.unwrapped.motion_loader.joint_mapping["motion2sim"]
        ]
        joint_vel = env.unwrapped.motion_loader_class.get_joint_vel_batch(frames)[
            ..., env.unwrapped.motion_loader.joint_mapping["motion2sim"]
        ]
    else:
        # get default root state
        root_states = asset.data.default_root_state[env_ids].clone()

        # poses
        range_list = [root_pos_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

        positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
        # base root velocities
        range_list = [root_vel_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

        velocities = root_states[:, 7:13] + rand_samples

        # get default joint state
        joint_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
        joint_vel = asset.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()

        # bias these values randomly
        joint_pos += math_utils.sample_uniform(*joint_pos_range, joint_pos.shape, joint_pos.device)
        joint_vel += math_utils.sample_uniform(*joint_vel_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def randomize_joint_default_pos(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        randomized_pos = _randomize_prop_by_op(
            asset.data.default_joint_pos.to(asset.device).clone(),
            pos_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )
        if isinstance(joint_ids, slice):
            asset.data.default_joint_pos[env_ids] = randomized_pos[env_ids]
        else:
            asset.data.default_joint_pos[env_ids[:, None], joint_ids] = randomized_pos[env_ids[:, None], joint_ids]

        # This environment splits the robot into policy and manual action terms.
        for term_name in env.action_manager.active_terms:
            term = env.action_manager.get_term(term_name)
            term_joint_ids = term._joint_ids
            term._offset[env_ids] = asset.data.default_joint_pos[env_ids][:, term_joint_ids]


def randomize_upper_body(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    alpha: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]

    # ----------------------------------------------------
    # 1. resolve env_ids shape
    # ----------------------------------------------------
    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    # ----------------------------------------------------
    # 2. sample frames from upper-body motion dataset
    # ----------------------------------------------------
    frames = env.unwrapped.motion_loader_upper.get_full_frame_batch(len(env_ids))

    # target motion (q_upper_target)
    q_target_full = env.unwrapped.motion_loader_upper.get_joint_pose_batch(frames)[
        ..., env.unwrapped.motion_loader_upper.joint_mapping["motion2sim"]
    ]

    # only select upper-body joints
    q_target = q_target_full[:, asset_cfg.joint_ids]



    # ----------------------------------------------------
    # 3. get default joint pose q0
    # ----------------------------------------------------
    q0 = asset.data._joint_pos.data[iter_env_ids, asset_cfg.joint_ids].clone()

    # ----------------------------------------------------
    # 4. curriculum scaling (Eq. 3)
    #     q_curr = q0 + α (q_target - q0)
    # ----------------------------------------------------
    q_curr = q0 + alpha * (q_target - q0)
    q_curr[:, 0] = 0
    q_curr[:, 1] = 0

    # ----------------------------------------------------
    # 5. joint velocity: set to zero or small random noise
    #    (the paper does NOT scale velocity)
    # ----------------------------------------------------
    joint_vel = torch.zeros_like(q_curr)

    # ----------------------------------------------------
    # 6. clamp to joint limits
    # ----------------------------------------------------
    joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]

    q_curr = q_curr.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # ----------------------------------------------------
    # 7. write to simulator
    # ----------------------------------------------------
    asset.write_joint_state_to_sim(q_curr, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
