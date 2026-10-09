# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_dtype,
    record_joint_names,
    record_shape,
    record_joint_pos_offsets,
    record_joint_vel_offsets,
)


def current_feet_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    feet_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
    return feet_contact


@generic_io_descriptor(
    units="rad", axes=["X", "Y"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def euler_angles(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    euler_angle = torch.stack([roll, pitch], dim=1)
    return euler_angle


@generic_io_descriptor(
    observation_type="JointState", on_inspect=[record_joint_names, record_dtype, record_shape], units="rad"
)
def amp_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if env.unwrapped.motion_loader is None:
        joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    else:
        joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids][
            ..., env.unwrapped.motion_loader.joint_mapping["sim2motion"]
        ]
    return joint_pos


@generic_io_descriptor(
    observation_type="JointState", on_inspect=[record_joint_names, record_dtype, record_shape], units="rad/s"
)
def amp_joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if env.unwrapped.motion_loader is None:
        joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    else:
        joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids][
            ..., env.unwrapped.motion_loader.joint_mapping["sim2motion"]
        ]
    return joint_vel


def amp_key_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Key pos on the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w[:, :3].unsqueeze(1)
    if env.unwrapped.motion_loader is None:
        key_pos = asset.data.body_pos_w[:, asset_cfg.body_ids].clone()
    else:
        key_pos = asset.data.body_pos_w[:, asset_cfg.body_ids].clone()[:, env.unwrapped.motion_loader.key_pos_mapping["sim2motion"], :]
    local_key_pos = (key_pos - root_pos).flatten(1, 2)
    return local_key_pos


def payload(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the added rigid-body mass relative to the default asset."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return (
        asset.root_physx_view._masses[:, asset_cfg.body_ids] - asset.data.default_mass[:, asset_cfg.body_ids]
    ).to(device=env.device)


def friction(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return static friction, dynamic friction, and restitution."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.root_physx_view._friction_properties[:, asset_cfg.joint_ids].squeeze(1).to(device=env.device)


def kp_params(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the current joint stiffness values."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.root_physx_view._dof_stiffnesses[:, asset_cfg.joint_ids].to(device=env.device)


def kd_params(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the current joint damping values."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.root_physx_view._dof_dampings[:, asset_cfg.joint_ids].to(device=env.device)


def effort_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the current joint effort limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.root_physx_view._dof_max_forces[:, asset_cfg.joint_ids].to(device=env.device)
