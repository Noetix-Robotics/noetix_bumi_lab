# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

# ============================================================================================
# 奖励函数模块 — 定义了所有 MDP 奖励项的具体计算逻辑
#
# 奖励函数是强化学习中定义任务目标的核心手段。每个函数接收环境状态，返回标量奖励值。
# 总奖励 = Σ(weight_i × reward_func_i)，其中 weight 在配置文件中定义。
#
# 奖励函数按功能分为以下类别：
#   1. 速度跟踪 — 鼓励机器人跟踪目标线速度和角速度
#   2. 基座姿态 — 保持躯干水平，防止倾斜
#   3. 足部行为 — 足端接触、滑动、腾空、冲击力等
#   4. 关节正则化 — 惩罚关节偏离默认姿态，鼓励节能和平滑
#   5. 非期望接触 — 惩罚脚踝以外的身体部位接触地面
#   6. 步态相关 — 基于步态相位的周期性足力、速度、摆腿高度控制
#   7. 场景化奖励 — 针对后退、行走、转向、侧移等特定场景的细粒度奖励
# ============================================================================================

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils  # Isaac Lab 数学工具（四元数运算、坐标变换等）
import torch
from isaaclab.assets import Articulation  # 关节体资产类型（用于获取关节状态、刚体位姿等）
from isaaclab.managers import SceneEntityCfg  # 场景实体配置（指定要读取哪个资产的哪些关节/刚体）
from isaaclab.sensors import ContactSensor  # 接触传感器类型（用于获取接触力、腾空时间等）

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv  # 仅类型检查时导入，避免运行时循环引用


# ============================================================================================
# 1. 通用辅助奖励
# ============================================================================================

def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """惩罚关节位置偏离目标值的 L2 范数。

    用于将指定关节保持在某个目标角度附近。

    Args:
        env: RL 环境实例。
        target: 目标关节角度（rad），所有指定关节使用同一目标值。
        asset_cfg: 指定要惩罚的资产和关节。

    Returns:
        奖励张量 (num_envs,)，值为各关节偏离量的平方和。值越大表示偏离越远（作为惩罚项时取负权重）。
    """
    # 提取资产引用（类型提示用）
    asset: Articulation = env.scene[asset_cfg.name]
    # 将关节位置包裹到 (-π, π) 范围，避免角度周期性带来的错误偏差
    joint_pos = math_utils.wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # 计算所有关节与目标值的平方偏差之和
    return torch.sum(torch.square(joint_pos - target), dim=1)


# ============================================================================================
# 2. 速度跟踪奖励 — 鼓励机器人跟踪命令速度
# ============================================================================================

def track_lin_vel_x_yaw_frame_exp(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Track commanded x velocity in the robot yaw frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    command = env.command_manager.get_command("base_velocity")
    lin_vel_error = torch.square(command[:, 0] - vel_yaw[:, 0])
    return torch.exp(-lin_vel_error / std**2)


def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """线速度 xy 跟踪（机器人偏航参考系，指数奖励）。

    在机器人的偏航参考系（即水平面投影方向）中计算 xy 线速度误差。
    为什么要用偏航参考系而不是世界坐标系？
      - 当机器人面向不同方向时，"前进"的定义应该始终是机头朝向。
      - 在偏航参考系下，命令的 vx 始终表示前进/后退，vy 始终表示左/右平移。

    奖励函数：reward = exp(-||cmd_xy - vel_xy||² / σ²)

    Args:
        env: RL 环境实例。
        std: 指数核的标准差 σ，控制奖励对误差的敏感度。σ 越小，对误差越敏感。
        asset_cfg: 机器人资产配置。

    Returns:
        奖励张量 (num_envs,)，值域 [0, 1]，1 表示完美跟踪。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # 将世界坐标系下的线速度转换到机器人的偏航参考系
    # math_utils.yaw_quat: 从根四元数中提取偏航分量（仅保留绕 z 轴的旋转）
    # quat_apply_inverse: 逆向旋转，即从世界坐标系转换到机体偏航坐标系
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    # 获取当前的速度命令 [vx_cmd, vy_cmd, ωz_cmd]
    command = env.command_manager.get_command("base_velocity")
    # 计算 xy 平面速度误差的平方和
    lin_vel_error = torch.sum(torch.square(command[:, :2] - vel_yaw[:, :2]), dim=1)
    # 指数奖励：误差越小，奖励越接近 1
    return torch.exp(-lin_vel_error / std**2)


def track_lin_vel_y_yaw_frame_exp(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """线速度 y 单独跟踪（机器人偏航参考系，指数奖励）。

    与 track_lin_vel_xy_yaw_frame_exp 类似，但仅跟踪侧向速度（y 方向）。
    用于对侧向运动进行额外的细粒度控制。

    Args:
        env: RL 环境实例。
        std: 指数核的标准差 σ。
        asset_cfg: 机器人资产配置。

    Returns:
        奖励张量 (num_envs,)，值域 [0, 1]。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # 世界线速度 → 机体偏航参考系
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    command = env.command_manager.get_command("base_velocity")
    # 仅计算 y 方向（侧向）速度误差的平方
    lin_vel_error = torch.square(command[:, 1] - vel_yaw[:, 1])
    return torch.exp(-lin_vel_error / std**2)


def command_tracking_error_over_threshold(
    env: ManagerBasedRLEnv, threshold: float = 0.5, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Return only the portion of command-tracking error above a threshold."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    command = env.command_manager.get_command("base_velocity")
    tracking_error = torch.stack(
        (
            torch.abs(command[:, 0] - vel_yaw[:, 0]),
            torch.abs(command[:, 1] - vel_yaw[:, 1]),
            torch.abs(command[:, 2] - asset.data.root_ang_vel_w[:, 2]),
        ),
        dim=1,
    )
    return (torch.amax(tracking_error, dim=1) - threshold).clamp(min=0.0)


def track_ang_vel_z_world_exp(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """角速度 z 跟踪（世界坐标系，指数奖励）。

    跟踪机器人绕世界 z 轴（偏航轴）的旋转角速度。
    角速度在偏航角速度跟踪中天然适合在世界坐标系下计算，因为转向是相对于世界的。

    Args:
        env: RL 环境实例。
        std: 指数核的标准差 σ。
        asset_cfg: 机器人资产配置。

    Returns:
        奖励张量 (num_envs,)，值域 [0, 1]。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    # 计算世界 z 轴角速度误差（command[:, 2] 即 ωz_cmd）
    ang_vel_error = torch.square(command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


# ============================================================================================
# 3. 站立/转向专项速度跟踪
# ============================================================================================

def stand_turn_ang_vel_z_world_exp(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """站立原地转向时的角速度跟踪（仅当线速度命令 ≈ 0 且角速度命令 ≠ 0 时激活）。

    逻辑：
      1. 计算角速度跟踪的指数奖励。
      2. 当 ||cmd_xy|| < 0.1（无前进/侧向命令）且 |cmd_ωz| > 0.1（有转向命令）时激活。

    Args:
        env: RL 环境实例。
        std: 指数核的标准差 σ。
        asset_cfg: 机器人资产配置。

    Returns:
        奖励张量 (num_envs,)。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    ang_vel_error = torch.square(command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    # 仅当无前进/侧移命令（线速度 < 0.1）时激活
    reward *= torch.norm(command[:, :2], dim=1) < 0.1
    # 仅当有转向命令（|ωz| > 0.1）时激活
    reward *= torch.abs(command[:, 2]) > 0.1
    return reward


def stand_turn_delta_yaw_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """站立转向时偏航角速度偏差惩罚（分段线性惩罚）。

    核心思想：当站立转向时，如果实际角速度与命令角速度偏差过大，给予惩罚。
    惩罚仅在以下条件同时满足时触发：
      - Δω > 0.1 rad/s（偏差超过阈值）
      - Δω > 0.5 × |cmd_ωz|（偏差超过命令值的一半）
      - ||cmd_xy|| < 0.1（无前进/侧向命令）
      - |cmd_ωz| > 0.1（有转向命令）

    Args:
        env: RL 环境实例。
        asset_cfg: 机器人资产配置。

    Returns:
        惩罚张量 (num_envs,)，0 表示无惩罚，正值表示需要惩罚的偏差量。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    # 实际与命令的角速度差值
    speed_yaw = asset.data.root_ang_vel_w[:, 2]
    command_speed_yaw = command[:, 2]
    delta_speed = torch.abs(speed_yaw - command_speed_yaw)
    absolute_command = torch.abs(command_speed_yaw)
    # 核心惩罚条件：偏差大且相对于命令值比例大
    punish = delta_speed * (delta_speed > 0.1) * (delta_speed > 0.5 * absolute_command)
    # 仅在站立转向时激活
    punish *= torch.norm(command[:, :2], dim=1) < 0.1
    punish *= torch.abs(command[:, 2]) > 0.1
    return punish


# ============================================================================================
# 4. 基座姿态奖励
# ============================================================================================

def flat_orientation_exp(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """躯干水平姿态奖励（指数核）。

    使用投影重力在 xy 平面的分量来衡量躯干的倾斜程度：
      - 完美水平时：projected_gravity = [0, 0, -9.81]，xy 分量 ≈ 0，奖励 ≈ 1
      - 倾斜时：xy 分量增大，奖励减小

    奖励函数：reward = exp(-||gravity_xy|| / σ²)

    Args:
        env: RL 环境实例。
        std: 指数核的标准差 σ。
        asset_cfg: 机器人资产配置。

    Returns:
        奖励张量 (num_envs,)，值域 [0, 1]。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # projected_gravity_b[:, :2] 是重力在机器人机体坐标系 xy 平面上的投影
    # 当机器人水平时，重力完全沿 z 轴，xy 投影为 0
    return torch.exp(-torch.norm(asset.data.projected_gravity_b[:, :2], dim=1) / std**2)


# ============================================================================================
# 5. 足部接触与姿态奖励
# ============================================================================================

def feet_contact_orientation_exp(
    env: ManagerBasedRLEnv, std: float,
    sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """足端着地时保持水平的奖励。

    对于着地的脚掌，奖励其保持水平（即脚掌局部 z 轴在世界坐标系中指向正上方）。
    仅在脚掌接地时（接触力 > 0.5N）计算惩罚。

    计算逻辑：
      1. 从接触传感器获取每个脚掌是否着地。
      2. 获取脚掌的四元数姿态，计算局部 z 轴在世界坐标系中的方向。
      3. 计算 z_world 的 xy 分量（理想情况下为 0，即完全垂直向上）。
      4. 对着地的脚掌施加水平偏差惩罚。

    Args:
        env: RL 环境实例。
        std: 指数核的标准差。
        sensor_cfg: 接触传感器配置，用于判断脚掌是否着地。
        asset_cfg: 机器人资产配置，必须恰好包含 2 个 body（左右脚掌）。

    Returns:
        惩罚张量 (num_envs,)，值越大表示脚掌越不平。
    """
    assert len(asset_cfg.body_ids) == 2

    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 获取历史接触力，沿时间维度取最大值，判断是否着地（阈值 0.5N）
    net_contact_forces = contact_sensor.data.net_forces_w_history
    feet_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids, :], dim=-1), dim=1)[0] > 0.5
    # 获取脚掌在世界坐标系中的姿态四元数 (num_envs, 2, 4)
    feet_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    # 局部 z 轴方向：[0, 0, 1]
    z_local = torch.tensor([0.0, 0.0, 1.0], device=env.device)
    num_envs = feet_quat.shape[0]
    z_world = torch.zeros(num_envs, 2, 3, device=env.device)
    # 将局部 z 轴旋转到世界坐标系
    for i in range(2):
        z_world[:, i, :] = math_utils.quat_apply(feet_quat[:, i, :], z_local.unsqueeze(0).expand(num_envs, -1))

    # 水平偏差 = z_world 在 xy 平面的投影长度（理想为 0）
    horizontal_deviation = torch.norm(z_world[:, :, :2], dim=2)  # (num_envs, 2)
    # 仅对着地的脚掌计算惩罚
    punishment = horizontal_deviation / std**2
    punishment = punishment * feet_contact.float()
    return torch.sum(punishment, dim=1)


def feet_contact_orientation_error(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """Penalize support-foot sole normals that do not align with contact forces."""
    if len(sensor_cfg.body_ids) != len(asset_cfg.body_ids):
        raise ValueError("sensor_cfg and asset_cfg must select the same feet")

    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    force_norm = torch.norm(contact_forces_w, dim=-1)
    feet_contact = force_norm > force_threshold
    contact_normal_w = contact_forces_w / force_norm.unsqueeze(-1).clamp(min=1.0e-6)

    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    sole_normal_b = torch.zeros_like(contact_normal_w)
    sole_normal_b[..., 2] = 1.0
    sole_normal_w = math_utils.quat_apply(foot_quat_w, sole_normal_b)
    alignment_error = 1.0 - torch.sum(sole_normal_w * contact_normal_w, dim=-1).clamp(-1.0, 1.0)
    orientation_error = alignment_error * feet_contact
    num_contact_feet = feet_contact.sum(dim=1).clamp_min(1)
    return orientation_error.sum(dim=1) / num_contact_feet


def pre_touchdown_feet_flat_error(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sole_points: tuple[tuple[float, float, float], ...] = (
        (0.105, 0.030, -0.050),
        (0.105, -0.030, -0.050),
        (-0.055, 0.030, -0.050),
        (-0.055, -0.030, -0.050),
    ),
    near_ground_height: float = 0.08,
    height_span_std: float = 0.008,
    contact_force_threshold: float = 1.0,
    descending_velocity_threshold: float = -0.02,
    ground_height: float = 0.0,
    command_name: str = "base_velocity",
    linear_command_threshold: float = 0.2,
    angular_command_threshold: float = 0.2,
) -> torch.Tensor:
    """Measure sole tilt immediately before touchdown under a motion command.

    The sole points are fixed in each foot-link frame. Their heights and velocities
    are evaluated in the world gravity frame so torso pitch and roll do not change
    the definition of a level foot. A negative term weight turns this non-negative
    error into a penalty without rewarding prolonged swing or extra foot lifts.
    """
    if len(sensor_cfg.body_ids) != len(asset_cfg.body_ids):
        raise ValueError("sensor_cfg and asset_cfg must select the same feet")
    if height_span_std <= 0.0:
        raise ValueError("height_span_std must be positive")

    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    num_envs, num_feet = foot_pos_w.shape[:2]
    num_points = len(sole_points)

    points_local = torch.tensor(sole_points, device=env.device, dtype=foot_pos_w.dtype)
    points_local = points_local.view(1, 1, num_points, 3).expand(num_envs, num_feet, -1, -1)
    foot_quat_expanded = foot_quat_w.unsqueeze(2).expand(-1, -1, num_points, -1)
    point_offsets_w = math_utils.quat_apply(
        foot_quat_expanded.reshape(-1, 4),
        points_local.reshape(-1, 3),
    ).view(num_envs, num_feet, num_points, 3)
    point_pos_w = foot_pos_w.unsqueeze(2) + point_offsets_w
    point_heights = point_pos_w[..., 2] - ground_height

    height_span = point_heights.amax(dim=-1) - point_heights.amin(dim=-1)
    min_height, lowest_point_ids = point_heights.min(dim=-1)

    foot_lin_vel_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]
    foot_ang_vel_w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
    point_vel_w = foot_lin_vel_w.unsqueeze(2) + torch.cross(
        foot_ang_vel_w.unsqueeze(2).expand_as(point_offsets_w),
        point_offsets_w,
        dim=-1,
    )
    lowest_point_vel_z = torch.gather(
        point_vel_w[..., 2],
        dim=-1,
        index=lowest_point_ids.unsqueeze(-1),
    ).squeeze(-1)

    contact_forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    no_contact = torch.norm(contact_forces_w, dim=-1) < contact_force_threshold
    near_ground = min_height < near_ground_height
    descending = lowest_point_vel_z < descending_velocity_threshold

    command = env.command_manager.get_command(command_name)
    linear_motion = torch.norm(command[:, :2], dim=1) > linear_command_threshold
    turning_motion = torch.abs(command[:, 2]) > angular_command_threshold
    moving_command = (linear_motion | turning_motion).unsqueeze(1)

    active = no_contact & near_ground & descending & moving_command
    flatness_error = 1.0 - torch.exp(-torch.square(height_span / height_span_std))
    active_error = flatness_error * active
    active_feet = active.sum(dim=1)
    return active_error.sum(dim=1) / active_feet.clamp_min(1)


def pre_touchdown_foot_horizontal_speed_error(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sole_points: tuple[tuple[float, float, float], ...] = (
        (0.105, 0.030, -0.050),
        (0.105, -0.030, -0.050),
        (-0.055, 0.030, -0.050),
        (-0.055, -0.030, -0.050),
    ),
    activation_height: float = 0.05,
    full_weight_height: float = 0.02,
    horizontal_speed_tolerance: float = 0.20,
    horizontal_speed_scale: float = 0.40,
    max_error: float = 4.0,
    contact_force_threshold: float = 1.0,
    descending_velocity_threshold: float = -0.02,
    ground_height: float = 0.0,
    command_name: str = "base_velocity",
    linear_command_threshold: float = 0.2,
    angular_command_threshold: float = 0.2,
) -> torch.Tensor:
    """Penalize horizontal velocity of the lowest sole point just before touchdown."""
    if len(sensor_cfg.body_ids) != len(asset_cfg.body_ids):
        raise ValueError("sensor_cfg and asset_cfg must select the same feet")
    if activation_height <= full_weight_height:
        raise ValueError("activation_height must be greater than full_weight_height")
    if horizontal_speed_scale <= 0.0:
        raise ValueError("horizontal_speed_scale must be positive")

    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    num_envs, num_feet = foot_pos_w.shape[:2]
    num_points = len(sole_points)

    points_local = torch.tensor(sole_points, device=env.device, dtype=foot_pos_w.dtype)
    points_local = points_local.view(1, 1, num_points, 3).expand(num_envs, num_feet, -1, -1)
    foot_quat_expanded = foot_quat_w.unsqueeze(2).expand(-1, -1, num_points, -1)
    point_offsets_w = math_utils.quat_apply(
        foot_quat_expanded.reshape(-1, 4),
        points_local.reshape(-1, 3),
    ).view(num_envs, num_feet, num_points, 3)
    point_pos_w = foot_pos_w.unsqueeze(2) + point_offsets_w
    point_heights = point_pos_w[..., 2] - ground_height
    min_height, lowest_point_ids = point_heights.min(dim=-1)

    foot_lin_vel_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]
    foot_ang_vel_w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
    point_vel_w = foot_lin_vel_w.unsqueeze(2) + torch.cross(
        foot_ang_vel_w.unsqueeze(2).expand_as(point_offsets_w),
        point_offsets_w,
        dim=-1,
    )
    lowest_point_index = lowest_point_ids.unsqueeze(-1).unsqueeze(-1)
    lowest_point_vel_w = torch.gather(
        point_vel_w,
        dim=2,
        index=lowest_point_index.expand(-1, -1, 1, 3),
    ).squeeze(2)

    horizontal_speed = torch.norm(lowest_point_vel_w[..., :2], dim=-1)
    excess_speed = (horizontal_speed - horizontal_speed_tolerance).clamp_min(0.0)
    speed_error = torch.square(excess_speed / horizontal_speed_scale).clamp(max=max_error)
    proximity = (
        (activation_height - min_height) / (activation_height - full_weight_height)
    ).clamp(0.0, 1.0)

    contact_forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    no_contact = torch.norm(contact_forces_w, dim=-1) < contact_force_threshold
    near_ground = min_height < activation_height
    descending = lowest_point_vel_w[..., 2] < descending_velocity_threshold

    command = env.command_manager.get_command(command_name)
    linear_motion = torch.norm(command[:, :2], dim=1) > linear_command_threshold
    turning_motion = torch.abs(command[:, 2]) > angular_command_threshold
    moving_command = (linear_motion | turning_motion).unsqueeze(1)

    active = no_contact & near_ground & descending & moving_command
    active_error = speed_error * proximity * active
    active_feet = active.sum(dim=1)
    return active_error.sum(dim=1) / active_feet.clamp_min(1)


def feet_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """足端接触模式奖励：鼓励双足机器人保持合理的支撑模式。

    奖励逻辑（三者满足其一即给奖励 1.0）：
      1. 单足支撑（恰好一只脚着地）— 行走的自然状态。
      2. 双足着地时间 < 0.2 秒 — 允许短暂的换脚过渡。
      3. 速度命令 ≈ 0 — 站立时允许双足着地。

    这种设计符合双足行走的生物力学规律：大部分时间是单腿支撑+摆动，
    仅在换脚时短暂双支撑，站立时可以双足着地。

    Args:
        env: RL 环境实例。
        sensor_cfg: 接触传感器配置。

    Returns:
        奖励张量 (num_envs,)，1.0 表示合理接触模式，0.0 表示不合理。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 获取历史接触力并判断着地状态
    net_contact_forces = contact_sensor.data.net_forces_w_history
    feet_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids, :], dim=-1), dim=1)[0] > 0.5

    # 双足都着地时累计维持时间，否则重置为 0
    both_feet_contact = torch.all(feet_contact, dim=1)
    env.unwrapped.feet_both_contact_time[both_feet_contact] += env.step_dt
    env.unwrapped.feet_both_contact_time[~both_feet_contact] = 0

    # 精确单足着地
    single_feet_contact = torch.sum(feet_contact, dim=1) == 1
    command = env.command_manager.get_command("base_velocity")
    # 三种合理情况取逻辑或
    reward = (
        single_feet_contact
        | (env.feet_both_contact_time < 0.2)  # 短时间双支撑（换脚过渡）
        | ((torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) < 0.1)  # 站立时双足着地也 OK
    ).float()
    return reward


def feet_clearance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_feet_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sole_points: tuple[tuple[float, float, float], ...] = (
        (0.105, 0.030, -0.050),
        (0.105, -0.030, -0.050),
        (-0.055, 0.030, -0.050),
        (-0.055, -0.030, -0.050),
    ),
    target_feet_height: float = 0.08,
    height_scale: float = 0.04,
    contact_force_threshold: float = 10.0,
    ground_height: float = 0.0,
    command_name: str = "base_velocity",
    linear_command_threshold: float = 0.1,
    angular_command_threshold: float = 0.3,
    max_error: float = 4.0,
) -> torch.Tensor:
    """纯旋转时摆动脚离地高度惩罚。

    仅在纯旋转命令下激活，未接触地面的脚被视为摆动脚。如果脚掌四角最低点
    低于目标离地高度，就产生非负惩罚；配置中应使用负 weight。
    """
    if len(sensor_cfg.body_ids) != len(asset_feet_cfg.body_ids):
        raise ValueError("sensor_cfg and asset_feet_cfg must select the same feet")
    if height_scale <= 0.0:
        raise ValueError("height_scale must be positive")

    asset_feet: Articulation = env.scene[asset_feet_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    feet_positions = asset_feet.data.body_pos_w[:, asset_feet_cfg.body_ids, :]
    feet_quat = asset_feet.data.body_quat_w[:, asset_feet_cfg.body_ids, :]
    num_envs, num_feet = feet_positions.shape[:2]
    num_points = len(sole_points)

    points_local = torch.tensor(sole_points, device=env.device, dtype=feet_positions.dtype)
    points_local = points_local.view(1, 1, num_points, 3).expand(num_envs, num_feet, -1, -1)
    feet_quat_expanded = feet_quat.unsqueeze(2).expand(-1, -1, num_points, -1)
    point_offsets_w = math_utils.quat_apply(
        feet_quat_expanded.reshape(-1, 4),
        points_local.reshape(-1, 3),
    ).view(num_envs, num_feet, num_points, 3)
    point_heights = (feet_positions.unsqueeze(2) + point_offsets_w)[..., 2] - ground_height
    min_sole_height = point_heights.amin(dim=-1)

    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = torch.linalg.vector_norm(contact_forces, dim=-1).amax(dim=1) > contact_force_threshold

    command = env.command_manager.get_command("base_velocity")
    not_walk = torch.norm(command[:, :2], dim=1) < linear_command_threshold
    turning = torch.abs(command[:, 2]) > angular_command_threshold
    pure_turn = (not_walk & turning).unsqueeze(1)

    swing_feet = (~in_contact) & pure_turn
    height_error = ((target_feet_height - min_sole_height) / height_scale).clamp(min=0.0, max=max_error)
    penalty = torch.square(height_error) * swing_feet
    active_feet = swing_feet.sum(dim=1)
    return penalty.sum(dim=1) / active_feet.clamp_min(1)


# ============================================================================================
# 6. 腾空时间相关奖励
# ============================================================================================

def feet_air_time_positive_biped(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """双足机器人腾空时间正奖励：鼓励合理的单腿支撑时间。

    核心思想：
      - 对于双足机器人，行走时大部分时间应该是单腿支撑（一条腿着地，另一条摆动）。
      - 记录每条腿处于当前模式（着地/腾空）的持续时间。
      - 取两条腿中较短的单支撑时间作为奖励（确保两腿行为对称）。
      - 奖励值上限为 threshold 秒，超过部分不再增加。

    Args:
        env: RL 环境实例。
        threshold: 腾空/着地时间奖励上限（秒）。超过此值不再增加奖励。
        sensor_cfg: 接触传感器配置。

    Returns:
        奖励张量 (num_envs,)，值域 [0, threshold]。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 获取当前的腾空时间和着地时间
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    # 判断每条腿处于着地模式还是腾空模式
    in_contact = contact_time > 0.0
    # 取当前模式的持续时间（着地 → contact_time，腾空 → air_time）
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    # 仅在单足支撑时计算奖励
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    # 取两条腿中较短的当前模式时间（鼓励对称步态）
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # 零速度命令时不给奖励（站立不需要摆动）
    command = env.command_manager.get_command("base_velocity")
    reward *= (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) > 0.1
    return reward


def feet_air_time_punish(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """足端模式时间惩罚：惩罚摆动时间过长或者地时间过短。

    与 feet_air_time_positive_biped 的正向奖励逻辑相反，这里采用惩罚方式：
      - 着地模式时：如果着地时间超过 threshold，不惩罚；没超过 threshold 则惩罚不足部分。
      - 腾空模式时：如果腾空时间没超过 threshold，不惩罚；超过 threshold 则惩罚超出部分。

    设计目的：确保足端摆动节奏合理，既不过快换脚也不过慢。

    Args:
        env: RL 环境实例。
        threshold: 目标模式持续时间（秒）。
        sensor_cfg: 接触传感器配置。

    Returns:
        惩罚张量 (num_envs,)。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    # 非单支撑模式时，用 threshold 替代（不产生惩罚）
    in_mode_time = torch.where(single_stance.unsqueeze(-1), in_mode_time, threshold)
    # 着地时惩罚不足 threshold，腾空时惩罚超过 threshold
    punish = torch.where(in_contact, (threshold - in_mode_time).clamp(min=0.0), (in_mode_time - threshold).clamp(min=0.0))
    punish = torch.sum(punish, dim=1)
    # 过滤小的数值噪声
    punish *= punish > 0.05
    command = env.command_manager.get_command("base_velocity")
    punish *= (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) > 0.1
    return punish


# ============================================================================================
# 7. 足部物理行为奖励
# ============================================================================================

def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """足端滑动惩罚：惩罚着地脚掌的水平滑动速度。

    只有当脚掌着地（接触力 > 1.0N）时才计算其水平滑动速度。
    着地时脚掌应该保持静止（相对于地面），滑动意味着打滑或拖地。

    Args:
        env: RL 环境实例。
        sensor_cfg: 接触传感器配置。
        asset_cfg: 机器人资产配置。

    Returns:
        惩罚张量 (num_envs,)，值为着地脚掌的水平速度范数之和。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 判断脚掌是否着地（历史最大接触力 > 1.0N）
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    # 获取脚掌在世界坐标系中的线速度（仅 xy 水平分量）
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # 滑动惩罚 = 各着地脚掌的水平速度 × 着地标志
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """足端绊倒/刮擦检测：检测脚掌是否受到过大的水平力。

    判断条件：水平接触力 > 5 × 垂直接触力
    这种情况通常表示脚掌碰到了障碍物或刮擦地面，是步态不平衡的信号。

    Args:
        env: RL 环境实例。
        sensor_cfg: 接触传感器配置。

    Returns:
        布尔张量 (num_envs,)，True 表示检测到绊倒。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def feet_impact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """足端冲击检测：检测脚掌着地时是否产生过大的水平冲击力。

    判断条件：水平力 > 0.5 × 垂直力
    正常的着地应该是垂直力为主，水平力过大说明着地角度不对或步态异常。

    Args:
        env: RL 环境实例。
        sensor_cfg: 接触传感器配置。

    Returns:
        布尔张量 (num_envs,)，True 表示检测到异常冲击。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    horizontal_force = torch.norm(feet_contact[:, :, :2], dim=-1)
    vertical_force = torch.abs(feet_contact[:, :, 2])
    impact = horizontal_force > 0.5 * vertical_force
    return torch.any(impact, dim=1)


# ============================================================================================
# 8. 足端力惩罚 — 控制接触力大小
# ============================================================================================

def normal_feet_force(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    """法向足端力惩罚（通用版本）：惩罚过大的足端垂直接触力。

    计算逻辑：
      1. 取两只脚掌垂直接触力的最大值。
      2. 如果最大力 < threshold，不惩罚（= 0）。
      3. 如果最大力 > threshold，惩罚 = min(最大力 - threshold, max_reward)。
      4. 对惩罚做 clamp(0, max_reward)。

    设计意图：过大的接触力意味着着地过猛，浪费能量且可能损伤硬件。

    Args:
        env: RL 环境实例。
        sensor_cfg: 接触传感器配置。
        threshold: 阈值（N），超过此值开始惩罚。
        max_reward: 惩罚上限（N），防止单次过大惩罚主导训练。

    Returns:
        惩罚张量 (num_envs,)。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 取两只脚掌垂直接触力（z 分量）的最大值
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].max(dim=-1)[0]
    # 小于阈值不惩罚
    reward[reward < threshold] = 0
    # 超出阈值的部分
    reward[reward > threshold] -= threshold
    # 限制惩罚上限
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def feet_force(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500,
    max_reward: float = 400, max_single_foot_force: float = 500.0,
) -> torch.Tensor:
    """足端力惩罚（增强版）：惩罚过大的总接触力，并过滤异常冲击帧。

    与 normal_feet_force 的区别：
      - 使用两只脚力的绝对值之和（而非最大值）。
      - 如果任一单脚力超过 max_single_foot_force，该帧不惩罚
        （避免空中坠落等异常帧产生极大力误导训练）。

    Args:
        env: RL 环境实例。
        sensor_cfg: 接触传感器配置。
        threshold: 总力阈值（N），超过此值开始惩罚。
        max_reward: 惩罚上限（N）。
        max_single_foot_force: 单脚最大允许力（N）。任一腳超过此值则该帧惩罚归零。

    Returns:
        惩罚张量 (num_envs,)。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_force_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    # 检测异常大冲击帧（空中坠落等），任一单脚力超过上限即标记
    invalid = (feet_force_z.abs() > max_single_foot_force).any(dim=1)
    # 计算总垂直接触力（绝对值之和）
    total_force = feet_force_z.abs().sum(dim=1)
    # 惩罚超出 threshold 的部分，但不超过 max_reward
    excess = torch.clamp(total_force - threshold, min=0.0)
    reward = excess.clamp(max=max_reward)
    # 屏蔽异常大冲击帧
    reward = torch.where(invalid, 0.0, reward)
    return reward


def feet_force_run(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    """跑步模式足端力惩罚：仅在跑步时激活的足力控制。

    与 normal_feet_force 的区别：
      - 额外的力上限保护：如果单脚力 > 1000N，奖励归零（过滤异常帧）。
      - 仅在目标线速度 > 0.1 m/s（跑步中）时激活。

    Args:
        env: RL 环境实例。
        sensor_cfg: 接触传感器配置。
        threshold: 阈值（N），超过此值开始惩罚。
        max_reward: 惩罚上限（N）。

    Returns:
        惩罚张量 (num_envs,)。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].max(dim=-1)[0]
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    # 过大力保护：> 1000N 可能是异常帧，不惩罚
    reward[reward > 1000] = 0
    reward = reward.clamp(min=0, max=max_reward)
    # 仅在走路/跑步（有前进命令）时激活
    command = env.command_manager.get_command("base_velocity")
    reward *= torch.norm(command[:, :2], dim=1) > 0.1
    return reward


# ============================================================================================
# 9. 足间距离控制
# ============================================================================================

def feet_too_near_humanoid(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    """双脚过近惩罚（人形机器人，仅前进行走时）。

    计算逻辑：
      1. 将双足位置从世界坐标系转换到机体坐标系。
      2. 计算机体坐标系下双足的 y 方向距离（左右间距）。
      3. 仅在直线前进（无转向，无侧移）时激活惩罚。
      4. 惩罚 = (threshold - 实际距离)，越近惩罚越大。

    设计意图：防止走路时双脚交叉或间距过小，保持稳定的支撑面。

    Args:
        env: RL 环境实例。
        asset_cfg: 机器人资产配置，必须恰好包含 2 个脚掌 body。
        threshold: 最小允许间距（m）。距离小于此值时开始惩罚。

    Returns:
        惩罚张量 (num_envs,)。
    """
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    # 获取双脚世界位置和基座位姿
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    base_pos_w = asset.data.root_pos_w
    base_quat_w = asset.data.root_quat_w
    # 脚掌相对基座的位置 → 转换到机体坐标系
    feet_pos_rel_w = feet_pos_w - base_pos_w.unsqueeze(1)
    feet_pos_b_0 = math_utils.quat_apply_inverse(base_quat_w, feet_pos_rel_w[:, 0])
    feet_pos_b_1 = math_utils.quat_apply_inverse(base_quat_w, feet_pos_rel_w[:, 1])
    # y 方向距离（左右间距）
    distance = torch.abs(feet_pos_b_0[:, 1] - feet_pos_b_1[:, 1])
    # 仅在前进行走（无转向、无侧移）时生效
    command = env.command_manager.get_command("base_velocity")
    is_forward = torch.abs(command[:, 2]) < 1e-6  # 无转向命令
    not_side = torch.abs(command[:, 1]) < 1e-6  # 无侧移命令
    reward = (threshold - distance).clamp(min=0)
    return is_forward.float() * not_side.float() * reward


def feet_too_near_humanoid_backward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    """双脚过近惩罚（后退专用版本）。

    与 feet_too_near_humanoid 类似，但仅在后退行走（vx_cmd < 0.1 且 |ωz_cmd| < 0.1）时激活。
    后退时双脚容易在不经意间交叉碰触，需要专门的惩罚来控制间距。

    Args:
        env: RL 环境实例。
        asset_cfg: 机器人资产配置。
        threshold: 最小允许间距（m）。

    Returns:
        惩罚张量 (num_envs,)。
    """
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    base_pos_w = asset.data.root_pos_w
    base_quat_w = asset.data.root_quat_w
    feet_pos_rel_w = feet_pos_w - base_pos_w.unsqueeze(1)
    feet_pos_b_0 = math_utils.quat_apply_inverse(base_quat_w, feet_pos_rel_w[:, 0])
    feet_pos_b_1 = math_utils.quat_apply_inverse(base_quat_w, feet_pos_rel_w[:, 1])
    distance = torch.abs(feet_pos_b_0[:, 1] - feet_pos_b_1[:, 1])
    # 仅在后退且无转向时生效
    command = env.command_manager.get_command("base_velocity")
    is_backward = (command[:, 0] < 0.1) & (torch.abs(command[:, 2]) < 0.1)
    reward = (threshold - distance).clamp(min=0)
    return is_backward.float() * reward


def feet_too_high_humanoid(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    """脚掌过高惩罚：惩罚脚掌离地过高。

    计算逻辑：
      1. 获取双脚在世界坐标系中的 z 坐标（高度）。
      2. 对超出 threshold 的高度部分求和作为惩罚。

    设计意图：过高的抬脚浪费能量且容易导致落地冲击过大。

    Args:
        env: RL 环境实例。
        asset_cfg: 机器人资产配置。
        threshold: 允许的最大离地高度（m）。

    Returns:
        惩罚张量 (num_envs,)。
    """
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    punish = (feet_height - threshold).clamp(min=0)
    punish = torch.sum(punish, dim=-1)
    return punish


def heels_too_near_humanoid(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.2, heel_offset: float = 0.05
) -> torch.Tensor:
    """脚跟过近惩罚：当双足脚跟水平距离过小时惩罚。

    脚跟位置 = 脚掌中心位置 + 脚掌局部坐标系下的后向偏移（-y 方向）。
    仅在 xy 平面（水平面）计算距离，忽略 z 方向（垂直）。

    设计意图：防止行走时两脚脚跟碰撞或交叉，特别是在前后站姿时。

    Args:
        env: RL 环境实例。
        asset_cfg: 机器人资产配置，必须恰好包含 2 个脚掌 body。
        threshold: 最小允许脚跟距离（m）。距离小于此值时施加惩罚。
        heel_offset: 脚跟相对脚掌中心的局部偏移量（m）。
                     正值表示向后（局部坐标系 -y 方向）。

    Returns:
        惩罚张量 (num_envs,)，距离 ≥ threshold 返回 0，否则返回正惩罚值。
    """
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取脚掌在世界坐标系中的位置和姿态
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # (num_envs, 2, 3)
    feet_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]  # (num_envs, 2, 4)

    # 定义脚跟局部偏移：在 Isaac Lab 的刚体坐标系中，x=前, y=左, z=上
    # 脚跟位于脚掌中心后方，即 -y 方向，并略微向下（-z）
    num_envs = feet_pos_w.shape[0]
    heel_offset_local = torch.tensor([0.0, -heel_offset, -0.05], device=env.device)
    heel_offset_local = heel_offset_local.unsqueeze(0).repeat(num_envs, 1)

    # 将脚跟偏移从局部坐标系变换到世界坐标系
    heel_offset_w_0 = math_utils.quat_apply(feet_quat_w[:, 0], heel_offset_local)
    heel_offset_w_1 = math_utils.quat_apply(feet_quat_w[:, 1], heel_offset_local)

    # 计算世界坐标系下的脚跟位置
    heel_pos_w_0 = feet_pos_w[:, 0, :] + heel_offset_w_0
    heel_pos_w_1 = feet_pos_w[:, 1, :] + heel_offset_w_1

    # 仅计算 xy 平面距离（忽略 z 方向）
    distance = torch.norm(heel_pos_w_0[:, :2] - heel_pos_w_1[:, :2], dim=-1)

    return (threshold - distance).clamp(min=0)


def feet_normal_standing(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.03
) -> torch.Tensor:
    """站立姿态双脚前后位置惩罚。

    计算逻辑：
      1. 将双脚位置从世界坐标系转换到机体坐标系。
      2. 计算机体坐标系下双足的 x 方向距离（前后差）。
      3. 仅在站立模式（速度命令 ≈ 0）时惩罚 x 方向间距过大。

    设计意图：站立时双脚应该在同一水平线上，一只脚在前一只脚在后是不稳定的站立姿态。

    Args:
        env: RL 环境实例。
        asset_cfg: 机器人资产配置。
        threshold: 允许的最大前后间距（m）。

    Returns:
        惩罚张量 (num_envs,)。
    """
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    base_pos_w = asset.data.root_pos_w
    base_quat_w = asset.data.root_quat_w
    feet_pos_rel_w = feet_pos_w - base_pos_w.unsqueeze(1)
    feet_pos_b_0 = math_utils.quat_apply_inverse(base_quat_w, feet_pos_rel_w[:, 0])
    feet_pos_b_1 = math_utils.quat_apply_inverse(base_quat_w, feet_pos_rel_w[:, 1])
    # 机体坐标系下，双脚 x 方向位置差（前后交错）
    distance = torch.abs(feet_pos_b_0[:, 0] - feet_pos_b_1[:, 0])

    # 仅在站立模式（无速度输入）下生效
    command = env.command_manager.get_command("base_velocity")
    is_standing = (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) < 0.01

    punish = (distance - threshold).clamp(min=0)
    return is_standing.float() * punish


# ============================================================================================
# 10. 关节正则化奖励 — 鼓励节能、平滑和安全的关节运动
# ============================================================================================

def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """关节偏差 L1 惩罚（始终激活）：惩罚关节偏离默认姿态。

    奖励 = Σ|joint_pos_i - default_pos_i|
    即各关节当前角度与默认姿态的绝对偏差之和。

    设计意图：鼓励机器人保持默认姿态（通常是最节能、最安全的姿态）。

    Args:
        env: RL 环境实例。
        asset_cfg: 机器人资产配置。

    Returns:
        惩罚张量 (num_envs,)。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def joint_deviation_l1_with_zero_flag(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """关节偏差 L1 惩罚（带零速标志）：在特定条件下豁免惩罚。

    豁免条件（满足其一即不惩罚 zero_flag = False）：
      1. 有非零的转向或侧移命令（||cmd[1:]|| > 0.1）。
      2. 完全静止命令（||cmd|| < 0.1，即站立）。

    惩罚条件（zero_flag = True）：
      仅有前进命令但无转向/侧移命令时。
      此时惩罚关节偏离默认姿态，鼓励保持直线行走的标准姿态。

    设计意图：前进时保持标准姿态，但转向/侧移/站立时允许关节自由调整。

    Args:
        env: RL 环境实例。
        asset_cfg: 机器人资产配置。

    Returns:
        惩罚张量 (num_envs,)。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    command = env.command_manager.get_command("base_velocity")
    # zero_flag = True 表示"不能豁免，需要惩罚"
    # 有侧移/转向命令或完全静止 → zero_flag = False（豁免）
    zero_flag = torch.logical_or(
        torch.norm(command[:, 1:], dim=1) > 0.1,  # 有侧移或转向命令
        (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) < 0.1,  # 完全静止
    )
    return torch.sum(torch.abs(angle), dim=1) * ~zero_flag


def joint_deviation_l1_when_standing(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """站立时关节偏差 L1 惩罚：仅在静止站立时惩罚关节偏离默认姿态。

    激活条件：||cmd|| < 0.1（速度命令接近零）。

    设计意图：站立时应该保持标准姿态，不要在静止时有多余的关节动作。

    Args:
        env: RL 环境实例。
        asset_cfg: 机器人资产配置。

    Returns:
        惩罚张量 (num_envs,)。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    command = env.command_manager.get_command("base_velocity")
    zero_flag = (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) < 0.1
    return torch.sum(torch.abs(angle), dim=1) * zero_flag


def joint_deviation_l1_when_notwalk(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize selected joints when no translation is commanded."""
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    command = env.command_manager.get_command("base_velocity")
    not_walk = torch.norm(command[:, :2], dim=1) < 0.1
    return torch.sum(torch.abs(angle), dim=1) * not_walk


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """能量消耗惩罚：惩罚高力矩高速度的关节运动。

    能量 = Σ|τ_i × ω_i|（各关节力矩绝对值与速度绝对值乘积之和）。
    这是一个简化的机械功率度量。

    设计意图：鼓励节能的步态和动作，避免不必要的肌肉紧张。

    Args:
        env: RL 环境实例。
        asset_cfg: 机器人资产配置。

    Returns:
        惩罚张量 (num_envs,)。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """关节加速度 L2 惩罚：惩罚过大的关节加速度。

    奖励 = Σ(joint_acc_i²)

    设计意图：鼓励平滑的关节运动轨迹，避免抖动和突变。
    加速度大意味着力矩变化剧烈，对电机和减速器不利。

    Args:
        env: RL 环境实例。
        asset_cfg: 机器人资产配置。

    Returns:
        惩罚张量 (num_envs,)。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """非期望接触惩罚：惩罚除了脚踝以外的身体部位接触地面。

    计算逻辑：
      1. 从接触传感器获取各身体部位的历史最大接触力。
      2. 统计接触力 > threshold 的身体部位数量。
      3. 通常配置 sensor_cfg 只监控非脚踝部位。

    设计意图：机器人只能用脚着地，膝盖、髋部、躯干等部位触地意味着摔倒或姿态异常。

    Args:
        env: RL 环境实例。
        threshold: 接触力阈值（N），超过此值视为接触。
        sensor_cfg: 接触传感器配置（通常配置为排除脚踝的身体部位）。

    Returns:
        惩罚张量 (num_envs,)，值为非期望接触的身体部位数量。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


# ============================================================================================
# 11. 步态相位系统 — 基于步态相位的周期奖励
#
# 步态相位（gait phase）是描述周期性运动（如行走）中当前所处阶段的值，范围 [0, 1)：
#   - phase ∈ [0, r): 摆动相（脚在空中）
#   - phase ∈ [r, 1): 支撑相（脚在地上）
#   其中 r = air_ratio（腾空比例）
#
# 以下函数实现了基于步态相位的梯形时钟信号和相应的足力、速度、摆腿高度奖励。
# ============================================================================================

def _gait_clock(
    num_envs: int, device: torch.device, gait_phase: torch.Tensor, phase_ratio: torch.Tensor, deltT: float = 0.02
) -> tuple[torch.Tensor, torch.Tensor]:
    """生成步态相位的梯形时钟信号。

    根据步态相位生成平滑的梯形时钟信号，用于驱动基于相位的控制：

    - I_frc（力时钟）: 摆动相 = 1.0，支撑相 = 0.0（带平滑过渡）
    - I_spd（速度时钟）: I_frc 的互补信号，支撑相 = 1.0，摆动相 = 0.0

    相位区域定义（deltT 为过渡时间，默认 0.02，即相位的 2%）：
      - [0, deltT):              过渡区 1（支撑末 → 摆动初）
      - [deltT, r-deltT):        摆动相中段（I_frc = 1.0）
      - [r-deltT, r+deltT):      过渡区 2（摆动末 → 支撑初）
      - [r+deltT, 1-deltT):      支撑相中段（I_frc = 0.0）
      - [1-deltT, 1):            过渡区 3（支撑末 → 下一周期摆动初）

    每个过渡区使用线性插值实现平滑切换，避免阶跃变化导致的力矩突变。

    Args:
        num_envs: 环境数量。
        device: 计算设备。
        gait_phase: 步态相位张量 (num_envs,)，值域 [0, 1)，其中：
                    - [0, r): 摆动相（脚在空中）
                    - [r, 1): 支撑相（脚在地面）
        phase_ratio: 相位比例（腾空比例/摆动比例）张量 (num_envs,)，值域 [0, 1]。
                     表示步态周期中摆动相所占的比例。
        deltT: 相位过渡时间（默认 0.02，占周期约 2%），用于平滑过渡。

    Returns:
        tuple[I_frc, I_spd]:
        - I_frc: 力时钟信号 (num_envs,)，摆动相为 1.0，支撑相为 0.0
        - I_spd: 速度时钟信号 (num_envs,)，支撑相为 1.0，摆动相为 0.0
    """
    phi = gait_phase  # (num_envs,)
    r = phase_ratio  # (num_envs,)

    I_frc = torch.zeros(num_envs, dtype=torch.float, device=device, requires_grad=False)

    # 定义相位区域标志
    swing_flag = (phi >= deltT) & (phi <= (r - deltT))  # 摆动相中段
    stand_flag = (phi >= (r + deltT)) & (phi <= (1 - deltT))  # 支撑相中段

    # 过渡区标志
    trans_flag1 = phi < deltT  # 过渡区 1：phi ∈ [0, deltT)，从支撑末到摆动初
    trans_flag2 = (phi > (r - deltT)) & (phi < (r + deltT))  # 过渡区 2：从摆动到支撑
    trans_flag3 = phi > (1 - deltT)  # 过渡区 3：phi ∈ [1-deltT, 1)，从支撑末到下一周期

    # 过渡区 2 的线性插值：从摆动到支撑
    # 公式：-(phi - r - deltT) / (2 * deltT)
    # 在 phi = r - deltT 时：-((r-deltT) - r - deltT)/(2*deltT) = 1.0（摆动末）
    # 在 phi = r + deltT 时：-((r+deltT) - r - deltT)/(2*deltT) = 0.0（支撑初）
    trans_flag2_value = -(phi - r - deltT) / (2.0 * deltT)

    # 合成最终时钟信号
    I_frc = (
        1.0 * swing_flag  # 摆动相中段：I_frc = 1.0（脚腾空，关注力控制）
        + (0.5 + phi / (2 * deltT)) * trans_flag1  # 过渡区 1：从 0.5 线性到 1.0
        + trans_flag2_value * trans_flag2  # 过渡区 2：从 1.0 线性到 0.0
        + 0.0 * stand_flag  # 支撑相中段：I_frc = 0.0（脚着地，关注速度控制）
        + ((phi - 1 + deltT) / (2 * deltT)) * trans_flag3  # 过渡区 3：从 0.0 线性到 0.5
    )

    # 速度时钟是力时钟的互补信号
    I_spd = 1.0 - I_frc

    return I_frc, I_spd


def gait_feet_frc_periodical(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_foot_frc: float = 500.0,
    force_scale: float = 10.0,
) -> torch.Tensor:
    """基于步态相位的周期性足力奖励。

    核心思想：
      - 支撑相时：足端应该承受体重（有一定接触力），奖励合理范围内的接触力。
      - 摆动相时：足端应该腾空（力为零），惩罚摆动时的不必要接触力。

    奖励计算：
      1. 将足力限制在 [0, max_foot_frc] 范围内。
      2. 使用指数衰减函数 exp(-force_scale × force²) 来奖励低力/惩罚高力。
      3. 乘以支撑相时钟（仅在支撑相时有非零时钟），实现相位耦合。

    Args:
        env: RL 环境实例（需要包含 gait_phase 和 avg_feet_force 属性）。
        sensor_cfg: 接触传感器配置。
        max_foot_frc: 足力归一化上限（N）。
        force_scale: 力惩罚的比例因子。值越大对高力越敏感。

    Returns:
        奖励张量 (num_envs,)，左右脚奖励之和。
    """
    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "gait_phase") or not hasattr(unwrapped, "avg_feet_force"):
        return torch.zeros(env.num_envs, device=env.device)

    # 获取环境维护的平均足力
    avg_feet_force = unwrapped.avg_feet_force  # (num_envs, 2)

    # 归一化足力，限制最大值
    normed_left_frc = avg_feet_force[:, 0].clamp(max=max_foot_frc)
    normed_right_frc = avg_feet_force[:, 1].clamp(max=max_foot_frc)

    # 获取左脚和右脚的支撑相时钟信号
    left_stance_clock, _ = _gait_clock(
        env.num_envs, env.device, unwrapped.gait_phase[:, 0],
        torch.full_like(unwrapped.gait_phase[:, 0], unwrapped.left_air_ratio)
    )
    right_stance_clock, _ = _gait_clock(
        env.num_envs, env.device, unwrapped.gait_phase[:, 1],
        torch.full_like(unwrapped.gait_phase[:, 1], unwrapped.right_air_ratio)
    )

    # 奖励计算：支撑相时奖励合理力，摆动相时不奖励（时钟为 0）
    left_frc_score = left_stance_clock * torch.exp(-force_scale * torch.square(normed_left_frc))
    right_frc_score = right_stance_clock * torch.exp(-force_scale * torch.square(normed_right_frc))

    return left_frc_score + right_frc_score


def gait_feet_spd_periodical(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    speed_scale: float = 200.0,
) -> torch.Tensor:
    """基于步态相位的周期性足端速度奖励。

    核心思想：
      - 支撑相时：脚掌应该保持静止（速度 ≈ 0），奖励低速。
      - 摆动相时：允许脚掌移动，但仍偏好较低速度。

    奖励计算：
      1. 获取环境维护的足端平均速度。
      2. 使用指数衰减 exp(-speed_scale × speed²) 奖励低速。
      3. 乘以速度时钟（支撑相 = 1.0，摆动相 = 0.0），实现相位耦合。

    Args:
        env: RL 环境实例（需要包含 gait_phase 和 avg_feet_speed 属性）。
        sensor_cfg: 接触传感器配置。
        asset_cfg: 机器人资产配置。
        speed_scale: 速度惩罚的比例因子。值越大对高速度越敏感。

    Returns:
        奖励张量 (num_envs,)，左右脚奖励之和。
    """
    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "gait_phase") or not hasattr(unwrapped, "avg_feet_speed"):
        return torch.zeros(env.num_envs, device=env.device)

    # 获取环境维护的足端平均速度
    avg_feet_speed = unwrapped.avg_feet_speed  # (num_envs, 2)

    # 获取速度时钟信号（支撑相 = 1.0，摆动相 = 0.0）
    _, left_spd_clock = _gait_clock(
        env.num_envs, env.device, unwrapped.gait_phase[:, 0],
        torch.full_like(unwrapped.gait_phase[:, 0], unwrapped.left_air_ratio)
    )
    _, right_spd_clock = _gait_clock(
        env.num_envs, env.device, unwrapped.gait_phase[:, 1],
        torch.full_like(unwrapped.gait_phase[:, 1], unwrapped.right_air_ratio)
    )

    # 奖励：支撑相时奖励足端静止，摆动相时不奖励
    left_spd_score = left_spd_clock * torch.exp(-speed_scale * torch.square(avg_feet_speed[:, 0]))
    right_spd_score = right_spd_clock * torch.exp(-speed_scale * torch.square(avg_feet_speed[:, 1]))

    return left_spd_score + right_spd_score


def gait_swing_up(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    swing_up_scale: float = 16.0,
) -> torch.Tensor:
    """摆动相初期的向上摆腿奖励。

    核心思想：在摆动相的前 25%，鼓励脚掌向上加速（抬脚动作）。

    计算逻辑：
      1. 计算摆动相进度 s = gait_phase / phase_ratio ∈ [0, 1]。
      2. 仅当 s < 0.25 时激活（摆动相初期）。
      3. 奖励 = swing_up_scale × s² × (向上速度)²。

    设计意图：在摆动相初期迅速抬脚，避免拖地。

    Args:
        env: RL 环境实例（需要包含 gait_phase 属性）。
        asset_cfg: 机器人资产配置。
        swing_up_scale: 摆腿奖励的比例因子。

    Returns:
        奖励张量 (num_envs,)，左右脚奖励之和。
    """
    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "gait_phase"):
        return torch.zeros(env.num_envs, device=env.device)

    asset: Articulation = env.scene[asset_cfg.name]
    # 获取脚掌在世界坐标系中的垂直速度（z 分量）
    feet_vels = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]  # (num_envs, 2)
    leftfoot_v = feet_vels[:, 0]
    rightfoot_v = feet_vels[:, 1]

    # 计算摆动相进度：gait_phase / air_ratio，限制在 [0, 1]
    left_phase_ratio = unwrapped.left_air_ratio
    right_phase_ratio = unwrapped.right_air_ratio

    s0 = torch.clamp(unwrapped.gait_phase[:, 0] / left_phase_ratio, 0.0, 1.0)
    s1 = torch.clamp(unwrapped.gait_phase[:, 1] / right_phase_ratio, 0.0, 1.0)

    # 仅在前 25% 摆动相激活
    s0 = s0 * (s0 < 0.25)
    s1 = s1 * (s1 < 0.25)
    s0 = torch.clamp(s0, 0.0, 0.2)
    s1 = torch.clamp(s1, 0.0, 0.2)

    # 奖励向上的速度（z 方向速度 > 0），使用平方使奖励更集中于有意义的向上运动
    left_score = swing_up_scale * s0 * s0 * torch.clamp(leftfoot_v, min=0.0) * torch.clamp(leftfoot_v, min=0.0)
    right_score = swing_up_scale * s1 * s1 * torch.clamp(rightfoot_v, min=0.0) * torch.clamp(rightfoot_v, min=0.0)

    return left_score + right_score


def gait_swing_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_diff_target: float = 0.02,
    height_scale: float = 25.0,
    swing_early_ratio: float = 0.3,
    reward_scale: float = 2.0,
) -> torch.Tensor:
    """摆动相初期双脚高度差奖励。

    核心思想：在摆动相的前 30%，鼓励摆动脚高于支撑脚。

    奖励计算：
      - 左脚摆动时：奖励 left_height - right_height ≈ height_diff_target
      - 右脚摆动时：奖励 right_height - left_height ≈ height_diff_target
      - 使用指数衰减：exp(-height_scale × |实际差 - 目标差|)

    设计意图：确保摆动脚有足够的离地间隙，防止两脚同时着地或高度相同。

    Args:
        env: RL 环境实例（需要包含 gait_phase 属性）。
        asset_cfg: 机器人资产配置，必须恰好包含 2 个脚掌 body。
        height_diff_target: 目标高度差（m）。理想情况下摆动脚比支撑脚高约 2cm。
        height_scale: 高度差惩罚的比例因子。
        swing_early_ratio: 摆动相前期比例（默认 0.3，即前 30%）。
        reward_scale: 整体奖励缩放因子。

    Returns:
        奖励张量 (num_envs,)，左右脚奖励之和。
    """
    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "gait_phase"):
        return torch.zeros(env.num_envs, device=env.device)

    asset: Articulation = env.scene[asset_cfg.name]
    assert len(asset_cfg.body_ids) == 2, "gait_swing_height 需要恰好 2 个脚掌 body"

    # 获取双脚在世界坐标系中的高度（z 坐标）
    feet_positions = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # (num_envs, 2, 3)
    leftfoot_height = feet_positions[:, 0, 2]
    rightfoot_height = feet_positions[:, 1, 2]

    # 获取摆动相比例
    left_phase_ratio = unwrapped.left_air_ratio
    right_phase_ratio = unwrapped.right_air_ratio

    # 计算摆动相进度
    s0 = unwrapped.gait_phase[:, 0] / left_phase_ratio
    s1 = unwrapped.gait_phase[:, 1] / right_phase_ratio

    # 仅在前 swing_early_ratio（默认 30%）的摆动相激活
    s0 = s0 * (s0 < swing_early_ratio)
    s1 = s1 * (s1 < swing_early_ratio)

    # 左脚摆动时的高度差奖励
    left_height_diff = leftfoot_height - rightfoot_height
    left_height_error = torch.abs(left_height_diff - height_diff_target)
    left_score = reward_scale * s0 * torch.exp(-height_scale * left_height_error)

    # 右脚摆动时的高度差奖励
    right_height_diff = rightfoot_height - leftfoot_height
    right_height_error = torch.abs(right_height_diff - height_diff_target)
    right_score = reward_scale * s1 * torch.exp(-height_scale * right_height_error)

    return left_score + right_score


def gait_swing_tracking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tracking_scale: float = 16.0,
) -> torch.Tensor:
    """摆动相中后期的足端速度跟踪奖励。

    核心思想：在摆动相的中后期（50%-60% 进度），鼓励脚掌产生速度（向前迈步）。

    计算逻辑：
      1. 计算摆动进度偏移：s = gait_phase / phase_ratio - 0.5，限制在 [0,1]。
      2. 仅当 s < 0.6 时激活（大约摆动相 50%-60% 阶段）。
      3. s 限制在 [0, 0.5]。
      4. 奖励 = tracking_scale × s² × clamp(v², 0, 1)。

    设计意图：在摆动相中后期加速迈步，确保足够大的步幅。

    Args:
        env: RL 环境实例（需要包含 gait_phase 属性）。
        asset_cfg: 机器人资产配置，必须恰好包含 2 个脚掌 body。
        tracking_scale: 跟踪奖励的比例因子。

    Returns:
        奖励张量 (num_envs,)，左右脚奖励之和。
    """
    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "gait_phase"):
        return torch.zeros(env.num_envs, device=env.device)

    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取脚掌垂直速度
    feet_vels = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]  # (num_envs, 2)
    leftfoot_v = feet_vels[:, 0]
    rightfoot_v = feet_vels[:, 1]

    # 获取相位比例
    left_phase_ratio = unwrapped.left_air_ratio
    right_phase_ratio = unwrapped.right_air_ratio

    # 计算 s = gait_phase / phase_ratio - 0.5，然后限制在 [0, 1]
    s0 = unwrapped.gait_phase[:, 0] / left_phase_ratio - 0.5
    s0 = torch.clamp(s0, min=0.0, max=1.0)
    s0 = s0 * (s0 < 0.6)
    s0 = torch.clamp(s0, min=0.0, max=0.5)

    s1 = unwrapped.gait_phase[:, 1] / right_phase_ratio - 0.5
    s1 = torch.clamp(s1, min=0.0, max=1.0)
    s1 = s1 * (s1 < 0.6)
    s1 = torch.clamp(s1, min=0.0, max=0.5)

    # 奖励 = tracking_scale × s² × v²（v² 限制在 [0, 1]）
    left_score = tracking_scale * s0 * s0 * torch.clamp(leftfoot_v * leftfoot_v, min=0.0, max=1.0)
    right_score = tracking_scale * s1 * s1 * torch.clamp(rightfoot_v * rightfoot_v, min=0.0, max=1.0)

    return left_score + right_score


# ============================================================================================
# 12. 基于全局相位的足端步态奖励（简化版）
# ============================================================================================

def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    """基于全局相位的足端步态奖励。

    使用全局时间生成统一的步态相位，检测脚掌接触状态是否匹配期望的步态模式。

    计算逻辑：
      1. 从 episode 时间计算全局相位：global_phase = (t % period) / period。
      2. 每条腿独立偏移：leg_phase = (global_phase + offset_i) % 1.0。
      3. 判断期望接触状态：leg_phase < threshold → 应该着地（支撑相）。
      4. 与实际接触状态比较：一致给 +1，不一致给 -1。

    如果指定了 command_name，零速度命令时将期望接触状态设为全支撑（全着地）。

    Args:
        env: RL 环境实例。
        period: 步态周期（秒）。
        offset: 每条腿的相位偏移列表，值域 [0, 1)。
        sensor_cfg: 接触传感器配置。
        threshold: 支撑相阈值（默认 0.5，即 50% 支撑，50% 摆动）。
        command_name: 可选，速度命令名称。零命令时全部期望支撑。

    Returns:
        奖励张量 (num_envs,)，每条腿 ±1 的总和。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    # 全局相位：基于 episode 时间和 period
    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)

    # 如果指定了命令名，零命令时将所有腿的期望置为支撑相
    cmd_norm = None
    if command_name is not None:
        cmd = env.command_manager.get_command(command_name)
        cmd_norm = torch.norm(cmd, dim=1)
        global_phase = torch.where((cmd_norm < 0.1).unsqueeze(1), torch.zeros_like(global_phase), global_phase)

    # 每条腿独立相位偏移
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    # 期望接触状态
    is_stance = leg_phase < threshold

    # 匹配奖励：一致 +1，不一致 -1
    reward = torch.where(is_stance == is_contact, 1.0, -1.0).sum(dim=1)
    return reward


def feet_speed_z(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), dis_threshold: float = 0.07):
    """足端离地速度惩罚（z 方向）。

    当脚掌离地高度低于 dis_threshold 时，惩罚脚掌向下的速度（避免拖地）。

    计算逻辑：
      1. 获取脚掌 z 方向速度和高度。
      2. 奖励 = 1 - exp(-10 × |向下速度|)，向下速度越大奖励越低。
      3. 仅当脚掌高度 < dis_threshold 时激活。

    Args:
        env: RL 环境实例。
        asset_cfg: 机器人资产配置。
        dis_threshold: 高度阈值（m），低于此高度时检查速度。

    Returns:
        惩罚张量 (num_envs,)。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
    body_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    # 奖励脚掌不向下运动（向下速度越大越差）
    rew_spd = 1 - torch.exp(-10.0 * torch.abs(body_vel.clip(max=0.)))
    # 仅当脚掌贴近地面时激活
    rew_spd *= body_pos < dis_threshold
    return rew_spd.sum(dim=1)


# ============================================================================================
# 13. 场景化角速度跟踪 — 根据运动模式启用不同的角速度跟踪策略
# ============================================================================================

def track_ang_vel_stand_world_exp(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lin_vel_threshold: float = 1.0
) -> torch.Tensor:
    """站立原地转向时的角速度跟踪（增强版，带线速度惩罚）。

    激活条件：vx_cmd ≈ 0 且 ωz_cmd ≠ 0（纯原地旋转命令）。

    奖励 = 角速度跟踪奖励 - lin_vel_threshold × 线速度惩罚

    与 stand_turn_ang_vel_z_world_exp 的区别：
      - 额外的线速度惩罚：转向时如果产生线速度偏移，给予惩罚。
      - 这鼓励真正的"原地"旋转而不产生意外的平移。

    Args:
        env: RL 环境实例。
        std: 指数核的标准差 σ。
        asset_cfg: 机器人资产配置。
        lin_vel_threshold: 线速度惩罚系数。

    Returns:
        奖励张量 (num_envs,)。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    # 判断是否是原地旋转（无前进命令，有转向命令）
    is_pure_rotation = (
            (torch.abs(command[:, 0]) < 1e-6) &  # vx ≈ 0
            (torch.abs(command[:, 2]) > 1e-6)  # ωz ≠ 0
    )
    # 线速度惩罚：转向时不应产生意外的平移速度
    lin_vel_penalty = torch.norm(env.robot.data.root_lin_vel_w[:, :2], dim=1)
    # 角速度跟踪奖励
    ang_vel_error = torch.square(command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    # 组合：角速度跟踪 - 线速度惩罚
    return is_pure_rotation.float() * (reward - lin_vel_threshold * lin_vel_penalty)


def track_ang_vel_backward_world_exp(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """后退时的角速度跟踪奖励。

    激活条件：
      - vx_cmd < -0.001（后退命令）
      - |vy_cmd| > 0.2（同时有侧移命令时激活）

    设计意图：后退时转向更加困难，需要专门的角速度跟踪奖励来稳定后退转向行为。

    Args:
        env: RL 环境实例。
        std: 指数核的标准差 σ。
        asset_cfg: 机器人资产配置。

    Returns:
        奖励张量 (num_envs,)。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    # 仅在后退且同时有侧向命令时启用
    is_backward = command[:, 0] < -1e-3
    is_side = torch.abs(command[:, 1]) > 0.2
    ang_vel_error = torch.square(command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    return is_backward.float() * is_side.float() * reward


def track_ang_vel_run_world_exp(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """跑步时的角速度跟踪奖励。

    激活条件：
      - |vx_cmd| > 0（有前进命令）
      - |ωz_cmd| > 0（有转向命令）

    即：跑步中同时转向时，给予角速度跟踪奖励。
    与站立转向相比，跑步中转向更难以精确控制方向。

    Args:
        env: RL 环境实例。
        std: 指数核的标准差 σ。
        asset_cfg: 机器人资产配置。

    Returns:
        奖励张量 (num_envs,)。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    # 判断是否为跑步中转向
    is_run = (
            (torch.abs(command[:, 0]) > 1e-6) &  # vx ≠ 0
            (torch.abs(command[:, 2]) > 1e-6)  # ωz ≠ 0
    )
    ang_vel_error = torch.square(command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return is_run.float() * torch.exp(-ang_vel_error / std**2)


# ============================================================================================
# 14. 特殊场景的关节速度与动作平滑惩罚
# ============================================================================================

def joint_vel_l2_when_zero_command(
    env: ManagerBasedRLEnv,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """零命令时关节速度 L2 惩罚：抑制静止站立时的关节抖动。

    当速度命令接近零（|vx| + |vy| + |ωz| < command_threshold）时，
    惩罚所有关节的速度。这有助于减少突然停止后的残余振动。

    Args:
        env: RL 环境实例。
        command_threshold: 命令阈值，低于此值视为零命令。
        asset_cfg: 机器人资产配置。

    Returns:
        惩罚张量 (num_envs,)。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")

    # 使用 |vx| + |vy| + |ωz| 作为零命令判定标准
    command_abs_sum = torch.abs(command[:, 0]) + torch.abs(command[:, 1]) + torch.abs(command[:, 2])
    zero_command_flag = command_abs_sum < command_threshold

    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    punish = torch.sum(torch.square(joint_vel), dim=1)
    return punish * zero_command_flag


def action_rate_l2_when_standing(
    env: ManagerBasedRLEnv,
    joint_ids: list[int] | None = None,
    asset_cfg: SceneEntityCfg | None = None,
    stand_cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """站立时动作变化率惩罚：仅在站立时惩罚动作的剧烈变化。

    支持指定关节子集：如果提供了 joint_ids 或 asset_cfg.joint_ids，
    则仅对指定的关节维度进行惩罚；否则对所有动作维度进行惩罚。

    设计意图：站立时应该输出稳定的控制信号，减少不必要的关节微调。

    Args:
        env: RL 环境实例。
        joint_ids: 要惩罚的关节索引列表（可选）。
        asset_cfg: 机器人资产配置（可选，用于提取 joint_ids）。
        stand_cmd_threshold: 站立判定阈值。

    Returns:
        惩罚张量 (num_envs,)。
    """
    command = env.command_manager.get_command("base_velocity")
    zero_flag = (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) < stand_cmd_threshold

    action_delta = env.action_manager.action - env.action_manager.prev_action

    # 选择要惩罚的关节维度
    selected_joint_ids = joint_ids
    if asset_cfg is not None and asset_cfg.joint_ids is not None and len(asset_cfg.joint_ids) > 0:
        selected_joint_ids = asset_cfg.joint_ids

    if selected_joint_ids is not None:
        action_delta = action_delta[:, selected_joint_ids]

    return torch.sum(torch.square(action_delta), dim=1) * zero_flag


# ============================================================================================
# 15. 后退专项奖励 — 针对后退步态的精细控制
# ============================================================================================

def waist_yaw_motion_backward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """后退时腰部偏航运动惩罚。

    当机器人后退行走（vx_cmd < -0.1）时，惩罚腰部偏航关节的速度。
    后退时腰部容易不自然地转动来补偿平衡，这种补偿应被抑制。

    Args:
        env: RL 环境实例。
        asset_cfg: 机器人资产配置（应指向 waist_yaw_joint）。

    Returns:
        惩罚张量 (num_envs,)。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")

    # 后退判定：x 命令 < -0.1
    is_backward = command[:, 0] < -0.1

    # 获取腰部偏航关节速度（单个关节）
    waist_vel = asset.data.joint_vel[:, asset_cfg.joint_ids][:, 0]

    # L1 惩罚 × 后退标志
    penalty = torch.abs(waist_vel) * is_backward

    return penalty


# ============================================================================================
# 16. 场景化足端力惩罚 — 根据运动模式（行走/侧移/转向）分别控制
# ============================================================================================

def normal_feet_force_walking(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 145, max_reward: float = 400
) -> torch.Tensor:
    """行走模式专用足端力惩罚。

    激活条件（同时满足）：
      - |vx_cmd| < 0.8（行走速度，非奔跑）
      - |ωz_cmd| < 0.1（无转向）
      - |vy_cmd| < 0.1（无侧移）

    即：仅在中低速直行时控制足力。相比通用版本阈值更低（145N vs 500N），
    因为行走时的着地力应该更轻柔。

    Args:
        env: RL 环境实例。
        sensor_cfg: 接触传感器配置。
        threshold: 阈值（N），超过此值开始惩罚。
        max_reward: 惩罚上限（N）。

    Returns:
        惩罚张量 (num_envs,)。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].max(dim=-1)[0]
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    command = env.command_manager.get_command("base_velocity")
    is_walking = torch.abs(command[:, 0]) < 0.8
    not_turning = torch.abs(command[:, 2]) < 0.1
    notside = torch.abs(command[:, 1]) < 0.1
    return reward * is_walking.float() * not_turning.float() * notside.float()


def normal_feet_force_side(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 180, max_reward: float = 200
) -> torch.Tensor:
    """侧移模式专用足端力惩罚。

    激活条件（同时满足）：
      - |vy_cmd| > 0.1（有侧移命令）
      - |ωz_cmd| < 0.1（无转向）
      - |vx_cmd| < 0.1（无前进）

    即：仅在纯侧移时控制足力。侧移时阈值 180N，比行走时稍高，
    因为侧移需要更大的支撑力来维持平衡。

    Args:
        env: RL 环境实例。
        sensor_cfg: 接触传感器配置。
        threshold: 阈值（N）。
        max_reward: 惩罚上限（N）。

    Returns:
        惩罚张量 (num_envs,)。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].max(dim=-1)[0]
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    command = env.command_manager.get_command("base_velocity")
    is_side = torch.abs(command[:, 1]) > 0.1
    not_turning = torch.abs(command[:, 2]) < 0.1
    not_walk = torch.abs(command[:, 0]) < 0.1
    return reward * is_side.float() * not_turning.float() * not_walk.float()


def normal_feet_force_turning(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 155, max_reward: float = 200
) -> torch.Tensor:
    """转向模式专用足端力惩罚。

    激活条件（同时满足）：
      - |ωz_cmd| > 0.1（有转向命令）
      - |vy_cmd| < 0.1（无侧移）
      - |vx_cmd| < 0.1（无前进）

    即：仅在原地转向时控制足力。转向时阈值 155N，
    因为转向涉及双脚交替支撑，着地力介于行走和侧移之间。

    Args:
        env: RL 环境实例。
        sensor_cfg: 接触传感器配置。
        threshold: 阈值（N）。
        max_reward: 惩罚上限（N）。

    Returns:
        惩罚张量 (num_envs,)。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].max(dim=-1)[0]
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    command = env.command_manager.get_command("base_velocity")
    is_turning = torch.abs(command[:, 2]) > 0.1
    not_side = torch.abs(command[:, 1]) < 0.1
    not_walk = torch.abs(command[:, 0]) < 0.1
    return reward * is_turning.float() * not_side.float() * not_walk.float()
