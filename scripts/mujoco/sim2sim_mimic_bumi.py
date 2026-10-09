# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

import argparse
import ast
import os
import sys
import time

import mujoco
import mujoco_viewer
import numpy as np
import onnx
import onnxruntime as ort
from pynput import keyboard


def matrix_to_quaternion_simple(matrix):
    """
    简化的矩阵转四元数实现
    """
    matrix = np.array(matrix)
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]

    trace = m00 + m11 + m22

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def subtract_frame_transforms_mujoco(quat_a, quat_b, init_to_world):
    """
    与IsaacLab中subtract_frame_transforms完全相同的实现（一维版本）
    计算从坐标系A到坐标系B的相对变换

    参数:
        pos_a: 坐标系A的位置 (3,)
        quat_a: 坐标系A的四元数 (4,) [w, x, y, z]格式
        pos_b: 坐标系B的位置 (3,)
        quat_b: 坐标系B的四元数 (4,) [w, x, y, z]格式

    返回:
        rel_pos: B相对于A的位置 (3,)
        rel_quat: B相对于A的旋转四元数 (4,) [w, x, y, z]格式
    """

    # 计算相对旋转: quat_B_to_A = quat_A^* ⊗ quat_B
    rel_quat = quaternion_multiply(matrix_to_quaternion_simple(init_to_world), quat_b)
    rel_quat = quaternion_multiply(quaternion_conjugate(quat_a), rel_quat)

    # 确保四元数归一化（与IsaacLab保持一致）
    rel_quat = rel_quat / np.linalg.norm(rel_quat)

    return rel_quat


def quaternion_conjugate(q):
    """四元数共轭: [w, x, y, z] -> [w, -x, -y, -z]"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_multiply(q1, q2):
    """四元数乘法: q1 ⊗ q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])


def yaw_quat(q):
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


ROOT_BODY_NAME = "base_link"
ANCHOR_BODY_NAME = "waist_yaw_link"

BUMI_BODY_NAMES = [
    "base_link",
    "l_leg_pitch_link",
    "r_leg_pitch_link",
    "waist_yaw_link",
    "l_leg_roll_link",
    "r_leg_roll_link",
    "l_arm_pitch_link",
    "r_arm_pitch_link",
    "l_leg_yaw_link",
    "r_leg_yaw_link",
    "l_arm_roll_link",
    "r_arm_roll_link",
    "l_knee_pitch_link",
    "r_knee_pitch_link",
    "l_arm_yaw_link",
    "r_arm_yaw_link",
    "l_ankle_pitch_link",
    "r_ankle_pitch_link",
    "l_elbow_pitch_link",
    "r_elbow_pitch_link",
    "l_ankle_roll_link",
    "r_ankle_roll_link",
]

MIMIC_BODY_NAMES = [
    "base_link",
    "l_leg_pitch_link",
    "l_leg_yaw_link",
    "l_knee_pitch_link",
    "l_ankle_roll_link",
    "r_leg_pitch_link",
    "r_leg_yaw_link",
    "r_knee_pitch_link",
    "r_ankle_roll_link",
    "waist_yaw_link",
    "l_arm_roll_link",
    "l_elbow_pitch_link",
    "r_arm_roll_link",
    "r_elbow_pitch_link",
]

mujoco_joint_index = [
    "waist_yaw_joint",
    "l_arm_pitch_joint",
    "l_arm_roll_joint",
    "l_arm_yaw_joint",
    "l_elbow_pitch_joint",
    "r_arm_pitch_joint",
    "r_arm_roll_joint",
    "r_arm_yaw_joint",
    "r_elbow_pitch_joint",
    "l_leg_pitch_joint",
    "l_leg_roll_joint",
    "l_leg_yaw_joint",
    "l_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "r_leg_pitch_joint",
    "r_leg_roll_joint",
    "r_leg_yaw_joint",
    "r_knee_pitch_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
]

isaacLab_joint_index = [
    "l_leg_pitch_joint",
    "r_leg_pitch_joint",
    "waist_yaw_joint",
    "l_leg_roll_joint",
    "r_leg_roll_joint",
    "l_arm_pitch_joint",
    "r_arm_pitch_joint",
    "l_leg_yaw_joint",
    "r_leg_yaw_joint",
    "l_arm_roll_joint",
    "r_arm_roll_joint",
    "l_knee_pitch_joint",
    "r_knee_pitch_joint",
    "l_arm_yaw_joint",
    "r_arm_yaw_joint",
    "l_ankle_pitch_joint",
    "r_ankle_pitch_joint",
    "l_elbow_pitch_joint",
    "r_elbow_pitch_joint",
    "l_ankle_roll_joint",
    "r_ankle_roll_joint",
]


def sanitize_variable_name(name):
    """将key名称转换为有效的Python变量名"""
    # 替换非法字符为下划线
    import re

    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    # 确保不以数字开头
    if name[0].isdigit():
        name = "_" + name
    return name.lower()  # 可选：转换为小写


def csv_str_to_list_advanced(csv_str, *, delimiter: str = ",", try_eval: bool = True):
    """
    将 CSV 字符串转换回列表，尝试识别更多数据类型

    Args:
        csv_str: CSV 格式的字符串
        delimiter: 分隔符，默认为逗号
        try_eval: 是否尝试使用 ast.literal_eval 解析复杂数据类型

    Returns:
        list: 包含转换后元素的列表
    """
    items = csv_str.split(delimiter)
    result = []

    for item in items:
        item = item.strip()

        if try_eval:
            # 尝试使用 ast.literal_eval 解析 Python 字面量
            try:
                evaluated = ast.literal_eval(item)
                result.append(evaluated)
                continue
            except (ValueError, SyntaxError):
                pass

        # 尝试转换为数字
        if item.isdigit():
            result.append(int(item))
        else:
            try:
                # 检查是否为浮点数
                float_val = float(item)
                # 检查是否真的是浮点数（不是整数）
                if float_val != int(float_val):
                    result.append(float_val)
                else:
                    result.append(int(float_val))
            except ValueError:
                # 保持为字符串
                result.append(item)

    return result


def resolve_motion_body_index(body_count: int, body_name: str) -> int:
    """Resolve a body name against known Bumi motion body layouts."""
    for body_names in (BUMI_BODY_NAMES, MIMIC_BODY_NAMES):
        if len(body_names) == body_count and body_name in body_names:
            return body_names.index(body_name)

    if body_count == 1:
        return 0

    raise ValueError(
        f"Cannot resolve body '{body_name}' in motion data with {body_count} bodies. "
        "Expected a Bumi motion npz generated by this project."
    )


class SimToSimCfg:
    """Configuration class for sim2sim parameters.

    Must be kept consistent with the training configuration.
    """

    class sim:
        sim_duration = 100.0
        num_obs_per_step = 114  # 21 (motion_joint_pos) + 21 (motion_joint_vel) + 6 (anchor_ori) + 3 (angular_vel) + 21 (dof_pos) + 21 (dof_vel) + 21 (actions) = 114
        actor_obs_history_length = 5
        dt = 0.005
        decimation = 4
        clip_actions = 18.0


class MujocoRunner:
    """
    Sim2Sim runner that loads a policy and a MuJoCo model
    to run real-time humanoid control simulation.

    Args:
        cfg (SimToSimCfg): Configuration object for simulation.
        policy_path (str): Path to the Onnx exported policy.
        model_path (str): Path to the MuJoCo XML model.
    """

    def __init__(self, cfg: SimToSimCfg, policy_path, model_path, motion_file):
        self.cfg = cfg
        network_path = policy_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.cfg.sim.dt
        self.motion = np.load(motion_file)

        self.root_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY_NAME)
        self.anchor_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ANCHOR_BODY_NAME)
        if self.root_body_id == -1:
            raise ValueError(f"Body {ROOT_BODY_NAME} not found in model")
        if self.anchor_body_id == -1:
            raise ValueError(f"Body {ANCHOR_BODY_NAME} not found in model")

        self.motion_joint_pos = self.motion["joint_pos"]
        self.motion_joint_vel = self.motion["joint_vel"]
        motion_body_count = self.motion["body_pos_w"].shape[1]
        self.motion_root_body_index = resolve_motion_body_index(motion_body_count, ROOT_BODY_NAME)
        self.motion_anchor_body_index = resolve_motion_body_index(motion_body_count, ANCHOR_BODY_NAME)
        self.motion_root_pos = self.motion["body_pos_w"][:, self.motion_root_body_index, :]
        self.motion_root_quat = self.motion["body_quat_w"][:, self.motion_root_body_index, :]
        self.motion_root_lin_vel = self.motion["body_lin_vel_w"][:, self.motion_root_body_index, :]
        self.motion_root_ang_vel = self.motion["body_ang_vel_w"][:, self.motion_root_body_index, :]
        self.motion_anchor_quat = self.motion["body_quat_w"][:, self.motion_anchor_body_index, :]
        print(
            f"[INFO] Motion body indexes: root {ROOT_BODY_NAME}={self.motion_root_body_index}, "
            f"anchor {ANCHOR_BODY_NAME}={self.motion_anchor_body_index}"
        )
        self.onnx_model = onnx.load(network_path)
        onnx.checker.check_model(self.onnx_model)
        self.policy = ort.InferenceSession(network_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer._render_every_frame = False
        mujoco.mj_step(self.model, self.data)
        self.init_variables()

        self.ref_tau = []
        self.real_tau = []
        self.is_paused = False
        self.reset_requested = False
        self.step_once_requested = False
        self.exit_requested = False
        self.listener = None

    def init_variables(self) -> None:
        """Initialize simulation variables and joint index mappings."""
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt

        assert self.onnx_model.metadata_props
        print("Loading robot properties from onnx: ")
        for prop in self.onnx_model.metadata_props:
            attr_name = sanitize_variable_name(prop.key)
            attr_value = csv_str_to_list_advanced(prop.value)
            setattr(self, attr_name, attr_value)

        self.mujoco_to_isaac_idx = [mujoco_joint_index.index(name) for name in isaacLab_joint_index]
        self.isaac_to_mujoco_idx = [isaacLab_joint_index.index(name) for name in mujoco_joint_index]

        self.num_action = len(self.default_joint_pos)
        self.default_joint_pos = np.array(self.default_joint_pos)[self.isaac_to_mujoco_idx]
        self.joint_stiffness = np.array(self.joint_stiffness)[self.isaac_to_mujoco_idx]
        self.joint_damping = np.array(self.joint_damping)[self.isaac_to_mujoco_idx]
        self.action_scale = np.array(self.action_scale)[self.isaac_to_mujoco_idx]
        # Bumi's action scale is defined as 0.25 * effort_limit / stiffness.
        # MuJoCo's XML uses a much larger generic actuator ctrlrange, so recover
        # and enforce the Isaac actuator effort limits explicitly.
        self.effort_limit = self.action_scale * self.joint_stiffness / 0.25

        if self.motion_joint_pos.shape[1] < self.num_action:
            raise ValueError(
                f"motion_joint_pos has {self.motion_joint_pos.shape[1]} joints, expected at least {self.num_action}"
            )
        self._resolve_mujoco_indices()
        self.apply_motion_frame(0)

        self.dof_pos = np.zeros(self.num_action)
        self.dof_vel = np.zeros(self.num_action)
        self.action = np.zeros(self.num_action)

        self.episode_length_buf = 0

        self.obs_history = np.zeros(
            (self.cfg.sim.actor_obs_history_length, self.cfg.sim.num_obs_per_step), dtype=np.float32
        )
        self.history_initialized = False

    def _resolve_mujoco_indices(self) -> None:
        """Resolve joint and actuator addresses by name instead of XML order."""
        joint_qpos_ids = []
        joint_qvel_ids = []
        actuator_ids = []
        for name in mujoco_joint_index:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if joint_id < 0 or actuator_id < 0:
                raise ValueError(f"MuJoCo model is missing joint or actuator: {name}")
            joint_qpos_ids.append(self.model.jnt_qposadr[joint_id])
            joint_qvel_ids.append(self.model.jnt_dofadr[joint_id])
            actuator_ids.append(actuator_id)

        self.qpos_indices = np.asarray(joint_qpos_ids, dtype=np.int64)
        self.qvel_indices = np.asarray(joint_qvel_ids, dtype=np.int64)
        self.actuator_indices = np.asarray(actuator_ids, dtype=np.int64)

        xml_ctrl_range = self.model.actuator_ctrlrange[self.actuator_indices]
        self.control_low = np.maximum(xml_ctrl_range[:, 0], -self.effort_limit)
        self.control_high = np.minimum(xml_ctrl_range[:, 1], self.effort_limit)

    def apply_motion_frame(self, timestep: int) -> None:
        """Set MuJoCo state from a Bumi motion frame."""
        motion_q_isaac = self.motion_joint_pos[timestep, : self.num_action]
        motion_dq_isaac = self.motion_joint_vel[timestep, : self.num_action]
        motion_q_mujoco = np.array(motion_q_isaac)[self.isaac_to_mujoco_idx]
        motion_dq_mujoco = np.array(motion_dq_isaac)[self.isaac_to_mujoco_idx]
        self.data.qpos[self.qpos_indices] = motion_q_mujoco
        self.data.qvel[self.qvel_indices] = motion_dq_mujoco

        if self.model.body_jntnum[self.root_body_id] > 0:
            root_jnt_id = self.model.body_jntadr[self.root_body_id]
            if self.model.jnt_type[root_jnt_id] == mujoco.mjtJoint.mjJNT_FREE:
                root_qpos_adr = self.model.jnt_qposadr[root_jnt_id]
                root_qvel_adr = self.model.jnt_dofadr[root_jnt_id]
                self.data.qpos[root_qpos_adr : root_qpos_adr + 3] = self.motion_root_pos[timestep].copy()
                self.data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = self.motion_root_quat[timestep].copy()
                self.data.qvel[root_qvel_adr : root_qvel_adr + 3] = self.motion_root_lin_vel[timestep].copy()
                self.data.qvel[root_qvel_adr + 3 : root_qvel_adr + 6] = self.motion_root_ang_vel[timestep].copy()

        mujoco.mj_forward(self.model, self.data)

    def get_obs(self, timestep) -> np.ndarray:
        """
        Compute current observation vector from MuJoCo sensors and internal state.

        Returns:
            np.ndarray: Normalized and clipped observation history.
        """
        if timestep < 2:

            ref_motion_quat = self.motion_anchor_quat[timestep]
            yaw_motion_quat = yaw_quat(ref_motion_quat)
            yaw_motion_matrix = np.zeros(9)
            mujoco.mju_quat2Mat(yaw_motion_matrix, yaw_motion_quat)
            yaw_motion_matrix = yaw_motion_matrix.reshape(3, 3)

            robot_quat = self.data.xquat[self.anchor_body_id]
            yaw_robot_quat = yaw_quat(robot_quat)
            yaw_robot_matrix = np.zeros(9)
            mujoco.mju_quat2Mat(yaw_robot_matrix, yaw_robot_quat)
            yaw_robot_matrix = yaw_robot_matrix.reshape(3, 3)

            self.init_to_world = yaw_robot_matrix @ yaw_motion_matrix.T

        self.dof_pos = self.data.qpos[self.qpos_indices].astype(np.double)
        self.dof_vel = self.data.qvel[self.qvel_indices].astype(np.double)

        ref_motion_quat = self.motion_anchor_quat[timestep]
        quaternion = self.data.xquat[self.anchor_body_id]
        anchor_quat = subtract_frame_transforms_mujoco(quaternion, ref_motion_quat, self.init_to_world)
        anchor_ori = np.zeros(9)
        mujoco.mju_quat2Mat(anchor_ori, anchor_quat)
        anchor_ori = anchor_ori.reshape(3, 3)[:, :2]
        anchor_ori = anchor_ori.reshape(
            -1,
        )

        obs = np.zeros((self.cfg.sim.num_obs_per_step,), dtype=np.float32)

        # command
        obs[0 : self.num_action] = self.motion_joint_pos[timestep, :]
        obs[self.num_action : 2 * self.num_action] = self.motion_joint_vel[timestep, :]
        # Motion_anchor_ori
        obs[2 * self.num_action : 6 + 2 * self.num_action] = anchor_ori
        # Angular vel
        obs[6 + 2 * self.num_action : 9 + 2 * self.num_action] = self.data.sensor("angular-velocity").data.astype(
            np.double
        )
        # Dof pos
        obs[9 + 2 * self.num_action : 9 + 3 * self.num_action] = (self.dof_pos - self.default_joint_pos)[
            self.mujoco_to_isaac_idx
        ]

        # Dof vel
        obs[9 + 3 * self.num_action : 9 + 4 * self.num_action] = self.dof_vel[self.mujoco_to_isaac_idx]

        # Action
        obs[9 + 4 * self.num_action : 9 + 5 * self.num_action] = self.action

        # Update observation history
        if not self.history_initialized:
            # Isaac Lab's CircularBuffer duplicates the first observation across
            # all history slots after reset.
            self.obs_history[:] = obs
            self.history_initialized = True
        else:
            self.obs_history[:-1] = self.obs_history[1:]
            self.obs_history[-1] = obs

        return self.obs_history.reshape(-1)

    def pd_control(self, target_q, q, init_q, target_dq, dq):
        """Calculates torques from position commands"""
        return (target_q + init_q - q) * self.joint_stiffness + (target_dq - dq) * self.joint_damping

    def run(self) -> None:
        """
        Run the simulation loop with keyboard-controlled commands.
        """
        self.setup_keyboard_listener()
        self.listener.start()
        self.print_keyboard_help()

        timestep = 0
        try:
            while self.data.time < self.cfg.sim.sim_duration and not self.exit_requested:
                if self.reset_requested:
                    timestep = 0
                    self.reset_env()
                    self.reset_requested = False

                if self.is_paused and not self.step_once_requested:
                    self.viewer.render()
                    time.sleep(self.cfg.sim.dt)
                    continue

                max_timestep = self.motion_joint_pos.shape[0]
                if timestep >= max_timestep:
                    timestep = 0

                observation = self.get_obs(timestep)
                obs = np.zeros([1, self.obs_history.size], dtype=np.float32)
                obs[0] = observation
                self.action[:] = self.policy.run(None, {"obs": obs})[0]
                self.action = np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)
                action = self.action.copy()[self.isaac_to_mujoco_idx]

                for sim_update in range(self.cfg.sim.decimation):
                    step_start_time = time.time()

                    q = self.data.qpos[self.qpos_indices].astype(np.double)
                    dq = self.data.qvel[self.qvel_indices].astype(np.double)
                    target_q = action * self.action_scale
                    target_dq = np.zeros((self.num_action), dtype=np.double)
                    tau = self.pd_control(target_q, q, self.default_joint_pos, target_dq, dq)  # Calc torques
                    tau = np.clip(tau, self.control_low, self.control_high)
                    self.data.ctrl[self.actuator_indices] = tau

                    mujoco.mj_step(self.model, self.data)
                    self.viewer.render()

                    elapsed = time.time() - step_start_time
                    sleep_time = self.cfg.sim.dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                self.episode_length_buf += 1
                timestep += 1
                self.step_once_requested = False
        finally:
            if self.listener is not None:
                self.listener.stop()
            self.viewer.close()

    def reset_env(self) -> None:
        """Reset MuJoCo state and policy history to the initial condition."""
        mujoco.mj_resetData(self.model, self.data)
        self.apply_motion_frame(0)
        self.data.ctrl[:] = np.zeros(self.num_action, dtype=np.double)

        self.dof_pos[:] = 0.0
        self.dof_vel[:] = 0.0
        self.action[:] = 0.0
        self.obs_history[:] = 0.0
        self.history_initialized = False
        self.episode_length_buf = 0
        self.step_once_requested = False
        print("[INFO] Environment reset.")

    def print_keyboard_help(self) -> None:
        """Print available keyboard controls."""
        print("\n[INFO] Keyboard controls:")
        print("  p : pause/resume simulation")
        print("  r : reset environment")
        print("  n : single step while paused")
        print("  h : print this help")
        print("  q : quit simulation\n")

    def setup_keyboard_listener(self) -> None:
        """Set up keyboard controls for simulation runtime management."""

        def on_press(key):
            try:
                key_char = key.char.lower()
            except AttributeError:
                return

            if key_char == "p":
                self.is_paused = not self.is_paused
                status = "paused" if self.is_paused else "running"
                print(f"[INFO] Simulation {status}.")
            elif key_char == "r":
                self.reset_requested = True
                print("[INFO] Reset requested.")
            elif key_char == "n":
                if self.is_paused:
                    self.step_once_requested = True
                    print("[INFO] Step once.")
            elif key_char == "h":
                self.print_keyboard_help()
            elif key_char == "q":
                self.exit_requested = True
                print("[INFO] Quit requested.")

        self.listener = keyboard.Listener(on_press=on_press)

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Rotate a vector by the inverse of a quaternion.

        Args:
            q (np.ndarray): Quaternion (x, y, z, w) format.
            v (np.ndarray): Vector to rotate.

        Returns:
            np.ndarray: Rotated vector.
        """
        q_w = q[-1]
        q_vec = q[:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0

        return a - b + c


if __name__ == "__main__":
    project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    parser = argparse.ArgumentParser(description="Run Bumi mimic sim2sim MuJoCo controller.")
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Path to policy.onnx.",
    )
    parser.add_argument("--motion_file", type=str, required=True, help="动作文件路径")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(
            project_root_dir, "source/NoetixBumi/NoetixBumi/assets/robots/bumi/bumi.xml"
        ),
        help="Path to model.xml",
    )
    parser.add_argument("--duration", type=float, default=100.0, help="Simulation duration in seconds")
    args = parser.parse_args()

    if not os.path.isfile(args.policy):
        print(f"[ERROR] Policy file not found: {args.policy}")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"[ERROR] MuJoCo model file not found: {args.model}")
        sys.exit(1)

    print(f"[INFO] Loaded policy: {args.policy}")
    print(f"[INFO] Loaded model: {args.model}")

    sim_cfg = SimToSimCfg()
    sim_cfg.sim.sim_duration = args.duration

    runner = MujocoRunner(cfg=sim_cfg, policy_path=args.policy, model_path=args.model, motion_file=args.motion_file)
    runner.run()
