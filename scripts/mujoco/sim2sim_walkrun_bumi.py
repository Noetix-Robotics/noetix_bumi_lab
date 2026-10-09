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


class SimToSimCfg:
    """Configuration class for sim2sim parameters.

    Must be kept consistent with the training configuration.
    """

    class sim:
        sim_duration = 100.0
        num_obs_per_step = 71  # 3 (vel_cmd) + 3 (ang_vel) + 2 (Euler) + 21 (joint_pos) + 21 (joint_vel) + 21 (actions) = 71
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

    def __init__(self, cfg: SimToSimCfg, policy_path, model_path):
        self.cfg = cfg
        network_path = policy_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.cfg.sim.dt

        self.onnx_model = onnx.load(network_path)
        onnx.checker.check_model(self.onnx_model)
        self.policy = ort.InferenceSession(network_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer._render_every_frame = False
        mujoco.mj_step(self.model, self.data)
        self.init_variables()

    def init_variables(self) -> None:
        """Initialize simulation variables and joint index mappings."""
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt

        assert self.onnx_model.metadata_props
        print("Loading robot properties from onnx: ")
        for prop in self.onnx_model.metadata_props:
            attr_name = sanitize_variable_name(prop.key)
            attr_value = csv_str_to_list_advanced(prop.value)
            setattr(self, attr_name, attr_value)

        self.mujoco_to_isaac_idx = [
            mujoco_joint_index.index(name) for name in isaacLab_joint_index
        ]
        self.isaac_to_mujoco_idx = [
            isaacLab_joint_index.index(name) for name in mujoco_joint_index
        ]

        self.num_action = len(self.default_joint_pos)
        self.default_joint_pos = np.array(self.default_joint_pos)[
            self.isaac_to_mujoco_idx
        ]
        self.joint_stiffness = np.array(self.joint_stiffness)[self.isaac_to_mujoco_idx]
        self.joint_damping = np.array(self.joint_damping)[self.isaac_to_mujoco_idx]
        self.action_scale = np.array(self.action_scale)[self.isaac_to_mujoco_idx]

        self.data.qpos[-self.num_action :] = self.default_joint_pos.copy()

        self.dof_pos = np.zeros(self.num_action)
        self.dof_vel = np.zeros(self.num_action)
        self.action = np.zeros(self.num_action)

        self.episode_length_buf = 0

        # Initial command vel
        self.command_vel = np.array([0.0, 0.0, 0.0])
        self.obs_history = np.zeros(
            (self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,),
            dtype=np.float32,
        )

    @staticmethod
    def euler_xyz_from_quat_single(
        quat: np.ndarray, wrap_to_2pi: bool = False
    ) -> tuple[float, float, float]:
        """Convert a single quaternion (w, x, y, z) to Euler angles in radians.

        Note:
            The euler angles are assumed in XYZ extrinsic convention.
            This function processes a SINGLE quaternion, not a batch.

        Args:
            quat: The quaternion orientation in (w, x, y, z). Shape is (4,).
            wrap_to_2pi (bool): Whether to wrap output Euler angles into [0, 2π). If
                False, angles are returned in the default range (−π, π]. Defaults to
                False.

        Returns:
            A tuple containing roll, pitch, yaw in radians.

        Reference:
            https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        """
        quat = np.asarray(quat, dtype=np.float64).reshape(-1)
        if quat.size != 4:
            raise ValueError(
                f"Expected quaternion with 4 elements, got shape {quat.shape}"
            )

        # 解包单个四元数分量 (w, x, y, z)
        q_w, q_x, q_y, q_z = quat

        # roll (x-axis rotation)
        sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
        cos_roll = 1.0 - 2.0 * (q_x * q_x + q_y * q_y)
        roll = np.arctan2(sin_roll, cos_roll)

        # pitch (y-axis rotation)
        sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
        # 处理数值误差导致的 asin 输入超出 [-1, 1] 的情况
        pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))

        # yaw (z-axis rotation)
        sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
        cos_yaw = 1.0 - 2.0 * (q_y * q_y + q_z * q_z)
        yaw = np.arctan2(sin_yaw, cos_yaw)

        if wrap_to_2pi:
            two_pi = 2.0 * np.pi
            return float(roll % two_pi), float(pitch % two_pi), float(yaw % two_pi)

        return float(roll), float(pitch), float(yaw)

    def get_obs(self) -> np.ndarray:
        """
        Compute current observation vector from MuJoCo sensors and internal state.

        Returns:
            np.ndarray: Normalized and clipped observation history.
        """
        self.dof_pos = self.data.qpos.astype(np.double)[-self.num_action :]
        self.dof_vel = self.data.qvel.astype(np.double)[-self.num_action :]

        obs = np.zeros((self.cfg.sim.num_obs_per_step,), dtype=np.float32)

        # Command velocity
        obs[0:3] = self.command_vel

        # Angular vel
        obs[3:6] = self.data.sensor("angular-velocity").data.astype(np.double)

        # Projected gravity
        orientation_sensor = self.data.sensor("orientation")
        quat_sensor_data = orientation_sensor.data.astype(
            np.double
        )  # 直接拿到数值数组，而非对象

        roll, pitch, yaw = self.euler_xyz_from_quat_single(quat_sensor_data)
        obs[6] = roll
        obs[7] = pitch

        # Dof pos
        obs[8 : 8 + self.num_action] = (self.dof_pos - self.default_joint_pos)[
            self.mujoco_to_isaac_idx
        ]

        # Dof vel
        obs[8 + self.num_action : 8 + self.num_action * 2] = self.dof_vel[
            self.mujoco_to_isaac_idx
        ]

        # Action
        obs[8 + self.num_action * 2 : 8 + self.num_action * 3] = self.action

        # Update observation history
        self.obs_history = np.roll(
            self.obs_history, shift=-self.cfg.sim.num_obs_per_step
        )
        self.obs_history[-self.cfg.sim.num_obs_per_step :] = obs.copy()

        return self.obs_history

    def pd_control(self, target_q, q, init_q, target_dq, dq):
        """Calculates torques from position commands"""
        return (target_q + init_q - q) * self.joint_stiffness + (
            target_dq - dq
        ) * self.joint_damping

    def run(self) -> None:
        """
        Run the simulation loop with keyboard-controlled commands.
        """
        self.setup_keyboard_listener()
        self.listener.start()

        while self.data.time < self.cfg.sim.sim_duration:
            self.obs_history = self.get_obs()
            obs = np.zeros([1, self.obs_history.size], dtype=np.float32)
            obs[0] = self.obs_history.copy()
            self.action[:] = self.policy.run(None, {"obs": obs})[0]
            self.action = np.clip(
                self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions
            )
            action = self.action.copy()[self.isaac_to_mujoco_idx]

            for sim_update in range(self.cfg.sim.decimation):
                step_start_time = time.time()

                q = self.data.qpos.astype(np.double)[-self.num_action :]
                dq = self.data.qvel.astype(np.double)[-self.num_action :]
                target_q = action * self.action_scale
                target_dq = np.zeros((self.num_action), dtype=np.double)
                tau = self.pd_control(
                    target_q, q, self.default_joint_pos, target_dq, dq
                )  # Calc torques
                self.data.ctrl = tau

                mujoco.mj_step(self.model, self.data)
                self.viewer.render()

                elapsed = time.time() - step_start_time
                sleep_time = self.cfg.sim.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            self.episode_length_buf += 1

        self.listener.stop()
        self.viewer.close()

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

    def adjust_command_vel(self, idx: int, increment: float) -> None:
        """
        Adjust command velocity vector.

        Args:
            idx (int): Index of velocity component (0=x, 1=y, 2=yaw).
            increment (float): Value to increment.
        """
        self.command_vel[idx] += increment
        if idx == 0:
            self.command_vel[idx] = np.clip(
                self.command_vel[idx], -0.5, 1.2
            )  # vel clip
        elif idx == 1:
            self.command_vel[idx] = np.clip(
                self.command_vel[idx], -0.7, 0.7
            )  # vel clip
        elif idx == 2:
            self.command_vel[idx] = np.clip(
                self.command_vel[idx], -1.0, 1.0
            )  # yaw rate clip
        print(f"Command velocity: {self.command_vel}")

    def setup_keyboard_listener(self) -> None:
        """
        Set up keyboard event listener for user control input.
        """

        def on_press(key):
            try:
                if key.char == "8":  # NumPad 8      x += 0.1
                    self.adjust_command_vel(0, 0.1)
                elif key.char == "2":  # NumPad 2      x -= 0.1
                    self.adjust_command_vel(0, -0.1)
                elif key.char == "4":  # NumPad 4      y -= 0.1
                    self.adjust_command_vel(1, -0.1)
                elif key.char == "6":  # NumPad 6      y += 0.1
                    self.adjust_command_vel(1, 0.1)
                elif key.char == "7":  # NumPad 7      yaw += 0.1
                    self.adjust_command_vel(2, 0.1)
                elif key.char == "9":  # NumPad 9      yaw -= 0.1
                    self.adjust_command_vel(2, -0.1)
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press)


if __name__ == "__main__":
    LEGGED_LAB_ROOT_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    )
    parser = argparse.ArgumentParser(description="Run sim2sim Mujoco controller.")
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Path to policy.onnx.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(
            LEGGED_LAB_ROOT_DIR,
            "noetix_bumi_lab/source/NoetixBumi/NoetixBumi/assets/robots/bumi/bumi.xml",
        ),
        help="Path to model.xml",
    )
    parser.add_argument(
        "--duration", type=float, default=100.0, help="Simulation duration in seconds"
    )
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

    runner = MujocoRunner(
        cfg=sim_cfg,
        policy_path=args.policy,
        model_path=args.model,
    )
    runner.run()

