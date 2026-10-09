# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""Run the 13-DOF walk-rand-arm policy in MuJoCo.

The policy controls the waist yaw joint and both legs. The eight arm joints are
held at the same fixed targets used by the Isaac Lab 13-DOF environment.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import mujoco
import mujoco_viewer
import numpy as np
import onnx
import onnxruntime as ort


MUJOCO_JOINT_NAMES = [
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

# Isaac Lab resolves the 13-DOF action and observation terms in robot joint order.
POLICY_JOINT_NAMES = [
    "l_leg_pitch_joint",
    "r_leg_pitch_joint",
    "waist_yaw_joint",
    "l_leg_roll_joint",
    "r_leg_roll_joint",
    "l_leg_yaw_joint",
    "r_leg_yaw_joint",
    "l_knee_pitch_joint",
    "r_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "r_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "r_ankle_roll_joint",
]

# Absolute upper-body targets from ManagerBasedRLAmpEnv when rand_arm=False.
FIXED_UPPER_BODY_TARGETS = {
    "l_arm_pitch_joint": 0.0,
    "r_arm_pitch_joint": 0.0,
    "l_arm_roll_joint": 0.2618,
    "r_arm_roll_joint": -0.2618,
    "l_arm_yaw_joint": 0.0,
    "r_arm_yaw_joint": 0.0,
    "l_elbow_pitch_joint": -0.3,
    "r_elbow_pitch_joint": -0.3,
}
UPPER_BODY_STIFFNESS = 12.0
UPPER_BODY_DAMPING = 0.4

# Nominal effort_limit_sim values from BUMI_21DOF_CFG.
JOINT_EFFORT_LIMITS = {
    "waist_yaw_joint": 27.0,
    "l_arm_pitch_joint": 5.5,
    "l_arm_roll_joint": 5.5,
    "l_arm_yaw_joint": 5.5,
    "l_elbow_pitch_joint": 5.5,
    "r_arm_pitch_joint": 5.5,
    "r_arm_roll_joint": 5.5,
    "r_arm_yaw_joint": 5.5,
    "r_elbow_pitch_joint": 5.5,
    "l_leg_pitch_joint": 60.0,
    "l_leg_roll_joint": 27.0,
    "l_leg_yaw_joint": 27.0,
    "l_knee_pitch_joint": 60.0,
    "l_ankle_pitch_joint": 13.0,
    "l_ankle_roll_joint": 13.0,
    "r_leg_pitch_joint": 60.0,
    "r_leg_roll_joint": 27.0,
    "r_leg_yaw_joint": 27.0,
    "r_knee_pitch_joint": 60.0,
    "r_ankle_pitch_joint": 13.0,
    "r_ankle_roll_joint": 13.0,
}


class SimToSimCfg:
    class sim:
        sim_duration = 100.0
        num_obs_per_step = 47
        actor_obs_history_length = 5
        dt = 0.005
        decimation = 4
        clip_actions = 18.0
        command_step = 0.1
        real_time = True


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",")]


def _parse_float_csv(value: str) -> np.ndarray:
    try:
        return np.asarray([float(item) for item in _parse_csv(value)], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric ONNX metadata value: {value[:120]}") from exc


def _euler_xyz_from_wxyz(quat: np.ndarray) -> tuple[float, float, float]:
    """Convert a MuJoCo wxyz quaternion to XYZ roll, pitch and yaw."""
    q_w, q_x, q_y, q_z = np.asarray(quat, dtype=np.float64)
    sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
    cos_roll = 1.0 - 2.0 * (q_x * q_x + q_y * q_y)
    roll = np.arctan2(sin_roll, cos_roll)

    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))

    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1.0 - 2.0 * (q_y * q_y + q_z * q_z)
    yaw = np.arctan2(sin_yaw, cos_yaw)
    return float(roll), float(pitch), float(yaw)


class WalkRandArmMujocoRunner:
    def __init__(self, cfg: SimToSimCfg, policy_path: str, model_path: str):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = cfg.sim.dt
        self.data = mujoco.MjData(self.model)

        self.onnx_model = onnx.load(policy_path)
        onnx.checker.check_model(self.onnx_model)
        self.policy = ort.InferenceSession(policy_path)
        self.input_name = self.policy.get_inputs()[0].name

        self._validate_policy_shape()
        self._load_metadata()
        self._resolve_mujoco_indices()

        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer._render_every_frame = False

        self.command_vel = np.zeros(3, dtype=np.float32)
        self.command_lock = threading.Lock()
        self.reset_requested = threading.Event()
        self.policy_action = np.zeros(len(POLICY_JOINT_NAMES), dtype=np.float32)
        self.obs_history = np.zeros(
            (cfg.sim.actor_obs_history_length, cfg.sim.num_obs_per_step), dtype=np.float32
        )
        self.history_initialized = False
        self.listener = None
        self.reset_robot()

    def _validate_policy_shape(self) -> None:
        expected_input = self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length
        input_shape = self.policy.get_inputs()[0].shape
        output_shape = self.policy.get_outputs()[0].shape

        if isinstance(input_shape[-1], int) and input_shape[-1] != expected_input:
            raise ValueError(f"Policy input is {input_shape[-1]}, expected {expected_input} (47 x 5).")
        if isinstance(output_shape[-1], int) and output_shape[-1] != len(POLICY_JOINT_NAMES):
            raise ValueError(f"Policy output is {output_shape[-1]}, expected 13 actions.")

    def _load_metadata(self) -> None:
        metadata = {item.key: item.value for item in self.onnx_model.metadata_props}
        required = {
            "joint_names",
            "joint_stiffness",
            "joint_damping",
            "default_joint_pos",
            "action_scale",
        }
        missing = sorted(required - metadata.keys())
        if missing:
            raise ValueError(f"ONNX metadata is missing: {', '.join(missing)}")

        metadata_joint_names = _parse_csv(metadata["joint_names"])
        values = {
            "joint_stiffness": _parse_float_csv(metadata["joint_stiffness"]),
            "joint_damping": _parse_float_csv(metadata["joint_damping"]),
            "default_joint_pos": _parse_float_csv(metadata["default_joint_pos"]),
            "action_scale": _parse_float_csv(metadata["action_scale"]),
        }
        for key, array in values.items():
            if len(array) != len(metadata_joint_names):
                raise ValueError(
                    f"ONNX metadata {key} has {len(array)} values, but joint_names has "
                    f"{len(metadata_joint_names)} entries. Re-export the policy with 21-DOF metadata."
                )

        missing_policy_joints = [name for name in POLICY_JOINT_NAMES if name not in metadata_joint_names]
        if missing_policy_joints:
            raise ValueError(
                "ONNX metadata does not contain all 13 policy joints. Missing: "
                + ", ".join(missing_policy_joints)
            )

        metadata_index = {name: index for index, name in enumerate(metadata_joint_names)}

        def reorder_required(key: str, names: list[str]) -> np.ndarray:
            return np.asarray([values[key][metadata_index[name]] for name in names], dtype=np.float64)

        self.policy_default_joint_pos = reorder_required("default_joint_pos", POLICY_JOINT_NAMES)
        self.policy_action_scale = reorder_required("action_scale", POLICY_JOINT_NAMES)

        default_joint_pos = []
        joint_stiffness = []
        joint_damping = []
        used_upper_body_fallback = False
        for name in MUJOCO_JOINT_NAMES:
            if name in metadata_index:
                index = metadata_index[name]
                default_joint_pos.append(values["default_joint_pos"][index])
                joint_stiffness.append(values["joint_stiffness"][index])
                joint_damping.append(values["joint_damping"][index])
            elif name in FIXED_UPPER_BODY_TARGETS:
                used_upper_body_fallback = True
                default_joint_pos.append(FIXED_UPPER_BODY_TARGETS[name])
                joint_stiffness.append(UPPER_BODY_STIFFNESS)
                joint_damping.append(UPPER_BODY_DAMPING)
            else:
                raise ValueError(f"No PD metadata or fallback is available for joint: {name}")

        self.default_joint_pos = np.asarray(default_joint_pos, dtype=np.float64)
        self.joint_stiffness = np.asarray(joint_stiffness, dtype=np.float64)
        self.joint_damping = np.asarray(joint_damping, dtype=np.float64)
        if used_upper_body_fallback:
            print("[INFO] ONNX has 13-DOF metadata; using Kp=12 and Kd=0.4 for the fixed arm joints.")

        self.fixed_target = self.default_joint_pos.copy()
        mujoco_index = {name: index for index, name in enumerate(MUJOCO_JOINT_NAMES)}
        for name, target in FIXED_UPPER_BODY_TARGETS.items():
            self.fixed_target[mujoco_index[name]] = target

    def _resolve_mujoco_indices(self) -> None:
        qpos_indices = []
        qvel_indices = []
        actuator_indices = []
        for name in MUJOCO_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if joint_id < 0 or actuator_id < 0:
                raise ValueError(f"MuJoCo model is missing joint or actuator: {name}")
            qpos_indices.append(self.model.jnt_qposadr[joint_id])
            qvel_indices.append(self.model.jnt_dofadr[joint_id])
            actuator_indices.append(actuator_id)

        self.qpos_indices = np.asarray(qpos_indices, dtype=np.int64)
        self.qvel_indices = np.asarray(qvel_indices, dtype=np.int64)
        self.actuator_indices = np.asarray(actuator_indices, dtype=np.int64)
        full_index = {name: index for index, name in enumerate(MUJOCO_JOINT_NAMES)}
        self.policy_full_indices = np.asarray(
            [full_index[name] for name in POLICY_JOINT_NAMES], dtype=np.int64
        )

        control_ranges = self.model.actuator_ctrlrange[self.actuator_indices]
        effort_limits = np.asarray(
            [JOINT_EFFORT_LIMITS[name] for name in MUJOCO_JOINT_NAMES], dtype=np.float64
        )
        self.control_low = np.maximum(control_ranges[:, 0], -effort_limits)
        self.control_high = np.minimum(control_ranges[:, 1], effort_limits)

    def reset_robot(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_indices] = self.fixed_target
        self.data.qvel[self.qvel_indices] = 0.0
        self.policy_action.fill(0.0)
        self.obs_history.fill(0.0)
        self.history_initialized = False
        mujoco.mj_forward(self.model, self.data)

    def _joint_position(self) -> np.ndarray:
        return self.data.qpos[self.qpos_indices].astype(np.float64, copy=True)

    def _joint_velocity(self) -> np.ndarray:
        return self.data.qvel[self.qvel_indices].astype(np.float64, copy=True)

    def get_observation(self) -> np.ndarray:
        obs = np.zeros(self.cfg.sim.num_obs_per_step, dtype=np.float32)
        with self.command_lock:
            obs[0:3] = self.command_vel

        obs[3:6] = self.data.sensor("angular-velocity").data.astype(np.float32)
        roll, pitch, _ = _euler_xyz_from_wxyz(self.data.sensor("orientation").data)
        obs[6:8] = (roll, pitch)

        q = self._joint_position()[self.policy_full_indices]
        dq = self._joint_velocity()[self.policy_full_indices]
        obs[8:21] = q - self.policy_default_joint_pos
        obs[21:34] = dq
        obs[34:47] = self.policy_action

        if not self.history_initialized:
            self.obs_history[:] = obs
            self.history_initialized = True
        else:
            self.obs_history[:-1] = self.obs_history[1:]
            self.obs_history[-1] = obs

        return self.obs_history.reshape(1, -1)

    def infer_action(self) -> None:
        observation = self.get_observation()
        output = self.policy.run(None, {self.input_name: observation})[0]
        action = np.asarray(output, dtype=np.float32).reshape(-1)
        if action.size != len(POLICY_JOINT_NAMES):
            raise ValueError(f"Policy returned {action.size} actions, expected 13.")
        self.policy_action = np.clip(
            action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions
        )

    def _apply_pd_control(self) -> None:
        q = self._joint_position()
        dq = self._joint_velocity()
        target = self.fixed_target.copy()
        target[self.policy_full_indices] = (
            self.policy_default_joint_pos + self.policy_action_scale * self.policy_action
        )
        torque = self.joint_stiffness * (target - q) - self.joint_damping * dq
        torque = np.clip(torque, self.control_low, self.control_high)
        self.data.ctrl[self.actuator_indices] = torque

    def adjust_command(self, component: int, delta: float) -> None:
        command_low = np.asarray([-1.0, -0.7, -1.0], dtype=np.float32)
        command_high = np.asarray([1.8, 0.7, 1.0], dtype=np.float32)
        with self.command_lock:
            self.command_vel[component] = np.clip(
                self.command_vel[component] + delta,
                command_low[component],
                command_high[component],
            )
            current = self.command_vel.copy()
        print(f"[CMD] vx={current[0]:+.2f}, vy={current[1]:+.2f}, wz={current[2]:+.2f}")

    def reset_command(self) -> None:
        with self.command_lock:
            self.command_vel.fill(0.0)
        print("[CMD] vx=+0.00, vy=+0.00, wz=+0.00")

    def setup_keyboard(self) -> None:
        try:
            from pynput import keyboard
        except Exception as exc:
            raise RuntimeError("Keyboard control requires pynput and an active X display.") from exc

        step = self.cfg.sim.command_step

        def on_press(key_event) -> None:
            try:
                char = key_event.char.lower()
            except AttributeError:
                return
            if char == "8":
                self.adjust_command(0, step)
            elif char == "2":
                self.adjust_command(0, -step)
            elif char == "4":
                self.adjust_command(1, step)
            elif char == "6":
                self.adjust_command(1, -step)
            elif char == "7":
                self.adjust_command(2, step)
            elif char == "9":
                self.adjust_command(2, -step)
            elif char == "r":
                self.reset_command()
            elif char == "t":
                self.reset_requested.set()

        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.daemon = True
        self.listener.start()

    def run(self) -> None:
        self.setup_keyboard()
        print("[INFO] Controls: 8/2 vx, 4/6 vy, 7/9 yaw, R zero command, T reset robot")
        try:
            while self.viewer.is_alive:
                if self.cfg.sim.sim_duration > 0.0 and self.data.time >= self.cfg.sim.sim_duration:
                    break
                if self.reset_requested.is_set():
                    self.reset_requested.clear()
                    self.reset_robot()
                    self.reset_command()
                    print("[INFO] Robot reset")

                self.infer_action()
                for _ in range(self.cfg.sim.decimation):
                    step_start = time.perf_counter()
                    self._apply_pd_control()
                    mujoco.mj_step(self.model, self.data)
                    self.viewer.render()
                    if self.cfg.sim.real_time:
                        sleep_time = self.cfg.sim.dt - (time.perf_counter() - step_start)
                        if sleep_time > 0.0:
                            time.sleep(sleep_time)
        finally:
            if self.listener is not None:
                self.listener.stop()
            self.viewer.close()


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def main() -> None:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Run the 13-DOF walk-rand-arm policy in MuJoCo.")
    parser.add_argument("--policy", required=True, help="Path to the exported ONNX policy.")
    parser.add_argument(
        "--model",
        default=os.path.join(
            root, "source/NoetixBumi/NoetixBumi/assets/robots/bumi/bumi.xml"
        ),
        help="Path to the MuJoCo XML model.",
    )
    parser.add_argument(
        "--duration", type=float, default=100.0, help="Simulation duration in seconds; <=0 runs until closed."
    )
    parser.add_argument("--command-step", type=float, default=0.1, help="Keyboard command increment.")
    parser.add_argument("--no-real-time", action="store_true", help="Run without wall-clock synchronization.")
    args = parser.parse_args()

    if not os.path.isfile(args.policy):
        print(f"[ERROR] Policy file not found: {args.policy}")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"[ERROR] MuJoCo model file not found: {args.model}")
        sys.exit(1)

    cfg = SimToSimCfg()
    cfg.sim.sim_duration = args.duration
    cfg.sim.command_step = args.command_step
    cfg.sim.real_time = not args.no_real_time

    print(f"[INFO] Policy: {os.path.abspath(args.policy)}")
    print(f"[INFO] Model:  {os.path.abspath(args.model)}")
    WalkRandArmMujocoRunner(cfg, args.policy, args.model).run()


if __name__ == "__main__":
    main()
