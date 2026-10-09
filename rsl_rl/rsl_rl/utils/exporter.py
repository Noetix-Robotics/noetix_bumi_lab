# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import os

import onnx
import torch
import torch.nn.functional as F


def export_policy_as_onnx(
    policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


def export_policy_as_onnx_fixed_upper_body(
    policy: object,
    env: object,
    path: str,
    normalizer: object | None = None,
    filename="policy_21dof_fixed_upper.onnx",
    action_name="joint_pos",
    verbose=False,
) -> bool:
    """Adapt a stacked subset-DOF policy to the full 21-DOF deployment ABI.

    The exported graph accepts full-body observations, selects the subset used
    during training in every history frame, runs the unchanged actor, and then
    expands its action to the full robot joint order. Upper-body constants
    reproduce the pose applied by the 13-DOF training environment.
    """
    full_joint_names = _robot_joint_names(env)
    policy_joint_names = _action_term_joint_names(env, action_name, full_joint_names)
    if len(policy_joint_names) >= len(full_joint_names):
        return False

    missing = [name for name in policy_joint_names if name not in full_joint_names]
    if missing:
        raise ValueError(f"Policy action joints are not in robot joint_names: {missing}")

    policy_to_full_indices = [full_joint_names.index(name) for name in policy_joint_names]
    action_term = env.action_manager.get_term(action_name)
    action_scales = _action_scale_to_full_joint_list(env, action_term, full_joint_names)
    default_joint_pos = env.scene["robot"].data.default_joint_pos[0].detach().cpu().tolist()
    upper_body_targets = {
        "l_arm_pitch_joint": 0.0,
        "r_arm_pitch_joint": 0.0,
        "l_arm_roll_joint": 0.2618,
        "r_arm_roll_joint": -0.2618,
        "l_arm_yaw_joint": 0.0,
        "r_arm_yaw_joint": 0.0,
        "l_elbow_pitch_joint": -0.3,
        "r_elbow_pitch_joint": -0.3,
    }
    fixed_action = [0.0] * len(full_joint_names)
    policy_joint_set = set(policy_joint_names)
    for index, joint_name in enumerate(full_joint_names):
        if joint_name in policy_joint_set:
            continue
        target = upper_body_targets.get(joint_name, default_joint_pos[index])
        scale = float(action_scales[index])
        if abs(scale) < 1.0e-8:
            raise ValueError(f"Cannot encode fixed target for {joint_name}: action scale is zero.")
        fixed_action[index] = (float(target) - float(default_joint_pos[index])) / scale

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxFixedUpperBodyPolicyExporter(
        policy,
        normalizer=normalizer,
        full_action_dim=len(full_joint_names),
        policy_to_full_indices=policy_to_full_indices,
        fixed_action=fixed_action,
        verbose=verbose,
    )
    policy_exporter.export(path, filename)
    print(
        f"[INFO] Exported fixed-upper-body ONNX: {filename} "
        f"(full observation -> {len(policy_joint_names)}-DOF actor -> "
        f"{len(full_joint_names)} full actions)."
    )
    return True


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.is_recurrent = policy.is_recurrent
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)


        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.rnn_type = type(self.rnn).__name__.lower()  # 'lstm' or 'gru'
            if self.rnn_type == "lstm":
                self.forward = self.forward_lstm
            elif self.rnn_type == "gru":
                self.forward = self.forward_gru
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward_gru(self, x_in, h_in):
        x_in = self.normalizer(x_in)
        x, h = self.rnn(x_in.unsqueeze(0), h_in)
        x = x.squeeze(0)
        return self.actor(x), h


    def forward(self, x):
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)

            if self.rnn_type == "lstm":
                c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
                torch.onnx.export(
                    self,
                    (obs, h_in, c_in),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=11,
                    verbose=self.verbose,
                    input_names=["obs", "h_in", "c_in"],
                    output_names=["actions", "h_out", "c_out"],
                    dynamic_axes={},
                )
            elif self.rnn_type == "gru":
                torch.onnx.export(
                    self,
                    (obs, h_in),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=11,
                    verbose=self.verbose,
                    input_names=["obs", "h_in"],
                    output_names=["actions", "h_out"],
                    dynamic_axes={},
                )
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        else:
            obs = torch.zeros(1, self.actor[0].in_features)
            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )


class _OnnxFixedUpperBodyPolicyExporter(_OnnxPolicyExporter):
    """Inference-only adapter from full observations to a subset-DOF actor."""

    def __init__(
        self,
        policy,
        normalizer=None,
        full_action_dim: int = 21,
        policy_to_full_indices: list[int] | None = None,
        fixed_action: list[float] | None = None,
        shared_observation_dim: int = 8,
        verbose=False,
    ):
        super().__init__(policy, normalizer=normalizer, verbose=verbose)
        if self.is_recurrent:
            raise NotImplementedError("Fixed-upper-body ONNX export currently supports non-recurrent policies only.")
        if policy_to_full_indices is None:
            raise ValueError("policy_to_full_indices must be provided.")
        if fixed_action is None or len(fixed_action) != full_action_dim:
            raise ValueError("fixed_action must contain one raw action for every full-body joint.")

        self.full_action_dim = full_action_dim
        policy_action_dim = len(policy_to_full_indices)
        actor_output_dim = getattr(self.actor[-1], "out_features", None)
        if actor_output_dim != policy_action_dim:
            raise ValueError(
                f"Actor output {actor_output_dim} does not match the {policy_action_dim} mapped policy joints."
            )
        policy_frame_dim = shared_observation_dim + 3 * policy_action_dim
        policy_observation_dim = self.actor[0].in_features
        if policy_observation_dim % policy_frame_dim != 0:
            raise ValueError(
                f"Actor input {policy_observation_dim} is not stacked {policy_frame_dim}-D 13-DOF observations."
            )
        stack_size = policy_observation_dim // policy_frame_dim
        full_frame_dim = shared_observation_dim + 3 * full_action_dim
        self.full_observation_dim = stack_size * full_frame_dim

        observation_indices = []
        for frame_index in range(stack_size):
            frame_start = frame_index * full_frame_dim
            observation_indices.extend(range(frame_start, frame_start + shared_observation_dim))
            for block_index in range(3):
                block_start = frame_start + shared_observation_dim + block_index * full_action_dim
                observation_indices.extend(block_start + index for index in policy_to_full_indices)

        self.register_buffer(
            "policy_to_full_indices",
            torch.tensor(policy_to_full_indices, dtype=torch.long).view(1, -1),
        )
        self.register_buffer(
            "policy_observation_indices",
            torch.tensor(observation_indices, dtype=torch.long),
        )
        self.register_buffer("fixed_action", torch.tensor(fixed_action, dtype=torch.float32).view(1, -1))

    def forward(self, x):
        policy_observation = torch.index_select(x, 1, self.policy_observation_indices)
        policy_action = self.actor(self.normalizer(policy_observation))
        full_action = self.fixed_action.expand(policy_action.shape[0], -1).clone()
        indices = self.policy_to_full_indices.expand(policy_action.shape[0], -1)
        return full_action.scatter(1, indices, policy_action)

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        obs = torch.zeros(1, self.full_observation_dim)
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr  # numbers → format, strings → as-is
    )


def _action_scale_to_list(action_term, expected_length: int) -> list[float]:
    """Convert scalar or tensor action scales to a flat metadata list."""
    scale = action_term._scale
    if isinstance(scale, torch.Tensor):
        scale = scale.detach()
        if scale.ndim > 1:
            scale = scale[0]
        values = scale.flatten().cpu().tolist()
    elif isinstance(scale, (int, float)):
        values = [float(scale)]
    else:
        values = list(scale)

    if len(values) == 1 and expected_length > 1:
        values *= expected_length
    return values


def _robot_joint_names(env) -> list[str]:
    robot = env.scene["robot"]
    if hasattr(robot.data, "joint_names"):
        return list(robot.data.joint_names)
    return list(robot.joint_names)


def _action_term_joint_ids(action_term, full_joint_count: int) -> list[int]:
    joint_ids = getattr(action_term, "_joint_ids", None)
    if joint_ids is None:
        if action_term.action_dim == full_joint_count:
            return list(range(full_joint_count))
        raise ValueError("Action term does not expose _joint_ids; cannot map subset action to full joints.")
    if isinstance(joint_ids, slice):
        return list(range(full_joint_count))[joint_ids]
    if isinstance(joint_ids, torch.Tensor):
        joint_ids = joint_ids.detach().cpu().tolist()
    return [int(index) for index in joint_ids]


def _action_term_joint_names(env, action_name: str, full_joint_names: list[str] | None = None) -> list[str]:
    if full_joint_names is None:
        full_joint_names = _robot_joint_names(env)
    action_term = env.action_manager.get_term(action_name)
    joint_ids = _action_term_joint_ids(action_term, len(full_joint_names))
    return [full_joint_names[index] for index in joint_ids]


def _action_scale_to_full_joint_list(env, action_term, full_joint_names: list[str]) -> list[float]:
    raw_scale = _action_scale_to_list(action_term, expected_length=action_term.action_dim)
    if len(raw_scale) == len(full_joint_names):
        return raw_scale

    joint_ids = _action_term_joint_ids(action_term, len(full_joint_names))
    if len(raw_scale) == len(joint_ids):
        full_scale = [0.25] * len(full_joint_names)
        for raw_index, joint_index in enumerate(joint_ids):
            full_scale[joint_index] = float(raw_scale[raw_index])
        return full_scale

    if len(raw_scale) == 1:
        return raw_scale * len(full_joint_names)

    raise ValueError(
        f"Cannot map action_scale with length {len(raw_scale)} to {len(full_joint_names)} robot joints."
    )


def attach_onnx_metadata(env, path: str, filename="policy.onnx") -> None:
    onnx_path = os.path.join(path, filename)
    robot = env.scene["robot"]
    action_term = env.action_manager.get_term("joint_pos")
    joint_names = _robot_joint_names(env)
    metadata = {
        "joint_names": joint_names,
        "joint_stiffness": robot.data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": robot.data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
        "action_scale": _action_scale_to_full_joint_list(env, action_term, joint_names),
    }

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)


def attach_onnx_metadata_12dof(env, path: str, filename="policy.onnx") -> None:
    """Attach metadata for 12-DOF policies while ensuring `action_scale` is expanded to full joint count.

    The environment robot may have a full joint set (e.g., 21 joints) while the policy controls a subset
    (e.g., 12 DOF). This function attempts to map the smaller `action_scale` vector back to the full
    joint ordering using information available on the action term (joint ids or joint names). If no
    mapping is found, it pads the remaining joints with zeros (safe fallback).
    """
    onnx_path = os.path.join(path, filename)

    joint_names = list(env.scene["robot"].data.joint_names)
    metadata = {
        "joint_names": joint_names,
        "joint_stiffness": env.scene["robot"].data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": env.scene["robot"].data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": env.scene["robot"].data.default_joint_pos[0].cpu().tolist(),
    }

    # Read raw action scale from the action term
    term = env.action_manager.get_term("joint_pos")
    raw_scale = None
    try:
        raw_scale = _action_scale_to_list(term, expected_length=len(joint_names))
    except Exception:
        raw_scale = None

    # Expand/pad to full joint count. Use 0.25 as default for unknown joints.
    DEFAULT_UNK_SCALE = 0.25
    if raw_scale is None:
        full_scale = [DEFAULT_UNK_SCALE] * len(joint_names)
        print("[WARN] Could not read action_scale from action term; filling with default 0.25 for full joint set.")
    else:
        # If sizes already match, use directly
        if len(raw_scale) == len(joint_names):
            full_scale = raw_scale
        # Known 12->21 mapping: place 12 scales at the provided indices and set others to DEFAULT_UNK_SCALE
        elif len(raw_scale) == 12 and len(joint_names) == 21:
            known_indices = [0, 1, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20]
            full_scale = [DEFAULT_UNK_SCALE] * len(joint_names)
            for i, v in enumerate(raw_scale):
                idx = known_indices[i]
                full_scale[idx] = float(v)
            print(f"[INFO] Applied known 12->21 mapping for action_scale; unknown joints set to {DEFAULT_UNK_SCALE}.")
        elif len(raw_scale) == 13 and len(joint_names) == 21:
            # 13‑DoF subset indices including waist
            known_indices = [0, 1, 2, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20]
            full_scale = [DEFAULT_UNK_SCALE] * len(joint_names)
            for i, v in enumerate(raw_scale):
                full_scale[known_indices[i]] = float(v)

    metadata["action_scale"] = full_scale

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
