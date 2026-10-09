# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import math

import numpy as np
import torch


class MotionLoaderE1:
    POS_SIZE = 3
    ROT_SIZE = 4

    JOINT_POS_SIZE = 23
    KEY_POS_LOCAL_SIZE = 72
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_VEL_SIZE = 23
    CONTACT_MASK_SIZE = 2

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + POS_SIZE

    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE

    JOINT_POSE_START_IDX = ROOT_ROT_END_IDX
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    KEY_POS_LOCAL_START_IDX = JOINT_POSE_END_IDX
    KEY_POS_LOCAL_END_IDX = KEY_POS_LOCAL_START_IDX + KEY_POS_LOCAL_SIZE

    LINEAR_VEL_START_IDX = KEY_POS_LOCAL_END_IDX
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE

    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE

    JOINT_VEL_START_IDX = ANGULAR_VEL_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    CONTACT_MASK_START_IDX = JOINT_VEL_END_IDX
    CONTACT_MASK_END_IDX = CONTACT_MASK_START_IDX + CONTACT_MASK_SIZE

    def __init__(
        self,
        device,
        time_between_frames,
        reference_observation_horizon=2,
        num_preload_transitions=10,
        motion_files="",
        sim_joint_names="",
        sim_key_pos_names="",
    ):
        """Expert dataset provides AMP observations from Human mocap dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames
        self.reference_observation_horizon = reference_observation_horizon
        self.num_preload_transitions = num_preload_transitions
        self.observation_start_dim = MotionLoaderE1.JOINT_POSE_START_IDX

        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split(".")[0])
            with open(motion_file) as f:
                motion_json = json.load(f)
                if i == 0:
                    motion_joint_names = motion_json["JointNames"]
                    motion_key_pos_names = motion_json["KeyPosNames"]
                    self.joint_mapping = create_index_mapping(motion_joint_names, sim_joint_names)
                    self.key_pos_mapping = create_index_mapping(motion_key_pos_names, sim_key_pos_names)
                motion_data = np.array(motion_json["Frames"])

                # Normalize and standardize quaternions.
                for f_i in range(motion_data.shape[0]):
                    root_rot = MotionLoaderE1.get_root_rot(motion_data[f_i])
                    root_rot = QuaternionNormalize(root_rot)
                    root_rot = standardize_quaternion(root_rot)
                    motion_data[f_i, MotionLoaderE1.POS_SIZE : (MotionLoaderE1.POS_SIZE + MotionLoaderE1.ROT_SIZE)] = (
                        root_rot
                    )

                # Remove first 7 observation dimensions (root_pos, root_rot).
                self.trajectories.append(
                    torch.tensor(
                        motion_data[:, MotionLoaderE1.ROOT_ROT_END_IDX : MotionLoaderE1.CONTACT_MASK_END_IDX],
                        dtype=torch.float32,
                        device=device,
                    )
                )
                self.trajectories_full.append(
                    torch.tensor(
                        motion_data[:, : MotionLoaderE1.CONTACT_MASK_END_IDX], dtype=torch.float32, device=device
                    )
                )
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f"Loaded {traj_len}s. motion from {motion_file}.")

        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        print(f"Preloading {self.num_preload_transitions} transitions")
        traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
        times = self.traj_time_sample_batch(traj_idxs)
        self.preloaded_states = torch.zeros(
            self.num_preload_transitions,
            self.reference_observation_horizon,
            self.trajectories_full[0].shape[1],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.motion_label = torch.nn.functional.one_hot(
            torch.tensor(traj_idxs, dtype=torch.long, device=self.device)
        ).float()

        for i in range(self.reference_observation_horizon):
            self.preloaded_states[:, i] = self.get_full_frame_at_time_batch(
                traj_idxs, times + self.time_between_frames * i
            )
        print("Finished preloading")

        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = (
            self.time_between_frames * (self.reference_observation_horizon - 1)
            + self.trajectory_frame_durations[traj_idx]
        )
        return max(0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = (
            self.time_between_frames * (self.reference_observation_horizon - 1)
            + self.trajectory_frame_durations[traj_idxs]
        )
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int32), np.ceil(p * n).astype(np.int32)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int32), np.ceil(p * n).astype(np.int32)
        all_frame_pos_starts = torch.zeros(len(traj_idxs), MotionLoaderE1.POS_SIZE, device=self.device)
        all_frame_pos_ends = torch.zeros(len(traj_idxs), MotionLoaderE1.POS_SIZE, device=self.device)
        all_frame_rot_starts = torch.zeros(len(traj_idxs), MotionLoaderE1.ROT_SIZE, device=self.device)
        all_frame_rot_ends = torch.zeros(len(traj_idxs), MotionLoaderE1.ROT_SIZE, device=self.device)
        all_frame_AMP_starts = torch.zeros(
            len(traj_idxs),
            MotionLoaderE1.CONTACT_MASK_END_IDX - MotionLoaderE1.JOINT_POSE_START_IDX,
            device=self.device,
        )
        all_frame_AMP_ends = torch.zeros(
            len(traj_idxs),
            MotionLoaderE1.CONTACT_MASK_END_IDX - MotionLoaderE1.JOINT_POSE_START_IDX,
            device=self.device,
        )
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_pos_starts[traj_mask] = MotionLoaderE1.get_root_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = MotionLoaderE1.get_root_pos_batch(trajectory[idx_high[traj_mask]])
            all_frame_rot_starts[traj_mask] = MotionLoaderE1.get_root_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = MotionLoaderE1.get_root_rot_batch(trajectory[idx_high[traj_mask]])
            all_frame_AMP_starts[traj_mask] = trajectory[idx_low[traj_mask]][
                :, MotionLoaderE1.JOINT_POSE_START_IDX : MotionLoaderE1.CONTACT_MASK_END_IDX
            ]
            all_frame_AMP_ends[traj_mask] = trajectory[idx_high[traj_mask]][
                :, MotionLoaderE1.JOINT_POSE_START_IDX : MotionLoaderE1.CONTACT_MASK_END_IDX
            ]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        rot_blend = torch_quaternion_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        AMP_blend = self.slerp(all_frame_AMP_starts, all_frame_AMP_ends, blend)
        return torch.cat([pos_blend, rot_blend, AMP_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        idxs = np.random.choice(self.preloaded_states.shape[0], size=num_frames)
        return self.preloaded_states[idxs, 0]

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """

        root_pos0, root_pos1 = MotionLoaderE1.get_root_pos(frame0), MotionLoaderE1.get_root_pos(frame1)
        root_rot0, root_rot1 = MotionLoaderE1.get_root_rot(frame0), MotionLoaderE1.get_root_rot(frame1)
        joints0, joints1 = MotionLoaderE1.get_joint_pose(frame0), MotionLoaderE1.get_joint_pose(frame1)
        tar_toe_pos_0, tar_toe_pos_1 = MotionLoaderE1.get_key_pos_local(frame0), MotionLoaderE1.get_key_pos_local(
            frame1
        )
        linear_vel_0, linear_vel_1 = MotionLoaderE1.get_linear_vel(frame0), MotionLoaderE1.get_linear_vel(frame1)
        angular_vel_0, angular_vel_1 = MotionLoaderE1.get_angular_vel(frame0), MotionLoaderE1.get_angular_vel(frame1)
        joint_vel_0, joint_vel_1 = MotionLoaderE1.get_joint_vel(frame0), MotionLoaderE1.get_joint_vel(frame1)

        blend_root_pos = self.slerp(root_pos0, root_pos1, blend)
        blend_root_rot = np_quaternion_slerp(root_rot0.cpu().numpy(), root_rot1.cpu().numpy(), blend)
        blend_root_rot = torch.tensor(standardize_quaternion(blend_root_rot), dtype=torch.float32, device=self.device)
        blend_joints = self.slerp(joints0, joints1, blend)
        blend_tar_toe_pos = self.slerp(tar_toe_pos_0, tar_toe_pos_1, blend)
        blend_linear_vel = self.slerp(linear_vel_0, linear_vel_1, blend)
        blend_angular_vel = self.slerp(angular_vel_0, angular_vel_1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat([
            blend_root_pos,
            blend_root_rot,
            blend_joints,
            blend_tar_toe_pos,
            blend_linear_vel,
            blend_angular_vel,
            blend_joints_vel,
        ])

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            ids = torch.randint(0, self.num_preload_transitions, (mini_batch_size,), device=self.device)
            states = self.preloaded_states[ids, :, self.observation_start_dim :]
            labels = self.motion_label[ids]
            yield states, labels

    @property
    def observation_full_dim(self):
        """Size of AMP full observations."""
        return self.trajectories_full[0].shape[1]

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1]

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_root_pos(pose):
        return pose[MotionLoaderE1.ROOT_POS_START_IDX : MotionLoaderE1.ROOT_POS_END_IDX]

    def get_root_pos_batch(poses):
        return poses[:, MotionLoaderE1.ROOT_POS_START_IDX : MotionLoaderE1.ROOT_POS_END_IDX]

    def get_root_rot(pose):
        return pose[MotionLoaderE1.ROOT_ROT_START_IDX : MotionLoaderE1.ROOT_ROT_END_IDX]

    def get_root_rot_batch(poses):
        return poses[:, MotionLoaderE1.ROOT_ROT_START_IDX : MotionLoaderE1.ROOT_ROT_END_IDX]

    def get_joint_pose(pose):
        return pose[MotionLoaderE1.JOINT_POSE_START_IDX : MotionLoaderE1.JOINT_POSE_END_IDX]

    def get_joint_pose_batch(poses):
        return poses[:, MotionLoaderE1.JOINT_POSE_START_IDX : MotionLoaderE1.JOINT_POSE_END_IDX]

    def get_key_pos_local(pose):
        return pose[MotionLoaderE1.KEY_POS_LOCAL_START_IDX : MotionLoaderE1.KEY_POS_LOCAL_END_IDX]

    def get_key_pos_local_batch(poses):
        return poses[:, MotionLoaderE1.KEY_POS_LOCAL_START_IDX : MotionLoaderE1.KEY_POS_LOCAL_END_IDX]

    def get_linear_vel(pose):
        return pose[MotionLoaderE1.LINEAR_VEL_START_IDX : MotionLoaderE1.LINEAR_VEL_END_IDX]

    def get_linear_vel_batch(poses):
        return poses[:, MotionLoaderE1.LINEAR_VEL_START_IDX : MotionLoaderE1.LINEAR_VEL_END_IDX]

    def get_angular_vel(pose):
        return pose[MotionLoaderE1.ANGULAR_VEL_START_IDX : MotionLoaderE1.ANGULAR_VEL_END_IDX]

    def get_angular_vel_batch(poses):
        return poses[:, MotionLoaderE1.ANGULAR_VEL_START_IDX : MotionLoaderE1.ANGULAR_VEL_END_IDX]

    def get_joint_vel(pose):
        return pose[MotionLoaderE1.JOINT_VEL_START_IDX : MotionLoaderE1.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(poses):
        return poses[:, MotionLoaderE1.JOINT_VEL_START_IDX : MotionLoaderE1.JOINT_VEL_END_IDX]

    def get_contact_mask_batch(poses):
        return poses[:, MotionLoaderE1.CONTACT_MASK_START_IDX : MotionLoaderE1.CONTACT_MASK_END_IDX]


class MotionLoaderE1_12DOF:
    POS_SIZE = 3
    ROT_SIZE = 4

    JOINT_POS_SIZE = 12
    KEY_POS_LOCAL_SIZE = 39
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_VEL_SIZE = 12
    CONTACT_MASK_SIZE = 2

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + POS_SIZE

    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE

    JOINT_POSE_START_IDX = ROOT_ROT_END_IDX
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    KEY_POS_LOCAL_START_IDX = JOINT_POSE_END_IDX
    KEY_POS_LOCAL_END_IDX = KEY_POS_LOCAL_START_IDX + KEY_POS_LOCAL_SIZE

    LINEAR_VEL_START_IDX = KEY_POS_LOCAL_END_IDX
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE

    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE

    JOINT_VEL_START_IDX = ANGULAR_VEL_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    CONTACT_MASK_START_IDX = JOINT_VEL_END_IDX
    CONTACT_MASK_END_IDX = CONTACT_MASK_START_IDX + CONTACT_MASK_SIZE

    def __init__(
        self,
        device,
        time_between_frames,
        reference_observation_horizon=2,
        num_preload_transitions=10,
        motion_files="",
        sim_joint_names="",
        sim_key_pos_names="",
    ):
        """Expert dataset provides AMP observations from Human mocap dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames
        self.reference_observation_horizon = reference_observation_horizon
        self.num_preload_transitions = num_preload_transitions
        self.observation_start_dim = MotionLoaderE1_12DOF.JOINT_POSE_START_IDX

        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split(".")[0])
            with open(motion_file) as f:
                motion_json = json.load(f)
                if i == 0:
                    motion_joint_names = motion_json["JointNames"]
                    motion_key_pos_names = motion_json["KeyPosNames"]
                    self.joint_mapping = create_index_mapping(motion_joint_names, sim_joint_names)
                    self.key_pos_mapping = create_index_mapping(motion_key_pos_names, sim_key_pos_names)
                motion_data = np.array(motion_json["Frames"])

                # Normalize and standardize quaternions.
                for f_i in range(motion_data.shape[0]):
                    root_rot = MotionLoaderE1_12DOF.get_root_rot(motion_data[f_i])
                    root_rot = QuaternionNormalize(root_rot)
                    root_rot = standardize_quaternion(root_rot)
                    motion_data[f_i, MotionLoaderE1_12DOF.POS_SIZE : (MotionLoaderE1_12DOF.POS_SIZE + MotionLoaderE1_12DOF.ROT_SIZE)] = (
                        root_rot
                    )

                # Remove first 7 observation dimensions (root_pos, root_rot).
                self.trajectories.append(
                    torch.tensor(
                        motion_data[:, MotionLoaderE1_12DOF.ROOT_ROT_END_IDX : MotionLoaderE1_12DOF.CONTACT_MASK_END_IDX],
                        dtype=torch.float32,
                        device=device,
                    )
                )
                self.trajectories_full.append(
                    torch.tensor(
                        motion_data[:, : MotionLoaderE1_12DOF.CONTACT_MASK_END_IDX], dtype=torch.float32, device=device
                    )
                )
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f"Loaded {traj_len}s. motion from {motion_file}.")

        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        print(f"Preloading {self.num_preload_transitions} transitions")
        traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
        times = self.traj_time_sample_batch(traj_idxs)
        self.preloaded_states = torch.zeros(
            self.num_preload_transitions,
            self.reference_observation_horizon,
            self.trajectories_full[0].shape[1],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.motion_label = torch.nn.functional.one_hot(
            torch.tensor(traj_idxs, dtype=torch.long, device=self.device)
        ).float()

        for i in range(self.reference_observation_horizon):
            self.preloaded_states[:, i] = self.get_full_frame_at_time_batch(
                traj_idxs, times + self.time_between_frames * i
            )
        print("Finished preloading")

        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = (
            self.time_between_frames * (self.reference_observation_horizon - 1)
            + self.trajectory_frame_durations[traj_idx]
        )
        return max(0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = (
            self.time_between_frames * (self.reference_observation_horizon - 1)
            + self.trajectory_frame_durations[traj_idxs]
        )
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int32), np.ceil(p * n).astype(np.int32)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int32), np.ceil(p * n).astype(np.int32)
        all_frame_pos_starts = torch.zeros(len(traj_idxs), MotionLoaderE1_12DOF.POS_SIZE, device=self.device)
        all_frame_pos_ends = torch.zeros(len(traj_idxs), MotionLoaderE1_12DOF.POS_SIZE, device=self.device)
        all_frame_rot_starts = torch.zeros(len(traj_idxs), MotionLoaderE1_12DOF.ROT_SIZE, device=self.device)
        all_frame_rot_ends = torch.zeros(len(traj_idxs), MotionLoaderE1_12DOF.ROT_SIZE, device=self.device)
        all_frame_AMP_starts = torch.zeros(
            len(traj_idxs),
            MotionLoaderE1_12DOF.CONTACT_MASK_END_IDX - MotionLoaderE1_12DOF.JOINT_POSE_START_IDX,
            device=self.device,
        )
        all_frame_AMP_ends = torch.zeros(
            len(traj_idxs),
            MotionLoaderE1_12DOF.CONTACT_MASK_END_IDX - MotionLoaderE1_12DOF.JOINT_POSE_START_IDX,
            device=self.device,
        )
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_pos_starts[traj_mask] = MotionLoaderE1_12DOF.get_root_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = MotionLoaderE1_12DOF.get_root_pos_batch(trajectory[idx_high[traj_mask]])
            all_frame_rot_starts[traj_mask] = MotionLoaderE1_12DOF.get_root_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = MotionLoaderE1_12DOF.get_root_rot_batch(trajectory[idx_high[traj_mask]])
            all_frame_AMP_starts[traj_mask] = trajectory[idx_low[traj_mask]][
                :, MotionLoaderE1_12DOF.JOINT_POSE_START_IDX : MotionLoaderE1_12DOF.CONTACT_MASK_END_IDX
            ]
            all_frame_AMP_ends[traj_mask] = trajectory[idx_high[traj_mask]][
                :, MotionLoaderE1_12DOF.JOINT_POSE_START_IDX : MotionLoaderE1_12DOF.CONTACT_MASK_END_IDX
            ]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        rot_blend = torch_quaternion_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        AMP_blend = self.slerp(all_frame_AMP_starts, all_frame_AMP_ends, blend)
        return torch.cat([pos_blend, rot_blend, AMP_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        idxs = np.random.choice(self.preloaded_states.shape[0], size=num_frames)
        return self.preloaded_states[idxs, 0]

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """

        root_pos0, root_pos1 = MotionLoaderE1_12DOF.get_root_pos(frame0), MotionLoaderE1_12DOF.get_root_pos(frame1)
        root_rot0, root_rot1 = MotionLoaderE1_12DOF.get_root_rot(frame0), MotionLoaderE1_12DOF.get_root_rot(frame1)
        joints0, joints1 = MotionLoaderE1_12DOF.get_joint_pose(frame0), MotionLoaderE1_12DOF.get_joint_pose(frame1)
        tar_toe_pos_0, tar_toe_pos_1 = MotionLoaderE1_12DOF.get_key_pos_local(frame0), MotionLoaderE1_12DOF.get_key_pos_local(
            frame1
        )
        linear_vel_0, linear_vel_1 = MotionLoaderE1_12DOF.get_linear_vel(frame0), MotionLoaderE1_12DOF.get_linear_vel(frame1)
        angular_vel_0, angular_vel_1 = MotionLoaderE1_12DOF.get_angular_vel(frame0), MotionLoaderE1_12DOF.get_angular_vel(frame1)
        joint_vel_0, joint_vel_1 = MotionLoaderE1_12DOF.get_joint_vel(frame0), MotionLoaderE1_12DOF.get_joint_vel(frame1)

        blend_root_pos = self.slerp(root_pos0, root_pos1, blend)
        blend_root_rot = np_quaternion_slerp(root_rot0.cpu().numpy(), root_rot1.cpu().numpy(), blend)
        blend_root_rot = torch.tensor(standardize_quaternion(blend_root_rot), dtype=torch.float32, device=self.device)
        blend_joints = self.slerp(joints0, joints1, blend)
        blend_tar_toe_pos = self.slerp(tar_toe_pos_0, tar_toe_pos_1, blend)
        blend_linear_vel = self.slerp(linear_vel_0, linear_vel_1, blend)
        blend_angular_vel = self.slerp(angular_vel_0, angular_vel_1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat([
            blend_root_pos,
            blend_root_rot,
            blend_joints,
            blend_tar_toe_pos,
            blend_linear_vel,
            blend_angular_vel,
            blend_joints_vel,
        ])

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            ids = torch.randint(0, self.num_preload_transitions, (mini_batch_size,), device=self.device)
            states = self.preloaded_states[ids, :, self.observation_start_dim :]
            labels = self.motion_label[ids]
            yield states, labels

    @property
    def observation_full_dim(self):
        """Size of AMP full observations."""
        return self.trajectories_full[0].shape[1]

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1]

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_root_pos(pose):
        return pose[MotionLoaderE1_12DOF.ROOT_POS_START_IDX : MotionLoaderE1_12DOF.ROOT_POS_END_IDX]

    def get_root_pos_batch(poses):
        return poses[:, MotionLoaderE1_12DOF.ROOT_POS_START_IDX : MotionLoaderE1_12DOF.ROOT_POS_END_IDX]

    def get_root_rot(pose):
        return pose[MotionLoaderE1_12DOF.ROOT_ROT_START_IDX : MotionLoaderE1_12DOF.ROOT_ROT_END_IDX]

    def get_root_rot_batch(poses):
        return poses[:, MotionLoaderE1_12DOF.ROOT_ROT_START_IDX : MotionLoaderE1_12DOF.ROOT_ROT_END_IDX]

    def get_joint_pose(pose):
        return pose[MotionLoaderE1_12DOF.JOINT_POSE_START_IDX : MotionLoaderE1_12DOF.JOINT_POSE_END_IDX]

    def get_joint_pose_batch(poses):
        return poses[:, MotionLoaderE1_12DOF.JOINT_POSE_START_IDX : MotionLoaderE1_12DOF.JOINT_POSE_END_IDX]

    def get_key_pos_local(pose):
        return pose[MotionLoaderE1_12DOF.KEY_POS_LOCAL_START_IDX : MotionLoaderE1_12DOF.KEY_POS_LOCAL_END_IDX]

    def get_key_pos_local_batch(poses):
        return poses[:, MotionLoaderE1_12DOF.KEY_POS_LOCAL_START_IDX : MotionLoaderE1_12DOF.KEY_POS_LOCAL_END_IDX]

    def get_linear_vel(pose):
        return pose[MotionLoaderE1_12DOF.LINEAR_VEL_START_IDX : MotionLoaderE1_12DOF.LINEAR_VEL_END_IDX]

    def get_linear_vel_batch(poses):
        return poses[:, MotionLoaderE1_12DOF.LINEAR_VEL_START_IDX : MotionLoaderE1_12DOF.LINEAR_VEL_END_IDX]

    def get_angular_vel(pose):
        return pose[MotionLoaderE1_12DOF.ANGULAR_VEL_START_IDX : MotionLoaderE1_12DOF.ANGULAR_VEL_END_IDX]

    def get_angular_vel_batch(poses):
        return poses[:, MotionLoaderE1_12DOF.ANGULAR_VEL_START_IDX : MotionLoaderE1_12DOF.ANGULAR_VEL_END_IDX]

    def get_joint_vel(pose):
        return pose[MotionLoaderE1_12DOF.JOINT_VEL_START_IDX : MotionLoaderE1_12DOF.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(poses):
        return poses[:, MotionLoaderE1_12DOF.JOINT_VEL_START_IDX : MotionLoaderE1_12DOF.JOINT_VEL_END_IDX]

    def get_contact_mask_batch(poses):
        return poses[:, MotionLoaderE1_12DOF.CONTACT_MASK_START_IDX : MotionLoaderE1.CONTACT_MASK_END_IDX]


# ---------------------------------------- utils ----------------------------------------
def create_index_mapping(motion_joint_names, sim_joint_names):
    if set(motion_joint_names) != set(sim_joint_names):
        raise ValueError("两个关节列表包含不同的关节名称")

    target_name_to_index = {name: idx for idx, name in enumerate(sim_joint_names)}

    sim2motion_map = []
    for source_idx, name in enumerate(motion_joint_names):
        target_idx = target_name_to_index[name]
        sim2motion_map.append(target_idx)

    motion2sim_map = []
    for target_idx, name in enumerate(sim_joint_names):
        source_idx = motion_joint_names.index(name)
        motion2sim_map.append(source_idx)

    return {"sim2motion": sim2motion_map, "motion2sim": motion2sim_map}


def standardize_quaternion(q):
    if q[-1] < 0:
        q = -q
    return q


def QuaternionNormalize(q):
    q_norm = np.linalg.norm(q)
    if np.isclose(q_norm, 0.0):
        raise ValueError(f"Quaternion may not be zero in QuaternionNormalize: |q| = {q_norm:f}, q = {q}")
    return q / q_norm


def unit_vector(data, axis=None, out=None):
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def np_quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True, _EPS=1e-6):
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        q1 *= -1.0
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0


def torch_quaternion_slerp(q0, q1, fraction, spin=0, shortestpath=True, _EPS=1e-6):
    """Batch quaternion spherical linear interpolation."""

    out = torch.zeros_like(q0)

    zero_mask = torch.isclose(fraction, torch.zeros_like(fraction)).squeeze()
    ones_mask = torch.isclose(fraction, torch.ones_like(fraction)).squeeze()
    out[zero_mask] = q0[zero_mask]
    out[ones_mask] = q1[ones_mask]

    d = torch.sum(q0 * q1, dim=-1, keepdim=True)
    dist_mask = (torch.abs(torch.abs(d) - 1.0) < _EPS).squeeze()
    out[dist_mask] = q0[dist_mask]

    if shortestpath:
        d_old = torch.clone(d)
        d = torch.where(d_old < 0, -d, d)
        q1 = torch.where(d_old < 0, -q1, q1)

    # TODO Clip d to be in the range [-1, 1] to avoid nan from acos
    d = torch.clamp(d, -1.0 + _EPS, 1.0 - _EPS)

    angle = torch.acos(d) + spin * torch.pi
    angle_mask = (torch.abs(angle) < _EPS).squeeze()
    out[angle_mask] = q0[angle_mask]

    final_mask = torch.logical_or(zero_mask, ones_mask)
    final_mask = torch.logical_or(final_mask, dist_mask)
    final_mask = torch.logical_or(final_mask, angle_mask)
    final_mask = torch.logical_not(final_mask)

    # isin = 1.0 / angle
    isin = 1.0 / (angle + _EPS)  # TODO Avoid division by zero
    q0 *= torch.sin((1.0 - fraction) * angle) * isin
    q1 *= torch.sin(fraction * angle) * isin
    q0 += q1
    out[final_mask] = q0[final_mask]
    return out


class MotionLoaderBumi1:
    POS_SIZE = 3
    ROT_SIZE = 4

    JOINT_POS_SIZE = 21
    KEY_POS_LOCAL_SIZE = 66  # 22 bodies * 3 (base_link + 21 other bodies)
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_VEL_SIZE = 21
    CONTACT_MASK_SIZE = 2

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + POS_SIZE

    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE

    JOINT_POSE_START_IDX = ROOT_ROT_END_IDX
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    KEY_POS_LOCAL_START_IDX = JOINT_POSE_END_IDX
    KEY_POS_LOCAL_END_IDX = KEY_POS_LOCAL_START_IDX + KEY_POS_LOCAL_SIZE

    LINEAR_VEL_START_IDX = KEY_POS_LOCAL_END_IDX
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE

    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE

    JOINT_VEL_START_IDX = ANGULAR_VEL_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    CONTACT_MASK_START_IDX = JOINT_VEL_END_IDX
    CONTACT_MASK_END_IDX = CONTACT_MASK_START_IDX + CONTACT_MASK_SIZE

    def __init__(
        self,
        device,
        time_between_frames,
        reference_observation_horizon=2,
        num_preload_transitions=10,
        motion_files="",
        sim_joint_names="",
        sim_key_pos_names="",
    ):
        """Expert dataset provides AMP observations from Human mocap dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames
        self.reference_observation_horizon = reference_observation_horizon
        self.num_preload_transitions = num_preload_transitions
        self.observation_start_dim = MotionLoaderBumi1.JOINT_POSE_START_IDX

        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split(".")[0])
            with open(motion_file) as f:
                motion_json = json.load(f)
                if i == 0:
                    motion_joint_names = motion_json["JointNames"]
                    motion_key_pos_names = motion_json["KeyPosNames"]
                    self.joint_mapping = create_index_mapping(motion_joint_names, sim_joint_names)
                    self.key_pos_mapping = create_index_mapping(motion_key_pos_names, sim_key_pos_names)
                motion_data = np.array(motion_json["Frames"])

                # Normalize and standardize quaternions.
                for f_i in range(motion_data.shape[0]):
                    root_rot = MotionLoaderBumi1.get_root_rot(motion_data[f_i])
                    root_rot = QuaternionNormalize(root_rot)
                    root_rot = standardize_quaternion(root_rot)
                    motion_data[f_i, MotionLoaderBumi1.POS_SIZE : (MotionLoaderBumi1.POS_SIZE + MotionLoaderBumi1.ROT_SIZE)] = (
                        root_rot
                    )

                # Remove first 7 observation dimensions (root_pos, root_rot).
                self.trajectories.append(
                    torch.tensor(
                        motion_data[:, MotionLoaderBumi1.ROOT_ROT_END_IDX : MotionLoaderBumi1.CONTACT_MASK_END_IDX],
                        dtype=torch.float32,
                        device=device,
                    )
                )
                self.trajectories_full.append(
                    torch.tensor(
                        motion_data[:, : MotionLoaderBumi1.CONTACT_MASK_END_IDX], dtype=torch.float32, device=device
                    )
                )
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f"Loaded {traj_len}s. motion from {motion_file}.")

        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        print(f"Preloading {self.num_preload_transitions} transitions")
        traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
        times = self.traj_time_sample_batch(traj_idxs)
        self.preloaded_states = torch.zeros(
            self.num_preload_transitions,
            self.reference_observation_horizon,
            self.trajectories_full[0].shape[1],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.motion_label = torch.nn.functional.one_hot(
            torch.tensor(traj_idxs, dtype=torch.long, device=self.device)
        ).float()

        for i in range(self.reference_observation_horizon):
            self.preloaded_states[:, i] = self.get_full_frame_at_time_batch(
                traj_idxs, times + self.time_between_frames * i
            )
        print("Finished preloading")

        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = (
            self.time_between_frames * (self.reference_observation_horizon - 1)
            + self.trajectory_frame_durations[traj_idx]
        )
        return max(0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = (
            self.time_between_frames * (self.reference_observation_horizon - 1)
            + self.trajectory_frame_durations[traj_idxs]
        )
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int32), np.ceil(p * n).astype(np.int32)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int32), np.ceil(p * n).astype(np.int32)
        all_frame_pos_starts = torch.zeros(len(traj_idxs), MotionLoaderBumi1.POS_SIZE, device=self.device)
        all_frame_pos_ends = torch.zeros(len(traj_idxs), MotionLoaderBumi1.POS_SIZE, device=self.device)
        all_frame_rot_starts = torch.zeros(len(traj_idxs), MotionLoaderBumi1.ROT_SIZE, device=self.device)
        all_frame_rot_ends = torch.zeros(len(traj_idxs), MotionLoaderBumi1.ROT_SIZE, device=self.device)
        all_frame_AMP_starts = torch.zeros(
            len(traj_idxs),
            MotionLoaderBumi1.CONTACT_MASK_END_IDX - MotionLoaderBumi1.JOINT_POSE_START_IDX,
            device=self.device,
        )
        all_frame_AMP_ends = torch.zeros(
            len(traj_idxs),
            MotionLoaderBumi1.CONTACT_MASK_END_IDX - MotionLoaderBumi1.JOINT_POSE_START_IDX,
            device=self.device,
        )
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_pos_starts[traj_mask] = MotionLoaderBumi1.get_root_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = MotionLoaderBumi1.get_root_pos_batch(trajectory[idx_high[traj_mask]])
            all_frame_rot_starts[traj_mask] = MotionLoaderBumi1.get_root_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = MotionLoaderBumi1.get_root_rot_batch(trajectory[idx_high[traj_mask]])
            all_frame_AMP_starts[traj_mask] = trajectory[idx_low[traj_mask]][
                :, MotionLoaderBumi1.JOINT_POSE_START_IDX : MotionLoaderBumi1.CONTACT_MASK_END_IDX
            ]
            all_frame_AMP_ends[traj_mask] = trajectory[idx_high[traj_mask]][
                :, MotionLoaderBumi1.JOINT_POSE_START_IDX : MotionLoaderBumi1.CONTACT_MASK_END_IDX
            ]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        rot_blend = torch_quaternion_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        AMP_blend = self.slerp(all_frame_AMP_starts, all_frame_AMP_ends, blend)
        return torch.cat([pos_blend, rot_blend, AMP_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        idxs = np.random.choice(self.preloaded_states.shape[0], size=num_frames)
        return self.preloaded_states[idxs, 0]

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """

        root_pos0, root_pos1 = MotionLoaderBumi1.get_root_pos(frame0), MotionLoaderBumi1.get_root_pos(frame1)
        root_rot0, root_rot1 = MotionLoaderBumi1.get_root_rot(frame0), MotionLoaderBumi1.get_root_rot(frame1)
        joints0, joints1 = MotionLoaderBumi1.get_joint_pose(frame0), MotionLoaderBumi1.get_joint_pose(frame1)
        tar_toe_pos_0, tar_toe_pos_1 = MotionLoaderBumi1.get_key_pos_local(frame0), MotionLoaderBumi1.get_key_pos_local(
            frame1
        )
        linear_vel_0, linear_vel_1 = MotionLoaderBumi1.get_linear_vel(frame0), MotionLoaderBumi1.get_linear_vel(frame1)
        angular_vel_0, angular_vel_1 = MotionLoaderBumi1.get_angular_vel(frame0), MotionLoaderBumi1.get_angular_vel(frame1)
        joint_vel_0, joint_vel_1 = MotionLoaderBumi1.get_joint_vel(frame0), MotionLoaderBumi1.get_joint_vel(frame1)

        blend_root_pos = self.slerp(root_pos0, root_pos1, blend)
        blend_root_rot = np_quaternion_slerp(root_rot0.cpu().numpy(), root_rot1.cpu().numpy(), blend)
        blend_root_rot = torch.tensor(standardize_quaternion(blend_root_rot), dtype=torch.float32, device=self.device)
        blend_joints = self.slerp(joints0, joints1, blend)
        blend_tar_toe_pos = self.slerp(tar_toe_pos_0, tar_toe_pos_1, blend)
        blend_linear_vel = self.slerp(linear_vel_0, linear_vel_1, blend)
        blend_angular_vel = self.slerp(angular_vel_0, angular_vel_1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat([
            blend_root_pos,
            blend_root_rot,
            blend_joints,
            blend_tar_toe_pos,
            blend_linear_vel,
            blend_angular_vel,
            blend_joints_vel,
        ])

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            ids = torch.randint(0, self.num_preload_transitions, (mini_batch_size,), device=self.device)
            states = self.preloaded_states[ids, :, self.observation_start_dim :]
            labels = self.motion_label[ids]
            yield states, labels

    @property
    def observation_full_dim(self):
        """Size of AMP full observations."""
        return self.trajectories_full[0].shape[1]

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1]

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_root_pos(pose):
        return pose[MotionLoaderBumi1.ROOT_POS_START_IDX : MotionLoaderBumi1.ROOT_POS_END_IDX]

    def get_root_pos_batch(poses):
        return poses[:, MotionLoaderBumi1.ROOT_POS_START_IDX : MotionLoaderBumi1.ROOT_POS_END_IDX]

    def get_root_rot(pose):
        return pose[MotionLoaderBumi1.ROOT_ROT_START_IDX : MotionLoaderBumi1.ROOT_ROT_END_IDX]

    def get_root_rot_batch(poses):
        return poses[:, MotionLoaderBumi1.ROOT_ROT_START_IDX : MotionLoaderBumi1.ROOT_ROT_END_IDX]

    def get_joint_pose(pose):
        return pose[MotionLoaderBumi1.JOINT_POSE_START_IDX : MotionLoaderBumi1.JOINT_POSE_END_IDX]

    def get_joint_pose_batch(poses):
        return poses[:, MotionLoaderBumi1.JOINT_POSE_START_IDX : MotionLoaderBumi1.JOINT_POSE_END_IDX]

    def get_key_pos_local(pose):
        return pose[MotionLoaderBumi1.KEY_POS_LOCAL_START_IDX : MotionLoaderBumi1.KEY_POS_LOCAL_END_IDX]

    def get_key_pos_local_batch(poses):
        return poses[:, MotionLoaderBumi1.KEY_POS_LOCAL_START_IDX : MotionLoaderBumi1.KEY_POS_LOCAL_END_IDX]

    def get_linear_vel(pose):
        return pose[MotionLoaderBumi1.LINEAR_VEL_START_IDX : MotionLoaderBumi1.LINEAR_VEL_END_IDX]

    def get_linear_vel_batch(poses):
        return poses[:, MotionLoaderBumi1.LINEAR_VEL_START_IDX : MotionLoaderBumi1.LINEAR_VEL_END_IDX]

    def get_angular_vel(pose):
        return pose[MotionLoaderBumi1.ANGULAR_VEL_START_IDX : MotionLoaderBumi1.ANGULAR_VEL_END_IDX]

    def get_angular_vel_batch(poses):
        return poses[:, MotionLoaderBumi1.ANGULAR_VEL_START_IDX : MotionLoaderBumi1.ANGULAR_VEL_END_IDX]

    def get_joint_vel(pose):
        return pose[MotionLoaderBumi1.JOINT_VEL_START_IDX : MotionLoaderBumi1.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(poses):
        return poses[:, MotionLoaderBumi1.JOINT_VEL_START_IDX : MotionLoaderBumi1.JOINT_VEL_END_IDX]

    def get_contact_mask_batch(poses):
        return poses[:, MotionLoaderBumi1.CONTACT_MASK_START_IDX : MotionLoaderBumi1.CONTACT_MASK_END_IDX]


class MotionLoaderBumi:
    """Motion loader for Bumi 0.5"""
    POS_SIZE = 3
    ROT_SIZE = 4

    JOINT_POS_SIZE = 21
    KEY_POS_LOCAL_SIZE = (JOINT_POS_SIZE + 1) * 3
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_VEL_SIZE = JOINT_POS_SIZE
    CONTACT_MASK_SIZE = 2

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + POS_SIZE

    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE

    JOINT_POSE_START_IDX = ROOT_ROT_END_IDX
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    KEY_POS_LOCAL_START_IDX = JOINT_POSE_END_IDX
    KEY_POS_LOCAL_END_IDX = KEY_POS_LOCAL_START_IDX + KEY_POS_LOCAL_SIZE

    LINEAR_VEL_START_IDX = KEY_POS_LOCAL_END_IDX
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE

    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE

    JOINT_VEL_START_IDX = ANGULAR_VEL_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    CONTACT_MASK_START_IDX = JOINT_VEL_END_IDX
    CONTACT_MASK_END_IDX = CONTACT_MASK_START_IDX + CONTACT_MASK_SIZE

    def __init__(
        self,
        device,
        time_between_frames,
        reference_observation_horizon=2,
        num_preload_transitions=10,
        motion_files="",
        sim_joint_names="",
        sim_key_pos_names="",
    ):
        """Expert dataset provides AMP observations from Human mocap dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames
        self.reference_observation_horizon = reference_observation_horizon
        self.num_preload_transitions = num_preload_transitions
        self.observation_start_dim = MotionLoaderBumi.JOINT_POSE_START_IDX

        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split(".")[0])
            with open(motion_file) as f:
                motion_json = json.load(f)
                if i == 0:
                    motion_joint_names = motion_json["JointNames"]
                    motion_key_pos_names = motion_json["KeyPosNames"]
                    self.joint_mapping = create_index_mapping(motion_joint_names, sim_joint_names)
                    self.key_pos_mapping = create_index_mapping(motion_key_pos_names, sim_key_pos_names)
                motion_data = np.array(motion_json["Frames"])

                # Normalize and standardize quaternions.
                for f_i in range(motion_data.shape[0]):
                    root_rot = MotionLoaderBumi.get_root_rot(motion_data[f_i])
                    root_rot = QuaternionNormalize(root_rot)
                    root_rot = standardize_quaternion(root_rot)
                    motion_data[f_i, MotionLoaderBumi.POS_SIZE : (MotionLoaderBumi.POS_SIZE + MotionLoaderBumi.ROT_SIZE)] = (
                        root_rot
                    )

                # Remove first 7 observation dimensions (root_pos, root_rot).
                self.trajectories.append(
                    torch.tensor(
                        motion_data[:, MotionLoaderBumi.ROOT_ROT_END_IDX : MotionLoaderBumi.CONTACT_MASK_END_IDX],
                        dtype=torch.float32,
                        device=device,
                    )
                )
                self.trajectories_full.append(
                    torch.tensor(
                        motion_data[:, : MotionLoaderBumi.CONTACT_MASK_END_IDX], dtype=torch.float32, device=device
                    )
                )
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f"Loaded {traj_len}s. motion from {motion_file}.")

        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        print(f"Preloading {self.num_preload_transitions} transitions")
        traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
        times = self.traj_time_sample_batch(traj_idxs)
        self.preloaded_states = torch.zeros(
            self.num_preload_transitions,
            self.reference_observation_horizon,
            self.trajectories_full[0].shape[1],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.motion_label = torch.nn.functional.one_hot(
            torch.tensor(traj_idxs, dtype=torch.long, device=self.device)
        ).float()

        for i in range(self.reference_observation_horizon):
            self.preloaded_states[:, i] = self.get_full_frame_at_time_batch(
                traj_idxs, times + self.time_between_frames * i
            )
        print("Finished preloading")

        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = (
            self.time_between_frames * (self.reference_observation_horizon - 1)
            + self.trajectory_frame_durations[traj_idx]
        )
        return max(0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = (
            self.time_between_frames * (self.reference_observation_horizon - 1)
            + self.trajectory_frame_durations[traj_idxs]
        )
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int32), np.ceil(p * n).astype(np.int32)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int32), np.ceil(p * n).astype(np.int32)
        all_frame_pos_starts = torch.zeros(len(traj_idxs), MotionLoaderBumi.POS_SIZE, device=self.device)
        all_frame_pos_ends = torch.zeros(len(traj_idxs), MotionLoaderBumi.POS_SIZE, device=self.device)
        all_frame_rot_starts = torch.zeros(len(traj_idxs), MotionLoaderBumi.ROT_SIZE, device=self.device)
        all_frame_rot_ends = torch.zeros(len(traj_idxs), MotionLoaderBumi.ROT_SIZE, device=self.device)
        all_frame_AMP_starts = torch.zeros(
            len(traj_idxs),
            MotionLoaderBumi.CONTACT_MASK_END_IDX - MotionLoaderBumi.JOINT_POSE_START_IDX,
            device=self.device,
        )
        all_frame_AMP_ends = torch.zeros(
            len(traj_idxs),
            MotionLoaderBumi.CONTACT_MASK_END_IDX - MotionLoaderBumi.JOINT_POSE_START_IDX,
            device=self.device,
        )
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_pos_starts[traj_mask] = MotionLoaderBumi.get_root_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = MotionLoaderBumi.get_root_pos_batch(trajectory[idx_high[traj_mask]])
            all_frame_rot_starts[traj_mask] = MotionLoaderBumi.get_root_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = MotionLoaderBumi.get_root_rot_batch(trajectory[idx_high[traj_mask]])
            all_frame_AMP_starts[traj_mask] = trajectory[idx_low[traj_mask]][
                :, MotionLoaderBumi.JOINT_POSE_START_IDX : MotionLoaderBumi.CONTACT_MASK_END_IDX
            ]
            all_frame_AMP_ends[traj_mask] = trajectory[idx_high[traj_mask]][
                :, MotionLoaderBumi.JOINT_POSE_START_IDX : MotionLoaderBumi.CONTACT_MASK_END_IDX
            ]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        rot_blend = torch_quaternion_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        AMP_blend = self.slerp(all_frame_AMP_starts, all_frame_AMP_ends, blend)
        return torch.cat([pos_blend, rot_blend, AMP_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        idxs = np.random.choice(self.preloaded_states.shape[0], size=num_frames)
        return self.preloaded_states[idxs, 0]

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """

        root_pos0, root_pos1 = MotionLoaderBumi.get_root_pos(frame0), MotionLoaderBumi.get_root_pos(frame1)
        root_rot0, root_rot1 = MotionLoaderBumi.get_root_rot(frame0), MotionLoaderBumi.get_root_rot(frame1)
        joints0, joints1 = MotionLoaderBumi.get_joint_pose(frame0), MotionLoaderBumi.get_joint_pose(frame1)
        tar_toe_pos_0, tar_toe_pos_1 = MotionLoaderBumi.get_key_pos_local(frame0), MotionLoaderBumi.get_key_pos_local(
            frame1
        )
        linear_vel_0, linear_vel_1 = MotionLoaderBumi.get_linear_vel(frame0), MotionLoaderBumi.get_linear_vel(frame1)
        angular_vel_0, angular_vel_1 = MotionLoaderBumi.get_angular_vel(frame0), MotionLoaderBumi.get_angular_vel(frame1)
        joint_vel_0, joint_vel_1 = MotionLoaderBumi.get_joint_vel(frame0), MotionLoaderBumi.get_joint_vel(frame1)

        blend_root_pos = self.slerp(root_pos0, root_pos1, blend)
        blend_root_rot = np_quaternion_slerp(root_rot0.cpu().numpy(), root_rot1.cpu().numpy(), blend)
        blend_root_rot = torch.tensor(standardize_quaternion(blend_root_rot), dtype=torch.float32, device=self.device)
        blend_joints = self.slerp(joints0, joints1, blend)
        blend_tar_toe_pos = self.slerp(tar_toe_pos_0, tar_toe_pos_1, blend)
        blend_linear_vel = self.slerp(linear_vel_0, linear_vel_1, blend)
        blend_angular_vel = self.slerp(angular_vel_0, angular_vel_1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat([
            blend_root_pos,
            blend_root_rot,
            blend_joints,
            blend_tar_toe_pos,
            blend_linear_vel,
            blend_angular_vel,
            blend_joints_vel,
        ])

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            ids = torch.randint(0, self.num_preload_transitions, (mini_batch_size,), device=self.device)
            states = self.preloaded_states[ids, :, self.observation_start_dim :]
            labels = self.motion_label[ids]
            yield states, labels

    @property
    def observation_full_dim(self):
        """Size of AMP full observations."""
        return self.trajectories_full[0].shape[1]

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1]

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_root_pos(pose):
        return pose[MotionLoaderBumi.ROOT_POS_START_IDX : MotionLoaderBumi.ROOT_POS_END_IDX]

    def get_root_pos_batch(poses):
        return poses[:, MotionLoaderBumi.ROOT_POS_START_IDX : MotionLoaderBumi.ROOT_POS_END_IDX]

    def get_root_rot(pose):
        return pose[MotionLoaderBumi.ROOT_ROT_START_IDX : MotionLoaderBumi.ROOT_ROT_END_IDX]

    def get_root_rot_batch(poses):
        return poses[:, MotionLoaderBumi.ROOT_ROT_START_IDX : MotionLoaderBumi.ROOT_ROT_END_IDX]

    def get_joint_pose(pose):
        return pose[MotionLoaderBumi.JOINT_POSE_START_IDX : MotionLoaderBumi.JOINT_POSE_END_IDX]

    def get_joint_pose_batch(poses):
        return poses[:, MotionLoaderBumi.JOINT_POSE_START_IDX : MotionLoaderBumi.JOINT_POSE_END_IDX]

    def get_key_pos_local(pose):
        return pose[MotionLoaderBumi.KEY_POS_LOCAL_START_IDX : MotionLoaderBumi.KEY_POS_LOCAL_END_IDX]

    def get_key_pos_local_batch(poses):
        return poses[:, MotionLoaderBumi.KEY_POS_LOCAL_START_IDX : MotionLoaderBumi.KEY_POS_LOCAL_END_IDX]

    def get_linear_vel(pose):
        return pose[MotionLoaderBumi.LINEAR_VEL_START_IDX : MotionLoaderBumi.LINEAR_VEL_END_IDX]

    def get_linear_vel_batch(poses):
        return poses[:, MotionLoaderBumi.LINEAR_VEL_START_IDX : MotionLoaderBumi.LINEAR_VEL_END_IDX]

    def get_angular_vel(pose):
        return pose[MotionLoaderBumi.ANGULAR_VEL_START_IDX : MotionLoaderBumi.ANGULAR_VEL_END_IDX]

    def get_angular_vel_batch(poses):
        return poses[:, MotionLoaderBumi.ANGULAR_VEL_START_IDX : MotionLoaderBumi.ANGULAR_VEL_END_IDX]

    def get_joint_vel(pose):
        return pose[MotionLoaderBumi.JOINT_VEL_START_IDX : MotionLoaderBumi.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(poses):
        return poses[:, MotionLoaderBumi.JOINT_VEL_START_IDX : MotionLoaderBumi.JOINT_VEL_END_IDX]

    def get_contact_mask_batch(poses):
        return poses[:, MotionLoaderBumi.CONTACT_MASK_START_IDX : MotionLoaderBumi.CONTACT_MASK_END_IDX]


class MotionLoaderBumi_12DOF:
    """Motion loader for Bumi 0.5 with 12DOF"""
    POS_SIZE = 3
    ROT_SIZE = 4

    JOINT_POS_SIZE = 12
    KEY_POS_LOCAL_SIZE = (JOINT_POS_SIZE + 1) * 3
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_VEL_SIZE = JOINT_POS_SIZE
    CONTACT_MASK_SIZE = 2

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + POS_SIZE

    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE

    JOINT_POSE_START_IDX = ROOT_ROT_END_IDX
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    KEY_POS_LOCAL_START_IDX = JOINT_POSE_END_IDX
    KEY_POS_LOCAL_END_IDX = KEY_POS_LOCAL_START_IDX + KEY_POS_LOCAL_SIZE

    LINEAR_VEL_START_IDX = KEY_POS_LOCAL_END_IDX
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE

    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE

    JOINT_VEL_START_IDX = ANGULAR_VEL_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    CONTACT_MASK_START_IDX = JOINT_VEL_END_IDX
    CONTACT_MASK_END_IDX = CONTACT_MASK_START_IDX + CONTACT_MASK_SIZE

    def __init__(
        self,
        device,
        time_between_frames,
        reference_observation_horizon=2,
        num_preload_transitions=10,
        motion_files="",
        sim_joint_names="",
        sim_key_pos_names="",
    ):
        """Expert dataset provides AMP observations from Human mocap dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames
        self.reference_observation_horizon = reference_observation_horizon
        self.num_preload_transitions = num_preload_transitions
        self.observation_start_dim = MotionLoaderBumi.JOINT_POSE_START_IDX

        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split(".")[0])
            with open(motion_file) as f:
                motion_json = json.load(f)
                if i == 0:
                    motion_12dof_joint_index = [i for i, joint_name in enumerate(motion_json["JointNames"]) if joint_name in sim_joint_names]
                    motion_12dof_key_pos_index = [i for i, key_pos_name in enumerate(motion_json["KeyPosNames"]) if key_pos_name in sim_key_pos_names]
                    motion_joint_names = [motion_json["JointNames"][i] for i in motion_12dof_joint_index]
                    motion_key_pos_names = [motion_json["KeyPosNames"][i] for i in motion_12dof_key_pos_index]
                    self.joint_mapping = create_index_mapping(motion_joint_names, sim_joint_names)
                    self.key_pos_mapping = create_index_mapping(motion_key_pos_names, sim_key_pos_names)
                motion_data = np.array(motion_json["Frames"])
                # 21dof to 12 dof
                motion_12dof_key_pos_data_index = []
                for key_pos_idx in motion_12dof_key_pos_index:
                    # 每个关键点位置在数据中占用3个连续位置
                    start_idx = key_pos_idx * 3
                    motion_12dof_key_pos_data_index.extend([start_idx, start_idx + 1, start_idx + 2])
                motion_data = np.concatenate(
                    [
                        motion_data[:, :MotionLoaderBumi.JOINT_POSE_START_IDX],
                        motion_data[:, MotionLoaderBumi.JOINT_POSE_START_IDX: MotionLoaderBumi.JOINT_POSE_END_IDX][:, motion_12dof_joint_index],
                        motion_data[:, MotionLoaderBumi.KEY_POS_LOCAL_START_IDX: MotionLoaderBumi.KEY_POS_LOCAL_END_IDX][:, motion_12dof_key_pos_data_index],
                        motion_data[:, MotionLoaderBumi.LINEAR_VEL_START_IDX: MotionLoaderBumi.LINEAR_VEL_END_IDX],
                        motion_data[:, MotionLoaderBumi.ANGULAR_VEL_START_IDX: MotionLoaderBumi.ANGULAR_VEL_END_IDX],
                        motion_data[:, MotionLoaderBumi.JOINT_VEL_START_IDX: MotionLoaderBumi.JOINT_VEL_END_IDX][:, motion_12dof_joint_index],
                        motion_data[:, MotionLoaderBumi.CONTACT_MASK_START_IDX: MotionLoaderBumi.CONTACT_MASK_END_IDX],
                    ],
                    axis=1)

                # Normalize and standardize quaternions.
                for f_i in range(motion_data.shape[0]):
                    root_rot = MotionLoaderBumi_12DOF.get_root_rot(motion_data[f_i])
                    root_rot = QuaternionNormalize(root_rot)
                    root_rot = standardize_quaternion(root_rot)
                    motion_data[f_i, MotionLoaderBumi_12DOF.POS_SIZE : (MotionLoaderBumi_12DOF.POS_SIZE + MotionLoaderBumi_12DOF.ROT_SIZE)] = (
                        root_rot
                    )

                # Remove first 7 observation dimensions (root_pos, root_rot).
                self.trajectories.append(
                    torch.tensor(
                        motion_data[:, MotionLoaderBumi_12DOF.ROOT_ROT_END_IDX : MotionLoaderBumi_12DOF.CONTACT_MASK_END_IDX],
                        dtype=torch.float32,
                        device=device,
                    )
                )
                self.trajectories_full.append(
                    torch.tensor(
                        motion_data[:, : MotionLoaderBumi_12DOF.CONTACT_MASK_END_IDX], dtype=torch.float32, device=device
                    )
                )
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f"Loaded {traj_len}s. motion from {motion_file}.")

        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        print(f"Preloading {self.num_preload_transitions} transitions")
        traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
        times = self.traj_time_sample_batch(traj_idxs)
        self.preloaded_states = torch.zeros(
            self.num_preload_transitions,
            self.reference_observation_horizon,
            self.trajectories_full[0].shape[1],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.motion_label = torch.nn.functional.one_hot(
            torch.tensor(traj_idxs, dtype=torch.long, device=self.device)
        ).float()

        for i in range(self.reference_observation_horizon):
            self.preloaded_states[:, i] = self.get_full_frame_at_time_batch(
                traj_idxs, times + self.time_between_frames * i
            )
        print("Finished preloading")

        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = (
            self.time_between_frames * (self.reference_observation_horizon - 1)
            + self.trajectory_frame_durations[traj_idx]
        )
        return max(0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = (
            self.time_between_frames * (self.reference_observation_horizon - 1)
            + self.trajectory_frame_durations[traj_idxs]
        )
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int32), np.ceil(p * n).astype(np.int32)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int32), np.ceil(p * n).astype(np.int32)
        all_frame_pos_starts = torch.zeros(len(traj_idxs), MotionLoaderBumi_12DOF.POS_SIZE, device=self.device)
        all_frame_pos_ends = torch.zeros(len(traj_idxs), MotionLoaderBumi_12DOF.POS_SIZE, device=self.device)
        all_frame_rot_starts = torch.zeros(len(traj_idxs), MotionLoaderBumi_12DOF.ROT_SIZE, device=self.device)
        all_frame_rot_ends = torch.zeros(len(traj_idxs), MotionLoaderBumi_12DOF.ROT_SIZE, device=self.device)
        all_frame_AMP_starts = torch.zeros(
            len(traj_idxs),
            MotionLoaderBumi_12DOF.CONTACT_MASK_END_IDX - MotionLoaderBumi_12DOF.JOINT_POSE_START_IDX,
            device=self.device,
        )
        all_frame_AMP_ends = torch.zeros(
            len(traj_idxs),
            MotionLoaderBumi_12DOF.CONTACT_MASK_END_IDX - MotionLoaderBumi_12DOF.JOINT_POSE_START_IDX,
            device=self.device,
        )
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_pos_starts[traj_mask] = MotionLoaderBumi_12DOF.get_root_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = MotionLoaderBumi_12DOF.get_root_pos_batch(trajectory[idx_high[traj_mask]])
            all_frame_rot_starts[traj_mask] = MotionLoaderBumi_12DOF.get_root_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = MotionLoaderBumi_12DOF.get_root_rot_batch(trajectory[idx_high[traj_mask]])
            all_frame_AMP_starts[traj_mask] = trajectory[idx_low[traj_mask]][
                :, MotionLoaderBumi_12DOF.JOINT_POSE_START_IDX : MotionLoaderBumi_12DOF.CONTACT_MASK_END_IDX
            ]
            all_frame_AMP_ends[traj_mask] = trajectory[idx_high[traj_mask]][
                :, MotionLoaderBumi_12DOF.JOINT_POSE_START_IDX : MotionLoaderBumi_12DOF.CONTACT_MASK_END_IDX
            ]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        rot_blend = torch_quaternion_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        AMP_blend = self.slerp(all_frame_AMP_starts, all_frame_AMP_ends, blend)
        return torch.cat([pos_blend, rot_blend, AMP_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        idxs = np.random.choice(self.preloaded_states.shape[0], size=num_frames)
        return self.preloaded_states[idxs, 0]

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """

        root_pos0, root_pos1 = MotionLoaderBumi_12DOF.get_root_pos(frame0), MotionLoaderBumi_12DOF.get_root_pos(frame1)
        root_rot0, root_rot1 = MotionLoaderBumi_12DOF.get_root_rot(frame0), MotionLoaderBumi_12DOF.get_root_rot(frame1)
        joints0, joints1 = MotionLoaderBumi_12DOF.get_joint_pose(frame0), MotionLoaderBumi_12DOF.get_joint_pose(frame1)
        tar_toe_pos_0, tar_toe_pos_1 = MotionLoaderBumi_12DOF.get_key_pos_local(frame0), MotionLoaderBumi_12DOF.get_key_pos_local(
            frame1
        )
        linear_vel_0, linear_vel_1 = MotionLoaderBumi_12DOF.get_linear_vel(frame0), MotionLoaderBumi_12DOF.get_linear_vel(frame1)
        angular_vel_0, angular_vel_1 = MotionLoaderBumi_12DOF.get_angular_vel(frame0), MotionLoaderBumi_12DOF.get_angular_vel(frame1)
        joint_vel_0, joint_vel_1 = MotionLoaderBumi_12DOF.get_joint_vel(frame0), MotionLoaderBumi_12DOF.get_joint_vel(frame1)

        blend_root_pos = self.slerp(root_pos0, root_pos1, blend)
        blend_root_rot = np_quaternion_slerp(root_rot0.cpu().numpy(), root_rot1.cpu().numpy(), blend)
        blend_root_rot = torch.tensor(standardize_quaternion(blend_root_rot), dtype=torch.float32, device=self.device)
        blend_joints = self.slerp(joints0, joints1, blend)
        blend_tar_toe_pos = self.slerp(tar_toe_pos_0, tar_toe_pos_1, blend)
        blend_linear_vel = self.slerp(linear_vel_0, linear_vel_1, blend)
        blend_angular_vel = self.slerp(angular_vel_0, angular_vel_1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat([
            blend_root_pos,
            blend_root_rot,
            blend_joints,
            blend_tar_toe_pos,
            blend_linear_vel,
            blend_angular_vel,
            blend_joints_vel,
        ])

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            ids = torch.randint(0, self.num_preload_transitions, (mini_batch_size,), device=self.device)
            states = self.preloaded_states[ids, :, self.observation_start_dim :]
            labels = self.motion_label[ids]
            yield states, labels

    @property
    def observation_full_dim(self):
        """Size of AMP full observations."""
        return self.trajectories_full[0].shape[1]

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1]

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_root_pos(pose):
        return pose[MotionLoaderBumi_12DOF.ROOT_POS_START_IDX : MotionLoaderBumi_12DOF.ROOT_POS_END_IDX]

    def get_root_pos_batch(poses):
        return poses[:, MotionLoaderBumi_12DOF.ROOT_POS_START_IDX : MotionLoaderBumi_12DOF.ROOT_POS_END_IDX]

    def get_root_rot(pose):
        return pose[MotionLoaderBumi_12DOF.ROOT_ROT_START_IDX : MotionLoaderBumi_12DOF.ROOT_ROT_END_IDX]

    def get_root_rot_batch(poses):
        return poses[:, MotionLoaderBumi_12DOF.ROOT_ROT_START_IDX : MotionLoaderBumi_12DOF.ROOT_ROT_END_IDX]

    def get_joint_pose(pose):
        return pose[MotionLoaderBumi_12DOF.JOINT_POSE_START_IDX : MotionLoaderBumi_12DOF.JOINT_POSE_END_IDX]

    def get_joint_pose_batch(poses):
        return poses[:, MotionLoaderBumi_12DOF.JOINT_POSE_START_IDX : MotionLoaderBumi_12DOF.JOINT_POSE_END_IDX]

    def get_key_pos_local(pose):
        return pose[MotionLoaderBumi_12DOF.KEY_POS_LOCAL_START_IDX : MotionLoaderBumi_12DOF.KEY_POS_LOCAL_END_IDX]

    def get_key_pos_local_batch(poses):
        return poses[:, MotionLoaderBumi_12DOF.KEY_POS_LOCAL_START_IDX : MotionLoaderBumi_12DOF.KEY_POS_LOCAL_END_IDX]

    def get_linear_vel(pose):
        return pose[MotionLoaderBumi_12DOF.LINEAR_VEL_START_IDX : MotionLoaderBumi_12DOF.LINEAR_VEL_END_IDX]

    def get_linear_vel_batch(poses):
        return poses[:, MotionLoaderBumi_12DOF.LINEAR_VEL_START_IDX : MotionLoaderBumi_12DOF.LINEAR_VEL_END_IDX]

    def get_angular_vel(pose):
        return pose[MotionLoaderBumi_12DOF.ANGULAR_VEL_START_IDX : MotionLoaderBumi_12DOF.ANGULAR_VEL_END_IDX]

    def get_angular_vel_batch(poses):
        return poses[:, MotionLoaderBumi_12DOF.ANGULAR_VEL_START_IDX : MotionLoaderBumi_12DOF.ANGULAR_VEL_END_IDX]

    def get_joint_vel(pose):
        return pose[MotionLoaderBumi_12DOF.JOINT_VEL_START_IDX : MotionLoaderBumi_12DOF.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(poses):
        return poses[:, MotionLoaderBumi_12DOF.JOINT_VEL_START_IDX : MotionLoaderBumi_12DOF.JOINT_VEL_END_IDX]

    def get_contact_mask_batch(poses):
        return poses[:, MotionLoaderBumi_12DOF.CONTACT_MASK_START_IDX : MotionLoaderBumi_12DOF.CONTACT_MASK_END_IDX]


class MotionLoaderBumi_13DOF:
    """Motion loader for Bumi with 13 DOF (waist + 12 lower-body joints)."""

    POS_SIZE = 3
    ROT_SIZE = 4

    JOINT_POS_SIZE = 13
    KEY_POS_LOCAL_SIZE = (JOINT_POS_SIZE + 1) * 3
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_VEL_SIZE = JOINT_POS_SIZE
    CONTACT_MASK_SIZE = 2

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + POS_SIZE

    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE

    JOINT_POSE_START_IDX = ROOT_ROT_END_IDX
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    KEY_POS_LOCAL_START_IDX = JOINT_POSE_END_IDX
    KEY_POS_LOCAL_END_IDX = KEY_POS_LOCAL_START_IDX + KEY_POS_LOCAL_SIZE

    LINEAR_VEL_START_IDX = KEY_POS_LOCAL_END_IDX
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE

    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE

    JOINT_VEL_START_IDX = ANGULAR_VEL_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    CONTACT_MASK_START_IDX = JOINT_VEL_END_IDX
    CONTACT_MASK_END_IDX = CONTACT_MASK_START_IDX + CONTACT_MASK_SIZE

    def __init__(
        self,
        device,
        time_between_frames,
        reference_observation_horizon=2,
        num_preload_transitions=10,
        motion_files="",
        sim_joint_names="",
        sim_key_pos_names="",
    ):
        """Expert dataset provides AMP observations from Bumi mocap dataset (13-DOF subset)."""
        self.device = device
        self.time_between_frames = time_between_frames
        self.reference_observation_horizon = reference_observation_horizon
        self.num_preload_transitions = num_preload_transitions
        self.observation_start_dim = MotionLoaderBumi_13DOF.JOINT_POSE_START_IDX

        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split(".")[0])
            with open(motion_file) as f:
                motion_json = json.load(f)
                if i == 0:
                    # Indices of the 13-DOF joints / key positions we care about.
                    joint_indices = [
                        j for j, joint_name in enumerate(motion_json["JointNames"]) if joint_name in sim_joint_names
                    ]
                    key_pos_indices = [
                        j for j, key_pos_name in enumerate(motion_json["KeyPosNames"]) if key_pos_name in sim_key_pos_names
                    ]
                    motion_joint_names = [motion_json["JointNames"][j] for j in joint_indices]
                    motion_key_pos_names = [motion_json["KeyPosNames"][j] for j in key_pos_indices]
                    self.joint_mapping = create_index_mapping(motion_joint_names, sim_joint_names)
                    self.key_pos_mapping = create_index_mapping(motion_key_pos_names, sim_key_pos_names)
                motion_data = np.array(motion_json["Frames"])

                # Build indices for key positions (each key pos uses 3 consecutive entries).
                key_pos_data_indices = []
                for key_pos_idx in key_pos_indices:
                    start_idx = key_pos_idx * 3
                    key_pos_data_indices.extend([start_idx, start_idx + 1, start_idx + 2])

                # Slice full motion data (21 DOF layout) down to 13-DOF structure.
                motion_data = np.concatenate(
                    [
                        motion_data[:, : MotionLoaderBumi.JOINT_POSE_START_IDX],
                        motion_data[
                            :, MotionLoaderBumi.JOINT_POSE_START_IDX : MotionLoaderBumi.JOINT_POSE_END_IDX
                        ][:, joint_indices],
                        motion_data[
                            :, MotionLoaderBumi.KEY_POS_LOCAL_START_IDX : MotionLoaderBumi.KEY_POS_LOCAL_END_IDX
                        ][:, key_pos_data_indices],
                        motion_data[
                            :, MotionLoaderBumi.LINEAR_VEL_START_IDX : MotionLoaderBumi.LINEAR_VEL_END_IDX
                        ],
                        motion_data[
                            :, MotionLoaderBumi.ANGULAR_VEL_START_IDX : MotionLoaderBumi.ANGULAR_VEL_END_IDX
                        ],
                        motion_data[
                            :, MotionLoaderBumi.JOINT_VEL_START_IDX : MotionLoaderBumi.JOINT_VEL_END_IDX
                        ][:, joint_indices],
                        motion_data[
                            :, MotionLoaderBumi.CONTACT_MASK_START_IDX : MotionLoaderBumi.CONTACT_MASK_END_IDX
                        ],
                    ],
                    axis=1,
                )

                # Normalize and standardize quaternions.
                for f_i in range(motion_data.shape[0]):
                    root_rot = MotionLoaderBumi_13DOF.get_root_rot(motion_data[f_i])
                    root_rot = QuaternionNormalize(root_rot)
                    root_rot = standardize_quaternion(root_rot)
                    motion_data[
                        f_i,
                        MotionLoaderBumi_13DOF.POS_SIZE : (
                            MotionLoaderBumi_13DOF.POS_SIZE + MotionLoaderBumi_13DOF.ROT_SIZE
                        ),
                    ] = root_rot

                # Remove first 7 observation dimensions (root_pos, root_rot).
                self.trajectories.append(
                    torch.tensor(
                        motion_data[
                            :,
                            MotionLoaderBumi_13DOF.ROOT_ROT_END_IDX : MotionLoaderBumi_13DOF.CONTACT_MASK_END_IDX,
                        ],
                        dtype=torch.float32,
                        device=device,
                    )
                )
                self.trajectories_full.append(
                    torch.tensor(
                        motion_data[:, : MotionLoaderBumi_13DOF.CONTACT_MASK_END_IDX],
                        dtype=torch.float32,
                        device=device,
                    )
                )
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f"Loaded {traj_len}s. motion from {motion_file}.")

        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        print(f"Preloading {self.num_preload_transitions} transitions")
        traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
        times = self.traj_time_sample_batch(traj_idxs)
        self.preloaded_states = torch.zeros(
            self.num_preload_transitions,
            self.reference_observation_horizon,
            self.trajectories_full[0].shape[1],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.motion_label = torch.nn.functional.one_hot(
            torch.tensor(traj_idxs, dtype=torch.long, device=self.device)
        ).float()

        for i in range(self.reference_observation_horizon):
            self.preloaded_states[:, i] = self.get_full_frame_at_time_batch(
                traj_idxs, times + self.time_between_frames * i
            )
        print("Finished preloading")

        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = (
            self.time_between_frames * (self.reference_observation_horizon - 1)
            + self.trajectory_frame_durations[traj_idx]
        )
        return max(0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = (
            self.time_between_frames * (self.reference_observation_horizon - 1)
            + self.trajectory_frame_durations[traj_idxs]
        )
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int32), np.ceil(p * n).astype(np.int32)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int32), np.ceil(p * n).astype(np.int32)
        all_frame_pos_starts = torch.zeros(len(traj_idxs), MotionLoaderBumi_13DOF.POS_SIZE, device=self.device)
        all_frame_pos_ends = torch.zeros(len(traj_idxs), MotionLoaderBumi_13DOF.POS_SIZE, device=self.device)
        all_frame_rot_starts = torch.zeros(len(traj_idxs), MotionLoaderBumi_13DOF.ROT_SIZE, device=self.device)
        all_frame_rot_ends = torch.zeros(len(traj_idxs), MotionLoaderBumi_13DOF.ROT_SIZE, device=self.device)
        all_frame_AMP_starts = torch.zeros(
            len(traj_idxs),
            MotionLoaderBumi_13DOF.CONTACT_MASK_END_IDX - MotionLoaderBumi_13DOF.JOINT_POSE_START_IDX,
            device=self.device,
        )
        all_frame_AMP_ends = torch.zeros(
            len(traj_idxs),
            MotionLoaderBumi_13DOF.CONTACT_MASK_END_IDX - MotionLoaderBumi_13DOF.JOINT_POSE_START_IDX,
            device=self.device,
        )
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_pos_starts[traj_mask] = MotionLoaderBumi_13DOF.get_root_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = MotionLoaderBumi_13DOF.get_root_pos_batch(trajectory[idx_high[traj_mask]])
            all_frame_rot_starts[traj_mask] = MotionLoaderBumi_13DOF.get_root_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = MotionLoaderBumi_13DOF.get_root_rot_batch(trajectory[idx_high[traj_mask]])
            all_frame_AMP_starts[traj_mask] = trajectory[idx_low[traj_mask]][
                :, MotionLoaderBumi_13DOF.JOINT_POSE_START_IDX : MotionLoaderBumi_13DOF.CONTACT_MASK_END_IDX
            ]
            all_frame_AMP_ends[traj_mask] = trajectory[idx_high[traj_mask]][
                :, MotionLoaderBumi_13DOF.JOINT_POSE_START_IDX : MotionLoaderBumi_13DOF.CONTACT_MASK_END_IDX
            ]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        rot_blend = torch_quaternion_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        AMP_blend = self.slerp(all_frame_AMP_starts, all_frame_AMP_ends, blend)
        return torch.cat([pos_blend, rot_blend, AMP_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        idxs = np.random.choice(self.preloaded_states.shape[0], size=num_frames)
        return self.preloaded_states[idxs, 0]

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation."""

        root_pos0, root_pos1 = MotionLoaderBumi_13DOF.get_root_pos(frame0), MotionLoaderBumi_13DOF.get_root_pos(frame1)
        root_rot0, root_rot1 = MotionLoaderBumi_13DOF.get_root_rot(frame0), MotionLoaderBumi_13DOF.get_root_rot(frame1)
        joints0, joints1 = MotionLoaderBumi_13DOF.get_joint_pose(frame0), MotionLoaderBumi_13DOF.get_joint_pose(frame1)
        tar_toe_pos_0, tar_toe_pos_1 = MotionLoaderBumi_13DOF.get_key_pos_local(
            frame0
        ), MotionLoaderBumi_13DOF.get_key_pos_local(frame1)
        linear_vel_0, linear_vel_1 = MotionLoaderBumi_13DOF.get_linear_vel(
            frame0
        ), MotionLoaderBumi_13DOF.get_linear_vel(frame1)
        angular_vel_0, angular_vel_1 = MotionLoaderBumi_13DOF.get_angular_vel(
            frame0
        ), MotionLoaderBumi_13DOF.get_angular_vel(frame1)
        joint_vel_0, joint_vel_1 = MotionLoaderBumi_13DOF.get_joint_vel(
            frame0
        ), MotionLoaderBumi_13DOF.get_joint_vel(frame1)

        blend_root_pos = self.slerp(root_pos0, root_pos1, blend)
        blend_root_rot = np_quaternion_slerp(root_rot0.cpu().numpy(), root_rot1.cpu().numpy(), blend)
        blend_root_rot = torch.tensor(standardize_quaternion(blend_root_rot), dtype=torch.float32, device=self.device)
        blend_joints = self.slerp(joints0, joints1, blend)
        blend_tar_toe_pos = self.slerp(tar_toe_pos_0, tar_toe_pos_1, blend)
        blend_linear_vel = self.slerp(linear_vel_0, linear_vel_1, blend)
        blend_angular_vel = self.slerp(angular_vel_0, angular_vel_1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat(
            [
                blend_root_pos,
                blend_root_rot,
                blend_joints,
                blend_tar_toe_pos,
                blend_linear_vel,
                blend_angular_vel,
                blend_joints_vel,
            ]
        )

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            ids = torch.randint(0, self.num_preload_transitions, (mini_batch_size,), device=self.device)
            states = self.preloaded_states[ids, :, self.observation_start_dim :]
            labels = self.motion_label[ids]
            yield states, labels

    @property
    def observation_full_dim(self):
        """Size of AMP full observations."""
        return self.trajectories_full[0].shape[1]

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1]

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_root_pos(pose):
        return pose[MotionLoaderBumi_13DOF.ROOT_POS_START_IDX : MotionLoaderBumi_13DOF.ROOT_POS_END_IDX]

    def get_root_pos_batch(poses):
        return poses[:, MotionLoaderBumi_13DOF.ROOT_POS_START_IDX : MotionLoaderBumi_13DOF.ROOT_POS_END_IDX]

    def get_root_rot(pose):
        return pose[MotionLoaderBumi_13DOF.ROOT_ROT_START_IDX : MotionLoaderBumi_13DOF.ROOT_ROT_END_IDX]

    def get_root_rot_batch(poses):
        return poses[:, MotionLoaderBumi_13DOF.ROOT_ROT_START_IDX : MotionLoaderBumi_13DOF.ROOT_ROT_END_IDX]

    def get_joint_pose(pose):
        return pose[MotionLoaderBumi_13DOF.JOINT_POSE_START_IDX : MotionLoaderBumi_13DOF.JOINT_POSE_END_IDX]

    def get_joint_pose_batch(poses):
        return poses[:, MotionLoaderBumi_13DOF.JOINT_POSE_START_IDX : MotionLoaderBumi_13DOF.JOINT_POSE_END_IDX]

    def get_key_pos_local(pose):
        return pose[MotionLoaderBumi_13DOF.KEY_POS_LOCAL_START_IDX : MotionLoaderBumi_13DOF.KEY_POS_LOCAL_END_IDX]

    def get_key_pos_local_batch(poses):
        return poses[:, MotionLoaderBumi_13DOF.KEY_POS_LOCAL_START_IDX : MotionLoaderBumi_13DOF.KEY_POS_LOCAL_END_IDX]

    def get_linear_vel(pose):
        return pose[MotionLoaderBumi_13DOF.LINEAR_VEL_START_IDX : MotionLoaderBumi_13DOF.LINEAR_VEL_END_IDX]

    def get_linear_vel_batch(poses):
        return poses[:, MotionLoaderBumi_13DOF.LINEAR_VEL_START_IDX : MotionLoaderBumi_13DOF.LINEAR_VEL_END_IDX]

    def get_angular_vel(pose):
        return pose[MotionLoaderBumi_13DOF.ANGULAR_VEL_START_IDX : MotionLoaderBumi_13DOF.ANGULAR_VEL_END_IDX]

    def get_angular_vel_batch(poses):
        return poses[:, MotionLoaderBumi_13DOF.ANGULAR_VEL_START_IDX : MotionLoaderBumi_13DOF.ANGULAR_VEL_END_IDX]

    def get_joint_vel(pose):
        return pose[MotionLoaderBumi_13DOF.JOINT_VEL_START_IDX : MotionLoaderBumi_13DOF.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(poses):
        return poses[:, MotionLoaderBumi_13DOF.JOINT_VEL_START_IDX : MotionLoaderBumi_13DOF.JOINT_VEL_END_IDX]

    def get_contact_mask_batch(poses):
        return poses[:, MotionLoaderBumi_13DOF.CONTACT_MASK_START_IDX : MotionLoaderBumi_13DOF.CONTACT_MASK_END_IDX]

# ---------------------------------------- utils ----------------------------------------


def create_index_mapping(motion_joint_names, sim_joint_names):
    if set(motion_joint_names) != set(sim_joint_names):
        raise ValueError("两个关节列表包含不同的关节名称")

    target_name_to_index = {name: idx for idx, name in enumerate(sim_joint_names)}

    sim2motion_map = []
    for source_idx, name in enumerate(motion_joint_names):
        target_idx = target_name_to_index[name]
        sim2motion_map.append(target_idx)

    motion2sim_map = []
    for target_idx, name in enumerate(sim_joint_names):
        source_idx = motion_joint_names.index(name)
        motion2sim_map.append(source_idx)

    return {"sim2motion": sim2motion_map, "motion2sim": motion2sim_map}


def standardize_quaternion(q):
    if q[-1] < 0:
        q = -q
    return q


def QuaternionNormalize(q):
    q_norm = np.linalg.norm(q)
    if np.isclose(q_norm, 0.0):
        raise ValueError(f"Quaternion may not be zero in QuaternionNormalize: |q| = {q_norm:f}, q = {q}")
    return q / q_norm


def unit_vector(data, axis=None, out=None):
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def np_quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True, _EPS=1e-6):
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        q1 *= -1.0
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0


def torch_quaternion_slerp(q0, q1, fraction, spin=0, shortestpath=True, _EPS=1e-6):
    """Batch quaternion spherical linear interpolation."""

    out = torch.zeros_like(q0)

    zero_mask = torch.isclose(fraction, torch.zeros_like(fraction)).squeeze()
    ones_mask = torch.isclose(fraction, torch.ones_like(fraction)).squeeze()
    out[zero_mask] = q0[zero_mask]
    out[ones_mask] = q1[ones_mask]

    d = torch.sum(q0 * q1, dim=-1, keepdim=True)
    dist_mask = (torch.abs(torch.abs(d) - 1.0) < _EPS).squeeze()
    out[dist_mask] = q0[dist_mask]

    if shortestpath:
        d_old = torch.clone(d)
        d = torch.where(d_old < 0, -d, d)
        q1 = torch.where(d_old < 0, -q1, q1)

    # TODO Clip d to be in the range [-1, 1] to avoid nan from acos
    d = torch.clamp(d, -1.0 + _EPS, 1.0 - _EPS)

    angle = torch.acos(d) + spin * torch.pi
    angle_mask = (torch.abs(angle) < _EPS).squeeze()
    out[angle_mask] = q0[angle_mask]

    final_mask = torch.logical_or(zero_mask, ones_mask)
    final_mask = torch.logical_or(final_mask, dist_mask)
    final_mask = torch.logical_or(final_mask, angle_mask)
    final_mask = torch.logical_not(final_mask)

    # isin = 1.0 / angle
    isin = 1.0 / (angle + _EPS)  # TODO Avoid division by zero
    q0 *= torch.sin((1.0 - fraction) * angle) * isin
    q1 *= torch.sin(fraction * angle) * isin
    q0 += q1
    out[final_mask] = q0[final_mask]
    return out