# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, obs_dim, obs_horizon, buffer_size, device):
        """Initialize a ReplayBuffer object.
        Arguments:
            buffer_size (int): maximum size of buffer
        """
        self.state_buf = torch.zeros(buffer_size, obs_horizon, obs_dim).to(device)
        self.buffer_size = buffer_size
        self.device = device

        self.step = 0
        self.num_samples = 0

    def get_buffer_size(self):
        return self.buffer_size

    def insert(self, state_buf):
        """Add new states to memory."""

        num_states = state_buf.shape[0]
        start_idx = self.step
        end_idx = self.step + num_states
        if end_idx > self.buffer_size:
            self.state_buf[self.step : self.buffer_size] = state_buf[: self.buffer_size - self.step]
            self.state_buf[: end_idx - self.buffer_size] = state_buf[self.buffer_size - self.step :]
        else:
            self.state_buf[start_idx:end_idx] = state_buf

        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_states) % self.buffer_size

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
            yield self.state_buf[sample_idxs, :].to(self.device)
