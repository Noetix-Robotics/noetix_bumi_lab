# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from torch import autograd

DISC_LOGIT_INIT_SCALE = 1.0


class Discriminator(nn.Module):
    def __init__(
        self,
        observation_dim,
        observation_horizon,
        device,
        reward_coef=0.1,
        reward_lerp=0.3,
        shape=[1024, 512],
        style_reward_function="quad_mapping",
        **kwargs,
    ):
        if kwargs:
            print(
                "Discriminator.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        self.observation_dim = observation_dim
        self.observation_horizon = observation_horizon
        self.input_dim = observation_dim * observation_horizon
        self.device = device
        self.reward_coef = reward_coef
        self.reward_lerp = reward_lerp
        self.style_reward_function = style_reward_function
        self.shape = shape

        discriminator_layers = []
        curr_in_dim = self.input_dim
        for hidden_dim in self.shape:
            discriminator_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            discriminator_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.architecture = nn.Sequential(*discriminator_layers).to(self.device)
        self.discriminator_logits = torch.nn.Linear(hidden_dim, 1)
        self.train()

    def forward(self, x):
        return self.discriminator_logits(self.architecture(x))

    def get_disc_weights(self):
        weights = []
        for m in self.architecture.modules():
            if isinstance(m, nn.Linear):
                weights.append(torch.flatten(m.weight))
        return weights

    def get_disc_logit_weights(self):
        return torch.flatten(self.discriminator_logits.weight)

    def eval_disc(self, x):
        return self.discriminator_logits(self.architecture(x))

    def compute_grad_pen(self, expert_data, lambda_=10):
        disc = self.eval_disc(expert_data)
        grad = autograd.grad(
            outputs=disc,
            inputs=expert_data,
            grad_outputs=torch.ones(disc.size(), device=disc.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * grad.norm(2, dim=1).pow(2).mean()
        return grad_pen

    def compute_wgan_div_grad_pen(self, expert_data, policy_data, p=6, k=2):
        # expert_d = self.eval_disc(expert_data)
        # expert_grad = autograd.grad(
        #     outputs=expert_d,
        #     inputs=expert_data,
        #     grad_outputs=torch.ones(expert_d.size(), device=expert_d.device),
        #     create_graph=True,
        #     retain_graph=True,
        #     only_inputs=True,
        # )[0]
        # expert_grad_norm = expert_grad.view(expert_grad.size(), -1).pow(2).sum(1) ** (p / 2)

        # policy_d = self.eval_disc(policy_data)
        # policy_grad = autograd.grad(
        #     outputs=policy_d,
        #     inputs=policy_data,
        #     grad_outputs=torch.ones(policy_d.size(), device=policy_d.device),
        #     create_graph=True,
        #     retain_graph=True,
        #     only_inputs=True,
        # )[0]
        # policy_grad_norm = policy_grad.view(policy_grad.size(), -1).pow(2).sum(1) ** (p / 2)

        # grad_pen = torch.mean(expert_grad_norm + policy_grad_norm) * k / 2

        batch_size = expert_data.size(0)
        device = expert_data.device

        # 随机生成插值系数 t ~ U(0, 1)
        t = torch.rand(batch_size, 1, device=device)
        t = t.expand_as(expert_data)

        # 生成插值点：hat_x = t * expert_data + (1 - t) * policy_data
        hat_x = t * expert_data + (1 - t) * policy_data
        hat_x.requires_grad_(True)

        # 计算判别器对插值点的输出
        disc = self.eval_disc(hat_x)

        grad = autograd.grad(
            outputs=disc,
            inputs=hat_x,
            grad_outputs=torch.ones(disc.size(), device=disc.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = 5 * (grad.norm(2, dim=1) - 1).pow(2).mean()

        return grad_pen

    def compute_weight_decay(self, lambda_=0.0001):
        disc_weights = self.get_disc_weights()
        disc_weights = torch.cat(disc_weights, dim=-1)
        weight_decay = lambda_ * torch.sum(torch.square(disc_weights))
        return weight_decay

    def compute_logit_reg(self, lambda_=0.05):
        logit_weights = self.get_disc_logit_weights()
        disc_logit_loss = lambda_ * torch.sum(torch.square(logit_weights))
        return disc_logit_loss

    def predict_amp_reward(self, state_buf, task_reward, dt=1, state_normalizer=None, style_reward_normalizer=None):
        with torch.no_grad():
            self.eval()
            state_buf = state_buf.clone()
            if state_normalizer is not None:
                for i in range(self.observation_horizon):
                    state_buf[:, i] = state_normalizer.normalize_torch(state_buf[:, i], self.device)
            d = self.eval_disc(state_buf.flatten(1, 2))
            if self.style_reward_function == "quad_mapping":
                style_reward = torch.clamp(1 - (1 / 4) * torch.square(d - 1), min=0)
            elif self.style_reward_function == "log_mapping":
                style_reward = -torch.log(
                    torch.maximum(1 - 1 / (1 + torch.exp(-d)), torch.tensor(0.0001, device=self.device))
                )
            elif self.style_reward_function == "wasserstein_mapping":
                if style_reward_normalizer is not None:
                    style_reward = style_reward_normalizer.normalize_torch(d.clone(), self.device)
                    style_reward_normalizer.update(d.cpu().numpy())
                else:
                    style_reward = torch.exp(torch.tanh(0.3 * d)) - torch.exp(-1 * torch.ones_like(d))
            else:
                raise ValueError("Unexpected style reward mapping specified")
            style_reward *= (1.0 - self.reward_lerp) * self.reward_coef
            task_reward = task_reward.unsqueeze(-1) * self.reward_lerp
            reward = style_reward + task_reward
            self.train()
        return reward.squeeze(), style_reward.squeeze()
    
