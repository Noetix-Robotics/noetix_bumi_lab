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


class SkillEncoder(nn.Module):
    def __init__(self, num_skills, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_skills, embedding_dim)

    def forward(self, skill_indices):
        return self.embedding(skill_indices)
    
class DiscriminatorCondition(nn.Module):
    def __init__(
        self,
        observation_dim,
        observation_horizon,
        num_motions,
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
        self.input_dim = observation_dim * observation_horizon + num_motions
        self.device = device
        self.reward_coef = reward_coef
        self.reward_lerp = reward_lerp
        self.style_reward_function = style_reward_function
        self.shape = shape
        self.softmax = nn.Softmax(dim=-1)

        use_skill_encoder = True
        skill_embedding_dim = 32
        if use_skill_encoder:
            self.skill_encoder = nn.Linear(num_motions, skill_embedding_dim)
            self.input_dim = observation_dim * observation_horizon + skill_embedding_dim
        discriminator_layers = []
        curr_in_dim = self.input_dim
        for hidden_dim in self.shape:
            discriminator_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            discriminator_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.architecture = nn.Sequential(*discriminator_layers).to(self.device)
        self.discriminator_logits = torch.nn.Linear(hidden_dim, 1)
        # self.discriminator_aux = torch.nn.Linear(hidden_dim, num_motions)

        # ------------------------- Skill Discriminator -------------------------
        use_skill_discriminator = True
        self.skill_discriminator_shape = [512, 256]
        if use_skill_discriminator:
            skill_discriminator_layers = []
            curr_in_dim = observation_dim * observation_horizon
            for hidden_dim in self.skill_discriminator_shape:
                skill_discriminator_layers.append(nn.Linear(curr_in_dim, hidden_dim))
                skill_discriminator_layers.append(nn.ReLU())
                curr_in_dim = hidden_dim
            self.skill_discriminator = nn.Sequential(*skill_discriminator_layers).to(self.device)
        self.skill_discriminator_logits = torch.nn.Linear(hidden_dim, skill_embedding_dim)
        self.train()

    def forward(self, x1, x2):
        '''
        x1: observations
        x2: motion one-hot vectors
        '''
        if hasattr(self, 'skill_encoder'):
            x2 = self.skill_encoder(x2)
        x = torch.cat((x1, x2), dim=1)
        hidden = self.architecture(x)
        fc_dis = self.discriminator_logits(hidden)
        # fc_aux = self.discriminator_aux(hidden)
        # classes = self.softmax(fc_aux)
        return fc_dis, None #, classes


    def get_disc_weights(self):
        weights = []
        for m in self.architecture.modules():
            if isinstance(m, nn.Linear):
                weights.append(torch.flatten(m.weight))
        return weights

    def get_disc_logit_weights(self):
        return torch.flatten(self.discriminator_logits.weight)

    def eval_disc(self, x1, x2):
        if hasattr(self, 'skill_encoder'):
            x2 = self.skill_encoder(x2)
        x = torch.cat((x1, x2), dim=1)
        hidden = self.architecture(x)
        fc_dis = self.discriminator_logits(hidden)
        # fc_aux = self.discriminator_aux(hidden)
        # classes = self.softmax(fc_aux)
        return fc_dis, None #, classes

    def eval_disc_skill(self, x1, x2):
        skill_hidden = self.skill_discriminator(x1)
        z_predict = self.skill_discriminator_logits(skill_hidden)
        z_target = self.skill_encoder(x2)
        return z_predict, z_target
    
    def compute_grad_pen(self, expert_data, skill_labels, lambda_=10):
        disc, _ = self.eval_disc(expert_data, skill_labels)
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

    def compute_wgan_div_grad_pen(self, expert_data, expert_skill_labels, policy_data, policy_skill_labels, p=6, k=2):
        expert_d, _ = self.eval_disc(expert_data, expert_skill_labels)
        expert_grad = autograd.grad(
            outputs=expert_d, inputs=expert_data,
            grad_outputs=torch.ones(expert_d.size(), device=expert_d.device), create_graph=True,
            retain_graph=True, only_inputs=True)[0]
        expert_grad_norm = expert_grad.view(expert_grad.size(), -1).pow(2).sum(1) ** (p / 2)

        policy_d, _ = self.eval_disc(policy_data, policy_skill_labels)
        policy_grad = autograd.grad(
            outputs=policy_d, inputs=policy_data,
            grad_outputs=torch.ones(policy_d.size(), device=policy_d.device), create_graph=True,
            retain_graph=True, only_inputs=True)[0]
        policy_grad_norm = policy_grad.view(policy_grad.size(), -1).pow(2).sum(1) ** (p / 2)

        grad_pen = torch.mean(expert_grad_norm + policy_grad_norm) * k / 2

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

    def predict_amp_reward(self, state_buf, skill_labels, task_reward, dt=1, state_normalizer=None, style_reward_normalizer=None):
        with torch.no_grad():
            self.eval()
            state_buf = state_buf.clone()
            if state_normalizer is not None:
                for i in range(self.observation_horizon):
                    state_buf[:, i] = state_normalizer.normalize_torch(state_buf[:, i], self.device)
            d, aux = self.eval_disc(state_buf.flatten(1, 2), skill_labels)
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

            if getattr(self, 'skill_discriminator', None) is not None:
                # 对应论文中的 \hat{z}
                skill_reward_coef = 0.3
                z_predict, z_target = self.eval_disc_skill(state_buf.flatten(1, 2), skill_labels)
                skill_reward = torch.cosine_similarity(z_predict, z_target, dim=1).unsqueeze(-1)

                dis_reward = style_reward + skill_reward * skill_reward_coef
                dis_reward *= (1.0 - self.reward_lerp) * self.reward_coef * dt
                task_reward = task_reward.unsqueeze(-1) * self.reward_lerp
                reward = dis_reward + task_reward
            self.train()
        return reward.squeeze(), style_reward.squeeze(), skill_reward.squeeze()
    
    def compute_skill_discriminator_loss(self, expert_data, expert_skill_labels, policy_data=None, policy_skill_labels=None):
        # z_predict, z_target = self.eval_disc_skill(expert_data, expert_skill_labels)
        # # loss = torch.nn.MSELoss()(z_predict, z_target)
        # cos_sim = torch.cosine_similarity(z_predict, z_target, dim=1)
        # loss = (1.0 - cos_sim).mean()
        # ---------- expert loss ----------
        z_predict_expert, z_target_expert = self.eval_disc_skill(expert_data, expert_skill_labels)
        loss_expert = (1.0 - torch.cosine_similarity(z_predict_expert, z_target_expert, dim=1)).mean()

        # ---------- policy loss ----------
        if policy_data is not None and policy_skill_labels is not None:
            z_predict_policy, z_target_policy = self.eval_disc_skill(policy_data, policy_skill_labels)
            loss_policy = (1.0 - torch.cosine_similarity(z_predict_policy, z_target_policy, dim=1)).mean()
            total_loss = loss_expert + loss_policy
        else:
            total_loss = loss_expert
        return total_loss
    
    def compute_skill_discriminator_grad_pen(self, expert_data, expert_skill_labels, policy_data=None, policy_skill_labels=None, p=6, k=2, lambda_=10):
        # disc, _ = self.eval_disc_skill(expert_data, expert_skill_labels)
        # grad = autograd.grad(
        #     outputs=disc,
        #     inputs=expert_data,
        #     grad_outputs=torch.ones(disc.size(), device=disc.device),
        #     create_graph=True,
        #     retain_graph=True,
        #     only_inputs=True,
        # )[0]

        # # Enforce that the grad norm approaches 0.
        # grad_pen = lambda_ * grad.norm(2, dim=1).pow(2).mean()
        # return grad_pen
    
        expert_d, _ = self.eval_disc_skill(expert_data, expert_skill_labels)
        expert_grad = autograd.grad(
            outputs=expert_d, inputs=expert_data,
            grad_outputs=torch.ones(expert_d.size(), device=expert_d.device), create_graph=True,
            retain_graph=True, only_inputs=True)[0]
        expert_grad_norm = expert_grad.view(expert_grad.size(), -1).pow(2).sum(1) ** (p / 2)

        policy_d, _ = self.eval_disc_skill(policy_data, policy_skill_labels)
        policy_grad = autograd.grad(
            outputs=policy_d, inputs=policy_data,
            grad_outputs=torch.ones(policy_d.size(), device=policy_d.device), create_graph=True,
            retain_graph=True, only_inputs=True)[0]
        policy_grad_norm = policy_grad.view(policy_grad.size(), -1).pow(2).sum(1) ** (p / 2)

        grad_pen = torch.mean(expert_grad_norm + policy_grad_norm) * k / 2

        return grad_pen
