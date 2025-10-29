import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class SoftQNetwork(nn.Module):
    def __init__(self, env, num_objectives):
        super().__init__()
        self.env = env
        obs_dim = 50
        discrete_action_dim = env.action_space[0].n + env.action_space[1].n + env.action_space[2].n
        continuous_action_dim = np.prod(env.action_space[3].shape)
        input_dim = obs_dim + discrete_action_dim + continuous_action_dim + num_objectives

        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, num_objectives)

    def forward(self, x, actions, preference):
        discrete_actions = actions[:, :3]
        continuous_action = actions[:, 3]

        discrete_one_hot = []
        for i in range(3):
            one_hot = F.one_hot(discrete_actions[:, i].long(), num_classes=self.env.action_space[i].n)
            discrete_one_hot.append(one_hot)
        discrete_one_hot = torch.cat(discrete_one_hot, dim=1)

        x = torch.cat([x, discrete_one_hot, continuous_action.unsqueeze(1), preference], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, num_objectives):
        super().__init__()
        obs_dim = 50

        self.fc1 = nn.Linear(obs_dim + num_objectives, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 512)

        self.fc_logits_1 = nn.Linear(512, env.action_space[0].n)
        self.fc_logits_2 = nn.Linear(512, env.action_space[1].n)
        self.fc_logits_3 = nn.Linear(512, env.action_space[2].n)

        self.fc_mean = nn.Linear(512, np.prod(env.action_space[3].shape))
        self.fc_logstd = nn.Linear(512, np.prod(env.action_space[3].shape))

        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space[3].high - env.action_space[3].low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space[3].high + env.action_space[3].low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x, preference):
        if preference.dim() == 1:
            x = torch.cat([x, preference.unsqueeze(0)], dim=1)
        else:
            x = torch.cat([x, preference], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        logits_1 = self.fc_logits_1(x)
        logits_2 = self.fc_logits_2(x)
        logits_3 = self.fc_logits_3(x)

        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return logits_1, logits_2, logits_3, mean, log_std

    def get_action(self, x, preference):
        logits_1, logits_2, logits_3, mean, log_std = self(x, preference)
        policy_dist_1 = Categorical(logits=logits_1)
        action_1 = policy_dist_1.sample()
        log_prob_1 = policy_dist_1.log_prob(action_1)

        policy_dist_2 = Categorical(logits=logits_2)
        action_2 = policy_dist_2.sample()
        log_prob_2 = policy_dist_2.log_prob(action_2)

        policy_dist_3 = Categorical(logits=logits_3)
        action_3 = policy_dist_3.sample()
        log_prob_3 = policy_dist_3.log_prob(action_3)

        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action_4 = y_t * self.action_scale + self.action_bias
        log_prob_4 = normal.log_prob(x_t)
        log_prob_4 -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob_4 = log_prob_4.sum(1, keepdim=True)

        actions = torch.cat(
            [action_1.unsqueeze(1), action_2.unsqueeze(1), action_3.unsqueeze(1), action_4],
            1)
        log_probs = torch.cat(
            [log_prob_1.unsqueeze(1), log_prob_2.unsqueeze(1), log_prob_3.unsqueeze(1), log_prob_4],
            1)
        return actions, log_probs
