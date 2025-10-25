import random
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os

import parameters as args
from replay_buffer import ReplayBuffer
from train_utils import *
from network import Actor, SoftQNetwork


run_name = f"version_3"

writer = SummaryWriter(f"runs/{run_name}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s"
    % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)

# TRY NOT TO MODIFY: seeding
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

device = torch.device("cpu")

# env setup
env = gym.make('CuttingEnv-v0')

actor = Actor(env, args.num_objectives).to(device)
actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

adversary = Actor(env, args.num_objectives)
adversary_optimizer = optim.Adam(list(adversary.parameters()), lr=args.adversarial_lr)

qf1 = SoftQNetwork(env, args.num_objectives).to(device)
qf2 = SoftQNetwork(env, args.num_objectives).to(device)
qf1_target = SoftQNetwork(env, args.num_objectives).to(device)
qf2_target = SoftQNetwork(env, args.num_objectives).to(device)
qf1_target.load_state_dict(qf1.state_dict())
qf2_target.load_state_dict(qf2.state_dict())
q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)

# eta = torch.tensor(1.0, requires_grad=False)
eta = torch.tensor(1.0, requires_grad=True)
eta_optimizer = optim.Adam([eta], lr=args.dual_lr)

delta = torch.zeros((args.batch_size, 50), requires_grad=False)

# Automatic entropy tuning
if args.autotune:
    # 计算离散动作和连续动作的目标熵
    discrete_target_entropy = 0
    for i in range(3):
        discrete_target_entropy += args.target_entropy_scale * torch.log(
            1 / torch.tensor(env.action_space[i].n)
        )
    continuous_target_entropy = -torch.prod(
        torch.Tensor(env.action_space[3].shape).to(device)
    ).item()
    target_entropy = discrete_target_entropy + continuous_target_entropy

    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
else:
    alpha = args.alpha

env.observation_space.dtype = np.float32
rb = ReplayBuffer(
    args.buffer_size,
    env.observation_space,
    env.action_space,
    device,
    n_envs=args.num_envs,
    num_objectives=args.num_objectives,
    handle_timeout_termination=False,
)

# 用于保存最佳模型的信息，(average_return, actor_filename, qf1_filename, qf2_filename)
best_models_info = []
# 记录每个评估周期内的所有回合回报
episodic_returns = []
# 评估周期（可以根据需要调整）
evaluation_period = 10

episode_length = 0
episode_count = 0
episode_return = 0

obs, infos = env.reset(seed=args.seed)
obs_array = np.reshape(infos["obs_array"], (1, -1))
for global_step in range(args.total_timesteps):
    omega_da = np.random.dirichlet(np.ones(args.num_objectives))
    omega_da = torch.tensor(omega_da, dtype=torch.float32).to(device)
    omega_da = omega_da.expand(args.batch_size, -1)

    omega_t = np.random.dirichlet(np.ones(args.num_objectives))
    # ALGO LOGIC: put action logic here
    if global_step < args.learning_starts:
        (a1, a2, a3, a4) = env.action_space.sample()
        actions = np.array([a1, a2, a3, a4[0]])
    else:
        if torch.rand(1).item() > args.adversarial_prob:
            actions, _ = actor.get_action(torch.Tensor(obs_array).to(device), torch.Tensor(omega_t).to(device))
        else:
            actions, _ = adversary.get_action(torch.Tensor(obs_array).to(device), torch.Tensor(omega_t).to(device))
        actions = actions[0].detach().cpu().numpy()

    # TRY NOT TO MODIFY: execute the game and log data.
    next_obs, _, terminations, truncations, infos = env.step(actions)

    next_obs_array = np.reshape(infos["obs_array"], (1, -1))
    rewards = infos["reward_vector"]
    epistemic_std = infos["tool_wear_epistemic_std"]
    aleatory_std = infos["tool_wear_aleatory_std"]
    done = terminations or truncations

    episode_length += 1
    episode_return += np.sum(rewards * omega_t).item()

    # reward_efficiency = rewards[0]
    # reward_tool = rewards[1]
    # reward_quality = rewards[2]
    # writer.add_scalar("charts/reward_efficiency", reward_efficiency, global_step)
    # writer.add_scalar("charts/reward_tool", reward_tool, global_step)
    # writer.add_scalar("charts/reward_quality", reward_quality, global_step)

    if done:
        print(
            f"global_step={global_step}, episodic_length={episode_length}, episodic_return={episode_return}"
        )
        writer.add_scalar("charts/episodic_return", episode_return, global_step)
        writer.add_scalar("charts/episodic_length", episode_length, global_step)
        episodic_returns.append(episode_return)

        episode_count += 1
        episode_length = 0
        episode_return = 0

    rb.add(obs_array, next_obs_array, actions, rewards, done, omega_t, infos, aleatory_std, epistemic_std)

    # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
    if done:
        obs, infos = env.reset(seed=args.seed)
        obs_array = np.reshape(infos["obs_array"], (1, -1))
    else:
        obs = next_obs
        obs_array = next_obs_array

    # ALGO LOGIC: training.
    if global_step > args.learning_starts:
        data = rb.sample(args.batch_size)
        state = data.observations
        action = data.actions
        next_state = data.next_observations
        done = data.dones
        reward = data.rewards
        omega_rb = data.preferences
        ale_std_rb = data.ale_stds
        epi_stds_rb = data.epi_stds

        qf_loss_l1_rb, qf_loss_l2_rb = cal_q_loss(
            actor, adversary, qf1, qf2, qf1_target, qf2_target,
            state, next_state, action, reward, omega_rb, done, alpha
        )
        qf_loss_l1_da, qf_loss_l2_da = cal_q_loss(
            actor, adversary, qf1, qf2, qf1_target, qf2_target,
            state, next_state, action, reward, omega_da, done, alpha
        )
        qf_loss_l1 = qf_loss_l1_rb + qf_loss_l1_da
        qf_loss_l2 = qf_loss_l2_rb + qf_loss_l2_da
        # optimize the model
        qf_loss = (1 - args.beta) * qf_loss_l1 + args.beta * qf_loss_l2
        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()

        actor_loss, C_rb, C_da, delta_update = cal_actor_loss(
            actor, adversary, qf1, qf2,
            state, omega_rb, omega_da, eta, alpha, ale_std_rb, delta
        )
        delta = delta_update.clone().detach()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # eta_loss = 0
        eta_loss = cal_eta_loss(eta, C_rb.detach(), C_da.detach())
        eta_optimizer.zero_grad()
        eta_loss.backward()
        eta_optimizer.step()
        eta.data.clamp_(min=0.1, max=5.0)  # 确保非负

        if args.autotune:
            with torch.no_grad():
                _, log_pi = actor.get_action(state, omega_rb)
            log_pi_sum = torch.sum(log_pi, 1, keepdim=True)
            alpha_loss = (-log_alpha.exp() * (log_pi_sum + target_entropy)).mean()
            a_optimizer.zero_grad()
            alpha_loss.backward()
            a_optimizer.step()
            alpha = log_alpha.exp().item()

        if global_step % 10 == 0:
            adversary_loss = cal_adv_loss(actor, adversary, qf1, qf2, state, omega_rb, omega_da, epi_stds_rb)
            adversary_optimizer.zero_grad()
            adversary_loss.backward()
            adversary_optimizer.step()

        # update the target networks
        if global_step % args.target_network_frequency == 0:
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        if global_step % 100 == 0:
            writer.add_scalar("losses/qf_loss_l1", qf_loss_l1.item(), global_step)
            writer.add_scalar("losses/qf_loss_l2", qf_loss_l2.item(), global_step)
            writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            writer.add_scalar("losses/adv_loss", adversary_loss.item(), global_step)
            writer.add_scalar("losses/eta_loss", eta_loss.item(), global_step)
            writer.add_scalar("values/Crb", C_rb.item(), global_step)
            writer.add_scalar("values/Cda", C_da.item(), global_step)
            writer.add_scalar("values/alpha", alpha, global_step)
            writer.add_scalar("values/eta", eta.item(), global_step)
            if args.autotune:
                writer.add_scalar(
                    "losses/alpha_loss", alpha_loss.item(), global_step
                )
            hv_value = evaluate_hypervolume(actor, env)
            writer.add_scalar("metrics/hypervolume", hv_value.item(), global_step)

        # 评估周期结束，计算平均回报并检查是否保存模型
        if episode_count % evaluation_period == 0 and len(episodic_returns) > 0:
            average_return = np.mean(episodic_returns)

            # 保存新的模型信息
            folder_path = f"./models/{run_name}"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)
            actor_filename = f"./models/{run_name}/actor_model_step_{global_step}_return_{average_return:.2f}.pth"
            qf1_filename = f"./models/{run_name}/qf1_model_step_{global_step}_return_{average_return:.2f}.pth"
            qf2_filename = f"./models/{run_name}/qf2_model_step_{global_step}_return_{average_return:.2f}.pth"

            new_model_info = (average_return, actor_filename, qf1_filename, qf2_filename)
            # 如果列表未满，直接添加
            if len(best_models_info) < 5:
                best_models_info.append(new_model_info)
                best_models_info.sort(key=lambda x: x[0], reverse=True)
                torch.save(actor.state_dict(), actor_filename)
                torch.save(qf1.state_dict(), qf1_filename)
                torch.save(qf2.state_dict(), qf2_filename)
            else:
                # 如果新模型比列表中最差的模型好，则替换
                if average_return > best_models_info[-1][0]:
                    # 删除旧的最差模型文件
                    _, old_actor_filename, old_qf1_filename, old_qf2_filename = (best_models_info.pop())

                    if os.path.exists(old_actor_filename):
                        os.remove(old_actor_filename)
                    if os.path.exists(old_qf1_filename):
                        os.remove(old_qf1_filename)
                    if os.path.exists(old_qf2_filename):
                        os.remove(old_qf2_filename)

                    best_models_info.append(new_model_info)
                    best_models_info.sort(key=lambda x: x[0], reverse=True)
                    torch.save(actor.state_dict(), actor_filename)
                    torch.save(qf1.state_dict(), qf1_filename)
                    torch.save(qf2.state_dict(), qf2_filename)

            episodic_returns = []

env.close()
writer.close()
