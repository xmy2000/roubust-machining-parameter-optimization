import torch
import torch.nn.functional as F
from pymoo.indicators.hv import Hypervolume
import copy
import numpy as np

import parameters as args


def evaluate_hypervolume(actor, env):
    """使用pymoo计算超体积"""
    reward_vectors = []
    eval_env = copy.deepcopy(env)  # 独立评估环境

    for _ in range(args.NUM_EVAL_EPISODES):
        obs, infos = eval_env.reset()
        obs_array = np.reshape(infos["obs_array"], (1, -1))
        episode_rewards = []
        done = False

        while not done:
            # 评估时用确定性策略（无探索噪声）
            with torch.no_grad():
                # 生成均匀分布的偏好权重（覆盖多目标空间）
                omega = np.random.dirichlet(np.ones(args.num_objectives))
                actions, _ = actor.get_action(
                    torch.Tensor(obs_array),
                    torch.Tensor(omega),
                )
            actions = actions[0].cpu().numpy()

            next_obs, _, terminations, truncations, infos = eval_env.step(actions)
            rewards = infos["reward_vector"]  # 多目标奖励向量
            episode_rewards.append(rewards)
            done = terminations or truncations
            obs_array = np.reshape(infos["obs_array"], (1, -1))

        # 累计episode总奖励向量
        total_reward = np.sum(episode_rewards, axis=0)
        reward_vectors.append(total_reward)

    eval_env.close()

    # 使用pymoo计算超体积
    if not reward_vectors:
        return 0.0
    solutions = np.array(reward_vectors)
    transformed_solutions = args.REF_POINT - solutions
    indicator = Hypervolume(ref_point=[0.0, 0.0, 0.0], normalize=False)
    return indicator.do(transformed_solutions)


def cal_q_mix(q, state, action, action_adv, omega):
    q_mix = (1 - args.adversarial_prob) * q(state, action, omega) + args.adversarial_prob * q(state, action_adv, omega)
    return q_mix


def cal_u_mix(q, state, action, action_adv, omega):
    return cal_q_mix(q, state, action, action_adv, omega) * omega


def compute_robustness_constraint(actor, state, preference, original_action, stds, delta):
    perturbed_state = state.clone().detach()
    delta = delta.clone().detach().requires_grad_(True)

    # 不确定性扰动
    # tool_wear_mu = state[:, args.tool_wear_idx]
    # tool_wear = tool_wear_mu + stds * torch.randn_like(tool_wear_mu)
    # tool_wear = torch.clamp(tool_wear, min=0.1)
    # perturbed_state[:, args.tool_wear_idx] = tool_wear
    perturbed_state = (1 - torch.randn_like(state) * args.zeta) * state

    perturbed_state = perturbed_state + delta
    perturbed_action, _ = actor.get_action(perturbed_state, preference)
    constraint = torch.norm(original_action - perturbed_action, p=2, dim=1).mean()

    constraint.backward(retain_graph=True)
    delta_update = delta + args.zeta_bar * torch.sign(delta.grad)

    return constraint, delta_update.detach()


def cal_q_target(
        actor, adversary, q1_target, q2_target, state, reward, omega, dones, alpha
):
    action_adv, _ = adversary.get_action(state, omega)
    action, log_probs = actor.get_action(state, omega)

    qf1_next_target = cal_q_mix(q1_target, state, action, action_adv, omega)
    qf2_next_target = cal_q_mix(q2_target, state, action, action_adv, omega)

    log_pi = torch.sum(log_probs, 1, keepdim=True)
    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * log_pi
    q_target = torch.zeros_like(min_qf_next_target)
    for i in range(args.num_objectives):
        q_target[:, i] = reward[:, i] + (1 - dones.flatten()) * args.gamma * (
            min_qf_next_target[:, i]
        ).view(-1)

    return q_target


def cal_q_loss(
        actor,
        adversary,
        q1,
        q2,
        q1_target,
        q2_target,
        state,
        next_state,
        action,
        reward,
        omega,
        dones,
        alpha,
):
    with torch.no_grad():
        next_q_value = cal_q_target(
            actor, adversary, q1_target, q2_target,
            next_state, reward, omega, dones, alpha,
        )
    qf1_a_values = q1(state, action.float(), omega)
    qf2_a_values = q2(state, action.float(), omega)

    qf1_loss_l1 = F.mse_loss(qf1_a_values, next_q_value)
    qf2_loss_l1 = F.mse_loss(qf2_a_values, next_q_value)
    q_loss_l1 = qf1_loss_l1 + qf2_loss_l1

    qf1_loss_l2 = F.l1_loss(omega * qf1_a_values, omega * next_q_value)
    qf2_loss_l2 = F.l1_loss(omega * qf2_a_values, omega * next_q_value)
    q_loss_l2 = qf1_loss_l2 + qf2_loss_l2

    return q_loss_l1, q_loss_l2


def cal_actor_loss(
        actor, adversary, qf1, qf2, state, omega_rb, omega_da, eta, alpha, stds, delta
):
    action_rb, log_prob_rb = actor.get_action(state, omega_rb)
    action_da, log_prob_da = actor.get_action(state, omega_da)
    with torch.no_grad():
        action_adv_rb, log_prob_adv_rb = adversary.get_action(state, omega_rb)
        action_adv_da, log_prob_adv_da = adversary.get_action(state, omega_da)

    # 策略鲁棒性约束
    C_rb, delta_update = compute_robustness_constraint(actor, state, omega_rb, action_rb, stds, delta)
    C_da, _ = compute_robustness_constraint(actor, state, omega_da, action_da, stds, delta)

    U_rb_1 = cal_u_mix(qf1, state, action_rb, action_adv_rb, omega_rb)
    U_rb_2 = cal_u_mix(qf2, state, action_rb, action_adv_rb, omega_rb)
    U_rb = torch.min(U_rb_1, U_rb_2)

    U_da_1 = cal_u_mix(qf1, state, action_da, action_adv_da, omega_da)
    U_da_2 = cal_u_mix(qf2, state, action_da, action_adv_da, omega_da)
    U_da = torch.min(U_da_1, U_da_2)

    L = 2 * (U_rb + U_da) + eta * (2 * args.robustness_epsilon - C_rb - C_da)

    log_pi_sum = torch.sum(log_prob_rb, 1, keepdim=True)
    actor_loss = ((alpha * log_pi_sum) - L).mean()

    return actor_loss, C_rb, C_da, delta_update


def cal_adv_loss(actor, adversary, qf1, qf2, state, omega_rb, omega_da, epi_std):
    with torch.no_grad():
        action_rb, log_prob_rb = actor.get_action(state, omega_rb)
        action_da, log_prob_da = actor.get_action(state, omega_da)
    action_adv_rb, log_prob_adv_rb = adversary.get_action(state, omega_rb)
    action_adv_da, log_prob_adv_da = adversary.get_action(state, omega_da)

    U_rb_1 = cal_u_mix(qf1, state, action_rb, action_adv_rb, omega_rb)
    U_rb_2 = cal_u_mix(qf2, state, action_rb, action_adv_rb, omega_rb)
    U_rb = torch.min(U_rb_1, U_rb_2)

    U_da_1 = cal_u_mix(qf1, state, action_da, action_adv_da, omega_da)
    U_da_2 = cal_u_mix(qf2, state, action_da, action_adv_da, omega_da)
    U_da = torch.min(U_da_1, U_da_2)

    log_pi_sum = torch.sum(log_prob_adv_rb, 1, keepdim=True)
    adversary_loss = U_rb + U_da + 0.2 * log_pi_sum
    return adversary_loss.mean()


def cal_eta_loss(eta, C_rb, C_da):
    e_loss = eta * (2 * args.robustness_epsilon - C_rb - C_da)
    return e_loss
