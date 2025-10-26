import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from parameters import *

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号


def adjacency_to_edge_index(adj_matrix):
    """
    将邻接矩阵转换为PyG的edge_index和edge_weight

    参数:
    adj_matrix (torch.Tensor): 邻接矩阵，形状为[N, N]，元素值在0到1之间

    返回:
    edge_index (torch.Tensor): 边索引，形状为[2, E]
    edge_weight (torch.Tensor): 边权重，形状为[E]
    """
    # 获取邻接矩阵中非零元素的坐标和值
    row_indices, col_indices = torch.nonzero(adj_matrix, as_tuple=True)
    edge_weight = adj_matrix[row_indices, col_indices]

    # 构建edge_index [2, E]
    edge_index = torch.stack([row_indices, col_indices], dim=0)

    return edge_index, edge_weight


def adjacency_to_edge_weight(adj_matrix, edge_index):
    edge_weight = torch.zeros(edge_index.shape[1], dtype=torch.float, device=DEVICE)
    for i in range(edge_index.shape[1]):
        row = edge_index[0, i]
        col = edge_index[1, i]
        edge_weight[i] = adj_matrix[row, col]
    return edge_weight


def matrix_poly(matrix, n):
    x = torch.eye(n, device=DEVICE) + torch.div(matrix, 1)
    return torch.matrix_power(x, n)


def h_A(A):
    m = A.shape[0]
    expm_A = matrix_poly(A * A, m)
    h_A = torch.trace(expm_A) - m
    return h_A


def boundary_penalty(matrix):
    """计算矩阵元素超出[0,1]范围的惩罚"""
    # 低于0的部分
    below_zero = torch.clamp(matrix, max=0) ** 2
    # 高于1的部分
    above_one = torch.clamp(matrix - 1, min=0) ** 2
    # 总惩罚（可根据需要加权）
    return torch.mean(below_zero + above_one)


def plot_error(name, error):
    mean = np.mean(error)

    plt.figure(figsize=(16, 8))
    plt.suptitle(name, fontsize=18)
    plt.subplot(121)
    plt.plot(error, 'o')
    plt.axhline(mean, color='r', linewidth=2)
    plt.subplot(122)
    sns.kdeplot(data=error, fill=True, common_norm=False, alpha=.5, linewidth=0)
    plt.axvline(mean, color='r', linewidth=2)
    plt.tight_layout()


def plot_CI(y_pred, y_mean, y_true):
    lower_bound = np.quantile(y_pred, 0.05, axis=1)
    upper_bound = np.quantile(y_pred, 0.95, axis=1)

    plt.figure()
    plt.scatter(np.arange(len(y_true)), y_true, label='True Values', color='blue', alpha=0.6)
    plt.scatter(np.arange(len(y_mean)), y_mean, label='Pred Values', color='red', alpha=0.6)
    plt.fill_between(np.arange(len(y_true)), lower_bound.flatten(), upper_bound.flatten(), color='gray', alpha=0.2,
                     label='95% Confidence Interval')
    plt.xlabel('Data Points')
    plt.ylabel('Values')
    plt.title('True Values with 95% Confidence Interval')
    plt.legend()


def plot_CI_errorbar(y_pred, y_mean, y_true):
    lower_bound = np.quantile(y_pred, 0.05, axis=1)
    upper_bound = np.quantile(y_pred, 0.95, axis=1)
    y_err = [y_mean - lower_bound, upper_bound - y_mean]

    plt.figure()
    plt.scatter(np.arange(len(y_true)), y_true, label='True Values', color='blue', alpha=0.6)
    # plt.scatter(np.arange(len(y_mean)), y_mean, label='Pred Values', color='red', alpha=0.6)
    plt.errorbar(np.arange(len(y_mean)), y_mean, yerr=y_err, fmt='o', capsize=5, capthick=2, color='red', alpha=0.6)

    plt.xlabel('Data Points')
    plt.ylabel('Values')
    plt.title('True Values with 95% Confidence Interval')
    plt.legend()
