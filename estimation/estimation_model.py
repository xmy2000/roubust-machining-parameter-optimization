import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data, Batch
import torch.distributions as dist
from scipy.stats import norm
from typing import Dict, List, Tuple
from GCN_layers import GCNModule


# --------------------------
# 完整加工指标估计模型（3.2节核心）
# --------------------------
class CausalMachingIndicatorEstimator(nn.Module):
    def __init__(
            self,
            node_type_dims: Dict[str, int],
            gcn_hidden: int = 16,
            causal_dim: int = 16,
            num_gcn_propagation: int = 2,
            prior_var: float = 1.0,
            gru_layers: int = 2,
            confidence_level: float = 0.95
    ):
        super().__init__()
        self.node_types = list(node_type_dims.keys())
        self.node_type_dims = node_type_dims
        self.causal_dim = causal_dim
        self.confidence_level = confidence_level
        self.z_alpha = torch.tensor(norm.ppf((1 + confidence_level) / 2), dtype=torch.float32)  # 95%置信度对应zα=1.96

        self.node_encoders = nn.ModuleDict()
        for node_type, in_dim in node_type_dims.items():
            self.node_encoders[node_type] = nn.Sequential(
                nn.Linear(in_dim, gcn_hidden),
                nn.ReLU(),
                nn.Linear(gcn_hidden, gcn_hidden)
            )

        self.graph_encoder = GCNModule(
            in_channels=gcn_hidden,
            hidden_channels=gcn_hidden,
            out_channels=gcn_hidden,
            num_propagation_layers=num_gcn_propagation,
            prior_var=prior_var
        )

        self.global_pool = global_mean_pool
        self.gru = nn.GRU(
            input_size=gcn_hidden,
            hidden_size=gcn_hidden,
            num_layers=gru_layers,
            batch_first=True
        )

        self.num_nodes_per_graph = len(self.node_types)  # 每个加工图的节点数
        self.causal_matrix = nn.Parameter(torch.randn(self.num_nodes_per_graph, self.num_nodes_per_graph) * 0.1)
        self.causal_matrix.data.fill_diagonal_(0.0)  # 无自环（DAG基础约束）
        self.causal_mlp = nn.Sequential(  # 公式13的g_i：子节点由父节点生成
            nn.Linear(gcn_hidden, causal_dim),
            nn.Mish(),  # 文档3.2.1节指定激活函数
            nn.Linear(causal_dim, causal_dim)
        )

        self.node_decoders = nn.ModuleDict()
        for node_type, out_dim in node_type_dims.items():
            self.node_decoders[node_type] = nn.Sequential(
                nn.Linear(causal_dim, gcn_hidden),
                nn.Mish(),
                nn.Linear(gcn_hidden, out_dim * 2)  # 输出：[均值, log(方差)]
            )

    def dag_constraint(self, edge_weight: torch.Tensor) -> torch.Tensor:
        ac = edge_weight * self.causal_matrix  # A⊙C（元素-wise乘法，整合图拓扑不确定性）
        i = torch.eye(self.num_nodes_per_graph, device=ac.device)
        mat = i + ac
        mat_pow = torch.matrix_power(mat, self.num_nodes_per_graph)  # (I+AC)^Nv
        h_c = torch.trace(mat_pow) - self.num_nodes_per_graph  # 公式19
        return h_c

    def calc_pi_metrics(self, y_true: torch.Tensor, y_mean: torch.Tensor, y_var: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        # 计算预测区间：[y_mean - zα*σ, y_mean + zα*σ]
        pi_low = y_mean - self.z_alpha * torch.sqrt(y_var + 1e-6)
        pi_up = y_mean + self.z_alpha * torch.sqrt(y_var + 1e-6)

        # PICP（公式17）
        k = ((y_true >= pi_low) & (y_true <= pi_up)).float()
        picp = k.mean()

        # PINAW（公式17，归一化到[y_min, y_max]范围）
        y_min = y_true.min(dim=0, keepdim=True)[0]
        y_max = y_true.max(dim=0, keepdim=True)[0]
        piw = (pi_up - pi_low) * k
        pinaw = (piw.sum(dim=0) / (k.sum(dim=0) + 1e-6)) / (y_max - y_min + 1e-6)
        pinaw = pinaw.mean()  # 平均PINAW

        return picp, pinaw

    def forward(
            self,
            data_list: List[Data],
            node_types_list: List[List[str]],  # 每个图的节点类型列表（与data_list对应）
            hist_gru_hidden: torch.Tensor = None  # GRU历史隐藏状态（多道次切削时传递）
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        device = next(self.parameters()).device
        batch = Batch.from_data_list(data_list)  # 批量处理图数据
        batch_size = batch.num_graphs
        z_alpha = self.z_alpha.to(device)

        flat_node_types = []
        for types in node_types_list:
            flat_node_types.extend(types)
        # 逐个节点编码
        x_encoded_list = []
        for i in range(batch.x.shape[0]):
            node_type = flat_node_types[i]
            x_encoded = self.node_encoders[node_type](batch.x[i].unsqueeze(0))
            x_encoded_list.append(x_encoded)
        x_encoded = torch.cat(x_encoded_list, dim=0)  # (total_nodes, gcn_hidden)

        edge_weight_matrix = torch.zeros(batch_size, self.num_nodes_per_graph, self.num_nodes_per_graph, device=device)
        for idx, data in enumerate(data_list):
            ew = data.edge_weight if hasattr(data, "edge_weight") else torch.ones(data.edge_index.shape[1],
                                                                                  device=device)
            for (u, v), w in zip(data.edge_index.T, ew):
                edge_weight_matrix[idx, u, v] = w
        node_mean, node_std, gcn_kl_loss = self.graph_encoder(
            x=x_encoded,
            edge_index=batch.edge_index,
            edge_weight=batch.edge_weight if hasattr(batch, "edge_weight") else torch.ones(batch.edge_index.shape[1],
                                                                                           device=device)
        )
        node_var = node_std ** 2  # 随机不确定性（方差）

        graph_feat = self.global_pool(node_mean, batch.batch)  # (batch_size, gcn_hidden)
        # GRU融合历史经验（初始隐藏状态为None时自动初始化）
        gru_out, new_gru_hidden = self.gru(graph_feat.unsqueeze(1), hist_gru_hidden)  # (batch_size, 1, gcn_hidden)
        gru_feat = gru_out.squeeze(1)  # (batch_size, gcn_hidden)

        causal_z_list = []  # 因果表示z（每个图的节点级z）
        causal_z_recon_list = []  # 掩码重建后的z（公式13）
        dag_loss_list = []  # DAG约束损失（每个图）
        for idx in range(batch_size):
            # 提取当前图的节点均值、边权重矩阵
            node_start = idx * self.num_nodes_per_graph
            node_end = (idx + 1) * self.num_nodes_per_graph
            current_node_mean = node_mean[node_start:node_end]  # (num_nodes, gcn_hidden)
            current_ew_matrix = edge_weight_matrix[idx]  # (num_nodes, num_nodes)

            # 1. 因果结构建模（公式12：z = (I - (A⊙C)^T)^{-1} · ε_ex）
            ac = current_ew_matrix * self.causal_matrix  # A⊙C
            i = torch.eye(self.num_nodes_per_graph, device=device)
            try:
                inv_mat = torch.inverse(i - ac.T)  # 逆矩阵（DAG确保可逆）
            except:
                inv_mat = torch.pinverse(i - ac.T)  # 数值稳定：伪逆
            eps_ex = current_node_mean  # 外源嵌入ε_ex（近似为节点均值，文档3.2.1节）
            causal_z = inv_mat @ eps_ex  # 因果表示z（num_nodes, gcn_hidden）

            # 2. 掩码自重建（公式13：z_i仅由父节点生成）
            causal_z_recon = []
            for node_idx in range(self.num_nodes_per_graph):
                # 掩码：仅保留当前节点的父节点（C[父节点, 当前节点]≠0）
                parent_mask = (self.causal_matrix[:, node_idx] != 0).float().unsqueeze(1)  # (num_nodes, 1)
                parent_z = causal_z * parent_mask  # 仅父节点信息
                z_i_recon = self.causal_mlp(parent_z.sum(dim=0, keepdim=True))  # 父节点聚合
                causal_z_recon.append(z_i_recon)
            causal_z_recon = torch.cat(causal_z_recon, dim=0)  # (num_nodes, causal_dim)

            # 3. DAG约束损失（公式19）
            dag_loss = self.dag_constraint(current_ew_matrix)

            # 收集结果
            causal_z_list.append(causal_z)
            causal_z_recon_list.append(causal_z_recon)
            dag_loss_list.append(dag_loss)

        # 批量整合因果表示
        causal_z_batch = torch.cat(causal_z_list, dim=0)  # (total_nodes, causal_dim)
        causal_z_recon_batch = torch.cat(causal_z_recon_list, dim=0)  # (total_nodes, causal_dim)
        avg_dag_loss = torch.stack(dag_loss_list).mean()  # 批量平均DAG损失

        y_pred_mean_list = []  # 加工指标预测均值
        y_pred_var_list = []  # 加工指标预测方差（随机不确定性）
        y_true_list = []  # 加工指标真实值（从data.y提取）
        for idx in range(batch_size):
            # 提取当前图的因果表示、真实标签
            node_start = idx * self.num_nodes_per_graph
            node_end = (idx + 1) * self.num_nodes_per_graph
            current_z = causal_z_batch[node_start:node_end]
            current_y_true = batch.y[node_start:node_end]  # 假设data.y存储节点真实值

            # 逐个节点解码
            for node_idx in range(self.num_nodes_per_graph):
                node_type = node_types_list[idx][node_idx]
                # 解码输出：[均值, log(方差)]
                dec_out = self.node_decoders[node_type](current_z[node_idx].unsqueeze(0))
                out_dim = self.node_type_dims[node_type]
                y_mean = dec_out[:, :out_dim]
                y_log_var = dec_out[:, out_dim:]
                y_var = torch.exp(y_log_var)  # 方差（确保非负）

                # 收集结果
                y_pred_mean_list.append(y_mean)
                y_pred_var_list.append(y_var)
                y_true_list.append(current_y_true[node_idx].unsqueeze(0))

        # 批量整合预测结果
        y_pred_mean = torch.cat(y_pred_mean_list, dim=0)  # (total_nodes, max_out_dim)
        y_pred_var = torch.cat(y_pred_var_list, dim=0)  # (total_nodes, max_out_dim)
        y_true = torch.cat(y_true_list, dim=0)  # (total_nodes, max_out_dim)

        log_likelihood = dist.Normal(y_pred_mean, torch.sqrt(y_pred_var + 1e-6)).log_prob(y_true).mean()
        elbo_loss = (gcn_kl_loss / batch_size) - log_likelihood  # KL按批量归一化

        l_reg = 0.0  # 数值节点（如tool wear、Ra、Δ）
        l_cls = 0.0  # 分类节点（如tool breakage）
        reg_count = 0
        cls_count = 0
        for idx in range(y_true.shape[0]):
            node_type = flat_node_types[idx % self.num_nodes_per_graph]
            y_t = y_true[idx].unsqueeze(0)
            y_m = y_pred_mean[idx].unsqueeze(0)
            if node_type in ["tool", "quality", "param", "allowance"]:  # 数值节点（文档4.1节）
                l_reg += F.mse_loss(y_m, y_t)
                reg_count += 1
            elif node_type == "breakage":  # 分类节点（0/1）
                l_cls += F.binary_cross_entropy_with_logits(y_m, y_t)
                cls_count += 1
        l_rec = (l_reg / (reg_count + 1e-6)) + (l_cls / (cls_count + 1e-6))

        picp, pinaw = self.calc_pi_metrics(y_true, y_pred_mean, y_pred_var)
        l_pi = pinaw - torch.sqrt(torch.tensor(self.num_nodes_per_graph, device=device)) * picp  # 公式17

        l_z = F.mse_loss(causal_z_batch, causal_z_recon_batch)

        dag_loss = avg_dag_loss

        total_loss = torch.pow(
            elbo_loss * l_rec * l_pi * l_z * (dag_loss + 1e-6),  # +1e-6避免0值
            1.0 / 5.0
        )

        epistemic_uncertainty = gcn_kl_loss / batch_size  # 认知不确定性（KL损失量化）
        aleatoric_uncertainty = y_pred_var.mean()  # 随机不确定性（预测方差均值）
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty

        pred_dict = {
            "y_pred_mean": y_pred_mean,  # 加工指标点估计
            "y_pred_var": y_pred_var,  # 加工指标方差（随机不确定性）
            "y_true": y_true,  # 加工指标真实值
            "epistemic_uncertainty": epistemic_uncertainty,  # 认知不确定性
            "aleatoric_uncertainty": aleatoric_uncertainty,  # 随机不确定性
            "total_uncertainty": total_uncertainty,  # 总不确定性
            "picp": picp,  # 预测区间覆盖率（文档4.2.2节）
            "pinaw": pinaw,  # 归一化平均区间宽度（文档4.2.2节）
            "new_gru_hidden": new_gru_hidden  # 新GRU隐藏状态（多道次切削传递）
        }

        loss_dict = {
            "total_loss": total_loss,  # 总损失（公式15）
            "elbo_loss": elbo_loss,  # ELBO损失（认知不确定性优化）
            "recon_loss": l_rec,  # 重建损失（预测 accuracy）
            "pi_loss": l_pi,  # PI损失（预测区间质量）
            "causal_loss": l_z,  # 因果损失（因果关系稳定性）
            "dag_loss": dag_loss  # DAG损失（因果结构有效性）
        }

        return pred_dict, loss_dict
