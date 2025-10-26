import torch.nn.functional as F
from torch import nn
import lightning as L
from torch_geometric.nn import MLP, global_mean_pool
from parameters import *
from BayesianGCN import GCNConv
from utils import adjacency_to_edge_weight
from scipy.stats import norm


class UnGCNLayer(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim):
        super().__init__()
        self.mean_gcn = GCNConv(mu=0, sigma=0.1, in_channels=node_feature_dim, out_channels=hidden_dim)
        self.std_gcn = GCNConv(mu=0, sigma=0.1, in_channels=node_feature_dim, out_channels=hidden_dim)

    def forward(self, mean, std, edge_index, edge_weight):
        new_mean, kl_mean = self.mean_gcn(mean, edge_index, edge_weight)
        var = std ** 2
        new_log_var, kl_std = self.std_gcn(var, edge_index, edge_weight ** 2)
        new_var = torch.exp(new_log_var) + 1e-6
        new_std = torch.sqrt(new_var)
        total_kl = kl_mean + kl_std
        return new_mean, new_std, total_kl


class GraphEncoder(nn.Module):
    def __init__(
            self,
            node_feature_dim: int,
            hidden_dim: int,
            latent_dim: int,
            num_propagation_layers: int,
            num_samples: int = 1
    ):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.num_samples = num_samples
        self.kl_total = 0.0

        edge_index = [[i, j] for i in range(NUM_NODES) for j in range(NUM_NODES) if i != j]
        self.edge_index = torch.tensor(edge_index, dtype=torch.long, device=DEVICE).t().contiguous()

        self.init_mean_gcn = GCNConv(mu=0, sigma=0.1, in_channels=node_feature_dim, out_channels=hidden_dim)
        self.init_std_gcn = GCNConv(mu=0, sigma=0.1, in_channels=node_feature_dim, out_channels=hidden_dim)

        self.propagation_layers = nn.ModuleList()
        for _ in range(num_propagation_layers):
            out_dim = hidden_dim if _ < num_propagation_layers - 1 else latent_dim
            self.propagation_layers.append(UnGCNLayer(node_feature_dim=hidden_dim, hidden_dim=out_dim))

    def kl_loss(self) -> torch.Tensor:
        return self.kl_total

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor):
        self.kl_total = 0.0
        edge_weight = adjacency_to_edge_weight(adj_matrix, self.edge_index)

        init_mean, kl_init_mean = self.init_mean_gcn(x, self.edge_index, edge_weight)
        init_log_var, kl_init_std = self.init_std_gcn(x, self.edge_index, edge_weight ** 2)
        init_std = torch.sqrt(torch.exp(init_log_var) + 1e-6)
        self.kl_total += kl_init_mean + kl_init_std

        current_mean, current_std = init_mean, init_std
        for prop_layer in self.propagation_layers:
            current_mean, current_std, kl_prop = prop_layer(
                mean=current_mean,
                std=current_std,
                edge_index=self.edge_index,
                edge_weight=edge_weight
            )
            self.kl_total += kl_prop  # 累计传播层KL

        normalized_kl = self.kl_total / self.num_samples
        return current_mean, current_std, normalized_kl


class NodeEncoder(nn.Module):
    def __init__(self, node_feature_dim):
        super(NodeEncoder, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.node_encoders = nn.ModuleDict()
        for k, v in NODES_DIM.items():
            self.node_encoders[k] = nn.Linear(v, node_feature_dim)

    def forward(self, x):
        nodes_feature = torch.zeros((NUM_NODES, self.node_feature_dim), dtype=torch.float, device=DEVICE)
        for n in NODES:
            index = NODES.index(n)
            nodes_feature[index] = self.node_encoders[n](x[index, :NODES_DIM[n]])
        return nodes_feature


class NodeDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(NodeDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.node_decoders = nn.ModuleDict()
        self.nn_logstd = nn.ModuleDict()
        for k, v in NODES_DIM.items():
            self.node_decoders[k] = MLP(
                in_channels=latent_dim,
                hidden_channels=hidden_dim,
                out_channels=v,
                num_layers=3,
                norm="layer_norm",
                act="swish"
            )
            self.nn_logstd[k] = MLP(
                in_channels=latent_dim,
                hidden_channels=hidden_dim,
                out_channels=v,
                num_layers=3,
                norm="layer_norm",
                act="swish"
            )

    def forward(self, z):
        x_recon = torch.zeros((NUM_NODES, MAX_NODE_DIM), dtype=torch.float, device=DEVICE)
        x_logstd = torch.zeros((NUM_NODES, MAX_NODE_DIM), dtype=torch.float, device=DEVICE)
        for n in NODES:
            index = NODES.index(n)
            x_recon[index, :NODES_DIM[n]] = self.node_decoders[n](z[index])
            x_logstd[index, :NODES_DIM[n]] = self.nn_logstd[n](z[index])
        return x_recon, x_logstd.exp()


class CausalEncoder(nn.Module):
    def __init__(
            self,
            latent_dim: int,
            num_nodes: int,
            adj_matrix: torch.Tensor = None  # 补充默认值
    ):
        super(CausalEncoder, self).__init__()
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim

        if adj_matrix is None:
            adj_matrix = torch.randn(num_nodes, num_nodes, device=DEVICE) * 0.1
        self.C = nn.Parameter(adj_matrix, requires_grad=True)
        self.C.data.fill_diagonal_(0.0)

        self.causal_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Mish(),
            nn.Linear(latent_dim, latent_dim)
        )

    def dag_constraint(self, A: torch.Tensor) -> torch.Tensor:
        AC = A * self.C
        I = torch.eye(self.num_nodes, device=DEVICE)
        mat = I + AC
        mat_pow = torch.matrix_power(mat, self.num_nodes)
        h_c = torch.trace(mat_pow) - self.num_nodes
        return h_c

    def forward(self, z: torch.Tensor, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        AC = A * self.C
        I = torch.eye(self.num_nodes, device=DEVICE)
        try:
            inv_mat = torch.inverse(I - AC.T)
        except:
            inv_mat = torch.pinverse(I - AC.T)
        z_causal = inv_mat @ z

        z_recon = []
        for node_idx in range(self.num_nodes):
            parent_mask = (self.C[:, node_idx] != 0).float().unsqueeze(1)
            parent_z = z_causal * parent_mask
            z_i_recon = self.causal_mlp(parent_z.sum(dim=0, keepdim=True))
            z_recon.append(z_i_recon)
        z_recon = torch.cat(z_recon, dim=0)

        dag_loss = self.dag_constraint(A)
        return z_causal, z_recon, dag_loss


class Model(L.LightningModule):
    def __init__(
            self,
            node_feature_dim: int,
            hidden_dim: int,
            latent_dim: int,
            dataset,
            num_propagation_layers: int = 2,
            num_uncertainty_samples: int = 30
    ):
        super(Model, self).__init__()
        self.save_hyperparameters()
        self.test_dataset = dataset
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_nodes = NUM_NODES
        self.num_propagation_layers = num_propagation_layers
        self.num_uncertainty_samples = num_uncertainty_samples
        self.confidence = 0.95
        self.z_alpha = torch.tensor(norm.ppf((1 + self.confidence) / 2), device=DEVICE)

        adj_matrix = torch.load("./data/new_adj_matrix.pt", weights_only=False)
        self.adj_matrix = nn.Parameter(adj_matrix, requires_grad=True)

        self.node_encoder = NodeEncoder(node_feature_dim=node_feature_dim)
        self.graph_encoder = GraphEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_propagation_layers=num_propagation_layers,
            num_samples=num_uncertainty_samples
        )
        self.node_decoder = NodeDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.causal_encoder = CausalEncoder(
            latent_dim=latent_dim,
            num_nodes=self.num_nodes,
            adj_matrix=self.adj_matrix
        )

        self.global_pool = global_mean_pool
        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=2,
            batch_first=True,
            device=DEVICE
        )

    def calc_pi_metrics(self, x: torch.Tensor, x_recon: torch.Tensor, x_std: torch.Tensor, node_mask: torch.Tensor):
        x, x_recon, x_std = x[node_mask], x_recon[node_mask], x_std[node_mask]
        pi_low = x_recon - self.z_alpha * x_std
        pi_up = x_recon + self.z_alpha * x_std

        k = ((x >= pi_low) & (x <= pi_up)).float()
        picp = k.mean()

        piw = (pi_up - pi_low) * k
        piw = piw.sum() / (k.sum() + 1e-6)
        y_range = x.max() - x.min() + 1e-6
        pinaw = piw / y_range

        return picp, pinaw, pi_low, pi_up

    def uncertainty_decomposition(self, z_mean: torch.Tensor, z_std: torch.Tensor, node_mask: torch.Tensor):
        x_std_valid = z_std[node_mask]
        u_a = x_std_valid.pow(2).mean()

        sample_means = []
        for _ in range(self.num_uncertainty_samples):
            sample_z_mean, _, _ = self.graph_encoder(self.node_encoder(z_mean), self.adj_matrix)
            sample_means.append(sample_z_mean[node_mask].unsqueeze(0))
        sample_means = torch.cat(sample_means, dim=0)
        u_e = sample_means.var(dim=0).mean()

        u_pred = u_a + u_e
        return u_e, u_a, u_pred

    def forward(self, graph_sequence: list):
        wear_memory = torch.zeros(NODES_DIM['tool_wear'], dtype=torch.float, device=DEVICE)
        breakage_memory = torch.zeros(NODES_DIM['tool_breakage'], dtype=torch.float, device=DEVICE)
        hist_emb = None

        rec_loss = 0.0,
        log_likelihood = 0.0
        interval_loss = 0.0
        z_recon_loss = 0.0
        dag_loss = 0.0
        kl_total = 0.0
        u_e_total = 0.0
        u_a_total = 0.0

        seq_len = len(graph_sequence)
        for graph in graph_sequence:
            x, node_mask = graph.x, graph.node_mask[0]
            x_clone = x.clone()

            x_clone[NODES.index('tool_wear'), :NODES_DIM['tool_wear']] = wear_memory
            x_clone[NODES.index('tool_breakage'), :NODES_DIM['tool_breakage']] = breakage_memory
            x_clone[NODES.index('cut'), :NODES_DIM['cut']] = 0.0
            x_clone[NODES.index('roughness'), :NODES_DIM['roughness']] = 0.0

            x_encode = self.node_encoder(x_clone)

            z_mean, z_std, gcn_kl = self.graph_encoder(x_encode, self.adj_matrix)
            kl_total += gcn_kl

            graph_feat = self.global_pool(z_mean, graph.batch)
            if hist_emb is None:
                hist_emb = torch.zeros(2, 1, graph_feat.shape[1])
            gru_out, hist_emb = self.gru(graph_feat.unsqueeze(1), hist_emb)
            z_fused = z_mean + gru_out.squeeze(1)

            z_causal, z_recon, dag_loss_step = self.causal_encoder(z_fused, self.adj_matrix)
            dag_loss += dag_loss_step
            z_recon_loss += F.mse_loss(z_causal, z_recon)

            x_recon, x_std = self.node_decoder(z_causal)

            wear_memory = x_recon[NODES.index('tool_wear'), :NODES_DIM['tool_wear']]
            breakage_memory = x_recon[NODES.index('tool_breakage'), :NODES_DIM['tool_breakage']]

            x_valid = x[node_mask]
            x_recon_valid = x_recon[node_mask]
            rec_loss += F.mse_loss(x_valid, x_recon_valid)

            x_std_valid = x_std[node_mask]
            log_like_step = -0.5 * torch.mean(
                torch.log(x_std_valid.pow(2) + 1e-6) +
                (x_valid - x_recon_valid).pow(2) / (x_std_valid.pow(2) + 1e-6)
            )
            log_likelihood += log_like_step

            picp, pinaw, _, _ = self.calc_pi_metrics(x, x_recon, x_std, node_mask)
            interval_loss += pinaw - torch.sqrt(torch.tensor(self.num_nodes, device=DEVICE)) * picp

            u_e, u_a, _ = self.uncertainty_decomposition(z_mean, z_std, node_mask)
            u_e_total += u_e
            u_a_total += u_a

        avg_metrics = {
            "rec_loss": rec_loss / seq_len,
            "elbo_loss": (kl_total / seq_len) - (log_likelihood / seq_len),
            "interval_loss": interval_loss / seq_len,
            "z_recon_loss": z_recon_loss / seq_len,
            "dag_loss": dag_loss / seq_len,
            "kl_loss": kl_total / seq_len,
            "u_e": u_e_total / seq_len,
            "u_a": u_a_total / seq_len,
            "u_pred": (u_e_total + u_a_total) / seq_len,
            "picp": picp,
            "pinaw": pinaw
        }
        return avg_metrics

    def cal_loss(self, graph_sequence: list):
        forward_out = self.forward(graph_sequence)

        elbo_loss = forward_out["elbo_loss"].clamp(min=1e-6)
        rec_loss = forward_out["rec_loss"].clamp(min=1e-6)
        pi_loss = forward_out["interval_loss"].abs().clamp(min=1e-6)
        z_loss = forward_out["z_recon_loss"].clamp(min=1e-6)
        dag_loss = (forward_out["dag_loss"] + 1.0).clamp(min=1e-6)

        total_loss = torch.pow(elbo_loss * rec_loss * pi_loss * z_loss * dag_loss, 1.0 / 5.0)

        return (
            total_loss,
            forward_out["rec_loss"],
            forward_out["elbo_loss"],
            forward_out["interval_loss"],
            forward_out["dag_loss"],
            forward_out["kl_loss"]
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer
