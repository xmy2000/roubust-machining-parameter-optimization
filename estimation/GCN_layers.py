import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class GCNModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            num_propagation_layers: int = 2,
            prior_var: float = 1.0,
            num_samples: int = 1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_propagation_layers = num_propagation_layers
        self.prior_var = prior_var
        self.num_samples = num_samples

        self.init_mean_gcn = BayesianGCNLayer(
            in_channels=in_channels,
            out_channels=hidden_channels,
            prior_var=prior_var
        )
        self.init_std_gcn = BayesianGCNLayer(
            in_channels=in_channels,
            out_channels=hidden_channels,
            prior_var=prior_var
        )

        self.propagation_layers = nn.ModuleList()
        for _ in range(num_propagation_layers):
            self.propagation_layers.append(
                PropGCNLayer(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels if _ < num_propagation_layers - 1 else out_channels,
                    prior_var=prior_var
                )
            )

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_weight if hasattr(data, "edge_weight") else torch.ones(edge_index.shape[1],
                                                                                       device=x.device)
        total_kl_loss = 0.0

        init_mean, kl_init_mean = self.init_mean_gcn(x, edge_index, edge_weight)
        init_log_var, kl_init_std = self.init_std_gcn(x, edge_index, edge_weight ** 2)
        init_std = torch.sqrt(torch.exp(init_log_var) + 1e-6)
        total_kl_loss += kl_init_mean + kl_init_std

        current_mean, current_std = init_mean, init_std
        for prop_layer in self.propagation_layers:
            current_mean, current_std, kl_prop = prop_layer(
                mean=current_mean,
                std=current_std,
                edge_index=edge_index,
                edge_weight=edge_weight
            )
            total_kl_loss += kl_prop

        return current_mean, current_std, total_kl_loss / self.num_samples


class BayesianGCNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, prior_var: float = 1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.prior_var = prior_var

        self.mu = nn.Parameter(torch.empty(in_channels, out_channels))
        self.log_sigma = nn.Parameter(torch.empty(in_channels, out_channels))

        nn.init.xavier_uniform_(self.mu)
        nn.init.constant_(self.log_sigma, -3.0)

    def reparameterize(self) -> torch.Tensor:
        sigma = torch.exp(self.log_sigma)
        eps = torch.randn_like(sigma)
        return self.mu + eps * sigma

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor]:
        weight = self.reparameterize()

        support = torch.matmul(x, weight)
        out = GCNConv(self.in_channels, self.out_channels, bias=False).propagate(
            edge_index, x=support, edge_weight=edge_weight
        )

        kl = 0.5 * (
                torch.exp(2 * self.log_sigma) / self.prior_var
                + self.mu ** 2 / self.prior_var
                - 2 * self.log_sigma
                - 2 * torch.log(torch.sqrt(torch.tensor(self.prior_var, device=x.device)))
                - 1
        )
        kl_loss = kl.sum()

        return out, kl_loss


class PropGCNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, prior_var: float = 1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.prior_var = prior_var

        self.mean_gcn = BayesianGCNLayer(in_channels, out_channels, prior_var)
        self.std_gcn = BayesianGCNLayer(in_channels, out_channels, prior_var)

    def forward(
            self,
            mean: torch.Tensor,
            std: torch.Tensor,
            edge_index: torch.Tensor,
            edge_weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        new_mean, kl_mean = self.mean_gcn(mean, edge_index, edge_weight)

        var = std ** 2
        new_log_var, kl_std = self.std_gcn(var, edge_index, edge_weight ** 2)
        new_var = torch.exp(new_log_var) + 1e-6  # 还原新方差（避免负数值）
        new_std = torch.sqrt(new_var)  # 新标准差（σ）

        total_kl = kl_mean + kl_std

        return new_mean, new_std, total_kl
