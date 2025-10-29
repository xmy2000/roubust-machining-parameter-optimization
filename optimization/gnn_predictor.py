import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected

NODES = ["cut_count", "feature", "tool", "tool_wear", "tool_breakage", "allowance", "feature_size", "depth",
         "cut", "roughness", "cut_setting"]
predict_nodes = ["tool_wear", "tool_breakage", "cut", "roughness"]
numerical_vars = {
    "cut_count": [0, 1],
    "tool_radius": [0.618, 0.216],
    "tool_degree": [57.546, 19.089],
    "tool_length": [74.75, 60.732],
    "tool_wear": [298.764, 371.12],
    "allowance": [1.347, 1.803],
    "feature_size": [210.866, 64.003],
    "depth": [0.646, 0.609],
    "cut": [0.0253, 0.0324],
    "roughness": [1.725, 0.944],
    "feed": [0.105, 0.025],
    "rotate": [47.291, 16.177],
    "cutting_distance": [14.045, 20.082],
}
classified_vars = {
    "feature_type": 11,
    "cutting_type": 2,
    "material": 2,
    "tool_type": 7,
    "tool_extension": 3,
    "tool_supplier": 5,
    "tool_breakage": 3,
    "machine": 4,
    "machine_type": 2,
}

edge_lst = [
    ("feature", "cut_count"),
    ("feature", "tool"),
    ("feature", "allowance"),
    ("feature", "feature_size"),
    ("feature", "depth"),
    ("feature", "cut"),
    ("feature", "roughness"),
    ("feature", "cut_setting"),
    ("tool", "tool_wear"),
    ("tool", "tool_breakage"),
]
NODES_DIM = {
    'cut_count': 1,
    'feature': 15,
    'tool': 18,
    'tool_wear': 1,
    'tool_breakage': 3,
    'allowance': 1,
    'feature_size': 1,
    'depth': 1,
    'cut': 1,
    'roughness': 1,
    'cut_setting': 9
}


class GNNPredictor:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def forward(self, loader):
        for graph in loader:
            x = graph.x
            x[NODES.index('cut'), :NODES_DIM['cut']] = 0.0
            x[NODES.index('roughness'), :NODES_DIM['roughness']] = 0.0
            with torch.no_grad():
                x_encode = self.model.node_encoder(x)
                z = self.model.graph_encoder(x_encode)
                x_decode, x_std = self.model.node_decoder(z)
        return x_decode, x_std

    def predict_with_uncertainty(self, data, num_samples=10):
        graph = self.create_graph_data(data)
        loader = DataLoader([graph], batch_size=1, shuffle=False)
        result = {}
        epistemic_samples = {}
        for node in predict_nodes:
            result[node] = {}
            epistemic_samples[node] = []

        self.model.graph_encoder.conv1.lin.unfreeze()
        self.model.graph_encoder.conv2.lin.unfreeze()
        for _ in range(num_samples):
            x_decode, x_std = self.forward(loader)
            for node in predict_nodes:
                idx = NODES.index(node)
                y_mu = x_decode[idx][:NODES_DIM[node]]
                if node != "tool_breakage":
                    node_mean = numerical_vars[node][0]
                    node_std = numerical_vars[node][1]
                    y = y_mu * node_std + node_mean
                    epistemic_samples[node].append(y.item())
        for k, v in epistemic_samples.items():
            if k != "tool_breakage":
                samples = np.array(v)
                result[k].update({"epistemic_samples": v})
                result[k].update({"epistemic_std": samples.std().item()})

        self.model.graph_encoder.conv1.lin.freeze()
        self.model.graph_encoder.conv2.lin.freeze()
        x_decode, x_std = self.forward(loader)
        for node in predict_nodes:
            idx = NODES.index(node)
            y_mu = x_decode[idx][:NODES_DIM[node]]
            y_std = x_std[idx][:NODES_DIM[node]]
            if node == "tool_breakage":
                y = F.softmax(y_mu, dim=-1).tolist()
                result[node].update({"mean": y})
            else:
                node_mean = numerical_vars[node][0]
                node_std = numerical_vars[node][1]
                y = Normal(y_mu, y_std).sample(torch.Size([num_samples])).squeeze(-1) * node_std + node_mean
                y_mu = y_mu * node_std + node_mean
                result[node].update({
                    "mean": y_mu.item(),
                    "aleatory_samples": y[y >= 0].tolist(),
                    "aleatory_std": y.std().item(),
                })

        self.model.graph_encoder.conv1.lin.unfreeze()
        self.model.graph_encoder.conv2.lin.unfreeze()
        return result

    def load_model(self, model_path):
        model = Model.load_from_checkpoint(model_path)
        model.eval()
        return model

    def create_graph_data(self, data):
        for k, v in data.items():
            if k in numerical_vars:
                data[k] = [(data[k] - numerical_vars[k][0]) / numerical_vars[k][1]]
            elif k in classified_vars:
                code = [0 for _ in range(classified_vars[k])]
                code[data[k]] = 1
                data[k] = code
            else:
                raise Exception(k)

        node_data = {
            "cut_count": data["cut_count"],
            "feature": data["feature_type"] + data["cutting_type"] + data["material"],
            "tool": data["tool_type"]
                    + data["tool_radius"]
                    + data["tool_degree"]
                    + data["tool_length"]
                    + data["tool_extension"]
                    + data["tool_supplier"],
            "tool_wear": data["tool_wear"],
            "tool_breakage": data["tool_breakage"],
            "allowance": data["allowance"],
            "feature_size": data["feature_size"],
            "depth": data["depth"],
            "cut": data["cut"],
            "roughness": data["roughness"],
            "cut_setting": data["machine"]
                           + data["machine_type"]
                           + data["feed"]
                           + data["rotate"]
                           + data["cutting_distance"],
        }

        num_nodes = len(NODES)
        feature_dim = 18
        node_name_lst = NODES.copy()

        x = torch.zeros((num_nodes, feature_dim), dtype=torch.float)
        for i in range(num_nodes):
            node_name = NODES[i]
            d = node_data[node_name]
            x[i][:len(d)] = torch.tensor(d, dtype=torch.float)

        edge_index = [[NODES.index(s), NODES.index(t)] for s, t in edge_lst if
                      s in node_name_lst and t in node_name_lst]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index)

        graph = Data(
            x=x,
            edge_index=edge_index,
            num_nodes=len(node_name_lst),
            node_lst=node_name_lst,
            node_mask=[NODES.index(n) for n in node_name_lst],
            tool_id="predict_tool",
            timestamp=1,
        )
        return graph

