import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch
from sympy.core.benchmarks.bench_sympify import timeit_sympify_x
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

df = pd.read_csv("../data/data.csv")
tool_lst = df['tool_id'].unique().tolist()
print("Data Shape=", df.shape)
print("Tool Count=", len(tool_lst))

node_lst = ['cut_count', 'feature', 'tool', 'tool_wear', 'tool_breakage', 'allowance', 'feature_size', 'depth', 'cut',
            'roughness', 'cut_setting']
tool_attr_lst = ['tool_type', 'tool_radius', 'tool_degree', 'tool_length', 'tool_extension', 'tool_supplier']  # 18
feature_attr_lst = ['feature_type', 'cutting_type', 'material']  # 15
cut_setting_attr_lst = ['machine', 'machine_type', 'feed', 'rotate', 'cutting_distance']  # 9

edge_lst = [
    ('feature', 'cut_count'),
    ('feature', 'tool'),
    ('feature', 'allowance'),
    ('feature', 'feature_size'),
    ('feature', 'depth'),
    ('feature', 'cut'),
    ('feature', 'roughness'),
    ('feature', 'cut_setting'),
    ('tool', 'tool_wear'),
    ('tool', 'tool_breakage'),
]

# visualization
graph = nx.DiGraph(edge_lst)
pos = nx.spring_layout(graph)
print(f"Original graph contains {len(graph.nodes)} nodes, {len(graph.edges)} edges")
plt.figure(figsize=[10, 8])
nx.draw_networkx_nodes(graph, pos, node_size=1000, alpha=0.8)
nx.draw_networkx_edges(graph, pos, arrows=True, alpha=0.6, arrowsize=20, arrowstyle='->')
nx.draw_networkx_labels(graph, pos, font_size=10, font_family='SimHei')
plt.savefig("graph.png", dpi=300)

material = pd.get_dummies(df['material'], prefix='material').astype(float)
machine = pd.get_dummies(df['machine'], prefix='machine').astype(float)
machine_type = pd.get_dummies(df['machine_type'], prefix='machine_type').astype(float)
cutting_type = pd.get_dummies(df['cutting_type'], prefix='cutting_type').astype(float)
feature_type = pd.get_dummies(df['feature_type'], prefix='feature_type').astype(float)
tool_type = pd.get_dummies(df['tool_type'], prefix='tool_type').astype(float)
tool_extension = pd.get_dummies(df['tool_extension'], prefix='tool_extension').astype(float)
tool_supplier = pd.get_dummies(df['tool_supplier'], prefix='tool_supplier').astype(float)
tool_breakage = pd.get_dummies(df['tool_breakage'], prefix='tool_breakage').astype(float)

cut_count = df['cut_count']
feature = pd.concat([feature_type, cutting_type, material], axis=1)
tool = pd.concat([tool_type, df['tool_radius'], df['tool_degree'], df['tool_length'], tool_extension, tool_supplier],
                 axis=1)
tool_wear = df['tool_wear']
tool_breakage = tool_breakage
allowance = df['allowance']
feature_size = df['feature_size']
depth = df['depth']
cut = df['cut']
roughness = df['roughness']
cut_setting = pd.concat([machine, machine_type, df['feed'], df['rotate'], df['cutting_distance']], axis=1)
node_data = {
    'cut_count': cut_count,
    'feature': feature,
    'tool': tool,
    'tool_wear': tool_wear,
    'tool_breakage': tool_breakage,
    'allowance': allowance,
    'feature_size': feature_size,
    'depth': depth,
    'cut': cut,
    'roughness': roughness,
    'cut_setting': cut_setting,
}

# create graph
num_nodes = len(node_lst)
feature_dim = 18
graph_dataset = []
for t in tool_lst:
    graph_sequence = []
    timestamp = 0
    data = df[df['tool_id'] == t]
    for index, row in data.iterrows():
        timestamp += 1
        node_name_lst = node_lst.copy()
        x = torch.zeros((num_nodes, feature_dim), dtype=torch.float)
        for i in range(num_nodes):
            node_name = node_lst[i]
            d = node_data[node_name].loc[index]

            if node_name == "tool_breakage" and d.eq(0).all():
                d = np.nan

            if isinstance(d, pd.Series):
                x[i][:len(d)] = torch.tensor(d, dtype=torch.float)
            elif pd.isna(d):
                node_name_lst.remove(node_name)
            else:
                x[i][0] = torch.tensor(d, dtype=torch.float)
        if torch.isnan(x).any().item():
            raise Exception('have nan')

        edge_index = [[node_lst.index(s), node_lst.index(t)] for s, t in edge_lst if
                      s in node_name_lst and t in node_name_lst]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index)

        sub_graph = Data(
            x=x,
            edge_index=edge_index,
            num_nodes=len(node_name_lst),
            node_lst=node_name_lst,
            node_mask=[node_lst.index(n) for n in node_name_lst],
            tool_id=t,
            timestamp=timestamp,
        )
        graph_sequence.append(sub_graph)
    graph_dataset.append(graph_sequence)
print("Dataset Count=", len(graph_dataset))
torch.save(graph_dataset, '../data/dataset.pt')
