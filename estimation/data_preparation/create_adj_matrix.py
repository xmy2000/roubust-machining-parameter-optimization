import torch
import matplotlib.pyplot as plt


def plot_heatmap(matrix):
    heatmap = matrix.numpy()
    fig, ax = plt.subplots()
    img = ax.imshow(heatmap)
    fig.colorbar(img, ax=ax)
    plt.show()


node_lst = ['cut_count', 'feature', 'tool', 'tool_wear', 'tool_breakage', 'allowance', 'feature_size', 'depth', 'cut',
            'roughness', 'cut_setting']
group1 = ['cut_count', 'feature', 'tool', 'depth', 'allowance', 'feature_size']
group2 = ['tool_wear', 'tool_breakage']
group3 = ['cut', 'roughness']
edge_lst = []
for source in group1:
    for target in group2:
        edge_lst.append((source, target))
for source in group2:
    for target in group3:
        edge_lst.append((source, target))
print(edge_lst)

adj_matrix = torch.zeros((len(node_lst), len(node_lst)), dtype=torch.float)
for i, j in edge_lst:
    row = node_lst.index(i)
    col = node_lst.index(j)
    adj_matrix[row, col] = 1.0
plot_heatmap(adj_matrix)
torch.save(adj_matrix, '../data/adj_matrix.pt')
