import torch

NODES = ['cut_count', 'feature', 'tool', 'tool_wear', 'tool_breakage', 'allowance', 'feature_size', 'depth', 'cut',
         'roughness', 'cut_setting']
NUM_NODES = len(NODES)

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

MAX_NODE_DIM = 18

# DEVICE = torch.device("cuda")
DEVICE = torch.device("cpu")

MAX_LOGSTD = 10
