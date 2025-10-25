import gymnasium as gym

gym.register(
    id="CuttingEnv-v0",
    entry_point="gnn_environment:CuttingEnv",
)
exp_name: str = 'RMORL_lightgbm'
"""the name of this experiment"""
seed: int = 1
"""seed of the experiment"""
torch_deterministic: bool = True
"""if toggled, `torch.backends.cudnn.deterministic=False`"""
cuda: bool = True
"""if toggled, cuda will be enabled by default"""

# Algorithm specific arguments
env_id: str = "CuttingEnv-v0"
"""the environment id of the task"""
total_timesteps: int = 1000000
"""total timesteps of the experiments"""
num_envs: int = 1
"""the number of parallel game environments"""
buffer_size: int = int(1e7)
"""the replay memory buffer size"""
gamma: float = 0.99
"""the discount factor gamma"""
tau: float = 0.005
"""target smoothing coefficient (default: 0.005)"""
batch_size: int = 128
"""the batch size of sample from the reply memory"""
learning_starts: int = 5e3
# learning_starts: int = 128  # for debug
"""timestep to start learning"""
policy_lr: float = 1e-4
"""the learning rate of the policy network optimizer"""
q_lr: float = 1e-4
"""the learning rate of the Q network network optimizer"""
policy_frequency: int = 1
"""the frequency of training policy (delayed)"""
target_network_frequency: int = 10
"""the frequency of updates for the target nerworks"""
alpha: float = 0.2
"""Entropy regularization coefficient."""
autotune: bool = False
"""automatic tuning of the entropy coefficient"""
target_entropy_scale: float = 0.5
"""coefficient for scaling the autotune entropy target"""

# RMORL核心参数（论文Section IV）
adversarial_prob: float = 0.1  # α：对抗动作混合概率
# adversarial_prob: float = 2.0  # for debug
adversarial_lr: float = 1e-4
dual_lr: float = 1e-4
robustness_epsilon: float = 1.0  # ε：策略变化约束上限
beta: float = 0.2  # L1/L2损失平衡系数
zeta: float = 0.05  # 随机观测噪声系数
zeta_bar: float = 0.1  # 对抗观测扰动系数

num_objectives: int = 3

obs_dim = {
    "material": 2,
    "machine": 4,
    "machine_type": 2,
    "cutting_type": 2,
    "feature_type": 11,
    "tool_type": 7,
    "tool_radius": 1,
    "tool_degree": 1,
    "tool_length": 1,
    "tool_extension": 2,
    "tool_supplier": 5,
    "cutting_distance": 1,
    "rotate": 1,
    "feed": 1,
    "tool_wear": 1,
    "tool_breakage": 3,
    "allowance": 1,
    "feature_size": 1,
    "lower_tolerance": 1,
    "upper_tolerance": 1,
    "roughness_requirement": 1,
}
tool_wear_idx = 44

model_path = "./checkpoints/epoch=1287-val_loss=7.2904.ckpt"

# 超体积评估参数
NUM_EVAL_EPISODES = 10  # 评估用的episode数量
REF_POINT = [-5.0, -3.0, -65.0]  # 参考点
