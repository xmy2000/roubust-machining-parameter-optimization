import numpy as np
import gymnasium as gym
from gymnasium import spaces

from gnn_predictor import GNNPredictor
from parameters import model_path


class CuttingEnv(gym.Env):
    def __init__(self):
        # 环境变量
        self.cut_count = -1
        self.total_time = -1
        self.material = -1
        self.machine = -1
        self.machine_type = -1
        self.cutting_type = -1
        self.feature_type = -1
        self.tool_type = -1
        self.tool_radius = -1
        self.tool_degree = -1
        self.tool_length = -1
        self.tool_extension = -1
        self.tool_supplier = -1
        self.cutting_distance = -1
        self.rotate = -1
        self.feed = -1
        self.tool_wear = -1
        self.tool_breakage = -1
        self.allowance = -1
        self.feature_size = -1
        self.cut = -1
        self.cutting_time = -1
        self.change_time = 120
        self.roughness = -1
        self.roughness_requirement = -1
        self.mrr = -1
        self.lower_tolerance = -1
        self.upper_tolerance = -1

        self.observation_space = spaces.Dict(
            {
                "material": spaces.Discrete(2),
                "machine": spaces.Discrete(4),
                "machine_type": spaces.Discrete(2),
                "cutting_type": spaces.Discrete(2),
                "feature_type": spaces.Discrete(11),
                "tool_type": spaces.Discrete(7),
                "tool_radius": spaces.Box(low=0.1, high=0.8, shape=(1,), dtype=np.float32),
                "tool_degree": spaces.Box(low=35, high=80, shape=(1,), dtype=np.float32),
                "tool_length": spaces.Box(low=25, high=215, shape=(1,), dtype=np.float32),
                "tool_extension": spaces.Discrete(2),
                "tool_supplier": spaces.Discrete(5),
                "cutting_distance": spaces.Box(low=0.5, high=200, shape=(1,), dtype=np.float32),
                "rotate": spaces.Box(low=20, high=80, shape=(1,), dtype=np.float32),
                "feed": spaces.Box(low=0.02, high=0.2, shape=(1,), dtype=np.float32),
                "tool_wear": spaces.Box(low=0.0, high=2000, shape=(1,), dtype=np.float32),
                "tool_breakage": spaces.Discrete(3),
                "allowance": spaces.Box(low=0, high=6.5, shape=(1,), dtype=np.float32),
                "feature_size": spaces.Box(low=50, high=360, shape=(1,), dtype=np.float32),
                "lower_tolerance": spaces.Box(low=-0.2, high=-0.05, shape=(1,), dtype=np.float32),
                "upper_tolerance": spaces.Box(low=0.05, high=0.2, shape=(1,), dtype=np.float32),
                "roughness_requirement": spaces.Box(low=0.8, high=6.4, shape=(1,), dtype=np.float32),
            }
        )

        action_1 = spaces.Discrete(2)
        action_2 = spaces.Discrete(13)
        action_3 = spaces.Discrete(13)
        action_4 = spaces.Box(low=0.0, high=2.5, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Tuple((action_1, action_2, action_3, action_4))

        self.predictor = GNNPredictor(model_path)

    def _get_predict_data(self, rotate, feed, depth):
        data = {
            "cut_count": self.cut_count,
            "feature_type": self.feature_type,
            "cutting_type": self.cutting_type,
            "material": self.material,
            "tool_type": self.tool_type,
            "tool_radius": self.tool_radius[0],
            "tool_degree": self.tool_degree[0],
            "tool_length": self.tool_length[0],
            "tool_extension": self.tool_extension,
            "tool_supplier": self.tool_supplier,
            "tool_wear": self.tool_wear[0],
            "tool_breakage": self.tool_breakage,
            "allowance": self.allowance[0],
            "feature_size": self.feature_size[0],
            "depth": depth,
            "cut": 0.0253,
            "roughness": 1.725,
            "machine": self.machine,
            "machine_type": self.machine_type,
            "feed": feed,
            "rotate": rotate,
            "cutting_distance": self.cutting_distance[0],
        }
        return data

    def _get_obs(self):
        obs_dict = {
            "material": self.material,
            "machine": self.machine,
            "machine_type": self.machine_type,
            "cutting_type": self.cutting_type,
            "feature_type": self.feature_type,
            "tool_type": self.tool_type,
            "tool_radius": self.tool_radius,
            "tool_degree": self.tool_degree,
            "tool_length": self.tool_length,
            "tool_extension": self.tool_extension,
            "tool_supplier": self.tool_supplier,
            "cutting_distance": self.cutting_distance,
            "rotate": self.rotate,
            "feed": self.feed,
            "tool_wear": self.tool_wear,
            "tool_breakage": self.tool_breakage,
            "allowance": self.allowance,
            "feature_size": self.feature_size,
            "lower_tolerance": self.lower_tolerance,
            "upper_tolerance": self.upper_tolerance,
            "roughness_requirement": self.roughness_requirement,
        }
        return obs_dict

    def obs_dict_to_array(self, obs_dict):
        array_parts = []

        discrete_vars = {
            "material": 2,
            "machine": 4,
            "machine_type": 2,
            "cutting_type": 2,
            "feature_type": 11,
            "tool_type": 7,
            "tool_extension": 2,
            "tool_supplier": 5,
            "tool_breakage": 3
        }

        for var, size in discrete_vars.items():
            one_hot = np.zeros(size, dtype=np.float32)
            one_hot[obs_dict[var]] = 1.0
            array_parts.append(one_hot)

        continuous_vars = [
            "tool_radius",
            "tool_degree",
            "tool_length",
            "cutting_distance",
            "rotate",
            "feed",
            "tool_wear",
            "allowance",
            "feature_size",
            "lower_tolerance",
            "upper_tolerance",
            "roughness_requirement"
        ]

        for var in continuous_vars:
            array_parts.append(np.array(obs_dict[var], dtype=np.float32).flatten())

        return np.concatenate(array_parts)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.cut_count = 0
        self.total_time = 0
        self.material = self.observation_space["material"].sample()
        self.machine = self.observation_space["machine"].sample()
        self.machine_type = self.observation_space["machine_type"].sample()
        self.cutting_type = self.observation_space["cutting_type"].sample()
        self.feature_type = np.random.choice([1, 5, 6, 9])
        self.tool_type = np.random.choice([0, 2, 4, 5, 6])
        self.tool_radius = np.array([np.random.choice([0.4, 0.8, 0.65, 0.1, 0.2])], dtype=np.float32)
        self.tool_degree = np.array([np.random.choice([35, 55, 80])], dtype=np.float32)
        self.tool_length = np.array([np.random.choice([110., 25., 42., 55., 30., 50., 215., 35.])], dtype=np.float32)
        self.tool_extension = self.observation_space["tool_extension"].sample()
        self.tool_supplier = self.observation_space["tool_supplier"].sample()
        self.cutting_distance = self.observation_space["cutting_distance"].sample()
        self.rotate = np.array([np.random.choice([45, 20, 25, 40, 50, 70, 65, 35, 29, 24, 30, 38, 60, 80, 75])],
                               dtype=np.float32)
        self.feed = np.array([np.random.choice([0.15, 0.12, 0.1, 0.13, 0.05, 0.03, 0.08, 0.02, 0.18])],
                             dtype=np.float32)
        self.tool_wear = np.array([0.0], dtype=np.float32)
        self.tool_breakage = 1
        self.feature_size = self.observation_space["feature_size"].sample()
        self.cut = np.array([0.0], dtype=np.float32)
        self.cutting_time = 0
        if self.machine_type == 0:
            self.allowance = np.array([np.random.uniform(1.0, 6.5)], dtype=np.float32)
            self.upper_tolerance = np.array([1.0], dtype=np.float32)
            self.lower_tolerance = np.array([-1.0], dtype=np.float32)
            self.roughness_requirement = np.array([np.random.choice([3.2, 6.4])], dtype=np.float32)
        else:
            self.allowance = np.array([np.random.uniform(0.0, 0.5)], dtype=np.float32)
            self.lower_tolerance = self.observation_space['lower_tolerance'].sample()
            self.upper_tolerance = self.observation_space['upper_tolerance'].sample()
            self.roughness_requirement = np.array([np.random.choice([0.8, 1.6, 3.2])], dtype=np.float32)

        observation = self._get_obs()
        obs_array = self.obs_dict_to_array(observation)
        info = {"obs_array": obs_array}

        return observation, info

    def step(self, action):
        rotate_ratio = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
        feed_ratio = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
        change_tool = int(action[0])  # 0-不换刀; 1-换刀
        rotate = rotate_ratio[int(action[1])] * self.rotate[0]
        feed = feed_ratio[int(action[2])] * self.feed[0]
        depth = np.clip(action[3].item(), self.action_space[3].low, self.action_space[3].high).item()

        if change_tool == 1 or self.cut_count == 0:
            self.tool_wear[0] = 0.0
            self.tool_breakage = 1

        predict_data = self._get_predict_data(rotate, feed, depth)
        predict_result = self.predictor.predict_with_uncertainty(predict_data, num_samples=10)
        self.tool_wear[0] = np.clip(predict_result["tool_wear"]["mean"], 0.0, 2000.0).item()
        tool_wear_aleatory_std = predict_result["tool_wear"]["aleatory_std"]
        tool_wear_epistemic_std = predict_result["tool_wear"]["epistemic_std"]
        tool_breakage_prob = predict_result["tool_breakage"]["mean"]
        self.tool_breakage = tool_breakage_prob.index(max(tool_breakage_prob))
        self.roughness = np.clip(predict_result["roughness"]["mean"], 0.8, 6.4).item()
        if self.machine_type == 0:
            self.cut = depth
        else:
            self.cut = np.clip(predict_result["cut"]["mean"], 0.0, 2.5).item()

        self.cut_count += 1
        self.cutting_time = self.cutting_distance / feed
        self.total_time += (self.cutting_time + change_tool * self.change_time)
        self.allowance -= self.cut
        if self.feature_type == 1:
            self.feature_size += 2 * self.cut
        elif self.feature_type == 6:
            self.feature_size -= 2 * self.cut
        elif self.feature_type == 5:
            self.feature_size += self.cut
        elif self.feature_type == 9:
            self.feature_size -= self.cut
        self.mrr = rotate * feed * depth
        observation = self._get_obs()

        terminated = False
        truncated = False
        if self.roughness <= self.roughness_requirement:
            if self.feature_type in [1, 5] and self.lower_tolerance <= -self.allowance <= self.upper_tolerance:
                terminated = True
            elif self.feature_type in [6, 9] and self.lower_tolerance <= self.allowance <= self.upper_tolerance:
                terminated = True
        if self.cut_count > 5:
            truncated = True
        if self.feature_type in [1, 5] and -self.allowance > self.upper_tolerance:
            truncated = True
        if self.feature_type in [6, 9] and self.allowance < self.lower_tolerance:
            truncated = True

        reward_time = self.cutting_distance / self.feed - self.cutting_time

        reward_breakage = 1 * tool_breakage_prob[1] - 1 * tool_breakage_prob[2] - 1 * tool_breakage_prob[0]

        if self.machine_type == 0:
            reward_wear = self._cal_continue_reward(self.tool_wear[0], 600.0)
        elif self.machine_type == 1:
            reward_wear = self._cal_continue_reward(self.tool_wear[0], 300.0)

        reward_roughness = self._cal_continue_reward(self.roughness, self.roughness_requirement[0])

        reward_mrr = self.mrr - self.rotate * self.feed * depth

        if self.feature_type in [1, 5]:
            cur = -self.allowance
        elif self.feature_type in [6, 9]:
            cur = self.allowance
        reward_size = self._cal_delta_reward(cur, self.lower_tolerance, self.upper_tolerance)

        reward_efficiency = (reward_time + reward_mrr).item()
        reward_tool = (reward_wear + reward_breakage).item()
        reward_quality = (reward_roughness + reward_size).item()
        reward_vector = np.array([reward_efficiency, reward_tool, reward_quality])

        info = {
            "cut_count": self.cut_count,
            "change_tool": change_tool,
            "rotate": rotate,
            "feed": feed,
            "cutting_depth": depth,
            "reward_vector": reward_vector,
            "reward": {
                "reward_time": reward_time,
                "reward_wear": reward_wear,
                "reward_breakage": reward_breakage,
                "reward_roughness": reward_roughness,
                "reward_mrr": reward_mrr,
                "reward_size": reward_size,
            },
            "cut": self.cut,
            "roughness": self.roughness,
            "tool_wear": self.tool_wear,
            "tool_wear_aleatory_std": tool_wear_aleatory_std,
            "tool_wear_epistemic_std": tool_wear_epistemic_std,
            "tool_breakage": self.tool_breakage,
            "tool_breakage_prob": tool_breakage_prob,
        }
        info.update(observation)
        obs_array = self.obs_dict_to_array(observation)
        info["obs_array"] = obs_array
        return observation, np.sum(reward_vector).item(), terminated, truncated, info

    def _cal_continue_reward(self, actual, target):
        return 1 - actual / target

    def _cal_delta_reward(self, delta, delta_min, delta_max):
        if delta < delta_min:
            return delta - delta_min
        elif delta_min <= delta < 0:
            return 1 - delta / delta_min
        elif 0 <= delta <= delta_max:
            return 1 - delta / delta_max
        elif delta >= delta_max:
            return delta_max - delta

    def predict(self, action):
        rotate_ratio = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
        feed_ratio = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
        change_tool = int(action[0])
        rotate = rotate_ratio[int(action[1])] * self.rotate[0]
        feed = feed_ratio[int(action[2])] * self.feed[0]
        depth = np.clip(action[3].item(), self.action_space[3].low, self.action_space[3].high).item()

        if change_tool == 1 or self.cut_count == 0:
            tool_wear = 0.0
            tool_breakage = 1
        else:
            tool_wear = self.tool_wear[0]
            tool_breakage = self.tool_breakage

        predict_data = {
            "cut_count": self.cut_count,
            "feature_type": self.feature_type,
            "cutting_type": self.cutting_type,
            "material": self.material,
            "tool_type": self.tool_type,
            "tool_radius": self.tool_radius[0],
            "tool_degree": self.tool_degree[0],
            "tool_length": self.tool_length[0],
            "tool_extension": self.tool_extension,
            "tool_supplier": self.tool_supplier,
            "tool_wear": tool_wear,
            "tool_breakage": tool_breakage,
            "allowance": self.allowance[0],
            "feature_size": self.feature_size[0],
            "depth": depth,
            "cut": 0.0253,
            "roughness": 1.725,
            "machine": self.machine,
            "machine_type": self.machine_type,
            "feed": feed,
            "rotate": rotate,
            "cutting_distance": self.cutting_distance[0],
        }
        predict_result = self.predictor.predict_with_uncertainty(predict_data, num_samples=10)
        return predict_result
