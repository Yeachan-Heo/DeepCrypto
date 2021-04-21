import os
import gym
import numpy as np
from gym import spaces
from .tree import Tree


def evaluate_for_one_ticker(factor, label):
    mask = np.where(~np.isnan(factor) & ~np.isnan(label))
    factor, label = factor[mask], label[mask]
    ic = np.corrcoef(factor, label)
    return ic[0][-1]


class GenStrategyEnv(gym.Env):
    def __init__(self, cfg):
        self._cfg = cfg

        self.engine = cfg["engine"]

        self._operation_nodes = cfg["operation_nodes"]
        self._data_nodes = cfg["data_nodes"]

        self.action_space = spaces.Discrete(
            len(self._operation_nodes) + len(self._data_nodes))

        self._n_operations = len(self._operation_nodes)
        self._n_data_nodes = len(self._data_nodes)

        self._nodes = self._operation_nodes + self._data_nodes

        if not os.path.exists(self._cfg["factor_save_path"]):
            os.makedirs(self._cfg["factor_save_path"])

    def evaluate(self):
        raise NotImplementedError

    def reset(self):
        self.tree = Tree()
        action_mask = [1] * 4 + [0] * (self._n_operations + self._n_data_nodes - 4)
        return {"action_mask": action_mask}

    def step(self, action):
        action_node_fn = self._nodes[action]

        self.tree.insert(action_node_fn)

        done = self.tree.iscompleted

        reward = 0

        if done: reward = self.evaluate()

        action_mask = [1] * (self._n_operations + self._n_data_nodes)

        if self.tree.current.depth - 1 >= self._cfg["max_depth"]:
            action_mask = [0] * (self._n_operations) + [1] * (self._n_data_nodes)

        return {"action_mask": np.array(action_mask), "obs": []}, reward, done, {}