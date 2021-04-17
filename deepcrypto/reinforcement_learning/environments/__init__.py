import gym
import numpy as np
import pandas as pd

from deepcrypto.backtest.lightning import LightningBroker, OHLCVData


class BasicRLEnv(gym.Env):
    def __init__(self, env_config):
        self.broker = LightningBroker(env_config["data"], **env_config["broker_config"])
        self.env_config = env_config

        self.action_space = self.make_action_space(env_config)
        self.observation_space = self.make_observation_space(env_config)

    def __getitem__(self, item):
        return self.env_config[item]

    def make_observation_space(self, *args, **kwargs):
        raise NotImplementedError

    def make_action_space(self, *args, **kwargs):
        raise NotImplementedError

    def get_observation(self, *args, **kwargs):
        raise NotImplementedError

    def get_reward(self, *args, **kwargs):
        raise NotImplementedError

    def check_terminal(self):
        if self.broker.data.idx + self["control_freq"] >= len(self.broker.data.df.index):
            return True
        return False

    def external_information(self, *args, **kwargs):
        raise NotImplementedError

    def apply_action(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        return self.get_observation()

    def step(self, action):
        self.apply_action(action)
        for i in range(self["control_freq"]): self.broker.step()
        return self.get_observation(), self.get_reward(), self.check_terminal(), self.external_information()


class DiscreteRLEnv(BasicRLEnv):
    def __init__(self, env_config):
        super(DiscreteRLEnv, self).__init__(env_config)

    def make_action_space(self, env_config):
        max_action = env_config["action_space_config"]["range"]
        discretization_level = env_config["discretization_level"]
        action_pt1 = [-max_action / discretization_level * i for i in reversed(range(1, discretization_level+1))]
        action_pt2 = [max_action / discretization_level * i for i in range(1, discretization_level + 1)]

        self.action_lst = action_pt1 + [0] + action_pt2

        self.action_space = gym.spaces.Discrete(len(self.action_lst))




