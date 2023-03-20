from typing import Optional

import gym
import numpy as np
from gym.utils import seeding


class DoorsEnv(gym.Env):
    def __init__(self, num_doors: int, num_stages: int):
        self.num_doors = num_doors
        self.num_stages = num_stages

        self.action_space = gym.spaces.Discrete(self.num_doors)
        self.observation_space = gym.spaces.Box(0, np.inf, dtype=np.float64)

        self.task0_path = None
        self.task1_path = None
        self.task2_path = None

        self.current_path = None
        self.current_stage = None

        self.state = None
        self.done = False
        self.seed()

    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_state(self) -> np.ndarray:
        self.state = np.zeros((3,))
        self.state[0] = 0.1 * self.task1_path[self.current_stage]
        self.state[1] = 0.3 * self.task2_path[self.current_stage]
        self.state[2] = 0.7 * self.task2_path[self.current_stage]
        return self.state

    def step(self, action: int):
        if self.done:
            raise RuntimeError
        self.current_path.append(action)
        self.current_stage += 1
        reward = 0.0
        if self.current_stage == self.num_stages:
            self.done = True
            current_path = np.array(self.current_path)
            if (self.task0_path == current_path).all():
                reward += 0.1
            if (self.task1_path == current_path).all():
                reward += 1.0
            if (self.task2_path == current_path).all():
                reward += 10.0
            self.state = np.zeros((3,))
        else:
            self.state = self.get_state()
        info = {}
        return self.state, reward, self.done, info

    def reset(self):
        self.current_path = []
        self.current_stage = 0
        self.task0_path = np.zeros((self.num_stages,))
        self.task1_path = self.np_random.integers(
            0, self.num_doors, size=(self.num_stages,)
        )
        self.task2_path = self.np_random.integers(
            0, self.num_doors, size=(self.num_stages,)
        )
        self.state = self.get_state()
        self.done = False
        return self.state
