import unittest

import gym
import numpy as np


class TestDoorsEnv(unittest.TestCase):
    def test_task_0(self):
        env = gym.make("gym_dimensional_doors:doors-v0")
        state = env.reset()
        self.assertEqual(env.observation_space.shape, state.shape)
        self.assertEqual(np.float32, state.dtype)
        done = False
        path_to_take = np.zeros(env.num_stages, dtype=int)
        expected_terminal_reward = 0
        if (path_to_take == env.task0_path).all():
            expected_terminal_reward += 0.1
        if (path_to_take == env.task1_path).all():
            expected_terminal_reward += 1.0
        if (path_to_take == env.task2_path).all():
            expected_terminal_reward += 10.0
        step_count = 0
        while not done:
            state, reward, done, info = env.step(path_to_take[step_count])
            self.assertEqual(env.observation_space.shape, state.shape)
            self.assertEqual(np.float32, state.dtype)
            self.assertEqual(bool, type(done))
            self.assertEqual(dict, type(info))
            if done:
                self.assertEqual(expected_terminal_reward, reward)
            else:
                self.assertEqual(0, reward)
            step_count += 1

    def test_task_1(self):
        env = gym.make("gym_dimensional_doors:doors-v0")
        state = env.reset()
        self.assertEqual(env.observation_space.shape, state.shape)
        self.assertEqual(np.float32, state.dtype)
        done = False
        path_to_take = np.array(env.task1_path)
        expected_terminal_reward = 0
        if (path_to_take == env.task0_path).all():
            expected_terminal_reward += 0.1
        if (path_to_take == env.task1_path).all():
            expected_terminal_reward += 1.0
        if (path_to_take == env.task2_path).all():
            expected_terminal_reward += 10.0
        step_count = 0
        while not done:
            state, reward, done, info = env.step(path_to_take[step_count])
            self.assertEqual(env.observation_space.shape, state.shape)
            self.assertEqual(np.float32, state.dtype)
            self.assertEqual(bool, type(done))
            self.assertEqual(dict, type(info))
            if done:
                self.assertEqual(expected_terminal_reward, reward)
            else:
                self.assertEqual(0, reward)
            step_count += 1

    def test_task_2(self):
        env = gym.make("gym_dimensional_doors:doors-v0")
        state = env.reset()
        self.assertEqual(env.observation_space.shape, state.shape)
        self.assertEqual(np.float32, state.dtype)
        done = False
        path_to_take = np.array(env.task2_path)
        expected_terminal_reward = 0
        if (path_to_take == env.task0_path).all():
            expected_terminal_reward += 0.1
        if (path_to_take == env.task1_path).all():
            expected_terminal_reward += 1.0
        if (path_to_take == env.task2_path).all():
            expected_terminal_reward += 10.0
        step_count = 0
        while not done:
            state, reward, done, info = env.step(path_to_take[step_count])
            self.assertEqual(env.observation_space.shape, state.shape)
            self.assertEqual(np.float32, state.dtype)
            self.assertEqual(bool, type(done))
            self.assertEqual(dict, type(info))
            if done:
                self.assertEqual(expected_terminal_reward, reward)
            else:
                self.assertEqual(0, reward)
            step_count += 1


if __name__ == "__main__":
    unittest.main()
