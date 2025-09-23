import unittest

import grid2op
from grid2op.Agent import DoNothingAgent, RandomAgent
from grid2op.Observation import BaseObservation

from nri.create_dataset import sample_trajectory, AgentFailsEarly, generate_dataset, \
    generator_features_from_observation, FEATURES_PER_GENERATOR, load_features_from_observation, FEATURES_PER_LOAD, \
    line_features_from_observation, FEATURES_PER_LINE


class TestCreateDataset(unittest.TestCase):
    def setUp(self):
        # https://grid2op.readthedocs.io/en/latest/available_envs.html for a list of available envs
        self.env = grid2op.make("l2rpn_case14_sandbox")
        # https://grid2op.readthedocs.io/en/master/user/agent.html for a list of available agents
        self.do_nothing_agent = DoNothingAgent(self.env.action_space)
        self.random_agent = RandomAgent(self.env.action_space)

    def test_sample_trajectory(self):
        length = 300
        max_retries = 10
        trajectory = sample_trajectory(length, self.do_nothing_agent, self.env, max_retries)
        self.assertEqual(len(trajectory), length)
        self.assertIsInstance(trajectory[0], BaseObservation)

    def test_sample_trajectories_agent_fails(self):
        length = 3000 # to long to reach
        max_retries = 0 # no retries
        self.assertRaises(AgentFailsEarly, sample_trajectory, length, self.random_agent, self.env, max_retries)

    def test_generator_features_from_obs(self):
        observation = self.env.reset()
        generator_features = generator_features_from_observation(observation)
        self.assertEqual(generator_features.shape, (self.env.n_gen, FEATURES_PER_GENERATOR))

    def test_load_features_from_obs(self):
        observation = self.env.reset()
        load_features = load_features_from_observation(observation)
        self.assertEqual(load_features.shape, (self.env.n_load, FEATURES_PER_LOAD))

    def test_line_features_from_obs(self):
        observation = self.env.reset()
        line_features = line_features_from_observation(observation)
        self.assertEqual(line_features.shape, (self.env.n_line, FEATURES_PER_LINE))

    def test_generate_dataset(self):
        num_trajectories = 10
        len_per_trajectory = 10
        generators_data, loads_data, lines_data = generate_dataset(num_trajectories, len_per_trajectory, self.do_nothing_agent, self.env)
        self.assertEqual(generators_data.shape, (num_trajectories, len_per_trajectory, self.env.n_gen, FEATURES_PER_GENERATOR))
        self.assertEqual(loads_data.shape, (num_trajectories, len_per_trajectory, self.env.n_load, FEATURES_PER_LOAD))
        self.assertEqual(lines_data.shape, (num_trajectories, len_per_trajectory, self.env.n_line, FEATURES_PER_LINE))

