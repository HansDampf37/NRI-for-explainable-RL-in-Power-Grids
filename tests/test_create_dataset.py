import unittest

import grid2op
from grid2op.Agent import DoNothingAgent, RandomAgent
from grid2op.Observation import BaseObservation

from nri.create_dataset import sample_trajectory, AgentFailsEarly, observation_to_np, FEATURES_PER_NODE, generate_dataset


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
        max_retries = 0
        self.assertRaises(AgentFailsEarly, sample_trajectory, length, self.random_agent, self.env, max_retries)

    def test_observation_to_numpy(self):
        observation = self.env.reset()
        observation_np = observation_to_np(observation)
        num_nodes = self.env.current_obs.n_gen + self.env.current_obs.n_load + self.env.current_obs.n_sub
        self.assertEqual(observation_np.shape, (num_nodes, FEATURES_PER_NODE))

    def test_generate_dataset(self):
        num_trajectories = 10
        len_per_trajectory = 10
        dataset = generate_dataset(num_trajectories, len_per_trajectory, self.random_agent, self.env)
        self.assertEqual(dataset.shape, )

