import unittest

import grid2op
from grid2op.Agent import DoNothingAgent, RandomAgent
from grid2op.Observation import BaseObservation

from src.common.graph_structured_observation_space import NODES, EDGE_INDEX, EDGE_MASK, BusConnectivityGraphObsSpace
from src.nri.create_dataset import sample_trajectory, AgentFailsEarly, generate_dataset


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
        length = 3000  # to long to reach
        max_retries = 0  # no retries
        self.assertRaises(AgentFailsEarly, sample_trajectory, length, self.random_agent, self.env, max_retries)

    def test_generate_dataset(self):
        num_traj = 10
        traj_len = 10
        obs_space = BusConnectivityGraphObsSpace(self.env.observation_space)
        data = generate_dataset(num_traj, traj_len, self.do_nothing_agent, self.env, obs_space)

        # edge mask and edge index are not added to the dataset
        spaces = {k for k in obs_space.spaces.keys() if k not in [EDGE_MASK, EDGE_INDEX]}
        self.assertSetEqual(set(data.keys()), spaces)
        self.assertEqual(data[NODES].shape, (num_traj, traj_len, obs_space.num_nodes, obs_space.x_dim))
