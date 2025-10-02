import unittest

import grid2op
from grid2op.Agent import DoNothingAgent, RandomAgent
from grid2op.Observation import BaseObservation

from common import GraphObservationSpace
from common.graph_structured_observation_space import GraphObservationSpace as Space, LINES_FEATURES, GENERATORS
from nri.create_dataset import sample_trajectory, AgentFailsEarly, generate_dataset

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

    def test_generate_dataset(self):
        num_traj = 10
        traj_len = 10
        obs_space = GraphObservationSpace(self.env.observation_space, [GENERATORS, LINES_FEATURES])
        data = generate_dataset(num_traj, traj_len, self.do_nothing_agent, self.env, obs_space)

        self.assertEqual(list(data.keys()), [GENERATORS, LINES_FEATURES])
        self.assertEqual(data[GENERATORS].shape, (num_traj, traj_len, self.env.n_gen, Space.NUM_FEATURES_PER_GENERATOR))
        self.assertEqual(data[LINES_FEATURES].shape, (num_traj, traj_len, self.env.n_line, Space.NUM_FEATURES_PER_LINE))

