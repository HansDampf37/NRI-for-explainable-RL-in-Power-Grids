import logging
import time
from typing import List, Tuple

import numpy as np
from grid2op.Agent import BaseAgent
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation

logger = logging.getLogger(__name__)

FEATURES_PER_NODE = 9  # TODO
FEATURES_PER_EDGE = 9  # TODO


class AgentFailsEarly(Exception):
    def __init__(self, agent: BaseAgent, env: Environment, length: int, max_tries: int):
        self.agent = agent
        self.env = env
        self.length = length
        self.max_tries = max_tries

    def __str__(self):
        return f"Agent {self.agent} failed {self.max_tries} times to produce a trajectory of length {self.length}"


def sample_trajectory(length: int, agent: BaseAgent, env: Environment, max_retries: int = 4) -> List[BaseObservation]:
    """
    Samples a trajectory of observations from a grid2op environment operated by an agent
    :param length: the maximum length of the trajectory. If the agent fails to operate the environment until this number is reached, we try again from scratch until either a trajectory with the required length is found or max_tries is reached.
    :param agent: the agent operating the environment
    :param env: the env that is operated on
    :param max_retries: the maximum number of times to retry to find a trajectory of length >= length
    :raise AgentFailedEarly: if the agent fails to operate the environment until the trajectory reaches its target length
    :return: the trajectory as a list of observations
    """
    trajectory = []
    obs, reward = env.reset(), 0
    trajectory.append(obs)
    retries = 0

    while len(trajectory) < length:
        action = agent.act(obs, reward)
        obs, reward, done, info = env.step(action)
        trajectory.append(obs)
        if done and retries < max_retries:
            logger.warning(
                f"Trajectory ended early after {length} timesteps. Restarting environment to find longer trajectory...")
            obs, reward = env.reset(), 0
            trajectory.append(obs)
            retries += 1
        elif done:
            raise AgentFailsEarly(agent=agent, env=env, length=length, max_tries=retries)

    return trajectory


def observation_to_np(obs: BaseObservation) -> Tuple[np.ndarray:, np.ndarray]:
    """
    Returns a numpy representation of the observation for nodes and edges. The returned tuple contains node features in
    the first entry and edge features in the second entry. Both entries are of shape (n, x) where n is the number of
    nodes/edges and x is the number of features per node/edge.

    :param obs: The observation
    :return: the numpy representation of the observation
    """
    generators = np.array([obs.gen_p, obs.gen_q, obs.gen_v, obs.gen_theta]).transpose()
    loads = np.array([obs.load_p, obs.load_q, obs.load_v, obs.load_theta]).transpose()
    substations = np.array([obs.time_before_cooldown_sub]).transpose()

    empty_gen_features = np.zeros((generators[0].shape[0],))
    empty_load_features = np.zeros((loads[0].shape[0],))
    empty_sub_features = np.zeros((substations[0].shape[0],))
    node_level_features = np.stack([
                                       np.concatenate([generator, empty_load_features, empty_sub_features]) for
                                       generator in generators] + [
                                       np.concatenate([empty_gen_features, load, empty_sub_features]) for load in
                                       loads] + [
                                       np.concatenate([empty_gen_features, empty_load_features, substation]) for
                                       substation in substations
                                   ], 0)
    return node_level_features


def generate_dataset(num_sims: int, length: int, agent: BaseAgent, env: Environment) -> Tuple[np.ndarray, np.ndarray]:
    node_trajectories_all = []
    edge_trajectories_all = []

    for i in range(num_sims):
        t = time.time()
        trajectory: List[BaseObservation] = sample_trajectory(length=length, agent=agent, env=env)
        node_edge_trajectories_np: List[Tuple[np.ndarray]] = [observation_to_np(obs) for obs in trajectory]
        node_trajectories, edge_trajectories = zip(*node_edge_trajectories_np)
        node_trajectories_all.append(node_trajectories)
        edge_trajectories_all.append(edge_trajectories)
        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))

    return np.stack(node_trajectories_all), np.stack(edge_trajectories_all)
