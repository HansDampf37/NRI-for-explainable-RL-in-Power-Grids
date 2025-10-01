"""
This script contains code to generate a dataset containing trajectories of the environment being operated by some agent.
This dataset can further be used for NRI. The contained dataset contains trajectories for the following entities:
- generators
- loads
- powerlines
"""

import logging
from datetime import datetime
from typing import List, Tuple, Dict

import grid2op
import hydra
import numpy as np
from grid2op.Agent import BaseAgent, RandomAgent, DoNothingAgent, RecoPowerlineAgent, TopologyGreedy
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from baselines.train_stable_dqn_baseline import build_agent
from common.graph_structured_observation_space import GraphObservationSpace

logger = logging.getLogger(__name__)


class AgentFailsEarly(Exception):
    """
    This exception is raised when an agent fails repeatedly to create a trajectory of a specified length.
    """

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
    obs, reward = env.reset(), 0
    trajectory = [obs]
    retries = 0

    while len(trajectory) < length:
        action = agent.act(obs, reward)
        obs, reward, done, info = env.step(action)
        trajectory.append(obs)
        if done and retries < max_retries:
            logger.warning(
                f"Trajectory ended early after {len(trajectory)} timesteps. Restarting environment to find longer trajectory...")
            obs, reward = env.reset(), 0
            trajectory = [obs]
            retries += 1
        elif done:
            raise AgentFailsEarly(agent=agent, env=env, length=length, max_tries=retries)

    return trajectory


def generate_dataset(num_sims: int, length: int, agent: BaseAgent, env: Environment, observation_converter: GraphObservationSpace) -> Dict:
    """
    Creates a dataset containing multiple trajectories of the environment being operated by some agent.
    :param num_sims: the amount of trajectories to generate
    :param length: the length of each trajectory
    :param agent: the agent to operate the environment
    :param env: the grid2op environment to operate
    :param observation_converter: the graph observation space to use
    :return: trajectory data for the observed grid entities
    """
    trajectories = {}

    for _ in tqdm(range(num_sims), f"Creating {num_sims} trajectories"):
        trajectory: List[BaseObservation] = sample_trajectory(length=length, agent=agent, env=env)
        converted_trajectory: List[Dict[str, np.ndarray]] = [observation_converter.to_gym(obs) for obs in trajectory]
        for grid_entity in observation_converter.spaces_to_keep:
            if grid_entity not in trajectories:
                trajectories[grid_entity] = []

            trajectories[grid_entity].append(np.array([obs[grid_entity] for obs in converted_trajectory]))

    return {grid_entity: np.stack(trajectories[grid_entity]) for grid_entity in observation_converter.spaces_to_keep}


@hydra.main(config_path="../hydra_configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(cfg)
    # create env + observation space
    env = grid2op.make(cfg.env.env_name)
    observation_converter: GraphObservationSpace = instantiate(
        cfg.nri.obs_space,
        grid2op_observation_space=env.observation_space
    )

    # create agent
    if cfg.nri.agent == 'random':
        agent = RandomAgent(env.action_space)
    elif cfg.nri.agent == 'do_nothing':
        agent = DoNothingAgent(env.action_space)
    elif cfg.nri.agent == 'reconnect':
        agent = RecoPowerlineAgent(env.action_space)
    elif cfg.nri.agent == 'topology_greedy':
        agent = TopologyGreedy(env.action_space)
        logger.warning("You have configured the topology greedy agent that will simulate every topology action. "
                       "This is only feasible for small environments.")
    elif cfg.nri.agent == 'baseline':
        agent = build_agent(cfg, load_weights_from=cfg.nri.model_path)
    else:
        raise NotImplementedError(f"Unknown agent '{cfg.nri.agent}'")

    total = cfg.nri.num_train_trajectories + cfg.nri.num_val_trajectories + cfg.nri.num_test_trajectories
    logger.info(f"Running {cfg.nri.agent} on {cfg.env.env_name} to produce {total} trajectories...")
    data = generate_dataset(total, cfg.nri.trajectory_length, agent, env, observation_converter)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")
    for grid_entity in data:
        train_data = data[grid_entity][:cfg.nri.num_train_trajectories]
        test_data = data[grid_entity][cfg.nri.num_train_trajectories:cfg.nri.num_train_trajectories + cfg.nri.num_test_trajectories]
        val_data = data[grid_entity][-cfg.nri.num_val_trajectories:]
        np.save(f'data/nri_dataset/train_{grid_entity}_{cfg.env.env_name}_{timestamp}_.npy', train_data)
        np.save(f'data/nri_dataset/test_{grid_entity}_{cfg.env.env_name}_{timestamp}_.npy', test_data)
        np.save(f'data/nri_dataset/val_{grid_entity}_{cfg.env.env_name}_{timestamp}_.npy', val_data)


if __name__ == '__main__':
    main()
