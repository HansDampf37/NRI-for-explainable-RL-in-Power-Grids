"""
This script contains code to generate a dataset containing trajectories of the environment being operated by some agent.
"""
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import grid2op
import hydra
import numpy as np
from grid2op.Agent import BaseAgent, RandomAgent, DoNothingAgent, RecoPowerlineAgent, TopologyGreedy
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation
from hydra.utils import instantiate
from lightsim2grid import LightSimBackend
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from common.constants import logger, set_experiment_name
from common.graph_structured_observation_space import EDGE_INDEX, EDGE_MASK, GraphObservationSpace
from common.rewards import MazeRLReward


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


def sample_trajectory(length: int, agent: BaseAgent, env: Environment, max_retries: int = 100) -> List[BaseObservation]:
    """
    Samples a trajectory of observations from a grid2op environment operated by an agent
    :param length: the maximum length of the trajectory. If the agent fails to operate the environment until this number is reached, we try again from scratch until either a trajectory with the required length is found or max_tries is reached.
    :param agent: the agent operating the environment
    :param env: the env that is operated on
    :param max_retries: the maximum number of times to retry to find a trajectory of length >= length
    :raise AgentFailedEarly: if the agent fails to operate the environment until the trajectory reaches its target length
    :return: the trajectory as a list of observations
    """
    obs = env.current_obs
    reward = 0
    trajectory = [obs]
    retries = 0

    while len(trajectory) < length:
        action = agent.act(obs, reward)
        obs, reward, done, info = env.step(action)
        trajectory.append(obs)
        if done and retries < max_retries:
            obs = env.reset()
            reward = 0
            trajectory = [obs]
            retries += 1
        elif done:
            raise AgentFailsEarly(agent=agent, env=env, length=length, max_tries=retries)

    logger.info(f"Successfully found trajectory of length {length} in chronic {env.chronics_handler.get_name()}")
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
    logger.info(f"Running {agent.__class__.__name__} on {env.env_name} to produce {num_sims} trajectories of length {length}")
    logger.info(f"Using observation converter: {observation_converter.__class__.__name__}")

    trajectories = {}
    env.reset()

    for _ in tqdm(range(num_sims), f"Creating {num_sims} trajectories"):
        trajectory: List[BaseObservation] = sample_trajectory(length=length, agent=agent, env=env)
        converted_trajectory: List[Dict[str, np.ndarray]] = [observation_converter.to_gym(obs) for obs in trajectory]
        for grid_entity in converted_trajectory[0].keys():
            if grid_entity == EDGE_INDEX or grid_entity == EDGE_MASK:
                continue
            if grid_entity not in trajectories:
                trajectories[grid_entity] = []

            trajectories[grid_entity].append(np.array([obs[grid_entity] for obs in converted_trajectory]))

    return {grid_entity: np.stack(trajectories[grid_entity]) for grid_entity in trajectories.keys()}


@hydra.main(config_path="../hydra_configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    set_experiment_name(cfg.experiment_name)
    from common.constants import NRI_DATASETS_PATH
    # create env + observation space
    env_train = grid2op.make(cfg.nri.dataset_creation.env_name + "_train", backend=LightSimBackend(), reward_class=MazeRLReward)
    env_test = grid2op.make(cfg.nri.dataset_creation.env_name + "_test", backend=LightSimBackend(), reward_class=MazeRLReward)
    env_val = grid2op.make(cfg.nri.dataset_creation.env_name + "_val", backend=LightSimBackend(), reward_class=MazeRLReward)
    observation_converter: GraphObservationSpace = instantiate(
        cfg.nri.dataset_creation.obs_space,
        grid2op_observation_space=env_train.observation_space
    )

    # create agent
    if cfg.nri.dataset_creation.agent == 'random':
        agent = RandomAgent(env_train.action_space)
    elif cfg.nri.dataset_creation.agent == 'do_nothing':
        agent = DoNothingAgent(env_train.action_space)
    elif cfg.nri.dataset_creation.agent == 'reconnect':
        agent = RecoPowerlineAgent(env_train.action_space)
    elif cfg.nri.dataset_creation.agent == 'topology_greedy':
        agent = TopologyGreedy(env_train.action_space)
        logger.warning("You have configured the topology greedy agent that will simulate every topology action. "
                       "This is only feasible for small environments.")
    else:
        raise NotImplementedError(f"Unknown agent '{cfg.nri.dataset_creation.agent}'")

    train_data = generate_dataset(cfg.nri.dataset_creation.num_train_trajectories, cfg.nri.dataset_creation.trajectory_length, agent, env_train, observation_converter)
    test_data = generate_dataset(cfg.nri.dataset_creation.num_test_trajectories, cfg.nri.dataset_creation.trajectory_length, agent, env_test, observation_converter)
    val_data = generate_dataset(cfg.nri.dataset_creation.num_val_trajectories, cfg.nri.dataset_creation.trajectory_length, agent, env_val, observation_converter)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")
    for grid_entity in train_data:
        np.save(Path(NRI_DATASETS_PATH, f'train_{grid_entity}_{cfg.nri.dataset_creation.env_name}_{timestamp}_.npy'), train_data[grid_entity])
        np.save(Path(NRI_DATASETS_PATH, f'test_{grid_entity}_{cfg.nri.dataset_creation.env_name}_{timestamp}_.npy'), test_data[grid_entity])
        np.save(Path(NRI_DATASETS_PATH, f'val_{grid_entity}_{cfg.nri.dataset_creation.env_name}_{timestamp}_.npy'), val_data[grid_entity])


if __name__ == '__main__':
    main()
