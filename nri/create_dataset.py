"""
This script contains code to generate a dataset containing trajectories of the environment being operated by some agent.
This dataset can further be used for NRI. The contained dataset contains trajectories for the following entities:
- generators
- loads
- powerlines
"""

import logging
from typing import List, Tuple, Dict

import grid2op
import hydra
import numpy as np
from grid2op.Agent import BaseAgent, RandomAgent, DoNothingAgent, RecoPowerlineAgent, TopologyGreedy
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation
from grid2op.gym_compat import GymEnv, DiscreteActSpace
from omegaconf import DictConfig
from tqdm import tqdm

from baselines.train_stable_dqn_baseline import build_agent, base_path_models
from common.graph_structured_observation_space import GraphObservationSpace, GENERATOR_FEATURES, LOAD_FEATURES, \
    LINES_FEATURES

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


def generate_dataset(num_sims: int, length: int, agent: BaseAgent, env: Environment) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates a dataset containing multiple trajectories of the environment being operated by some agent.
    :param num_sims: the amount of trajectories to generate
    :param length: the length of each trajectory
    :param agent: the agent to operate the environment
    :param env: the grid2op environment to operate
    :return: trajectory data for generators, loads, and powerlines
    """
    observation_converter = GraphObservationSpace(env.observation_space, [GENERATOR_FEATURES, LOAD_FEATURES, LINES_FEATURES])
    generator_trajectories_all = []
    load_trajectories_all = []
    line_trajectories_all = []

    for _ in tqdm(range(num_sims), f"Creating {num_sims} trajectories"):
        trajectory: List[BaseObservation] = sample_trajectory(length=length, agent=agent, env=env)
        converted_trajectory: List[Dict[str, np.ndarray]] = [observation_converter.to_gym(obs) for obs in trajectory]
        generator_trajectory: np.ndarray = np.array([obs[GENERATOR_FEATURES] for obs in converted_trajectory])
        load_trajectory: np.ndarray = np.array([obs[LOAD_FEATURES] for obs in converted_trajectory])
        line_trajectory: np.ndarray = np.array([obs[LINES_FEATURES] for obs in converted_trajectory])

        generator_trajectories_all.append(generator_trajectory)
        load_trajectories_all.append(load_trajectory)
        line_trajectories_all.append(line_trajectory)

    return (np.stack(generator_trajectories_all),
            np.stack(load_trajectories_all),
            np.stack(line_trajectories_all))

@hydra.main(config_path="../hydra_configs", config_name="nri_training", version_base="1.3")
def main(cfg: DictConfig):
    print(cfg)
    # create env
    env = grid2op.make(cfg.env.name)

    # create agent
    if cfg.agent == 'random':
        agent = RandomAgent(env.action_space)
    elif cfg.agent == 'do_nothing':
        agent = DoNothingAgent(env.action_space)
    elif cfg.agent == 'reconnect':
        agent = RecoPowerlineAgent(env.action_space)
    elif cfg.agent == 'topology_greedy':
        agent = TopologyGreedy(env.action_space)
        logger.warning("You have configured the topology greedy agent that will simulate every topology action. "
                       "This is only feasible for small environments.")
    elif cfg.agent == 'baseline':
        gym_env = GymEnv(env)
        gym_env.observation_space.close()
        gym_env.observation_space = GraphObservationSpace(env.observation_space)
        gym_env.action_space.close()
        gym_env.action_space = DiscreteActSpace()
        model_path = base_path_models.joinpath(cfg.baseline_name)
        agent = build_agent(cfg, load_weights_from=model_path)
    else:
        raise NotImplementedError(f"Unknown agent '{cfg.agent}'")

    total = cfg.num_train_trajecctories + cfg.num_val_trajectories + cfg.num_test_trajectories
    logger.info(f"Running {cfg.agent} on {cfg.env.name} to produce {total} trajectories...")
    gen_traj, load_traj, line_traj = generate_dataset(total, cfg.trajectory_length, agent=agent, env=env)

    gen_traj_train = gen_traj[:cfg.num_train_trajecctorie]
    load_traj_train = load_traj[:cfg.num_train_trajecctorie]
    line_traj_train = line_traj[:cfg.num_train_trajecctorie]

    gen_traj_test = gen_traj[cfg.num_train_trajecctorie:cfg.num_train_trajecctorie + cfg.num_test_trajectories]
    load_traj_test = load_traj[cfg.num_train_trajecctorie:cfg.num_train_trajecctorie + cfg.num_test_trajectories]
    line_traj_test = line_traj[cfg.num_train_trajecctorie:cfg.num_train_trajecctorie + cfg.num_test_trajectories]

    gen_traj_valid = gen_traj[-cfg.num_val_trajectories:]
    load_traj_valid = load_traj[-cfg.num_val_trajectories:]
    line_traj_valid = line_traj[-cfg.num_val_trajectories:]

    np.save('dataset/gen_traj_train_' + cfg.env.name + '_.npy', gen_traj_train)
    np.save('dataset/load_traj_train_' + cfg.env.name + '_.npy', load_traj_train)
    np.save('dataset/line_traj_train_' + cfg.env.name + '_.npy', line_traj_train)
    np.save('dataset/gen_traj_valid_' + cfg.env.name + '_.npy', gen_traj_valid)
    np.save('dataset/load_traj_valid_' + cfg.env.name + '_.npy', load_traj_valid)
    np.save('dataset/line_traj_valid_' + cfg.env.name + '_.npy', line_traj_valid)
    np.save('dataset/gen_traj_test_' + cfg.env.name + '_.npy', gen_traj_test)
    np.save('dataset/load_traj_test_' + cfg.env.name + '_.npy', load_traj_test)
    np.save('dataset/line_traj_test_' + cfg.env.name + '_.npy', line_traj_test)

if __name__ == '__main__':
    main()
