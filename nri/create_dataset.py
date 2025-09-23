"""
This script contains code to generate a dataset containing trajectories of the environment being operated by some agent.
This dataset can further be used for NRI. The contained dataset contains trajectories for the following entities:
- generators
- loads
- powerlines
"""

import argparse
import logging
from typing import List, Tuple

import numpy as np
from grid2op.Agent import BaseAgent, RandomAgent, DoNothingAgent, RecoPowerlineAgent, TopologyGreedy
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation
import grid2op
from tqdm import tqdm

from baseline_gnn_agent.baseline_agent import BaselineGNNAgent, TopologyPolicy

logger = logging.getLogger(__name__)

FEATURES_PER_GENERATOR = 9
FEATURES_PER_LOAD = 5
FEATURES_PER_LINE = 13


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


def generator_features_from_observation(obs: BaseObservation) -> np.ndarray:
    """
    Returns a numpy representation of the features for generators in the observation. The returned numpy array
    is of shape (n, x) where n is the number of generators and x is the number of features per generator.

    :param obs: The observation
    :return: the numpy representation of the generators in the observation
    """
    return np.array([
        obs.gen_pmax,  # the maximum power (in MW)
        obs.gen_p,  # the current power (in MW)
        obs.gen_q,  # the current reactive power (in MVar)
        obs.gen_v,  # the voltage at the bus (in kV)
        obs.gen_theta,  # the voltage angle at the bus (in deg)
        obs.target_dispatch,  # the dispatch asked for by the agent
        obs.actual_dispatch,
        # the dispatch implemented by the environment (might differ to the above due to physical constraints)
        obs.curtailment_limit_mw,  # the production limit of the renewable generator (in MW)
        obs.gen_bus  # the bus that this generator is connected to (1, 2 or -1)
    ]).transpose()


def load_features_from_observation(obs: BaseObservation) -> np.ndarray:
    """
    Returns a numpy representation of the features for loads in the observation. The returned numpy array
    is of shape (n, x) where n is the number of loads and x is the number of features per load.

    :param obs: The observation
    :return: the numpy representation of the loads in the observation
    """
    return np.array([
        obs.load_p,  # the power (in MW)
        obs.load_q,  # the reactive power (in MVar)
        obs.load_v,  # the voltage at the bus (in kV)
        obs.load_theta,  # the voltage angle at the bus (in deg)
        obs.load_bus  # the bus that this load is connected to (1, 2 or -1)
    ]).transpose()


def line_features_from_observation(obs: BaseObservation) -> np.ndarray:
    """
    Returns a numpy representation of the features for lines in the observation. The returned numpy array
    is of shape (n, x) where n is the number of lines and x is the number of features per line.

    :param obs: The observation
    :return: the numpy representation of the lines in the observation
    """
    return np.array([
        obs.line_status,  # whether the line is connected
        obs.rho,  # the load on the line from 0 to 1
        obs.thermal_limit,  # the thermal limit of each line (in A)
        obs.p_or,  # the power at the origin (in MW)
        obs.q_or,  # the reactive power at the origin (in MVar)
        obs.a_or,  # the flow at the origin (in A)
        obs.theta_or,  # the voltage angle at the origin (in deg)
        obs.line_or_bus,  # the bus that the origin of the line is connected to (1, 2 or -1)
        obs.p_ex,  # the power at the extremity (in MW)
        obs.q_ex,  # the reactive power at the extremity (in MVar)
        obs.a_ex,  # the flow at the extremity (in A)
        obs.theta_ex,  # the voltage angle at the extremity (in deg)
        obs.line_ex_bus,  # the bus that the extremity of the line is connected to (1, 2 or -1)
    ]).transpose()


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
    generator_trajectories_all = []
    load_trajectories_all = []
    line_trajectories_all = []

    for _ in tqdm(range(num_sims), f"Creating {num_sims} trajectories"):
        trajectory: List[BaseObservation] = sample_trajectory(length=length, agent=agent, env=env)
        generator_trajectory: np.ndarray = np.array([generator_features_from_observation(obs) for obs in trajectory])
        load_trajectory: np.ndarray = np.array([load_features_from_observation(obs) for obs in trajectory])
        line_trajectory: np.ndarray = np.array([line_features_from_observation(obs) for obs in trajectory])

        generator_trajectories_all.append(generator_trajectory)
        load_trajectories_all.append(load_trajectory)
        line_trajectories_all.append(line_trajectory)

    return (np.stack(generator_trajectories_all),
            np.stack(load_trajectories_all),
            np.stack(line_trajectories_all))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='l2rpn_case14_sandbox', help='The name of the env.')
    parser.add_argument('--agent', type=str, default='do_nothing', help='One of "do_nothing", "random",'
                                                                        ' "reconnect", "topology_greedy", "baseline"')
    parser.add_argument('--num-train', type=int, default=500, help='Number of training trajectories to generate.')
    parser.add_argument('--num-valid', type=int, default=100, help='Number of validation trajectories to generate.')
    parser.add_argument('--num-test', type=int, default=100, help='Number of test trajectories to generate.')
    parser.add_argument('--length', type=int, default=400, help='Length of trajectory.')
    args = parser.parse_args()

    # create env
    env = grid2op.make(args.env)

    # create agent
    if args.agent == 'random':
        agent = RandomAgent(env.action_space)
    elif args.agent == 'do_nothing':
        agent = DoNothingAgent(env.action_space)
    elif args.agent == 'reconnect':
        agent = RecoPowerlineAgent(env.action_space)
    elif args.agent == 'topology_greedy':
        agent = TopologyGreedy(env.action_space)
        logger.warning("You have configured the topology greedy agent that will simulate every topology action. "
                       "This is only feasible for small environments.")
    elif args.agent == 'baseline':
        agent = BaselineGNNAgent(env.action_space, TopologyPolicy(action_space=env.action_space))
    else:
        raise NotImplementedError(f"Unknown agent '{args.agent}'")

    total = args.num_train + args.num_valid + args.num_test
    logger.info(f"Running {args.agent} on {args.env} to produce {total} trajectories...")
    gen_traj, load_traj, line_traj = generate_dataset(args.num_train, args.length, agent=agent, env=env)

    gen_traj_train = gen_traj[:args.num_train]
    load_traj_train = load_traj[:args.num_train]
    line_traj_train = line_traj[:args.num_train]

    gen_traj_test = gen_traj[args.num_train:args.num_train + args.num_test]
    load_traj_test = load_traj[args.num_train:args.num_train + args.num_test]
    line_traj_test = line_traj[args.num_train:args.num_train + args.num_test]

    gen_traj_valid = gen_traj[-args.num_valid:]
    load_traj_valid = load_traj[-args.num_valid:]
    line_traj_valid = line_traj[-args.num_valid:]

    np.save('dataset/gen_traj_train_' + env + '_.npy', gen_traj_train)
    np.save('dataset/load_traj_train_' + env + '_.npy', load_traj_train)
    np.save('dataset/line_traj_train_' + env + '_.npy', line_traj_train)
    np.save('dataset/gen_traj_valid_' + env + '_.npy', gen_traj_valid)
    np.save('dataset/load_traj_valid_' + env + '_.npy', load_traj_valid)
    np.save('dataset/line_traj_valid_' + env + '_.npy', line_traj_valid)
    np.save('dataset/gen_traj_test_' + env + '_.npy', gen_traj_test)
    np.save('dataset/load_traj_test_' + env + '_.npy', load_traj_test)
    np.save('dataset/line_traj_test_' + env + '_.npy', line_traj_test)

if __name__ == '__main__':
    main()
