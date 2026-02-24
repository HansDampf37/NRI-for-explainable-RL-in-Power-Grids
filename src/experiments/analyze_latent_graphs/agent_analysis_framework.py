import abc
import logging
import time
from abc import abstractmethod
from typing import Any

import numpy.typing as npt
import torch
from grid2op.Action import BaseAction
from grid2op.Agent import BaseAgent
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation
from tqdm import tqdm

from src.common.observation_space import EDGE_INDEX, EDGE_MASK
from src.rl4pnc.evaluation.evaluation_agents import RllibAgent
from src.rl4pnc.grid2op_env.observation_converter import ObservationConverter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PosteriorAnalyzer(abc.ABC):
    @abstractmethod
    def on_rl_step(self, posterior: npt.NDArray, powergrid_graph: npt.NDArray, observation: BaseObservation, environment: Environment):
        pass

    @abstractmethod
    def on_heuristic_step(self, powergrid_graph: npt.NDArray, observation: BaseObservation, environment: Environment):
        pass

    @abstractmethod
    def on_new_episode(self, chronic_id: str):
        pass

    @abstractmethod
    def on_evaluation_end(self):
        pass


class LatentGraphAnalysisAgent(BaseAgent):
    """Wrapper around RllibAgent that invokes an analysis component when agent acts."""

    def __init__(self, rllib_agent: RllibAgent, gym_wrapper: ObservationConverter, analyser: PosteriorAnalyzer):
        """Initialize Agent."""
        BaseAgent.__init__(self, rllib_agent.action_space)

        self.rllib_agent = rllib_agent
        self.analyser = analyser
        self.gym_wrapper = gym_wrapper

        # Get the policy model for accessing posterior
        self.policy_model = rllib_agent._rllib_agent.model

        # Get environment and node styles for graph visualization
        self.g2op_env = gym_wrapper.env_gym.init_env

    def act(self, observation: BaseObservation, reward: float, done: bool = False) -> BaseAction:
        """Returns action and visualizes posterior when RL model is used."""
        action = self.rllib_agent.act(observation, reward, done)

        # Check if RL agent will be activated (same logic as RllibAgent)
        use_rl_component = self.rllib_agent.activate_agent(observation)

        # get powergrid edge index
        self.rllib_agent.gym_wrapper.update_obs(observation)
        powergrid_edge_index = self.rllib_agent.gym_wrapper.cur_gym_obs[EDGE_INDEX]
        edge_mask = self.rllib_agent.gym_wrapper.cur_gym_obs[EDGE_MASK]
        powergrid_edge_index = torch.from_numpy(powergrid_edge_index[..., edge_mask])

        # invoke analyzer
        if use_rl_component:
            posterior = self._get_posterior()
            self.analyser.on_rl_step(
                posterior,
                powergrid_edge_index.detach().cpu().numpy(),
                observation,
                self.g2op_env,
            )
        else:
            self.analyser.on_heuristic_step(
                powergrid_edge_index.detach().cpu().numpy(),
                observation,
                self.g2op_env
            )

        return action

    def on_new_episode(self, chronic_id: str):
        """Notify analyser of new episode."""
        self.analyser.on_new_episode(chronic_id)

    def on_evaluation_end(self):
        """Notify analyser of evaluation end."""
        self.analyser.on_evaluation_end()

    def _get_posterior(self) -> Any:
        posterior = self.policy_model.get_posterior()  # [B, E, K]

        if posterior.dim() == 3:
            posterior = posterior[0]  # [E, K]

        posterior_np = posterior.cpu().detach().numpy()
        return posterior_np

    def analyze(self, max_total_duration_s: int | None = None, num_episodes: int = 999999):
        """
        Runs analysis on the agents environment until either num episodes are finished, or the time is over.
        :param max_total_duration_s: max total duration in seconds (default disabled)
        :param num_episodes: the number of episodes (default all)
        """
        # Run episodes
        start_time = time.time()
        num_episodes = min(num_episodes, len(self.g2op_env.chronics_handler.available_chronics()))
        for episode in range(num_episodes):
            max_steps = self.g2op_env.max_episode_duration()
            chronic_id = self.g2op_env.chronics_handler.get_name()
            pbar = tqdm(total=max_steps, desc=f"Episode {chronic_id} ({episode + 1}/{num_episodes} episodes)", unit="step")

            obs = self.g2op_env.reset()
            self.on_new_episode(chronic_id)
            done = False
            total_reward = 0

            while not done:
                action = self.act(obs, total_reward, done)
                obs, reward, done, info = self.g2op_env.step(action)
                total_reward += reward
                pbar.update(1)

            print(f"Episode {episode + 1} ended with total reward: {total_reward} after {self.g2op_env.nb_time_step}/{self.g2op_env.max_episode_duration()} steps\n")
            if max_total_duration_s is not None:
                elapsed_time = time.time() - start_time
                if elapsed_time >= max_total_duration_s:
                    logger.info(f"Reached maximum total duration of {max_total_duration_s} seconds. Stopping evaluation.")
                    break

        self.on_evaluation_end()

        try:
            self.gym_wrapper.env_gym.close()
        except Exception as e:
            logger.debug(f"Environment cleanup error (ignored): {e}")