import logging
from pathlib import Path
from typing import List

import grid2op
import numpy as np
import torch
from grid2op.Action import TopologySetAction
from grid2op.Observation import BaseObservation
from grid2op.gym_compat import DiscreteActSpace, BoxGymObsSpace, GymnasiumActionSpace, GymnasiumObservationSpace
from stable_baselines3 import DQN

from baselines.baseline_agent import BaselineAgent, TopologyPolicy, evaluate
from common import Grid2OpEnvWrapper, GraphObservationSpace
from common.GNN import SB3GNNWrapper
from common.graph_structured_observation_space import NODE_FEATURES, EDGE_FEATURES, EDGE_INDEX

_default_env_name = "l2rpn_case14_sandbox"
_default_obs_attr_to_keep = ["rho", "p_or", "gen_p", "load_p"]
_default_act_attr_to_keep = ["set_line_status_simple", "set_bus"]
_model_name = "dqn-mlp"
_model_path = Path(f"data/models/stable-baselines/{_model_name}")
_safe_max_rho = 0.95

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TopoPolicyStableDQN(TopologyPolicy[GymnasiumActionSpace, GymnasiumObservationSpace]):
    """
    Implements the interface of a topology policy. Uses a DQN trained by the stable baselines 3 library.
    """
    def __init__(self, dqn: DQN, gym_action_space: GymnasiumActionSpace, gym_observation_space: GymnasiumObservationSpace):
        """
        Initializes the topology policy that can be used to search for promising moves

        :param dqn: The sb3 DQN
        :param gym_action_space: the gymnasium compatible action space that is used to transform actions to grid2op actions
        :param gym_observation_space: the gymnasium compatible observation space that is used to transform observations to gymnasium observations
        """
        TopologyPolicy.__init__(self, gym_action_space, gym_observation_space)
        self.dqn: DQN = dqn

    def get_k_best_actions(self, observation: BaseObservation, k: int = 3) -> List[TopologySetAction]:
        # Convert observation to a tensor
        obs_batch = torch.from_numpy(self.observation_space.to_gym(observation)).unsqueeze(0).to(dtype=torch.float32)
        # Get Q-values for each action
        q_values = self.dqn.policy.q_net.forward(obs_batch).squeeze().cpu().detach().numpy()
        # Get the indices of the top k actions based on their Q-values
        top_k_indices = np.argsort(q_values)[-k:][::-1]  # Sort in descending order
        top_k_actions = [self.action_space.from_gym(idx) for idx in top_k_indices]

        return top_k_actions


def build_agent(path: Path = _model_path, env_name: str = _default_env_name) -> BaselineAgent:
    """
    Given a model path, builds an agent with a topology policy that can be used to search for promising actions.

    :param path: The path of the model
    :param env_name: The name of the gym environment
    :return: a baseline agent using the trained topology policy
    """
    env = grid2op.make(env_name)
    action_space = DiscreteActSpace(env.action_space, attr_to_keep=_default_act_attr_to_keep)
    obs_space = BoxGymObsSpace(env.observation_space, attr_to_keep=_default_obs_attr_to_keep)
    dqn = DQN.load(path)
    dqn.action_space = action_space
    dqn.observation_space = obs_space
    baseline_agent = BaselineAgent(
        env.action_space,
        TopoPolicyStableDQN(dqn, action_space, obs_space)
    )
    return baseline_agent


def train():
    """
    Trains a model
    """
    env = Grid2OpEnvWrapper(
        env_name=_default_env_name,
        safe_max_rho=_safe_max_rho,
        observation_space_creation=lambda e: GraphObservationSpace(e.observation_space, [NODE_FEATURES, EDGE_FEATURES, EDGE_INDEX])
    )
    dqn = DQN(
        "MultiInputPolicy",
        env,
        verbose=1,
        train_freq=16,
        gradient_steps=8,
        gamma=0.99,
        exploration_fraction=0.2,
        exploration_final_eps=0.07,
        target_update_interval=600,
        learning_starts=1000,
        buffer_size=10000,
        batch_size=128,
        learning_rate=4e-3,
        tensorboard_log=f"data/logs/stable-baselines/{_model_name}",
        policy_kwargs=dict(
            features_extractor_class=SB3GNNWrapper,
            features_extractor_kwargs=dict(
                x_dim = GraphObservationSpace.NUM_FEATURES_PER_NODE,
                e_dim = GraphObservationSpace.NUM_FEATURES_PER_EDGE,
                hidden_x_dim = 16,
                hidden_e_dim = 16,
                node_out_dim = 128,
                edge_out_dim = 128,
                n_layers = 3,
                dropout_prob = 0.1,
                residual=True
            ),
        ),
        seed=2,
    )
    dqn.learn(120_000, log_interval=10)
    dqn.save(_model_path)


if __name__ == '__main__':
    train()
    evaluate(
        agent=build_agent(_model_path, _default_env_name),
        env=grid2op.make(_default_env_name),
        path_results=f"data/evaluations/stable-baselines/{_model_name}",
        nb_episode=3,
        max_iter=500,
        nb_process=2,
        pbar=True,
    )
