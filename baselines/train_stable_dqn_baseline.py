import logging
from pathlib import Path
from typing import List, Dict, Optional

import grid2op
import numpy as np
import torch
from grid2op.Action import TopologySetAction
from grid2op.Observation import BaseObservation
from grid2op.gym_compat import DiscreteActSpace
from stable_baselines3 import DQN

from baselines.baseline_agent import BaselineAgent, TopologyPolicy, evaluate
from common import Grid2OpEnvWrapper, GraphObservationSpace
from common.GNN import SB3GNNWrapper
from common.graph_structured_observation_space import NODE_FEATURES, EDGE_FEATURES, EDGE_INDEX

_default_env_name = "l2rpn_case14_sandbox"
_model_name = "dqn-gnn"
_model_path = Path(f"data/models/stable-baselines/{_model_name}")

_default_act_attr_to_keep = ["set_bus"]
_default_obs_spaces_to_keep = [NODE_FEATURES, EDGE_FEATURES, EDGE_INDEX]
_safe_max_rho = 0.9

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TopoPolicyStableDQN(TopologyPolicy):
    """
    Implements the interface of a topology policy. Uses a DQN trained by the stable baselines 3 library.
    """

    def __init__(self, dqn: DQN):
        """
        Initializes the topology policy that can be used to search for promising moves

        :param dqn: The sb3 DQN
        """
        TopologyPolicy.__init__(self)
        self.dqn: DQN = dqn
        if not hasattr(dqn.observation_space, "to_gym") or not hasattr(dqn.action_space, "from_gym"):
            raise ValueError("In order to use your DQN in a grid2op topo policy the DQN observation space needs a"
                             "method 'to_gym' and the action space needs a method 'from_gym'")  # TODO bad design

    def get_k_best_actions(self, observation: BaseObservation, k: int = 3) -> List[TopologySetAction]:
        # Convert observation to a tensor
        gym_obs = self.dqn.observation_space.to_gym(observation)
        if isinstance(gym_obs, np.ndarray):
            obs_batch = torch.from_numpy(gym_obs).unsqueeze(0).to(dtype=torch.float32)
        elif isinstance(gym_obs, Dict):
            obs_batch = {}
            for key, value in gym_obs.items():
                obs_batch[key] = torch.from_numpy(value).unsqueeze(0).to(dtype=torch.float32)
        else:
            raise NotImplementedError(f"Unknown gym obs type {gym_obs.__class__.__name__}")

        # Get Q-values for each action
        with torch.no_grad():
            features = self.dqn.policy.q_net.features_extractor(obs_batch)
            q_values = self.dqn.policy.q_net.q_net(features)
            q_values = q_values.squeeze().cpu().numpy()
        # Get the indices of the top k actions based on their Q-values
        top_k_indices = np.argsort(q_values)[-k:][::-1]  # Sort in descending order
        top_k_actions = [self.dqn.action_space.from_gym(idx) for idx in top_k_indices]

        return top_k_actions


def build_agent(load_weights_from: Path = _model_path, env_name: str = _default_env_name) -> BaselineAgent:
    """
    Given a model path, builds an agent with a topology policy that can be used to search for promising actions.

    :param load_weights_from: load the model weights from here
    :param env_name: The name of the gym environment
    :return: a baseline agent using the trained topology policy
    """
    dqn = model_setup(load_weights_from)
    return BaselineAgent(
        grid2op.make(env_name).action_space,
        TopoPolicyStableDQN(dqn),
    )


def train():
    """
    Creates a model using the setup_method, trains it and saves it.
    """
    dqn = model_setup()
    dqn.learn(300_000, log_interval=10)
    dqn.save(_model_path)


def model_setup(load_weights_from: Optional[Path] = None) -> DQN:
    """
    Reproducible model setup. Creates a dqn model to train / evaluate.
    :param: load_weights_from: Optionally provide a path to load weights
    :return: The untrained DQN model
    """
    env = Grid2OpEnvWrapper(
        env_name=_default_env_name + "_train",
        safe_max_rho=_safe_max_rho,
        action_space_creation=lambda e: DiscreteActSpace(e.action_space, attr_to_keep=_default_act_attr_to_keep),
        observation_space_creation=lambda e: GraphObservationSpace(e.observation_space, _default_obs_spaces_to_keep)
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
            net_arch=[100],
            features_extractor_class=SB3GNNWrapper,
            features_extractor_kwargs=dict(
                x_dim=GraphObservationSpace.NUM_FEATURES_PER_NODE,
                e_dim=GraphObservationSpace.NUM_FEATURES_PER_EDGE,
                hidden_x_dim=16,
                hidden_e_dim=16,
                node_out_dim=128,
                edge_out_dim=128,
                n_layers=3,
                dropout_prob=0.1,
                residual=True
            ),
        ),
        seed=2,
    )
    if load_weights_from is not None:
        dqn.load(load_weights_from)
    return dqn


if __name__ == '__main__':
    train()
    evaluate(
        agent=build_agent(_model_path, _default_env_name),
        env=grid2op.make(_default_env_name + "_test"),
        num_episodes=3,
        max_episode_length=20,
        path_results=f"data/evaluations/stable-baselines/{_model_name}"
    )
