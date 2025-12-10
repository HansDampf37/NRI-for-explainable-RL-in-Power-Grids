"""
Power systems differ fundamentally from the closed physical systems studied by Kipf et al. in the original NRI paper.
They continuously interact with external agents and unmodeled processes. Most notably, they are operated by system
controllers and influenced by time-varying injections and consumptions at generators and loads.

If an NRI model were trained to predict next-state values of these injections or consumptions, it would waste capacity
modeling external processes rather than internal system dynamics. Yet, these quantities still affect the evolution of the
grid and must be available as conditioning information.

To separate these roles, we decompose each timestep into two feature sets:
x_t – dynamic node features whose next-state values we aim to predict.
θ_t – exogenous node features that we want to provide to the model but that we do not predict

The training objective is to maximize
p(x_{t+1} | x_t, θ_t),
that is, to predict the next system state conditioned on the current state and external factors.

This separation can be achieved by applying a feature mask that restricts the loss function to the predictive subset
x_t. This function creates the respective masks.
"""
import numpy as np

from .observation_space import GraphObservationSpace


def get_feature_mask(obs_space: GraphObservationSpace, predict_features: list[str]) -> np.ndarray:
    """
    @param obs_space: masked observations must come from this observation space
    @param predict_features: feature names that we want to keep after masking
    @return: the mask as numpy array
    """
    name_to_idx = {name: i for i, name in enumerate(obs_space.node_feature_names)}
    mask = np.zeros(obs_space.x_dim, dtype=bool)
    for name in predict_features:
        mask[name_to_idx[name]] = True
    return mask
