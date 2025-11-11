from abc import abstractmethod
from typing import Union

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from torch import Tensor


class RARL(BaseAlgorithm):
    @abstractmethod
    def get_edge_type_posterior(self, obs: Union[np.ndarray, dict[str, np.ndarray]]) -> Tensor:
        """
        predicts type distributions for each edge of the fully connected graph such "exists" / "doesn't exist"
        @param obs: the input
        @return: edge type distributions [E, K]
        """
        pass
