"""
This script can be used to retrieve and save edge type probabilities by running a RARL algorithm on a specified env.
The final edge type probabilities are averaged for each edge-type combination across all samples in the observations encountered.
"""
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from gymnasium import Env

from common import EDGE_PROBS_PATH, logger
from .RARL import RARL


def get_edge_type_probabilities(
        RARL_model: RARL,
        env: Env,
        verbose: bool = True,
) -> np.ndarray:
    """
    Runs the specified RARL alg on the specified environment and returns the averaged edge type probabilities.
    :param RARL_model: the RARL alg to run
    :param env: the env to run the module on
    :param verbose: whether to print info
    :return: the averaged edge type probabilities as numpy array of shape [E, NUM_EDGE_TYPES]
    """
    # Accumulate sum over all samples to compute a true dataset-wide average
    running_sum: Optional[np.ndarray] = None
    total_count: int = 0

    # Move edge_index once if provided
    with torch.no_grad():
        while total_count < 1000:
            obs, _ = env.reset()
            done = False
            while not done:
                act, _ = RARL_model.predict(obs)
                latent_edges = RARL_model.get_edge_type_posterior(obs)
                if running_sum is None:
                    running_sum = latent_edges
                else:
                    running_sum += latent_edges
                total_count += 1
                # step
                obs, _, done, truncated, _ = env.step(act)
                done = done or truncated


        if running_sum is None or total_count == 0:
            latent_edge_probs = np.ndarray([])
        else:
            latent_edge_probs = running_sum / float(total_count)

    if verbose:
        logger.info(
            f"------------------------ Edge detection complete. ------------------------\n"
            f"The latent edge type distributions are:\n{latent_edge_probs}"
        )

    return latent_edge_probs


def save_edge_probs(
        RARL_model: RARL,
        env: Env,
        save_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Runs the specified RARL alg on the specified environment and saves the averaged edge type probabilities.
    :param RARL_model: the RARL alg to run
    :param env: the env to run the module on
    :param save_path: optional output path. If None, uses data/edge_probabilities if it exists, otherwise data/edge_probs
    :return: the averaged edge type probabilities as numpy array of shape [E, NUM_EDGE_TYPES]
    """
    edge_probs = get_edge_type_probabilities(
        RARL_model,
        env,
        verbose=True,
    )

    # Smart default for output directory
    if save_path is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        file_name = f"edge_probs_{timestamp}.npy"
        save_path = Path(EDGE_PROBS_PATH, file_name)

    os.makedirs(save_path.parent, exist_ok=True)
    np.save(save_path, edge_probs)
    return edge_probs
