"""
This script can be used to retrieve and save edge type probabilities by running a NRI module on a specified dataset.
The final edge type probabilities are averaged for each edge-type combination across all samples in the dataset.
"""
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm

from common.constants import MODELS_PATH, EDGE_PROBS_PATH
from .NRI import NRIModule

logger = logging.getLogger(__name__)


def get_edge_type_probabilities(
        nri_module: NRIModule,
        dataset: Dataset,
        batch_size: int = 64,
        verbose: bool = True,
        edge_index: Optional[Tensor] = None,
) -> np.ndarray:
    """
    Runs the specified nri module on the specified dataset and returns the averaged edge type probabilities.
    :param nri_module: the NRI module to run (already on correct device)
    :param dataset: the dataset to run the module on
    :param batch_size: the batch size
    :param verbose: whether to print info
    :param edge_index: optional edge index [2, E] to restrict latent edges considered
    :return: the averaged edge type probabilities as numpy array of shape [E, NUM_EDGE_TYPES]
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nri_module.to(device)
    nri_module.eval()

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Accumulate sum over all samples to compute a true dataset-wide average
    running_sum: Optional[np.ndarray] = None
    total_count: int = 0

    # Move edge_index once if provided
    edge_index_device: Optional[Tensor] = edge_index.to(device) if edge_index is not None else None

    with torch.no_grad():
        for batch_ in tqdm(data_loader, desc="Detecting edges"):
            batch: Tensor = batch_[0].to(device=device, dtype=torch.float32)
            predictions, edge_type_distributions = nri_module.forward(batch, edge_index=edge_index_device)
            # Sum over batch dimension to allow exact dataset average even with uneven last batch
            batch_sum_probs = edge_type_distributions.sum(dim=0)  # [E, K]
            batch_size_actual = edge_type_distributions.shape[0] if edge_type_distributions.dim() == 3 else 1

            if running_sum is None:
                running_sum = batch_sum_probs.detach().cpu().numpy()
            else:
                running_sum += batch_sum_probs.detach().cpu().numpy()

            total_count += batch_size_actual

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
        nri_module: NRIModule,
        dataset: Dataset,
        batch_size: int = 64,
        edge_index: Optional[Tensor] = None,
        save_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Runs the specified nri module on the specified dataset and saves the averaged edge type probabilities.
    :param nri_module: the NRI module to run
    :param dataset: the dataset to run the module on
    :param batch_size: the batch size
    :param edge_index: optional edge index [2, E] to restrict latent edges considered
    :param save_path: optional output path. If None, uses data/edge_probabilities if it exists, otherwise data/edge_probs
    :return: the averaged edge type probabilities as numpy array of shape [E, NUM_EDGE_TYPES]
    """
    edge_probs = get_edge_type_probabilities(
        nri_module,
        dataset,
        batch_size=batch_size,
        verbose=True,
        edge_index=edge_index,
    )

    # Smart default for output directory
    if save_path is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        file_name = f"edge_probs_{timestamp}.npy"
        save_path = Path(EDGE_PROBS_PATH, file_name)

    os.makedirs(save_path, exist_ok=True)
    np.save(save_path, edge_probs)
    return edge_probs


def _find_latest_checkpoint_by_name(name_prefix: str) -> Optional[str]:
    """
    Finds the newest checkpoint file in data/models/nri whose filename starts with the given name_prefix.
    Returns absolute path or None if not found.
    """
    base_dir = to_absolute_path(os.path.join(MODELS_PATH, "nri"))
    if not os.path.isdir(base_dir):
        return None

    candidates = [
        os.path.join(base_dir, f)
        for f in os.listdir(base_dir)
        if f.startswith(name_prefix) and f.endswith('.pt')
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


@hydra.main(config_path="../hydra_configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # prepare data
    # Prefer dataset paths defined in Hydra config; fall back to the explicit ones if missing
    train_path = to_absolute_path(cfg.nri.train.training_dataset_path)
    test_path = to_absolute_path(cfg.nri.train.testing_dataset_path)

    train_data = np.load(train_path)
    test_data = np.load(test_path)
    train_dataset = TensorDataset(torch.from_numpy(train_data))
    test_dataset = TensorDataset(torch.from_numpy(test_data))

    # prepare model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nri_module: NRIModule = instantiate(cfg.nri.model, x_dim=train_data.shape[-1])
    nri_module.to(device)

    # Try to infer a matching checkpoint from the configured name; otherwise fall back to a known path
    name_prefix = getattr(cfg.nri, 'name', None) or "NRI"
    checkpoint_path = _find_latest_checkpoint_by_name(name_prefix)
    if checkpoint_path is None:
        # fallback to a known checkpoint if available
        fallback_ckpt = 'data/models/nri/NRI_2025-10-21_23-22_500_steps_with_forecast.pt'
        fallback_abs = to_absolute_path(fallback_ckpt)
        if os.path.isfile(fallback_abs):
            checkpoint_path = fallback_abs
        else:
            raise FileNotFoundError(
                f"No checkpoint found matching prefix '{name_prefix}' in data/models/nri and fallback '{fallback_ckpt}' missing."
            )

    nri_weights = torch.load(checkpoint_path, map_location=device)
    nri_module.load_state_dict(nri_weights)

    # determine and save edge probs
    save_edge_probs(nri_module=nri_module, dataset=test_dataset, batch_size=cfg.nri.train.batch_size)
    save_edge_probs(nri_module=nri_module, dataset=train_dataset, batch_size=cfg.nri.train.batch_size)


if __name__ == "__main__":
    main()
