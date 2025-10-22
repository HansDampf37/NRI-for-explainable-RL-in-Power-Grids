# python
"""
Dieses Script kann verwendet werden, um die durchschnittlichen Edge-Probabilities
für ein gegebenes Dataset und ein NRI-Modul zu berechnen und zu speichern.
"""
import logging
import os
from datetime import datetime
from typing import Optional

import hydra
import numpy as np
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm

from nri.NRI import NRIModule

logger = logging.getLogger(__name__)


def get_edge_type_probabilities(nri_module: NRIModule, dataset: Dataset, batch_size: int = 64,
                                verbose: bool = True) -> np.ndarray:
    """
    Runs the specified nri module on the specified dataset and returns the averaged edge type probabilities.
    :param nri_module: the NRI module to run (already on correct device)
    :param dataset: the dataset to run the module on
    :param batch_size: the batch size
    :param verbose: whether to print info
    :return: the averaged edge type probabilities as numpy array of shape [E, NUM_EDGE_TYPES]
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nri_module.to(device)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    nri_module.eval()

    running_mse = torch.tensor(0.0, device=device)
    running_entropy = torch.tensor(0.0, device=device)
    running_latent_edge_probs: Optional[Tensor] = None

    with torch.no_grad():
        for batch_ in tqdm(data_loader, desc="Detecting edges"):
            batch: Tensor = batch_[0].to(device=device, dtype=torch.float32)
            predictions, edge_type_distributions = nri_module.forward(batch)
            target = batch[:, 1:, :, :]

            batch_mse = ((predictions - target) ** 2).sum(dim=-1).mean()
            running_mse += batch_mse

            batch_entropy = -(edge_type_distributions * (edge_type_distributions + 1e-10).log()).sum(dim=-1).mean()
            running_entropy += batch_entropy

            batch_mean_probs = edge_type_distributions.mean(dim=0)  # mean over batch -> shape [E, K]
            if running_latent_edge_probs is None:
                running_latent_edge_probs = batch_mean_probs.clone()
            else:
                running_latent_edge_probs += batch_mean_probs

        num_batches = len(data_loader) if len(data_loader) > 0 else 1
        latent_edge_probs = (running_latent_edge_probs / num_batches).cpu() if running_latent_edge_probs is not None else torch.tensor([])
        mse = (running_mse / num_batches).cpu().item()
        entropy = (running_entropy / num_batches).cpu().item()

    if verbose:
        logger.info(
            f"------------------------ Edge detection complete. ------------------------\n"
            f"The NRI-module achieved an average MSE of {mse:.4f} with its predictions across the provided dataset.\n"
            f"The entropy of the latent edge type distributions averages {entropy:.4f} across the provided dataset.\n"
            f"The latent edge type distributions are:\n{latent_edge_probs:.4f}"
        )

    return latent_edge_probs.numpy()


def save_edge_probs(nri_module: NRIModule, dataset: Dataset, file_name: Optional[str] = None,
                    batch_size: int = 64) -> None:
    """
    Runs the specified nri module on the specified dataset and saves the averaged edge type probabilities.
    :param nri_module: the NRI module to run
    :param dataset: the dataset to run the module on
    :param file_name: the name of the file to save the averaged edge type probabilities to
    :param batch_size: the batch size
    """
    edge_probs = get_edge_type_probabilities(nri_module, dataset, batch_size=batch_size, verbose=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    file_name = (file_name or "edge_probs") + f"_{timestamp}.npy"
    out_dir = to_absolute_path('../data/edge_probs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, file_name)
    np.save(out_path, edge_probs)


@hydra.main(config_path="../hydra_configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # prepare data
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
    checkpoint_path = to_absolute_path('../data/models/NRI_2025-10-21_23-22_500_steps_with_forecast.pt')
    nri_weights = torch.load(checkpoint_path, map_location=device)
    nri_module.load_state_dict(nri_weights)

    # determine and save edge probs
    save_edge_probs(nri_module=nri_module, dataset=test_dataset, file_name="test_edges")
    save_edge_probs(nri_module=nri_module, dataset=train_dataset, file_name="training_edges")


if __name__ == "__main__":
    main()
