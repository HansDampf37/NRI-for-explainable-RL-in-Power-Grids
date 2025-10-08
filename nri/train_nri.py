from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from nri.ElboObjective import ElboLoss
from nri.NRI import NRIModule
from visualization.utils import visualize_latent_graph


def train(
        nri_module: NRIModule,
        training_set: Dataset,
        testing_set: Optional[Dataset] = None,
        prior: Optional[np.ndarray] = None,
        edge_index: Optional[Tensor] = None,
        logger: Optional[SummaryWriter] = None,
        num_epochs: int = 100,
        batch_size: int = 64,
        learning_rate: int = 0.01,
        evaluate_every_k_steps: int = 10,
        show_latent_edges_on_eval: bool = False) -> NRIModule:
    """
    This function trains a given NRI model on a training set. During training, it is iteratively evaluate on the test set.
    Since the nri_module is a VAE we can provide prior beliefs about the latent distributions. If none are provided a
    uniform distribution is assumed. Furthermore, edge_index defines which edges are included in the encoder and decoder
    effectively limiting the possible latent edges to the edges provided in this argument.

    :param nri_module: NRI module to train
    :param training_set: Dataset to train on
    :param testing_set: Dataset to test on (optional, if no dataset is provided, there won't be evaluation)
    :param prior: Prior distribution to use for the ELBO objective (defaults to uniform distributions)
    :param edge_index: Edge indices to use for the NRI module (defaults to fully meshed)
    :param logger: Logging object to use (defaults to None)
    :param num_epochs: Number of epochs to train (defaults to 100)
    :param batch_size: Batch size (defaults to 64)
    :param learning_rate: Learning rate (defaults to 0.01)
    :param evaluate_every_k_steps: Evaluate model every k training steps (defaults to 10) (evaluation needs test set)
    :param show_latent_edges_on_eval: If set to true this method will display the latent edges even without logger (defaults to False)
    """
    dataloader_train = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    criterion = ElboLoss(prior)
    optimizer = torch.optim.Adam(nri_module.parameters(), lr=learning_rate)
    num_epochs = num_epochs
    for epoch in range(num_epochs):
        if epoch % evaluate_every_k_steps == 0 and testing_set is not None:
            evaluate_nri_module(nri_module, testing_set, prior, edge_index, logger, epoch, show_latent_edges_on_eval)

        running_loss = 0.0
        running_nll = 0.0
        running_kl_div = 0.0
        running_mse = 0.0

        nri_module.train()
        for batch in dataloader_train:
            batch: Tensor = batch[0]
            optimizer.zero_grad()
            predictions, edge_type_distributions = nri_module.forward(batch, edge_index)
            target = batch[:, 1:, :, :]
            loss = criterion(predictions, target, edge_type_distributions)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_nll += criterion.neg_log_likelihood(predictions, target)
            running_kl_div += criterion.kl_divergence_to_prior(edge_type_distributions)
            running_mse += ((predictions - target) ** 2).sum(dim=-1).mean()

        if logger is not None:
            logger.add_scalar("Loss", running_loss / len(dataloader_train), epoch)
            logger.add_scalar("Negative Log Likelihood", running_nll / len(dataloader_train), epoch)
            logger.add_scalar("KL-Divergence to Prior", running_kl_div / len(dataloader_train), epoch)
            logger.add_scalar("MSE", running_mse / len(dataloader_train), epoch)
        else:
            print(f"Epoch {epoch}: Training loss: {running_loss / len(dataloader_train):.2f} "
                  f"Neg Log Likelihood: {running_nll / len(dataloader_train):.2f} "
                  f"KL Divergence: {running_kl_div / len(dataloader_train):.2f} "
                  f"MSE: {running_mse / len(dataloader_train):.2f}")

    return nri_module


def evaluate_nri_module(
        nri_module: NRIModule,
        testing_set: Dataset,
        prior: Optional[np.ndarray] = None,
        edge_index: Optional[Tensor] = None,
        logger: Optional[SummaryWriter] = None,
        current_epoch: Optional[int] = None,
        show_latent_edges: bool = False,):
    """
    Examines the latent edges discovered by the encoder.
    :param nri_module: NRI module to evaluate
    :param testing_set: Dataset to evaluate on
    :param prior: Prior assumption for edge type distributions (defaults to uniform distributions)
    :param edge_index: Edge indices to use for the NRI module
    :param logger: Logging object to use (defaults to None)
    :param current_epoch: Current epoch (if this is invoked during training - if not None)
    :param show_latent_edges: If set to true this method will display the latent edges even without logger
    """
    with torch.no_grad():
        criterion = ElboLoss(prior)
        data_loader = DataLoader(testing_set, batch_size=64, shuffle=True)
        nri_module.eval()
        running_loss = 0.0
        running_nll = 0.0
        running_kl_div = 0.0
        running_mse = 0.0
        sampled_edges = []

        for batch_ in data_loader:
            batch: Tensor = batch_[0]
            predictions, edge_type_distributions = nri_module.forward(batch, edge_index)
            target = batch[:, 1:, :, :]
            loss = criterion(predictions, target, edge_type_distributions)

            running_loss += loss
            running_nll += criterion.neg_log_likelihood(predictions, target)
            running_kl_div += criterion.kl_divergence_to_prior(edge_type_distributions)
            running_mse += ((predictions - target) ** 2).sum(dim=-1).mean()

            sampled_edges.append(nri_module.get_latent_edges(batch, edge_index)) # shape [B, 3, E]

        # plot latent graph
        fig = visualize_latent_graph(
            typed_edge_index=torch.cat(sampled_edges, dim=0),
            ground_truth_edge_index=edge_index,
            skip_first_edge_type=nri_module.skip_first
        )

        if logger is not None:
            logger.add_figure("Predicted latent Graph (aggregated test set)", fig, global_step=current_epoch)
            logger.add_scalar("Loss eval", running_loss / len(data_loader), current_epoch)
            logger.add_scalar("Negative Log Likelihood", running_nll / len(data_loader), current_epoch)
            logger.add_scalar("KL-Divergence to Prior", running_kl_div / len(data_loader), current_epoch)
            logger.add_scalar("MSE", running_mse / len(data_loader), current_epoch)
        else:
            print(f"Epoch {current_epoch}: Training loss: {running_loss / len(data_loader):.2f} "
                  f"Neg Log Likelihood: {running_nll / len(data_loader):.2f} "
                  f"KL Divergence: {running_kl_div / len(data_loader):.2f} "
                  f"MSE: {running_mse / len(data_loader):.2f}")
            if show_latent_edges:
                plt.show()

        plt.close(fig)
