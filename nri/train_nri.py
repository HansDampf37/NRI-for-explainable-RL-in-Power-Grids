from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from nri.ElboObjective import ElboLoss
from nri.NRI import NRIModule


def train(
        nri_module: NRIModule,
        training_set: Dataset,
        testing_set: Optional[Dataset] = None,
        prior: Optional[np.ndarray] = None,
        edge_index: Optional[Tensor] = None,
        logger: Optional[SummaryWriter] = None,
        num_epochs: int = 100,
        batch_size: int = 64,
        learning_rate: int = 0.01) -> NRIModule:
    """
    This function trains a given NRI model on a training set. During training, it is iteratively evaluate on the test set.
    Since the nri_module is a VAE we can provide prior beliefs about the latent distributions. If none are provided a
    uniform distribution is assumed. Furthermore, edge_index defines which edges are included in the encoder and decoder
    effectively limiting the possible latent edges to the edges provided in this argument.

    :param nri_module: NRI module to train
    :param training_set: Dataset to train on
    :param testing_set: Dataset to test on (optional)
    :param prior: Prior distribution to use for the ELBO objective (defaults to uniform distributions)
    :param edge_index: Edge indices to use for the NRI module (defaults to fully meshed)
    :param logger: Logging object to use (defaults to None)
    :param num_epochs: Number of epochs to train (defaults to 100)
    :param batch_size: Batch size (defaults to 64)
    :param learning_rate: Learning rate (defaults to 0.01)
    """
    dataloader_train = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(testing_set, batch_size=batch_size, shuffle=True)
    criterion = ElboLoss(prior)
    optimizer = torch.optim.Adam(nri_module.parameters(), lr=learning_rate)
    num_epochs = num_epochs
    for epoch in range(num_epochs):
        running_loss = 0.0
        nri_module.train()
        for batch in dataloader_train:
            batch = batch[0]
            optimizer.zero_grad()
            predictions, latent_edges = nri_module.forward(batch, edge_index)
            target = batch[:, 1:, :, :]
            loss = criterion(predictions, target, latent_edges)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if logger is not None:
            logger.add_scalar("Loss", running_loss / len(dataloader_train), epoch)
        else:
            print(f"Epoch {epoch} training loss: {running_loss / len(dataloader_train)}")

        if epoch % 10 == 0 and testing_set is not None:
            running_loss_eval = 0.0
            nri_module.eval()
            for batch in dataloader_test:
                batch = batch[0]
                predictions, latent_edges = nri_module.forward(batch, edge_index)
                target = batch[:, 1:, :, :]
                loss = criterion(predictions, target, latent_edges)
                running_loss_eval += loss.item()

            if logger is not None:
                logger.add_scalar("Loss", running_loss_eval / len(dataloader_test), epoch)
            else:
                print(f"Epoch {epoch} test loss: {running_loss_eval / len(dataloader_test)}")


    return nri_module
