import logging
from typing import Optional

import grid2op
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nri.ElboObjective import ElboLoss
from nri.NRI import NRIModule
from visualization.utils import visualize_powergrid, get_node_positions

logger = logging.getLogger(__name__)

def train(
        nri_module: NRIModule,
        training_set: Dataset,
        testing_set: Optional[Dataset] = None,
        prior: Optional[np.ndarray] = None,
        edge_index: Optional[Tensor] = None,
        tensorboard_logger: Optional[SummaryWriter] = None,
        num_epochs: int = 100,
        batch_size: int = 64,
        learning_rate: int = 0.01,
        evaluate_every_k_steps: int = 10,
        show_latent_edges_on_eval: bool = False,
        node_positions: Optional[np.ndarray] = None) -> NRIModule:
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
    :param tensorboard_logger: Logging object to use (defaults to None)
    :param num_epochs: Number of epochs to train (defaults to 100)
    :param batch_size: Batch size (defaults to 64)
    :param learning_rate: Learning rate (defaults to 0.01)
    :param evaluate_every_k_steps: Evaluate model every k training steps (defaults to 10) (evaluation needs test set)
    :param show_latent_edges_on_eval: If set to true this method will display the latent edges even without logger (defaults to False)
    :param node_positions: custom node positions to use when displaying the latent graph (defaults to None)
    """
    num_params = sum(p.numel() for p in nri_module.parameters() if p.requires_grad)
    logger.info(f"Starting training of NRI module with {num_params} trainable parameters")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nri_module.to(device=device, dtype=torch.float32)

    dataloader_train = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    criterion = ElboLoss(prior)
    optimizer = torch.optim.Adam(nri_module.parameters(), lr=learning_rate)
    num_epochs = num_epochs
    for epoch in range(num_epochs):
        if epoch % evaluate_every_k_steps == 0 and testing_set is not None:
            evaluate_nri_module(
                nri_module=nri_module,
                testing_set=testing_set,
                prior=prior,
                edge_index=edge_index,
                tensorboard_logger=tensorboard_logger,
                current_epoch=epoch,
                show_latent_edges=show_latent_edges_on_eval,
                batch_size=batch_size,
                node_positions=node_positions
            )

        running_loss = 0.0
        running_nll = 0.0
        running_kl_div = 0.0
        running_mse = 0.0

        nri_module.train()
        for batch_ in tqdm(dataloader_train, f"Epoch {epoch} training"):
            batch: Tensor = batch_[0].to(device=device, dtype=torch.float32)
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

        if tensorboard_logger is not None:
            tensorboard_logger.add_scalar("Loss/training set", running_loss / len(dataloader_train), epoch)
            tensorboard_logger.add_scalar("Negative Log Likelihood/training set", running_nll / len(dataloader_train), epoch)
            tensorboard_logger.add_scalar("KL-Divergence to Prior/training set", running_kl_div / len(dataloader_train), epoch)
            tensorboard_logger.add_scalar("MSE/training set", running_mse / len(dataloader_train), epoch)
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
        tensorboard_logger: Optional[SummaryWriter] = None,
        current_epoch: Optional[int] = None,
        show_latent_edges: bool = False,
        batch_size: int = 64,
        node_positions: Optional[np.ndarray] = None):
    """
    Examines the latent edges discovered by the encoder.
    :param nri_module: NRI module to evaluate
    :param testing_set: Dataset to evaluate on
    :param prior: Prior assumption for edge type distributions (defaults to uniform distributions)
    :param edge_index: Edge indices to use for the NRI module
    :param tensorboard_logger: Logging object to use (defaults to None)
    :param current_epoch: Current epoch (if this is invoked during training - if not None)
    :param show_latent_edges: If set to true this method will display the latent edges even without logger
    :param batch_size: Batch size (defaults to 64)
    :param node_positions: custom node positions to use when displaying the latent graph (defaults to None)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nri_module.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        criterion = ElboLoss(prior)
        data_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=True)
        nri_module.eval()
        running_loss = 0.0
        running_nll = 0.0
        running_kl_div = 0.0
        running_mse = 0.0
        sampled_edges = None

        for batch_ in tqdm(data_loader, "Testing"):
            batch: Tensor = batch_[0].to(device=device, dtype=torch.float32)
            predictions, edge_type_distributions = nri_module.forward(batch, edge_index)
            target = batch[:, 1:, :, :]
            loss = criterion(predictions, target, edge_type_distributions)

            running_loss += loss.item()
            running_nll += criterion.neg_log_likelihood(predictions, target)
            running_kl_div += criterion.kl_divergence_to_prior(edge_type_distributions)
            running_mse += ((predictions - target) ** 2).sum(dim=-1).mean()

            latent_edges = nri_module.get_latent_edges(batch, edge_index) # shape [B, 3, E]
            latent_edges = latent_edges.permute(1, 0, 2).reshape(3, -1) # shape [3, B * E]
            if sampled_edges is not None:
                sampled_edges = torch.cat((sampled_edges, latent_edges), dim=-1)
            else:
                sampled_edges = latent_edges

        # plot latent graph
        fig = visualize_powergrid(
            typed_edge_index=sampled_edges,
            ground_truth_edge_index=edge_index,
            skip_first_edge_type=nri_module.skip_first,
            node_positions=node_positions
        )

        if tensorboard_logger is not None:
            tensorboard_logger.add_figure("Predicted latent Graph/test set", fig, global_step=current_epoch)
            tensorboard_logger.add_scalar("Loss/test set", running_loss / len(data_loader), current_epoch)
            tensorboard_logger.add_scalar("Negative Log Likelihood/test set", running_nll / len(data_loader), current_epoch)
            tensorboard_logger.add_scalar("KL-Divergence to Prior/test set", running_kl_div / len(data_loader), current_epoch)
            tensorboard_logger.add_scalar("MSE/test set", running_mse / len(data_loader), current_epoch)
        else:
            print(f"Epoch {current_epoch}: Testing loss: {running_loss / len(data_loader):.2f} "
                  f"Neg Log Likelihood: {running_nll / len(data_loader):.2f} "
                  f"KL Divergence: {running_kl_div / len(data_loader):.2f} "
                  f"MSE: {running_mse / len(data_loader):.2f}")
            if show_latent_edges:
                plt.show()

        plt.close(fig)

@hydra.main(config_path="../hydra_configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    env = grid2op.make(cfg.env.env_name)
    observation_space = instantiate(cfg.nri.obs_space, grid2op_observation_space=env.observation_space)
    node_positions = get_node_positions(env, observation_space.__class__)
    tensorboard_logger = SummaryWriter('data/logs/nri')
    train_data = np.load('data/nri_dataset/train_node_features_l2rpn_case14_sandbox_2025-10-16_09:28_.npy')
    test_data = np.load('data/nri_dataset/test_node_features_l2rpn_case14_sandbox_2025-10-16_09:28_.npy')
    train_dataset = TensorDataset(torch.from_numpy(train_data))
    test_dataset = TensorDataset(torch.from_numpy(test_data))
    nri_module = instantiate(cfg.nri.model, x_dim=train_data.shape[-1])
    train(
        nri_module=nri_module,
        training_set=train_dataset,
        testing_set=test_dataset,
        prior=np.array(cfg.nri.train.prior),
        tensorboard_logger=tensorboard_logger,
        num_epochs=cfg.nri.train.num_epochs,
        batch_size=cfg.nri.train.batch_size,
        learning_rate=cfg.nri.train.learning_rate,
        evaluate_every_k_steps=cfg.nri.train.evaluate_every_k_steps,
        show_latent_edges_on_eval=True,
        node_positions=node_positions
    )

if __name__ == "__main__":
    main()
