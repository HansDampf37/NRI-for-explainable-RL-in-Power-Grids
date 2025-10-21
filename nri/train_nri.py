import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union

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

from common.graph_structured_observation_space import EDGE_INDEX, GymnasiumObservationConverter
from nri.ElboObjective import ElboLoss
from nri.NRI import NRIModule
from visualization.utils import visualize_graph, get_node_positions, latent_edge_hist

logger = logging.getLogger(__name__)

@dataclass
class PlottingArgs:
    num_nodes: int
    powerline_edge_index: Optional[Union[np.ndarray, Tensor]] = None
    node_positions: Optional[np.ndarray] = None
    edge_weight: float = 5.0
    do_weight_sweep: bool = False


def train(
        nri_module: NRIModule,
        training_set: Dataset,
        testing_set: Optional[Dataset] = None,
        criterion: Optional[ElboLoss] = None,
        edge_index: Optional[Tensor] = None,
        tensorboard_logger: Optional[SummaryWriter] = None,
        num_epochs: int = 100,
        batch_size: int = 64,
        learning_rate: int = 0.01,
        evaluate_every_k_epochs: int = 10,
        plotting_args: Optional[PlottingArgs] = None) -> NRIModule:
    """
    This function trains a given NRI model on a training set. During training, it is iteratively evaluate on the test set.
    The NRI module is trained to minimize the provided Elbo Loss. Furthermore, edge_index defines which edges are included
    in the encoder and decoder effectively limiting the possible latent edges to the edges provided in this argument.

    :param nri_module: NRI module to train
    :param training_set: Dataset to train on
    :param testing_set: Dataset to test on (optional, if no dataset is provided, there won't be evaluation)
    :param criterion: The loss function to use (defaults to ELBO with uniform prior)
    :param edge_index: Edge indices to use for the NRI module (defaults to fully meshed)
    :param tensorboard_logger: Logging object to use (defaults to None)
    :param num_epochs: Number of epochs to train (defaults to 100)
    :param batch_size: Batch size (defaults to 64)
    :param learning_rate: Learning rate (defaults to 0.01)
    :param evaluate_every_k_epochs: Evaluate model every k training steps (defaults to 10) (evaluation needs test set)
    :param plotting_args: args for creating plots (only needed for evaluations) (optional)
    """
    num_params = sum(p.numel() for p in nri_module.parameters() if p.requires_grad)
    logger.info(f"Starting training of NRI module with {num_params} trainable parameters")

    plotting_args = plotting_args or PlottingArgs(num_nodes=training_set[0][0].shape[-2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nri_module.to(device=device, dtype=torch.float32)

    dataloader_train = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    criterion = criterion if criterion is not None else ElboLoss()
    optimizer = torch.optim.Adam(nri_module.parameters(), lr=learning_rate)
    num_epochs = num_epochs
    for epoch in range(num_epochs):
        if epoch % evaluate_every_k_epochs == 0 and testing_set is not None:
            evaluate_nri_module(
                nri_module=nri_module,
                testing_set=testing_set,
                criterion=criterion,
                edge_index=edge_index,
                tensorboard_logger=tensorboard_logger,
                current_epoch=epoch,
                batch_size=batch_size,
                plotting_args=plotting_args
            )

        running_loss = 0.0
        running_nll = 0.0
        running_kl_div = 0.0
        running_mse = 0.0
        running_entropy = 0.0

        nri_module.train()
        for batch_ in tqdm(dataloader_train, f"Epoch {epoch} training"):
            batch: Tensor = batch_[0].to(device=device, dtype=torch.float32)
            optimizer.zero_grad()
            predictions, edge_type_distributions = nri_module.forward(batch, edge_index)
            target = batch[:, 1:, :, :]
            loss = criterion(predictions, target, edge_type_distributions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nri_module.parameters(), max_norm=10.0)
            optimizer.step()

            with torch.no_grad():
                running_loss += loss.item()
                running_nll += criterion.neg_log_likelihood(predictions, target)
                running_kl_div += criterion.kl_divergence_to_prior(edge_type_distributions)
                running_mse += ((predictions - target) ** 2).sum(dim=-1).mean()
                running_entropy += -(edge_type_distributions * (edge_type_distributions + 1e-10).log()).sum(dim=-1).mean()
                avg_probs = edge_type_distributions.mean(dim=tuple(range(edge_type_distributions.ndim - 1)))

        grad_norm = sum(p.grad.norm().item() ** 2 for p in nri_module.parameters() if p.grad is not None) ** 0.5
        if tensorboard_logger is not None:
            tensorboard_logger.add_scalar("Loss/training", running_loss / len(dataloader_train), epoch)
            tensorboard_logger.add_scalar("Negative Log Likelihood/training", running_nll / len(dataloader_train), epoch)
            tensorboard_logger.add_scalar("KL-Divergence to Prior/training", running_kl_div / len(dataloader_train), epoch)
            tensorboard_logger.add_scalar("MSE/training", running_mse / len(dataloader_train), epoch)
            tensorboard_logger.add_scalar("Gradient Norm/training", grad_norm, epoch)
            tensorboard_logger.add_scalar("Entropy of edge type predictions/training", running_entropy / len(dataloader_train), epoch)
            for k in range(avg_probs.shape[0]):
                tensorboard_logger.add_scalar(f"Average occurrence of edge type/{k} (training)", avg_probs[k], epoch)
        else:
            logger.info(f"Epoch {epoch}: Training loss: {running_loss / len(dataloader_train):.2f} "
                        f"Neg Log Likelihood: {running_nll / len(dataloader_train):.2f} "
                        f"KL Divergence: {running_kl_div / len(dataloader_train):.2f} "
                        f"MSE: {running_mse / len(dataloader_train):.2f} "
                        f"Gradient Norm: {grad_norm:.2f} "
                        f"Entropy of edge type predictions: {running_entropy / len(dataloader_train):.2f}")

    # evaluate for one last time with weight sweep
    plotting_args.do_weight_sweep = True
    evaluate_nri_module(
        nri_module=nri_module,
        testing_set=testing_set,
        criterion=criterion,
        edge_index=edge_index,
        tensorboard_logger=tensorboard_logger,
        current_epoch=num_epochs - 1,
        batch_size=batch_size,
        plotting_args=plotting_args
    )

    return nri_module


def evaluate_nri_module(
        nri_module: NRIModule,
        testing_set: Dataset,
        criterion: Optional[ElboLoss] = None,
        edge_index: Optional[Tensor] = None,
        tensorboard_logger: Optional[SummaryWriter] = None,
        current_epoch: Optional[int] = None,
        batch_size: int = 64,
        plotting_args: Optional[PlottingArgs] = None):
    """
    Evaluates the NRI module on a test set.

    :param nri_module: NRI module to evaluate
    :param testing_set: Dataset to evaluate on
    :param criterion: The loss function to use (defaults to ELBO with uniform prior)
    :param edge_index: Edge indices to use for the NRI module
    :param tensorboard_logger: Logging object to use (defaults to None)
    :param current_epoch: Current epoch (if this is invoked during training - if not None)
    :param batch_size: Batch size (defaults to 64)
    :param plotting_args: args for creating plots (optional)
    """
    plotting_args = plotting_args or PlottingArgs(num_nodes=testing_set[0][0].shape[-2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nri_module.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        criterion = criterion if criterion is not None else ElboLoss()
        data_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=True)
        nri_module.eval()
        running_loss = 0.0
        running_nll = 0.0
        running_kl_div = 0.0
        running_mse = 0.0
        running_entropy = 0.0
        running_latent_edge_probs = 0.0

        for batch_ in tqdm(data_loader, "Testing"):
            batch: Tensor = batch_[0].to(device=device, dtype=torch.float32)
            predictions, edge_type_distributions = nri_module.forward(batch, edge_index)
            target = batch[:, 1:, :, :]
            loss = criterion(predictions, target, edge_type_distributions)

            running_loss += loss.item()
            running_nll += criterion.neg_log_likelihood(predictions, target)
            running_kl_div += criterion.kl_divergence_to_prior(edge_type_distributions)
            running_mse += ((predictions - target) ** 2).sum(dim=-1).mean()
            running_entropy += -(edge_type_distributions * (edge_type_distributions + 1e-10).log()).sum(dim=-1).mean()
            running_latent_edge_probs += edge_type_distributions.mean(dim=0)  # mean over batch of edge type probs

        running_latent_edge_probs = running_latent_edge_probs.cpu() / len(data_loader)
        if tensorboard_logger is not None:
            # create figures
            if not plotting_args.do_weight_sweep:
                latent_edges_fig = visualize_graph(
                    num_nodes=plotting_args.num_nodes,
                    edge_index=plotting_args.powerline_edge_index,
                    latent_edge_probs=running_latent_edge_probs,
                    skip_first_edge_type=nri_module.skip_first,
                    node_positions=plotting_args.node_positions,
                    edge_weight=plotting_args.edge_weight,
                )
                tensorboard_logger.add_figure("Predicted latent Graph/testing", latent_edges_fig, current_epoch)
            else:
                steps = np.linspace(0, 1, 100)
                weights: List[float] = plotting_args.edge_weight * 100 ** (steps * 2 - 1) / (1 + 100 ** (steps * 2 - 1)).tolist()
                for i, weight in enumerate(weights):
                    latent_edges_fig = visualize_graph(
                        num_nodes=plotting_args.num_nodes,
                        edge_index=plotting_args.powerline_edge_index,
                        latent_edge_probs=running_latent_edge_probs,
                        skip_first_edge_type=nri_module.skip_first,
                        node_positions=plotting_args.node_positions,
                        edge_weight=weight,
                    )
                    tensorboard_logger.add_figure("Predicted latent Graph sweep/testing", latent_edges_fig, i)
                    plt.close(latent_edges_fig)

            latent_edges_hist = latent_edge_hist(running_latent_edge_probs)
            tensorboard_logger.add_figure("Histogram of predicted latent edges/testing", latent_edges_hist, current_epoch)
            tensorboard_logger.add_scalar("Loss/testing", running_loss / len(data_loader), current_epoch)
            tensorboard_logger.add_scalar("Negative Log Likelihood/testing", running_nll / len(data_loader), current_epoch)
            tensorboard_logger.add_scalar("KL-Divergence to Prior/testing", running_kl_div / len(data_loader), current_epoch)
            tensorboard_logger.add_scalar("MSE/testing", running_mse / len(data_loader), current_epoch)
            tensorboard_logger.add_scalar("Entropy of edge type predictions/testing", running_entropy / len(data_loader), current_epoch)
            avg_probs = edge_type_distributions.mean(dim=tuple(range(edge_type_distributions.ndim - 1)))
            for k in range(avg_probs.shape[0]):
                tensorboard_logger.add_scalar(f"Average occurrence of edge type/{k} (testing)", avg_probs[k], current_epoch)
        else:
            logger.info(f"Epoch {current_epoch}: Testing loss: {running_loss / len(data_loader):.2f} "
                        f"Neg Log Likelihood: {running_nll / len(data_loader):.2f} "
                        f"KL Divergence: {running_kl_div / len(data_loader):.2f} "
                        f"MSE: {running_mse / len(data_loader):.2f} "
                        f"Entropy of edge type predictions: {running_entropy / len(data_loader):.2f}")


@hydra.main(config_path="../hydra_configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    tensorboard_logger = SummaryWriter(f'data/logs/nri/{timestamp}')

    # prepare training data
    train_data = np.load(cfg.nri.train.training_dataset_path)
    test_data = np.load(cfg.nri.train.testing_dataset_path)
    train_dataset = TensorDataset(torch.from_numpy(train_data))
    test_dataset = TensorDataset(torch.from_numpy(test_data))

    # prepare plotting args
    env = grid2op.make(cfg.env.env_name)
    observation_space: GymnasiumObservationConverter = instantiate(cfg.nri.obs_space, grid2op_observation_space=env.observation_space)
    node_positions = get_node_positions(env, observation_space.__class__)
    edge_index = observation_space.to_gym(env.reset())[EDGE_INDEX]
    plotting_args = PlottingArgs(
        num_nodes=train_data.shape[-2],
        node_positions=node_positions,
        powerline_edge_index=edge_index
    )

    # prepare model
    nri_module = instantiate(cfg.nri.model, x_dim=train_data.shape[-1])
    criterion = ElboLoss(
        prior=np.array(cfg.nri.train.prior),
        alpha=cfg.nri.train.alpha,
        beta=cfg.nri.train.beta,
    )

    train(
        nri_module=nri_module,
        training_set=train_dataset,
        testing_set=test_dataset,
        criterion=criterion,
        tensorboard_logger=tensorboard_logger,
        num_epochs=cfg.nri.train.num_epochs,
        batch_size=cfg.nri.train.batch_size,
        learning_rate=cfg.nri.train.learning_rate,
        evaluate_every_k_epochs=cfg.nri.train.evaluate_every_k_epochs,
        plotting_args=plotting_args
    )

    model_path = f"data/models/nri/NRI_{timestamp}.pt"
    torch.save(nri_module.state_dict(), model_path)


if __name__ == "__main__":
    main()
