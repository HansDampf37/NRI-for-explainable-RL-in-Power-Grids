import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import grid2op
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common import LOGS_PATH, MODELS_PATH, EDGE_PROBS_PATH, logger, EDGE_INDEX, GraphObservationSpace, get_feature_mask
from .ElboObjective import ElboLoss
from .NRI import NRIModule
from .utils import warn_large_loss
from .get_edge_probs import save_edge_probs
from visualization import visualize_graph, get_node_styles, latent_edge_hist, PlottingArgs

max_loss_without_warning = 200


class RunningMetrics:
    """
    Metrics can be accumulated for an epoch and then updated.
    """

    def __init__(self, summary_writer: Optional[SummaryWriter], plotting_args: Optional[PlottingArgs]):
        """
        Constructor.

        @param summary_writer: the summary writer
        @param plotting_args: the plotting arguments in order to log figures (optional if no provided no figures will be logged)
        """
        self.summary_writer = summary_writer
        self.plotting_args = plotting_args
        self.running_loss = 0.0
        self.running_nll = 0.0
        self.running_kl_div = 0.0
        self.running_mse = 0.0
        self.running_entropy = 0.0
        self.running_type_probs: Optional[np.ndarray] = None
        self.running_edge_type_probs: Optional[np.ndarray] = None
        self.grad_norm: Optional[float] = None
        self.num_iter = 0
        self.logger = logging.getLogger(__name__)

    def new_iter(self, loss: float, nll: float, kl_div: float, mse: float, edge_type_dists: np.ndarray):
        """
        Adds properties from one batch iteration to the metrics.

        @param loss: the loss
        @param nll: the negative log likelihood
        @param kl_div: the kl div
        @param mse: the mean squared error
        @param edge_type_dists: the edge type distributions
        """
        self.num_iter += 1
        self.running_loss += loss
        self.running_nll += nll
        self.running_kl_div += kl_div
        self.running_mse += mse
        self.running_entropy += -(edge_type_dists * np.log(edge_type_dists + 1e-10)).sum(axis=-1).mean()

        type_probs = edge_type_dists.mean(axis=tuple(range(edge_type_dists.ndim - 1)))
        edge_type_probs = edge_type_dists.mean(axis=tuple(range(edge_type_dists.ndim - 2)))
        if self.running_type_probs is not None:
            self.running_type_probs += type_probs
        else:
            self.running_type_probs = type_probs

        if self.running_edge_type_probs is not None:
            self.running_edge_type_probs += edge_type_probs
        else:
            self.running_edge_type_probs = edge_type_probs

    def log(self, epoch: int, tensorboard_tag: str = "training"):
        """
        Averages the accumulated metrics and logs them to the summary writer.
        @param tensorboard_tag: the group tag
        @param epoch: the current epoch
        """
        if self.num_iter == 0:
            self.logger.info("No iterations to log")
            return

        if self.summary_writer is not None:
            self.summary_writer.add_scalar(f"Loss/{tensorboard_tag}", self.running_loss / self.num_iter, epoch)
            self.summary_writer.add_scalar(f"Negative Log Likelihood/{tensorboard_tag}",
                                           self.running_nll / self.num_iter, epoch)
            self.summary_writer.add_scalar(f"KL-Divergence to Prior/{tensorboard_tag}",
                                           self.running_kl_div / self.num_iter, epoch)
            self.summary_writer.add_scalar(f"MSE/{tensorboard_tag}", self.running_mse / self.num_iter, epoch)
            self.summary_writer.add_scalar(f"Entropy of edge type predictions/{tensorboard_tag}",
                                           self.running_entropy / self.num_iter, epoch)

            if self.grad_norm is not None:
                self.summary_writer.add_scalar(f"Gradient Norm/{tensorboard_tag}", self.grad_norm, epoch)
            if self.running_type_probs is not None:
                self.summary_writer.add_histogram(f"Average type probabilities/{tensorboard_tag}",
                                                  self.running_type_probs / self.num_iter, epoch)
            if self.running_edge_type_probs is not None and self.plotting_args is not None:
                self.plotting_args.latent_edge_probs = self.running_edge_type_probs / self.num_iter
                if self.plotting_args.do_weight_sweep:
                    steps = np.linspace(0, 1, 10)
                    weights: List[float] = self.plotting_args.latent_edge_weight * 100 ** (steps * 2 - 1) / (
                                1 + 100 ** (steps * 2 - 1)).tolist()
                    for i, weight in enumerate(weights):
                        self.plotting_args.latent_edge_weight = weight
                        latent_edges_fig = visualize_graph(self.plotting_args)
                        self.summary_writer.add_figure(f"Predicted latent Graph sweep/{tensorboard_tag}",
                                                       latent_edges_fig, i)
                else:
                    latent_edges_fig = visualize_graph(self.plotting_args)
                    self.summary_writer.add_figure(f"Predicted latent Graph/{tensorboard_tag}", latent_edges_fig, epoch)

                latent_edges_hist = latent_edge_hist(self.running_edge_type_probs / self.num_iter)
                self.summary_writer.add_figure(f"Histogram of predicted latent edges/{tensorboard_tag}",
                                               latent_edges_hist, epoch)

        msg = (f"{tensorboard_tag} Epoch {epoch}: loss: {self.running_loss / self.num_iter:.2f} "
               f"Neg Log Likelihood: {self.running_nll / self.num_iter:.2f} "
               f"KL Divergence: {self.running_kl_div / self.num_iter:.2f} "
               f"MSE: {self.running_mse / self.num_iter:.2f} "
               f"Entropy of edge type predictions: {self.running_entropy / self.num_iter:.2f} ")
        if self.grad_norm is not None:
            msg += f"Gradient Norm: {self.grad_norm:.2f}"
        self.logger.info(msg)


def train(
        nri_module: NRIModule,
        training_set: Dataset,
        testing_set: Optional[Dataset] = None,
        criterion: Optional[ElboLoss] = None,
        feature_mask: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        tensorboard_logger: Optional[SummaryWriter] = None,
        num_epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.01,
        evaluate_every_k_epochs: int = 10,
        plotting_args: Optional[PlottingArgs] = None) -> NRIModule:
    """
    This function trains a given NRI model on a training set. During training, it is iteratively evaluated on the test set.
    The NRI module is trained to minimize the provided Elbo Loss. Furthermore, edge_index defines which edges are included
    in the encoder and decoder effectively limiting the possibly inferred latent edges.

    :param nri_module: NRI module to train
    :param training_set: Dataset to train on
    :param testing_set: Dataset to test on (optional, if no dataset is provided, there won't be evaluation)
    :param criterion: The loss function to use (defaults to ELBO with uniform prior)
    :param feature_mask: Optional mask to select features' predictions are included in the loss (defaults to all)
    :param edge_index: Edge indices to use for the NRI module (defaults to fully meshed)
    :param tensorboard_logger: Logging object to use (defaults to cli logger)
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
    if feature_mask is not None:
        feature_mask = feature_mask.bool().to(device=device)

    dataloader_train = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    criterion = criterion if criterion is not None else ElboLoss()
    optimizer = torch.optim.Adam(nri_module.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        # train
        nri_module.train()
        metrics = RunningMetrics(summary_writer=tensorboard_logger, plotting_args=plotting_args)
        for batch_ in tqdm(dataloader_train, desc=f"Epoch {epoch} training"):
            optimizer.zero_grad()

            batch: Tensor = batch_[0].to(device=device, dtype=torch.float32)
            predictions, edge_type_distributions = nri_module.forward(batch, edge_index)
            target = batch[:, 1:, :, :]  # target is next time step
            if feature_mask is not None:
                target = target[..., feature_mask]
                predictions = predictions[..., feature_mask]

            loss, nll, kl = criterion(predictions, target, edge_type_distributions, with_nll_kl=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nri_module.parameters(), max_norm=10.0)
            optimizer.step()

            with torch.no_grad():
                metrics.new_iter(
                    loss=loss.item(),
                    nll=nll.item(),
                    kl_div=kl.item(),
                    mse=((predictions - target) ** 2).sum(dim=-1).mean().item(),
                    edge_type_dists=edge_type_distributions.detach().cpu().numpy(),
                )
        metrics.grad_norm = sum(p.grad.norm().item() ** 2 for p in nri_module.parameters() if p.grad is not None) ** 0.5
        metrics.log(epoch=epoch, tensorboard_tag="training")

        # evaluate
        if epoch % evaluate_every_k_epochs == 0 and testing_set is not None:
            # do latent edge weight sweep on last evaluation
            is_last_eval = epoch + evaluate_every_k_epochs >= num_epochs
            plotting_args.do_weight_sweep = is_last_eval
            evaluate_nri_module(
                nri_module=nri_module,
                testing_set=testing_set,
                criterion=criterion,
                feature_mask=feature_mask,
                edge_index=edge_index,
                tensorboard_logger=tensorboard_logger,
                current_epoch=epoch,
                batch_size=batch_size,
                plotting_args=plotting_args
            )

    return nri_module


def evaluate_nri_module(
        nri_module: NRIModule,
        testing_set: Dataset,
        criterion: Optional[ElboLoss] = None,
        feature_mask: Optional[Tensor] = None,
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
    :param feature_mask: Optional mask to select features' predictions are included in the loss (defaults to all)
    :param edge_index: Edge indices to use for the NRI module
    :param tensorboard_logger: Logging object to use (defaults to None)
    :param current_epoch: Current epoch (if this is invoked during training - if not None)
    :param batch_size: Batch size (defaults to 64)
    :param plotting_args: args for creating plots (optional)
    """
    plotting_args = plotting_args or PlottingArgs(num_nodes=testing_set[0][0].shape[-2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nri_module.to(device=device, dtype=torch.float32)
    nri_module.eval()
    criterion = criterion if criterion is not None else ElboLoss()
    data_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=True)
    metrics = RunningMetrics(summary_writer=tensorboard_logger, plotting_args=plotting_args)

    with torch.no_grad():
        for batch_ in tqdm(data_loader, desc="Testing"):
            batch: Tensor = batch_[0].to(device=device, dtype=torch.float32)
            predictions, edge_type_distributions = nri_module.forward(batch, edge_index)
            target = batch[:, 1:, :, :]
            if feature_mask is not None:
                target = target[..., feature_mask]
                predictions = predictions[..., feature_mask]

            loss, nll, kl = criterion(predictions, target, edge_type_distributions, with_nll_kl=True)

            if loss.item() > max_loss_without_warning:
                warn_large_loss(predictions, target)

            metrics.new_iter(
                loss=loss.item(),
                nll=nll.item(),
                kl_div=kl.item(),
                mse=((predictions - target) ** 2).sum(dim=-1).mean().item(),
                edge_type_dists=edge_type_distributions.cpu().numpy(),
            )

        metrics.log(epoch=current_epoch, tensorboard_tag="testing")


@hydra.main(config_path="../hydra_configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    group = "nri"
    name = f"{cfg.nri.name}_{timestamp}_{uuid.uuid4().hex}"

    tensorboard_logger = SummaryWriter(os.path.join(LOGS_PATH, group, name))

    # prepare training data
    train_data = np.load(cfg.nri.train.training_dataset_path)
    test_data = np.load(cfg.nri.train.testing_dataset_path)
    train_dataset = TensorDataset(torch.from_numpy(train_data))
    test_dataset = TensorDataset(torch.from_numpy(test_data))

    # prepare plotting args
    env = grid2op.make(cfg.env.env_name)
    observation_space: GraphObservationSpace = instantiate(cfg.nri.dataset_creation.obs_space,
                                                           grid2op_observation_space=env.observation_space)
    edge_index = observation_space.to_gym(env.reset())[EDGE_INDEX]
    plotting_args = PlottingArgs(
        num_nodes=train_data.shape[-2],
        node_styles=get_node_styles(env, observation_space.__class__),
        powerline_edge_index=edge_index,
        latent_edge_weight=5.0,
        skip_last_edge_type=cfg.nri.model.skip_last,
    )

    # Prepare feature mask
    feature_mask = torch.from_numpy(get_feature_mask(observation_space, cfg.nri.train.features_to_predict))

    # prepare model
    nri_module: NRIModule = instantiate(cfg.nri.model, x_dim=train_data.shape[-1])
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
        feature_mask=feature_mask,
        tensorboard_logger=tensorboard_logger,
        num_epochs=cfg.nri.train.num_epochs,
        batch_size=cfg.nri.train.batch_size,
        learning_rate=cfg.nri.train.learning_rate,
        evaluate_every_k_epochs=cfg.nri.train.evaluate_every_k_epochs,
        plotting_args=plotting_args
    )

    save_path = Path(MODELS_PATH, group, name + ".pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(nri_module.state_dict(), save_path)
    save_edge_probs(nri_module, train_dataset, cfg.nri.train.batch_size,
                    save_path=Path(EDGE_PROBS_PATH, group, name + "_training.npy"))
    save_edge_probs(nri_module, test_dataset, cfg.nri.train.batch_size,
                    save_path=Path(EDGE_PROBS_PATH, group, name + "_testing.npy"))
    tensorboard_logger.close()


if __name__ == "__main__":
    main()
