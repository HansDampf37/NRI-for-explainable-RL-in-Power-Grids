"""
Pretrains an encoder on observations obtained by resetting the given environment. Since there is no training signal
from a downstream agent this pretraining trains the encoder solely on the objective of reproducing the prior (the
prior is a per edge distribution that the posterior distribution is pushed towards) given various observations sampled
from the environment.
"""
import uuid
from datetime import datetime
from pathlib import Path
from typing import Union, Optional, List

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from common import NODES, EDGE_INDEX, MODELS_PATH, LOGS_PATH, G2OpGymEnv, logger, EDGE_MASK
from nri import prior_from_env
from nri.agent import Encoder
from nri.agent.graphormer.GraphormerEncoder import GraphormerNRIEncoder
from nri.agent.ppo.train_RAPPO import get_env


class GraphDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(GraphDataset, self).__init__(root=".")
        self.data, self.slices = self.collate(data_list)


@hydra.main(config_path="../../hydra_configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """
    Pretrains and returns a graphormer encoder.
    :param cfg: the hydra config
    :return: the graphormer encoder
    """
    print(OmegaConf.to_yaml(cfg))
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    group = "rl/encoder/"
    name_suffix = cfg.rl.model.name_suffix
    if name_suffix is None:
        name = f"graphormer_{timestamp}_{uuid.uuid4().hex}"
    else:
        name = f"graphormer_{timestamp}_{name_suffix}_{uuid.uuid4().hex}"

    # train encoder
    log_path = Path(LOGS_PATH, group, name)
    tensorboard_logger = None #SummaryWriter(log_path)
    logger.info(f"Logging to {log_path}")

    env: G2OpGymEnv = get_env(cfg)
    encoder = GraphormerNRIEncoder(
        x_dim=env.observation_space.x_dim,
        hidden_dim=cfg.rl.model.features_extractor_kwargs.hidden_dim,
        num_layers=cfg.rl.model.features_extractor_kwargs.num_layers,
        num_edge_types=cfg.rl.model.features_extractor_kwargs.num_edge_types,
        max_degree=cfg.rl.model.features_extractor_kwargs.max_degree,
        max_path_distance=cfg.rl.model.features_extractor_kwargs.max_path_distance
    )
    prior = prior_from_env(cfg.rl.model.prior_for_graph_edges_existing, env)

    ds: GraphDataset = create_dataset(env, prior, 1000)
    train(encoder, ds, tensorboard_logger)

    # save encoder
    save_path = Path(MODELS_PATH, group, name + ".pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), save_path)
    return encoder


def train(
        encoder: Union[GraphormerNRIEncoder, Encoder],
        ds: GraphDataset,
        tensorboard_logger: Optional[SummaryWriter] = None,
        batch_size: int = 32,
        num_epochs: int = 100,
        lr: float = 0.005) -> List[float]:
    """
    Pretrains an encoder on observations obtained by resetting the given environment. Since there is no training signal
    from a downstream agent this pretraining trains the encoder solely on the objective of reproducing the prior (the
    prior is a per edge distribution that the posterior distribution is pushed towards) given various observations sampled
    from the environment.

    @param encoder: the encoder that should predict graph structures
    @param ds: the dataset
    @param tensorboard_logger: Optional Tensorboard logger.
    @param batch_size: the batch size to use
    @param num_epochs: the number of epochs to train
    @param lr: the learning rate
    @return: the training loss curve
    """
    # pretrain encoder to predict prior for every observation
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    eps = 0.00001
    loss_per_episode = []
    for epoch in range(num_epochs): # tqdm(range(num_epochs), f"Training {num_epochs} epochs"):
        total_loss = 0.0

        for batch_ in loader:
            x = batch_.x
            edge_index = batch_.edge_index
            batch = batch_.batch
            y = batch_.y

            optimizer.zero_grad()

            edge_logits = encoder.forward(x, batch=batch, powerline_edge_index=edge_index)
            edge_probs = F.softmax(edge_logits, dim=-1)

            loss = (edge_probs * (torch.log(edge_probs + eps) - torch.log(y + eps))).sum(dim=-1).mean()
            loss.backward()
            clip_grad_norm_(encoder.parameters(), 1)
            optimizer.step()
            total_loss += loss.item()

        loss_per_episode.append(total_loss)
        grads_abs = torch.cat([
            p.grad.abs().view(-1)
            for p in encoder.parameters()
            if p.requires_grad and p.grad is not None
        ])

        if tensorboard_logger is not None:
            tensorboard_logger.add_scalar("max-logit", edge_logits.max(), epoch)
            tensorboard_logger.add_histogram("grads_abs/global", grads_abs, epoch)
            tensorboard_logger.add_scalar("loss", total_loss, epoch)
        else:
            logger.info(f"Epoch {epoch}, "
                        f"Loss: {total_loss / len(ds):.8f}, "
                        f"Max Grads: {grads_abs.max():.2f}, "
                        f"logits max: {edge_logits.max():.2f}, ")

    return loss_per_episode


def create_dataset(env: G2OpGymEnv, prior: Tensor, ds_size: int) -> GraphDataset:
    """
    Creates a dataset for pretraining the encoder. Inputs are various observations sampled from the environment.
    As a target we use the given prior Tensor.
    :param env: The environment to sample observations from
    :param prior: the prior
    :param ds_size: the dataset size
    :return: the dataset
    """
    # create dataset
    data_list = []
    pbar = tqdm(range(ds_size), desc="Create dataset")
    while len(data_list) < ds_size:
        done = False
        observation, _ = env.reset()
        while not done and len(data_list) < ds_size:
            x = Tensor(observation[NODES])
            edge_mask = Tensor(observation[EDGE_MASK]).bool()
            edge_index = Tensor(observation[EDGE_INDEX]).long()[:, edge_mask]
            data = Data(x=x, edge_index=edge_index, y=prior)
            data_list.append(data)
            pbar.update(1)
            observation, _, done, _, _ = env.do_nothing()

    return GraphDataset(data_list)


if __name__ == "__main__":
    main()
