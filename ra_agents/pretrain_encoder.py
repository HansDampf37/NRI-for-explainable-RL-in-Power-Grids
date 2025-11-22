"""
Pretrains an encoder on observations obtained by resetting the given environment. Since there is no training signal
from a downstream agent this pretraining trains the encoder solely on the objective of reproducing the prior (the
prior is a per edge distribution that the posterior distribution is pushed towards) given various observations sampled
from the environment.
"""
import uuid
from datetime import datetime
from pathlib import Path
from typing import Union, Optional, List, Tuple

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

from common.constants import MODELS_PATH, LOGS_PATH, logger
from common.env import G2OpGymEnv
from common.graph_structured_observation_space import NODES, EDGE_INDEX, EDGE_MASK
from nri.utils import prior_from_env
from .Encoder import Encoder
from .graphormer.GraphormerEncoder import GraphormerNRIEncoder
from .utils import get_env

_eps = 0.00001

class GraphDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(GraphDataset, self).__init__(root=".")
        self.data, self.slices = self.collate(data_list)


@hydra.main(config_path="../hydra_configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """
    Pretrains a graphormer encoder.
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

    # setup logging
    log_path = Path(LOGS_PATH, group, name)
    tensorboard_logger = SummaryWriter(log_path)
    logger.info(f"Logging to {log_path}")

    # collect data
    env: G2OpGymEnv = get_env(cfg)
    prior = prior_from_env(cfg.rl.model.prior_for_graph_edges_existing, env)
    ds: GraphDataset = create_dataset(env, prior, cfg.rl.train.pretrain_encoder.ds_size)
    eval_ds: GraphDataset = create_dataset(get_env(cfg), prior, cfg.rl.train.pretrain_encoder.test_ds_size)

    # train encoder
    if cfg.rl.model.use_graphormer:
        encoder = GraphormerNRIEncoder(
            x_dim=env.observation_space.x_dim,
            hidden_dim=cfg.rl.model.features_extractor_kwargs.hidden_dim,
            num_layers=cfg.rl.model.features_extractor_kwargs.num_layers,
            num_edge_types=cfg.rl.model.features_extractor_kwargs.num_edge_types,
            max_degree=cfg.rl.model.features_extractor_kwargs.max_degree,
            max_path_distance=cfg.rl.model.features_extractor_kwargs.max_path_distance
        )
    else:
        encoder = Encoder(
            x_dim=env.observation_space.x_dim,
            hidden_dim=cfg.rl.model.features_extractor_kwargs.hidden_dim,
            num_edge_types=cfg.rl.model.features_extractor_kwargs.num_edge_types,
        )

    train(
        encoder=encoder,
        ds=ds,
        testing_ds=eval_ds,
        tensorboard_logger=tensorboard_logger,
        evaluate_every_k_epochs=cfg.rl.train.pretrain_encoder.evaluate_every_k_epochs,
        batch_size=cfg.rl.train.pretrain_encoder.batch_size,
        num_epochs=cfg.rl.train.pretrain_encoder.num_epochs,
        lr=cfg.rl.train.pretrain_encoder.learning_rate,
    )

    # save encoder
    save_path = Path(MODELS_PATH, group, name + ".pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), save_path)
    return encoder


def train(
        encoder: Union[GraphormerNRIEncoder, Encoder],
        ds: GraphDataset,
        testing_ds: Optional[GraphDataset] = None,
        evaluate_every_k_epochs: int = 10,
        tensorboard_logger: Optional[SummaryWriter] = None,
        batch_size: int = 32,
        num_epochs: int = 100,
        lr: float = 0.005) -> List[Tuple[float, float]]:
    """
    Pretrains an encoder on observations obtained by observing the given environment. Since there is no training signal
    from a downstream agent this pretraining trains the encoder solely on the objective of reproducing the prior (the
    prior is a per edge distribution that the posterior distribution is pushed towards) given various observations sampled
    from the environment.

    @param encoder: the encoder that should predict graph structures
    @param ds: the dataset to train on
    @param testing_ds: the dataset to test on
    @param evaluate_every_k_epochs: the number of training epochs between one evaluation
    @param tensorboard_logger: Optional Tensorboard logger.
    @param batch_size: the batch size to use
    @param num_epochs: the number of epochs to train
    @param lr: the learning rate
    @return: the loss curves (training, testing)
    """
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    loss_per_episode = []
    pbar = tqdm(total=num_epochs)
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_ in loader:
            x = batch_.x
            edge_index = batch_.edge_index
            batch = batch_.batch
            y = batch_.y

            optimizer.zero_grad()

            # compute loss
            edge_logits = encoder.forward(x, batch=batch, powerline_edge_index=edge_index)
            edge_log_probs = F.log_softmax(edge_logits, dim=-1)
            edge_probs = edge_log_probs.exp()
            loss = (0.9 * (edge_probs * (edge_log_probs - torch.log(y + _eps))).sum(dim=-1).mean() +
                    0.1 * (y * (torch.log(y + _eps) - edge_log_probs)).sum(dim=-1).mean())
            total_loss += loss.item()

            # update weights
            loss.backward()
            clip_grad_norm_(encoder.parameters(), 1)
            optimizer.step()

        # evaluate occasionally
        if epoch % evaluate_every_k_epochs == 0 and testing_ds is not None:
            testing_loss = evaluate(encoder, testing_ds, batch_size)
        else:
            testing_loss = None

        training_loss = total_loss / len(loader)
        loss_per_episode.append((training_loss, testing_loss))

        # logging results for this epoch
        grads_abs = torch.cat([
            p.grad.abs().view(-1)
            for p in encoder.parameters()
            if p.requires_grad and p.grad is not None
        ])
        if tensorboard_logger is not None:
            tensorboard_logger.add_scalar("max-logit", edge_logits.max(), epoch)
            tensorboard_logger.add_histogram("grads_abs/global", grads_abs, epoch)
            tensorboard_logger.add_scalar("loss/train", training_loss, epoch)
            if testing_loss is not None:
                tensorboard_logger.add_scalar("loss/eval", testing_loss, epoch)

        message = f"Epoch {epoch}, " \
             f"Loss: {training_loss:.8f}, " \
             f"Max Grads: {grads_abs.max():.2f}, " \
             f"logits max: {edge_logits.max():.2f}"
        if testing_loss is not None:
            message += f", Evaluation loss: {testing_loss:.8f}"
        pbar.update(1)
        pbar.desc = message

    return loss_per_episode

def evaluate(
        encoder: Union[GraphormerNRIEncoder, Encoder],
        eval_ds: GraphDataset,
        batch_size: int = 32) -> float:
    """
    Evaluate an encoder on the evaluation dataset.

    @param encoder: The encoder
    @param eval_ds: the dataset to evaluate on
    @param batch_size: the batch size to use
    @return: the evaluation loss
    """
    eval_loss = 0
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_ in eval_loader:
            x = batch_.x
            edge_index = batch_.edge_index
            batch = batch_.batch
            y = batch_.y

            edge_logits = encoder.forward(x, batch=batch, powerline_edge_index=edge_index)
            edge_log_probs = F.log_softmax(edge_logits, dim=-1)
            edge_probs = edge_log_probs.exp()
            loss = (edge_probs * (edge_log_probs - torch.log(y + _eps))).sum(dim=-1).mean()
            eval_loss += loss.item()

    return eval_loss / len(eval_loader)


def create_dataset(env: G2OpGymEnv, prior: Tensor, ds_size: int) -> GraphDataset:
    """
    Creates a dataset for pretraining the encoder. Inputs are various observations sampled from the environment.
    As a target we use the given prior Tensor.
    :param env: The environment to sample observations from
    :param prior: the prior
    :param ds_size: the dataset size
    :return: the dataset
    """
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
