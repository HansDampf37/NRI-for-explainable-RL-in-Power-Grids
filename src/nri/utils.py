import logging
from typing import Union, Tuple, Optional

import numpy as np
import numpy.typing as npt
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor, nn
from torch_geometric.utils import dense_to_sparse

from src.common.MLP import MLP
from src.common.env import G2OpGymEnv
from src.common.observation_space import GraphObservationSpace, EDGE_INDEX

logger = logging.getLogger(__name__)


def get_env(cfg: DictConfig, env_name: Optional[str] = None) -> G2OpGymEnv:
    """
    Creates a G2OpEnv wrapped in a monitor with fitting action and observation spaces from hydra config.

    :param cfg: The hydra config
    :param env_name: Optional override for the environment name
    :return: The environment
    """
    env: G2OpGymEnv = G2OpGymEnv(
        cfg.env.training_env.env_name if env_name is None else env_name,
        obs_space_creation=lambda e: instantiate(cfg.nri.dataset_creation.obs_space, grid2op_observation_space=e.observation_space),
        rule_config=cfg.env.rule_config,
        curriculum_level_settings=cfg.env.training_env.curriculum_level_settings,
    )
    return env


def fully_connected_edge_index(num_nodes: int, device: str = "cpu", self_loops: bool = False) -> Tensor:
    """
    Create an edge index representing a fully connected (directed) graph with num_nodes nodes.
    :param num_nodes: Number of nodes in the graph.
    :param self_loops: If true, create self-loops.
    :param device: Device to use.
    """
    senders, receivers = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing="ij")
    edge_index = torch.stack([senders.flatten(), receivers.flatten()], dim=0)
    if not self_loops:
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    return edge_index.to(device=device)


def fully_connected_edge_index_per_batch(batch: Tensor, device: Union[str, torch.device, int] = "cpu",
                                         self_loops: bool = False) -> Tensor:
    """
    Create an edge index representing batch of fully connected edge indices.
    :param batch: graph index per node.
    :param self_loops: If true, create self-loops.
    :param device: Device to use.
    """
    # batch: [N]  integer graph IDs
    num_graphs = int(batch.max()) + 1
    edge_indices = []
    for g in range(num_graphs):
        node_idx = (batch == g).nonzero(as_tuple=False).view(-1)
        n = node_idx.numel()
        if n == 0:
            continue
        adj = torch.ones((n, n), dtype=torch.bool, device=device)
        if not self_loops:
            adj.fill_diagonal_(False)
        edge_index_local, _ = dense_to_sparse(adj)
        edge_index_global = node_idx[edge_index_local]
        edge_indices.append(edge_index_global)
    return torch.cat(edge_indices, dim=1)


def get_priors(prob_graph_edges_exist: float, num_graph_edges: int, num_non_graph_edges: int,
               temperature: float = 0.2) -> Tuple[Tensor, Tensor]:
    """
    Creates the two prior distributions for graph edges and non-graph edges respectively while ensuring that the average
    distributions remains constant. The average distribution is defined as:

    num_graph_edges * p1 + num_non_graph_edges * p2 / (num_graph_edges + num_non_graph_edges)

    where p1 is the prior for graph edges and p2 for non-graph edges.
    @param prob_graph_edges_exist: Probability that a graph edge exists.
    @param num_graph_edges: Number of graph edges.
    @param num_non_graph_edges: Number of non-graph edges.
    @param temperature: ranges from 0 to num_non_graph_edges / num_graph_edges and indicates how many non graph edges should be predicted next to graph edges on average.
    @return: prior distribution for graph edges, prior distribution for non-graph edges
    """

    assert num_non_graph_edges > 0
    assert num_graph_edges >= 0
    assert 0 <= temperature < num_non_graph_edges / (num_graph_edges + 1)
    assert 0.0 <= prob_graph_edges_exist <= 1.0

    num_total_edges = num_graph_edges + num_non_graph_edges
    # Prior for true graph edges
    p1 = np.array([prob_graph_edges_exist, 1 - prob_graph_edges_exist], dtype=np.float64)
    # Average prior over all edges
    average_existence_prob = (1 + temperature) * num_graph_edges / num_total_edges
    p_hat = np.array([average_existence_prob, 1 - average_existence_prob], dtype=np.float64)
    # Solve for prior for non-graph edges
    p2 = (num_total_edges * p_hat - num_graph_edges * p1) / num_non_graph_edges

    p1 = np.clip(p1, 0, 1)
    p1 = p1 / np.sum(p1)
    p1 = p1.astype(np.float32)
    p2 = np.clip(p2, 0, 1)
    p2 = p2 / np.sum(p2)
    p2 = p2.astype(np.float32)

    epsilon = 1e-6
    assert np.all(-epsilon <= p1) and np.all(p1 <= 1 + epsilon) and np.isclose(np.sum(p1), 1.0, rtol=0.0, atol=1e-6), f"Prior for graph edges is not a probability distribution. {p1}, p_hat: {p_hat}, p2: {p2}, num_graph_edges: {num_graph_edges}, num_non_graph_edges: {num_non_graph_edges}, total: {num_total_edges}"
    assert np.all(-epsilon <= p2) and np.all(p2 <= 1 + epsilon) and np.isclose(np.sum(p2), 1.0, rtol=0.0, atol=1e-6), f"Prior for non graph edges is not a probability distribution. {p2}, p_hat: {p_hat}, p1: {p1}, num_graph_edges: {num_graph_edges}, num_non_graph_edges: {num_non_graph_edges}, total: {num_total_edges}"

    return Tensor(p1), Tensor(p2)


def prior_from_env(prob_graph_edge_exists: float, env: G2OpGymEnv, temperature: float = 0.2, num_edge_types: int = 2, verbose=True) -> Tensor:
    """
    Create prior distributions given the environment and existence probability for graph edges.
    These priors are used to condition the relation aware agents in their edge type predictions.

    :param prob_graph_edge_exists: the probability of latent dependencies on graph edges.
    :param env: The environment
    :param temperature: The amount of predicted edges according to the prior will be (1 + temperature) * num_graph_edges.
    :param num_edge_types: Number of edge types. Defaults to 2 (edge exists or doesn't exist).
    :param verbose: print extra explanatory or diagnostic information
    :return: prior distributions
    """
    obs_space: GraphObservationSpace = env.observation_space
    N = obs_space.num_nodes
    num_graph_edges = obs_space.max_num_edges
    num_non_graph_edges = N * (N - 1) // 2 - num_graph_edges
    prior_for_graph_edges, prior_for_non_graph_edges = get_priors(prob_graph_edge_exists, num_graph_edges, num_non_graph_edges, temperature)
    if verbose:
        logger.info(f"Prior for graph edges: {prior_for_graph_edges}, "
                    f"Prior for non graph edges: {prior_for_non_graph_edges}")
    powergrid_edge_index = torch.from_numpy(env.reset()[0][EDGE_INDEX])  # [2, E]
    all_edges = fully_connected_edge_index(N)  # [2, E']
    prior = get_prior_tensor(powergrid_edge_index, all_edges, prior_for_graph_edges, prior_for_non_graph_edges, num_edge_types=num_edge_types)
    return prior


def get_prior_tensor(graph_edges: Tensor, all_edges: Tensor, prior_for_graph_edges: Tensor,
                     prior_for_non_graph_edges: Tensor, num_edge_types: int = 2) -> Tensor:
    """
    Given edge indices for graph edges [2, E] and all considered edges [2, E'] return a tensor of shape [E', K] containing
    prior distribution for each considered edge in E'. If the edge exists as part of the graph it receives the distribution
    `prior_for_graph_edges`. Otherwise, its distribution is set to `prior_for_non_graph_edges`. For both cases the graph edge
    probability is distributed among the first K-1 classes
    @param graph_edges: Edge index for graph edges [2, E]
    @param num_edge_types: Number of edge types K
    @param all_edges: Edge index for all considered edges [2, E'] (typically fully connected)
    @param prior_for_graph_edges: prior distribution for graph edges [2,]
    @param prior_for_non_graph_edges: prior distribution for non-graph edges [2,]
    @return: prior distribution for all considered edges in E'
    """
    assert prior_for_graph_edges.shape == prior_for_non_graph_edges.shape
    all_edges = all_edges.T
    E, _ = all_edges.shape
    mask = torch.zeros((E,), dtype=torch.bool)
    reversed_graph_edges = graph_edges[[1, 0], :]
    for e in range(graph_edges.shape[1]):
        mask = torch.logical_or(torch.all(all_edges == graph_edges[:, e].unsqueeze(0), dim=1), mask)
        mask = torch.logical_or(torch.all(all_edges == reversed_graph_edges[:, e].unsqueeze(0), dim=1), mask)

    prior = torch.zeros((E, num_edge_types), dtype=torch.float32)
    prior[mask, :num_edge_types - 1] = prior_for_graph_edges[0] / (num_edge_types - 1)
    prior[mask, -1] = prior_for_graph_edges[1]
    prior[torch.logical_not(mask), :num_edge_types - 1] = prior_for_non_graph_edges[0] / (num_edge_types - 1)
    prior[torch.logical_not(mask), -1] = prior_for_non_graph_edges[1]
    return prior


class Node2Edge(nn.Module):
    """
   Implements the Node to edge message passing by Kipf et al.
   """

    def __init__(self, x_dim: int, hidden_dim: int, e_dim: int, dropout_prob=0.):
        # maps 2 node embeddings to one edge embedding
        super(Node2Edge, self).__init__()
        self.psi = MLP(
            input_features=x_dim * 2,
            hidden_dim=hidden_dim,
            output_features=e_dim,
            dropout_prob=dropout_prob
        )

    def forward(self, x: Tensor, edge_index: Union[np.array, Tensor]) -> Tensor:
        """
        Aggregates node features into edge features.
        :param x: node features [... N, X_dim]
        :param edge_index: adjacency information [2, E]
        :return: edge features [..., E, E_dim]
        """
        senders = edge_index[0]  # j
        receivers = edge_index[1]  # i
        # gather x_j and x_i per edge
        x_j = x[..., senders, :]  # [B, T, E, x_dim]
        x_i = x[..., receivers, :]  # [B, T, E, x_dim]
        node_aggr = torch.cat([x_i, x_j], dim=-1)
        return self.psi(node_aggr)


class Edge2Node(nn.Module):
    """
    Implements the edge to node message passing by Kipf et al. in their encoder
    """

    def __init__(self, e_dim: int, hidden_dim: int, x_dim: int, dropout_prob=0.):
        """
        Constructor.
        :param e_dim: edge features in the input
        :param hidden_dim: hidden dimension for the MLP
        :param x_dim: node features in the output of this module
        :param dropout_prob: dropout probability
        """
        # maps adjacent edge embeddings onto a new node embedding
        super(Edge2Node, self).__init__()
        self.phi = MLP(
            input_features=e_dim,
            hidden_dim=hidden_dim,
            output_features=x_dim,
            dropout_prob=dropout_prob
        )

    def forward(self, e: Tensor, edge_index: Union[np.array, Tensor]) -> Tensor:
        """
        Aggregates edge features into node features.

        :param e: edge features [..., E, E_dim]
        :param edge_index: adjacency information [2, E]
        :return: node features [..., N, X_dim]
        """
        receivers = edge_index[1]
        # aggregate edge messages into nodes by receiver index
        N = int(edge_index.max().item()) + 1
        target_shape = list(e.size())
        target_shape[-2] = N
        agg = e.new_zeros(tuple(target_shape))
        # index_reduce_ to average messages into receivers rows
        agg.index_reduce_(dim=-2, index=receivers, source=e, reduce="mean")
        return self.phi(agg)


def edge_membership_mask(super_edge_set: torch.Tensor, sub_edge_set: torch.Tensor) -> torch.Tensor:
    """
    Given two edge indices returns a mask indicating which elements from the superset are also included in the subset
    @param super_edge_set: the superset
    @param sub_edge_set: the subset
    @return: a mask
    """
    num_nodes = super_edge_set.max() + 1
    a = super_edge_set[0] * num_nodes + super_edge_set[1]  # [E]
    b = sub_edge_set[0] * num_nodes + sub_edge_set[1]  # [E_sub]
    return torch.isin(a, b)


class EdgeNode2Node(nn.Module):
    """
    Implements the edge to node message passing by Kipf et al. in their decoder.
    In contrast to the simple Edge2Node module here node features do not only depend on adjacent edge features but
    also on previous node features.
    """

    def __init__(self, x_dim: int, e_dim: int, hidden_dim: int, x_out_dim: int, dropout_prob=0.):
        """
        Constructor.
        :param x_dim: node features in the input
        :param e_dim: edge features in the input
        :param hidden_dim: hidden dimension for the MLP
        :param x_out_dim: node features in the output of this module
        :param dropout_prob: dropout probability
        """
        # maps adjacent edge embeddings + node embeddings onto a new node embedding
        super(EdgeNode2Node, self).__init__()
        self.phi = MLP(
            input_features=e_dim + x_dim,
            hidden_dim=hidden_dim,
            output_features=x_out_dim,
            dropout_prob=dropout_prob
        )

    def forward(self, x: Tensor, e: Tensor, edge_index: Union[np.array, Tensor]) -> Tensor:
        """
        Aggregates node + edge features into node features.

        :param x: node features [..., N, X_dim]
        :param e: edge features [..., E, E_dim]
        :param edge_index: adjacency information [2, E]
        :return: node features [..., N, X_dim]
        """
        receivers = edge_index[1]
        # aggregate edge messages into nodes by receiver index
        N = x.size(-2)
        target_shape = list(e.size())
        target_shape[-2] = N
        agg = e.new_zeros(tuple(target_shape))
        # index_reduce_ to average messages into receivers rows
        agg.index_reduce_(dim=-2, index=receivers, source=e, reduce="mean")
        return self.phi(torch.cat([agg, x], dim=-1))


def warn_large_loss(predictions: Tensor, target: Tensor) -> npt.NDArray[np.float32]:
    with torch.no_grad():
        per_feature_mse = ((target - predictions).pow(2)).mean(dim=(0, 1, 2)).cpu().numpy()
        logger.warning(f"Large Loss: {per_feature_mse.mean()}\n"
                       f"The loss per feature is:\n{per_feature_mse}")
        return per_feature_mse
