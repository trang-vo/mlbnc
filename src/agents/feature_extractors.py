from typing import *

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, dense_mincut_pool, global_mean_pool, GINEConv
from torch_geometric.utils import to_dense_adj, to_dense_batch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def preprocess_graph_representation(
    node_feature: torch.Tensor,
    edge_index: torch.Tensor,
    edge_feature: torch.Tensor,
    lens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Keys, values of observation
    node_feature: (batch, num_nodes, node_dim)
    edge_index: (batch, 2, num_edges)
    edge_feature: (batch, num_edges, edge_dim)
    lens: (batch, 1, 2)
    """
    lens = lens.long()
    n_nodes = []
    n_edges = []
    for i in range(len(lens)):
        n_nodes.append(lens[i][0])
        n_edges.append(lens[i][1])

    node_fea_batch = torch.cat(
        [node_feature[i][: n_nodes[i]] for i in range(len(node_feature))]
    )
    nNodes_cumsum = torch.cat(
        [torch.Tensor([0]), torch.cumsum(torch.Tensor(n_nodes).int(), dim=0)[:-1]]
    ).to(torch.int64)
    edge_index_batch = torch.cat(
        [
            edge_index[i][:, : n_edges[i]] + nNodes_cumsum[i]
            for i in range(len(edge_index))
        ],
        dim=1,
    ).long()
    edge_fea_batch = torch.cat(
        [edge_feature[i][: n_edges[i]] for i in range(len(edge_feature))]
    )
    batch_index = torch.cat(
        [torch.Tensor([i] * n_nodes[i]) for i in range(len(n_nodes))]
    ).to(torch.int64)

    return node_fea_batch, edge_index_batch, edge_fea_batch, batch_index


class GNNGraphExtractor(nn.Module):
    def __init__(
        self,
        node_dim: int,
        hidden_size: int,
        num_layers: int,
        n_clusters: int = 2,
        dropout: float = 0,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()

        self.pre_mp = nn.Linear(node_dim, hidden_size)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GraphConv(hidden_size, hidden_size))
            self.bns.append(nn.BatchNorm1d(hidden_size))

        self.pool = nn.Linear(hidden_size, n_clusters)

        self.dropout = dropout
        self.output_size = hidden_size
        self.device = device

    def forward(
        self, x: np.array, edge_index: np.array, edge_feature: np.array, lens: np.array
    ):
        x, edge_index, edge_feature, batch = preprocess_graph_representation(
            x, edge_index, edge_feature, lens
        )
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_feature = edge_feature.to(self.device)
        batch = batch.to(self.device)

        x = self.pre_mp(x)

        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index, edge_feature)
            x = self.bns[i](x)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # mincut pooling
        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch)

        s = self.pool(x)
        x, adj, mc_loss, o_loss = dense_mincut_pool(x, adj, s, mask)
        x = F.leaky_relu(x)

        # get graph embedding, size x = (batch_size, hidden_size)
        x = x.mean(dim=1)
        x = F.normalize(x)

        return x


class GINEGraphExtractor(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.pre_mp = nn.Linear(node_dim, hidden_size)
        self.pre_edge_feature = nn.Linear(edge_dim, hidden_size)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for l in range(self.num_layers):
            layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.convs.append(GINEConv(layer))
            self.bns.append(nn.BatchNorm1d(hidden_size))
        self.post_mp = nn.Linear(hidden_size, hidden_size)

        self.output_size = hidden_size
        self.device = device

    def forward(
        self, x: np.array, edge_index: np.array, edge_feature: np.array, lens: np.array
    ):
        x, edge_index, edge_feature, batch = preprocess_graph_representation(
            x, edge_index, edge_feature, lens
        )
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_feature = edge_feature.to(self.device)
        batch = batch.to(self.device)

        x = self.pre_mp(x)
        edge_feature = self.pre_edge_feature(edge_feature)
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index, edge_feature)
            x = self.bns[i](x)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        x = F.normalize(x)

        return x


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout: float = 0,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes) - 2):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.dropout = dropout
        self.output_size = output_size

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout)

        x = F.normalize(x)
        return x


class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        sup_feature_extractor: nn.Module,
        ori_feature_extractor: nn.Module,
        statistic_extractor: nn.Module,
        features_dim: int = 1,
        **kwargs,
    ):
        super().__init__(observation_space, features_dim)
        self.device = kwargs["device"]
        self.sup_feature_extractor = sup_feature_extractor.to(self.device)
        self.ori_feature_extractor = ori_feature_extractor.to(self.device)
        self.statistic_extractor = statistic_extractor.to(self.device)

        self._features_dim = (
            self.sup_feature_extractor.output_size
            + self.ori_feature_extractor.output_size
            + self.statistic_extractor.output_size
        )

    def forward(self, observations):
        sup_vec = self.sup_feature_extractor(
            observations["sup_node_feature"],
            observations["sup_edge_index"],
            observations["sup_edge_feature"],
            observations["sup_lens"],
        )

        ori_vec = self.ori_feature_extractor(
            observations["ori_node_feature"],
            observations["ori_edge_index"],
            observations["ori_edge_feature"],
            observations["ori_lens"],
        )

        statistic_vec = self.statistic_extractor(observations["statistic"])

        features = [sup_vec, ori_vec, statistic_vec]
        x = torch.cat(features, dim=1)

        return x


class EvalFeatureExtractor(nn.Module):
    def __init__(
        self,
        sup_feature_extractor: nn.Module,
        ori_feature_extractor: nn.Module,
        statistic_extractor: nn.Module,
    ) -> None:
        super().__init__()

        self.sup_feature_extractor = sup_feature_extractor
        self.ori_feature_extractor = ori_feature_extractor
        self.statistic_extractor = statistic_extractor

    def forward(self, observations):
        sup_vec = self.sup_feature_extractor(
            observations["sup_node_feature"],
            observations["sup_edge_index"],
            observations["sup_edge_feature"],
            observations["sup_lens"],
        )

        ori_vec = self.ori_feature_extractor(
            observations["ori_node_feature"],
            observations["ori_edge_index"],
            observations["ori_edge_feature"],
            observations["ori_lens"],
        )

        statistic_vec = self.statistic_extractor(observations["statistic"])

        features = [sup_vec, ori_vec, statistic_vec]
        x = torch.cat(features, dim=1)

        return x
