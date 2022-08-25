from typing import *
from typing import Dict

import numpy as np
import networkx as nx
from numpy import ndarray

from solvers.callbacks.state_extractor.base import StateExtractor
from solvers.callbacks.separators.cycle import CycleSeparator
from solvers.cplex_api import UserCutCallback
from problems.maxcut import MaxcutProblem
from utils import nodes2edge
from constant import TOLERANCE


class CycleStateExtractor(StateExtractor):
    def __init__(
        self, separator: CycleSeparator, padding=False, *args, **kwds
    ) -> None:
        super().__init__(separator, padding, *args, **kwds)

    def initialize_original_graph(
        self, problem: MaxcutProblem, var2idx: Dict[Tuple[int, int], int], **kwargs
    ) -> None:
        graph = problem.graph
        max_weight = max([graph.edges[edge]["weight"] for edge in graph.edges])

        self.ori_node_feature = []
        self.ori_edge_index = [[], []]
        self.ori_edge_weight = []
        self.ori_edge_index_list = []

        for node in graph.nodes:
            self.ori_node_feature.append([len(graph.adj[node])])
            for neighbor in graph.adj[node]:
                self.ori_edge_index[0].append(node)
                self.ori_edge_index[1].append(node)
                self.ori_edge_weight.append(
                    graph.adj[node][neighbor]["weight"] / max_weight
                )
                self.ori_edge_index_list.append(var2idx[nodes2edge(node, neighbor)])

        self.ori_node_feature = np.asarray(self.ori_node_feature)
        self.ori_edge_index = np.asarray(self.ori_edge_index)
        self.ori_lens = np.array(
            [self.ori_node_feature.shape[0], self.ori_edge_index.shape[1]]
        )

        if self.padding:
            assert (
                    self.init_config.ori_nEdges >= self.ori_edge_index.shape[1]
            ), "ori_nEdges {} < ori_edge_index {}".format(
                self.init_config.ori_nEdges, self.ori_edge_index.shape[1]
            )
            self.ori_edge_index = np.concatenate(
                [
                    self.ori_edge_index,
                    np.zeros(
                        (2, self.init_config.ori_nEdges - self.ori_edge_index.shape[1])
                    ),
                ],
                axis=1,
            )

    def get_support_graph_representation(self, support_graph: nx.Graph) -> Dict[str, ndarray]:
        node_feature = np.array(
            [[len(support_graph.adj[node])] for node in support_graph.nodes]
        )
        edge_feature = []
        edge_index = [[], []]

        node_label = {}
        for node in support_graph.nodes:
            node_label[node] = len(node_label)
        support_graph = nx.relabel_nodes(support_graph, node_label)

        for node in support_graph.nodes:
            for neighbor in support_graph.adj[node]:
                edge_index[0].append(node)
                edge_index[1].append(neighbor)
                edge_feature.append([support_graph.adj[node][neighbor]["weight"]])

        edge_index = np.asarray(edge_index)
        edge_feature = np.asarray(edge_feature)
        lens = np.asarray([node_feature.shape[0], edge_feature.shape[0]])

        if self.padding:
            assert (
                    self.init_config.sup_nNodes >= node_feature.shape[0]
            ), "sup_nNodes {} < node feature {}".format(
                self.init_config.sup_nNodes, node_feature.shape[0]
            )
            assert (
                    self.init_config.sup_nEdges >= edge_index.shape[1]
            ), "sup_nEdges {} < edge index {}".format(
                self.init_config.sup_nEdges, edge_index.shape[1]
            )
            assert (
                    self.init_config.sup_nEdges >= edge_feature.shape[0]
            ), "sup_nEdges {} < edge feature {}".format(
                self.init_config.sup_nEdges, edge_feature.shape[0]
            )
            node_feature = np.concatenate(
                [
                    node_feature,
                    np.zeros(
                        (
                            self.init_config.sup_nNodes - node_feature.shape[0],
                            self.init_config.sup_node_dim,
                        )
                    ),
                ]
            )
            edge_index = np.concatenate(
                [
                    edge_index,
                    np.zeros((2, self.init_config.sup_nEdges - edge_index.shape[1])),
                ],
                axis=1,
            )
            edge_feature = np.concatenate(
                [
                    edge_feature,
                    np.zeros(
                        (
                            self.init_config.sup_nEdges - edge_feature.shape[0],
                            self.init_config.sup_edge_dim,
                        )
                    ),
                ]
            )

        return dict(sup_node_feature=node_feature, sup_edge_index=edge_index, sup_edge_feature=edge_feature,
                    sup_lens=lens)

    def get_original_graph_representation(self, solution: np.array, lb: np.array, ub: np.array) -> Dict[str, np.array]:
        ori_edge_feature = []
        for idx, edge_idx in enumerate(self.ori_edge_index_list):
            ori_edge_feature.append(
                [
                    self.ori_edge_weight[idx],
                    solution[edge_idx],
                    lb[edge_idx],
                    ub[edge_idx],
                ]
            )

        ori_edge_feature = np.asarray(ori_edge_feature)
        if self.padding:
            assert (
                    self.init_config.ori_nEdges >= ori_edge_feature.shape[0]
            ), "ori_nEdges {} < ori_edge_feature {}".format(
                self.init_config.ori_nEdges, ori_edge_feature.shape[0]
            )
            ori_edge_feature = np.concatenate(
                [
                    ori_edge_feature,
                    np.zeros(
                        (
                            self.init_config.ori_nEdges - ori_edge_feature.shape[0],
                            ori_edge_feature.shape[1],
                        )
                    ),
                ]
            )

        return dict(
            ori_node_feature=self.ori_node_feature,
            ori_edge_index=self.ori_edge_index,
            ori_edge_feature=ori_edge_feature,
            ori_lens=self.ori_lens,
        )

    def get_state_representation(
        self, callback: UserCutCallback
    ) -> Tuple[Dict[str, np.array], nx.Graph]:
        solution = np.asarray(callback.get_values())
        support_graph = self.separator.create_support_graph(solution)
        sup_representation = self.get_support_graph_representation(support_graph)

        lb: np.array = np.asarray(callback.get_lower_bounds())
        ub: np.array = np.asarray(callback.get_upper_bounds())
        ori_representation = self.get_original_graph_representation(solution, lb, ub)

        processed_leaves: int = callback.get_num_nodes()
        remain_leaves: int = callback.get_num_remaining_nodes()
        total_leaves: int = processed_leaves + remain_leaves
        gap = callback.get_MIP_relative_gap()
        gap = gap if gap < 1 else 1
        obj = callback.get_objective_value()

        statistic = [
            callback.get_current_node_depth() / self.instance_size,
            callback.has_incumbent(),
            gap,
            sum(np.where(abs(solution - 1) < TOLERANCE, 1, 0))
            / self.instance_size,
            min(1, callback.get_cutoff() / obj)
            if abs(obj) > 1e-4 else 0,
            remain_leaves / total_leaves if total_leaves > 0 else 0,
            processed_leaves / total_leaves if total_leaves > 0 else 0,
            np.sum(lb) / self.instance_size,
            (lb.shape[0] - np.sum(lb)) / solution.shape[0],
            np.sum(ub) / solution.shape[0],
            (ub.shape[0] - np.sum(ub)) / solution.shape[0],
            sum(np.where(abs(lb - ub) < TOLERANCE, 1, 0)) / solution.shape[0],
        ]

        state: Dict[str, np.array] = {"statistic": np.asarray(statistic)}

        for representation in [sup_representation, ori_representation]:
            for key, value in representation.items():
                state[key] = value

        return state, support_graph
