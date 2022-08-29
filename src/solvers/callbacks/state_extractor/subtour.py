from typing import *

import networkx as nx
import numpy as np

from solvers.callbacks.state_extractor.base import StateExtractor
from problems.tsp import TSPProblem
from solvers.callbacks.separators.subtour import SubtourSeparator
from solvers.cplex_api import UserCutCallback
from constant import TOLERANCE
from utils import nodes2edge
from config import EnvConfig


class SubtourStateExtractor(StateExtractor):
    def __init__(
        self, separator: SubtourSeparator, padding=False, *args, **kwds
    ) -> None:
        super().__init__(separator, padding, *args, **kwds)

    def initialize_original_graph(
        self, problem: TSPProblem, var2idx: Dict[Tuple[int, int], int], k=10
    ) -> None:
        """Create a graph representation involving (node feature, edge feature, edge index) for the original graph.
        If the graph is completed, use k-nearest neighbors graph."""
        k_nearest_neighbors: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
        for node in problem.graph.nodes:
            k_nearest_neighbors[node] = sorted(
                problem.graph.adj[node].items(), key=lambda e: e[1]["weight"]
            )[:k]

        max_weight: int = max(
            [problem.graph.edges[edge]["weight"] for edge in problem.graph.edges]
        )
        knn_graph = nx.Graph()
        for node in k_nearest_neighbors:
            for neighbor, node_dict in k_nearest_neighbors[node]:
                knn_graph.add_edge(
                    node, neighbor, weight=node_dict["weight"] / max_weight
                )

        self.ori_node_feature = []
        self.ori_edge_index = [[], []]
        self.ori_edge_weight = []
        self.ori_edge_index_list = []

        for node in knn_graph:
            self.ori_node_feature.append([len(knn_graph.adj[node])])
            for neighbor in knn_graph.adj[node]:
                self.ori_edge_index[0].append(node)
                self.ori_edge_index[1].append(neighbor)
                self.ori_edge_weight.append(knn_graph[node][neighbor]["weight"])
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

    def compress_support_graph(self, support_graph: nx.Graph) -> nx.Graph:
        g = nx.Graph()
        for node in support_graph.nodes:
            if len(support_graph.adj[node]) == 2:
                for neighbor in support_graph.adj[node]:
                    g.add_edge(node, neighbor)

        compress_node_label = {}
        infor_component = {}
        components = list(nx.connected_components(g))
        for idx, component in enumerate(components):
            if len(component) > 4:
                endpoints = []
                for node in component:
                    if len(g.adj[node]) == 1:
                        compress_node_label[node] = (idx, 0)
                        for neighbor in g.adj[node]:
                            compress_node_label[neighbor] = (idx, 1)
                            endpoints.append(neighbor)

                if len(endpoints) == 2:
                    infor_component[idx] = endpoints
                else:
                    component = list(component)
                    tmp_node = component[0]
                    compress_node_label[tmp_node] = (idx, 0)
                    for neighbor in g.adj[tmp_node]:
                        compress_node_label[neighbor] = (idx, 1)
                    infor_component[idx] = list(g.adj[tmp_node])
            else:
                for node in component:
                    compress_node_label[node] = (idx, 0)

        output = nx.Graph()
        for node in support_graph.nodes:
            if len(support_graph.adj[node]) > 2:
                for neighbor in support_graph.adj[node]:
                    output.add_edge(
                        node,
                        neighbor,
                        weight=support_graph.adj[node][neighbor]["weight"],
                    )
            else:
                if node in compress_node_label:
                    idx, label = compress_node_label[node]
                    if label == 1:
                        for neighbor in support_graph.adj[node]:
                            if neighbor in compress_node_label:
                                output.add_edge(
                                    node,
                                    neighbor,
                                    weight=support_graph[node][neighbor]["weight"],
                                )

                        endpoints = infor_component[idx]
                        output.add_edge(endpoints[0], endpoints[1], weight=1)
                    else:
                        for neighbor in support_graph.adj[node]:
                            output.add_edge(
                                node,
                                neighbor,
                                weight=support_graph.adj[node][neighbor]["weight"],
                            )

        new_label = {}
        for node in output.nodes:
            new_label[node] = len(new_label)

        return nx.relabel_nodes(output, new_label)

    def get_support_graph_representation(self, support_graph: nx.Graph) -> Dict[str, np.array]:
        node_feature = []
        edge_feature = []
        edge_index = [[], []]

        g = self.compress_support_graph(support_graph)
        node_feature = np.array([[len(g.adj[node])] for node in g.nodes])
        edge_index = [[], []]
        edge_feature = []
        for node in g.nodes:
            for neighbor in g.adj[node]:
                edge_index[0].append(node)
                edge_index[1].append(neighbor)
                edge_feature.append([g.adj[node][neighbor]["weight"]])

        edge_index = np.asarray(edge_index)
        edge_feature = np.asarray(edge_feature)
        lens = np.asarray([node_feature.shape[0], edge_feature.shape[0]])

        if self.padding:
            assert (
                self.init_config.instance_size >= node_feature.shape[0]
            ), "instance size {} < node feature {}".format(
                self.init_config.instance_size, node_feature.shape[0]
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
                            self.init_config.instance_size - node_feature.shape[0],
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

        return dict(
            sup_node_feature=node_feature,
            sup_edge_index=edge_index,
            sup_edge_feature=edge_feature,
            sup_lens=lens,
        )

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

    def get_state_representation(self, callback: UserCutCallback):
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

        statistic = [
            callback.get_current_node_depth() / self.init_config.instance_size,
            callback.has_incumbent(),
            gap,
            sum(np.where(abs(solution - 1) < TOLERANCE, 1, 0))
            / self.init_config.instance_size,
            callback.get_objective_value() / callback.get_cutoff(),
            remain_leaves / total_leaves,
            processed_leaves / total_leaves,
            np.sum(lb) / self.init_config.instance_size,
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


class PriorSubtourStateExtractor(SubtourStateExtractor):
    def __init__(
            self, separator: SubtourSeparator, padding=False, *args, **kwds
    ) -> None:
        super().__init__(separator, padding, *args, **kwds)

    def initialize_original_graph(
        self, problem: TSPProblem, var2idx: Dict[Tuple[int, int], int], k=10
    ) -> None:
        super(PriorSubtourStateExtractor, self).initialize_original_graph(problem, var2idx, k)

    def compress_support_graph(self, support_graph: nx.Graph) -> nx.Graph:
        return super(PriorSubtourStateExtractor, self).compress_support_graph(support_graph)

    def get_support_graph_representation(self, support_graph: nx.Graph) -> Dict[str, np.array]:
        return super(PriorSubtourStateExtractor, self).get_support_graph_representation(support_graph)

    def get_original_graph_representation(self, solution: np.array, lb: np.array, ub: np.array) -> Dict[str, np.array]:
        return super(PriorSubtourStateExtractor, self).get_original_graph_representation(solution, lb, ub)

    def get_state_representation(self, callback: UserCutCallback):
        state, support_graph = super(PriorSubtourStateExtractor, self).get_state_representation(callback)

        if callback.prev_cuts > 0:
            state["prior"] = 5
        else:
            state["prior"] = 1

        return state, support_graph




