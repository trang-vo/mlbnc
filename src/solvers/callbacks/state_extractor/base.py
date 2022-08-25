from typing import *

import numpy as np
import networkx as nx

from problems.base import Problem
from config import EnvConfig


class StateExtractor:
    def __init__(self, separator, padding=False, *args, **kwds) -> None:
        self.init_config = None
        if padding:
            if "config" not in kwds:
                raise KeyError("Need to provide environment configs to pad")

        if "config" in kwds:
            self.init_config = EnvConfig(kwds["config"])
            self.instance_size = self.init_config.instance_size
        else:
            if "instance_size" not in kwds:
                raise KeyError("Need to provide the instance size")
            self.instance_size = kwds["instance_size"]

        self.padding = padding
        self.separator = separator

        self.ori_edge_index = None
        self.ori_lens = None
        self.ori_node_feature = None
        self.ori_edge_index_list = None
        self.ori_edge_weight = None

    def initialize_original_graph(self, problem: Problem, var2idx: Dict[Tuple[int, int], int]) -> None:
        raise NotImplementedError

    def get_support_graph_representation(
        self, support_graph: nx.Graph
    ) -> Dict[str, np.array]:
        raise NotImplementedError

    def get_original_graph_representation(
        self, solution: np.array, **kwargs
    ) -> Dict[str, np.array]:
        raise NotImplementedError

    def get_state_representation(
        self, callback
    ) -> Tuple[Dict[str, np.array], nx.Graph]:
        raise NotImplementedError
