from typing import *

import numpy as np
import networkx as nx

# from problems.base import Problem
from problems.base import Problem

class StateExtractor:
    def __init__(self, *args, **kwds) -> None:
        pass

    def initialize_original_graph(self, problem: Problem) -> None:
        raise NotImplementedError

    def get_support_graph_representation(
        self, support_graph: nx.Graph
    ) -> Tuple[np.array, np.array, np.array, np.array]:
        raise NotImplementedError

    def get_original_graph_representation(
        self, solution: np.array
    ) -> Tuple[np.array, np.array, np.array, np.array]:
        raise NotImplementedError

    def get_state_representation(
        self, callback
    ) -> Tuple[nx.Graph, Dict[str, np.array]]:
        raise NotImplementedError
