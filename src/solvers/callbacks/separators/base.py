from typing import *

import numpy as np
import networkx as nx


class Separator:
    def __init__(self, var2idx: Dict[Any, int]) -> None:
        self.cut_type = ""
        self.var2idx = var2idx
        self.idx2var = {idx: var for var, idx in var2idx.items()}

    def create_support_graph(self, solution: np.array):
        """Create a support graph to find violated constraints"""
        raise NotImplementedError

    def get_lazy_constraints(self, support_graph: nx.Graph):
        """Find constraints of self.cut_type violated by a feasible integer solution (mandatory)"""
        raise NotImplementedError

    def get_user_cuts(self, support_graph: nx.Graph):
        """Find constraints of self.cut_type violated by a fractional solution (optional)"""
        raise NotImplementedError
