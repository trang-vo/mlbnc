from .base import Problem

import tsplib95
import networkx as nx


class TSPProblem(Problem):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.type = "tsp"

    def read_file(self, path: str):
        """Read a TSP instance from a file in the tsplib format. Save the instance in a nx.graph object."""
        problem = tsplib95.load(path)
        self.graph: nx.Graph = problem.get_graph(normalize=True)

        for edge in self.graph.edges:
            if self.graph.edges[edge]["weight"] == 0:
                self.graph.remove_edge(*edge)
