from problems.base import Problem
from .callbacks.callback_name import CALLBACK_NAME
from .cplex_api import cplex


class Solver(cplex.Cplex):
    def __init__(self, problem: Problem) -> None:
        super().__init__()
        self.parameters.preprocessing.presolve.set(
            self.parameters.preprocessing.presolve.values.off
        )
        self.parameters.threads.set(1)
        self.parameters.mip.strategy.search.set(
            self.parameters.mip.strategy.search.values.traditional
        )
        self.parameters.mip.interval.set(1)
        self.parameters.mip.limits.cutsfactor.set(0)
        self.parameters.mip.display.set(4)

        self.edge2idx = {}
        self.graph = problem.graph
        self.create_mip_formulation()

    def create_mip_formulation(self):
        raise NotImplementedError

    def basic_solve(self, *args, **kwargs):
        raise NotImplementedError
