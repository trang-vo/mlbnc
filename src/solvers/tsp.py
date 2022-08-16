from collections import defaultdict
from time import time

from .base import Solver, cplex, CALLBACK_NAME
from problems.tsp import TSPProblem
from .callbacks.subtour import SubtourLazyCallback
from .callbacks.separators.subtour import SubtourSeparator


class TSPSolver(Solver):
    def __init__(self, problem: TSPProblem) -> None:
        super().__init__(problem)
        self.separator = SubtourSeparator(self.edge2idx)

        lazy_constraint = self.register_callback(SubtourLazyCallback)
        lazy_constraint.set_attribute(self.separator)

    def create_mip_formulation(self):
        degree_constraints = defaultdict(list)

        self.objective.set_sense(self.objective.sense.minimize)
        for edge in self.graph.edges:
            var_name = "x.{}.{}".format(*edge)
            self.edge2idx[edge] = self.variables.add(
                obj=[self.graph.edges[edge]["weight"]],
                lb=[0.0],
                ub=[1.0],
                types=["B"],
                names=[var_name],
            )[0]
            degree_constraints[edge[0]].append(self.edge2idx[edge])
            degree_constraints[edge[1]].append(self.edge2idx[edge])

        for cons in degree_constraints:
            self.linear_constraints.add(
                lin_expr=[
                    cplex.SparsePair(
                        degree_constraints[cons], [1] * len(degree_constraints[cons])
                    )
                ],
                senses=["E"],
                rhs=[2],
            )

    def basic_solve(self, *args, **kwargs):
        if "user_callback" in kwargs:
            user_callback = self.register_callback(
                CALLBACK_NAME[kwargs["user_callback"]]
            )
            user_callback.set_attribute(self.separator, **kwargs["user_cb_kwargs"])

        s = time()
        self.solve()
        t = time() - s
        print("Time to solve model", t)
        print("The objective value is", self.solution.get_objective_value())

        return t, self.solution.get_objective_value()
