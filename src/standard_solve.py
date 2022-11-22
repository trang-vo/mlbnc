import os.path
import sys
from time import time

import typer

from problems.problem_name import PROBLEM_NAME
from solvers.solver_name import SOLVER_NAME
from solvers.callbacks.base import BaseUserCallback, RandomUserCallback

app = typer.Typer()


@app.command()
def solve(problem_type: str, cut_type: str, instance_path: str, use_callback: int = True, frequent: int = 1,
          terminal_gap: float = 0.01, result_path: str = "", display_log: int = 1, log_path=""):
    print("Display log", display_log)
    prob = PROBLEM_NAME[problem_type](instance_path)
    if display_log == 0:
        display_log = False
    else:
        display_log = True
    solver = SOLVER_NAME[problem_type](prob, cut_type, display_log=display_log, log_path=log_path, time_limit=10800)

    user_callback = None
    if use_callback:
        user_callback = solver.register_callback(BaseUserCallback)
        user_callback.set_attribute(solver.separator, terminal_gap=terminal_gap, frequent=frequent,
                                    logger=solver.logger)

    s = time()
    solver.solve()
    t = time() - s

    mode = 1

    if result_path != "":
        if use_callback:
            if not os.path.isfile(result_path):
                with open(result_path, "w") as file:
                    file.write("name,mode,gap,time,total_nodes,total_cuts,action0,action1\n")
            with open(result_path, "a") as file:
                file.write("{},{},{:0.2f},{:0.3f},{},{},{},{}\n".format(prob.graph.graph["name"], mode,
                                                                        solver.solution.MIP.get_mip_relative_gap() * 100,
                                                                        t,
                                                                        user_callback.processed_nodes,
                                                                        user_callback.total_cuts,
                                                                        user_callback.actions[0],
                                                                        user_callback.actions[1]))
        else:
            if not os.path.isfile(result_path):
                with open(result_path, "w") as file:
                    file.write("name,time\n")
            with open(result_path, "a") as file:
                file.write("{},{:0.3f}\n".format(prob.graph.graph["name"], t))


@app.command()
def solve_random(problem_type: str, cut_type: str, instance_path: str, use_callback: int = True, frequent: int = 1,
                 terminal_gap: float = 0.01, result_path: str = "", display_log: int = 1, log_path=""):
    print("Display log", display_log)
    prob = PROBLEM_NAME[problem_type](instance_path)
    if display_log == 0:
        display_log = False
    else:
        display_log = True
    solver = SOLVER_NAME[problem_type](prob, cut_type, display_log=display_log, log_path=log_path, time_limit=10800)

    user_callback = None
    if use_callback:
        user_callback = solver.register_callback(RandomUserCallback)
        user_callback.set_attribute(solver.separator, terminal_gap=terminal_gap, frequent=frequent,
                                    logger=solver.logger)

    s = time()
    solver.solve()
    t = time() - s

    mode = 1

    if result_path != "":
        if use_callback:
            if not os.path.isfile(result_path):
                with open(result_path, "w") as file:
                    file.write("name,mode,gap,time,total_nodes,total_cuts,action0,action1\n")
            with open(result_path, "a") as file:
                file.write("{},{},{:0.2f},{:0.3f},{},{},{},{}\n".format(prob.graph.graph["name"], mode,
                                                                        solver.solution.MIP.get_mip_relative_gap() * 100,
                                                                        t,
                                                                        user_callback.processed_nodes,
                                                                        user_callback.total_cuts,
                                                                        user_callback.actions[0],
                                                                        user_callback.actions[1]))
        else:
            if not os.path.isfile(result_path):
                with open(result_path, "w") as file:
                    file.write("name,time\n")
            with open(result_path, "a") as file:
                file.write("{},{:0.3f}\n".format(prob.graph.graph["name"], t))


if __name__ == "__main__":
    typer.run(solve)
