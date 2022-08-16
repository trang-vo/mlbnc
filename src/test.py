from glob import glob

from problems.tsp import TSPProblem
from solvers.tsp import TSPSolver
from solvers.callbacks.subtour import SubtourUserCallback
from solvers.maxcut import MaxcutSolver
from problems.maxcut import MaxcutProblem
from environments.subtour import SubtourEnv

import numpy as np
from lescode.config import load_config, get_config

# load_config(name="env", path="configs/env/subtour.yaml")
# ENV_CONFIG = get_config(name="env").detail

if __name__ == "__main__":
    instance_paths = glob("../data/tsp_instances/200/eval/*.tsp")
    for path in instance_paths:
        print(path)
        prob = TSPProblem(path)
        solver = TSPSolver(prob)
        solver.basic_solve(user_callback="SubtourUserCallback", user_cb_kwargs={"terminal_gap": 0.01})

    # prob = MaxcutProblem("../data/maxcut/rudy_all/pm1s_100.1")
    # solver = MaxcutSolver(prob)
    # solver.basic_solve(user_callback="CycleUserCallback", user_cb_kwargs={"origin_graph": prob.graph})

    # env = SubtourEnv(ENV_CONFIG.SubtourEnv, mode="train")
    # done = False
    # obs = env.reset("../data/tsplib/kroB200.tsp")
    # action = np.random.randint(0, 2)
    # while not done:
    #     obs, r, done, _ = env.step(action)
    #     action = np.random.randint(0, 2)
