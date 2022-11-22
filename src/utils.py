import _pickle
import re
from glob import glob
from typing import *
import typer

import numpy as np

app = typer.Typer()


@app.command()
def divide_tsp_instances(min_node: int, max_node: int, ngroups: int):
    output = {i: [] for i in range(ngroups)}
    paths = glob("../data/tsplib/*.tsp")
    size_instances = {}
    for path in paths:
        nNode = get_num_nodes_from_path(path)
        if nNode < min_node or nNode >= max_node:
            continue

        size_instances[path] = nNode

    size_instances = {k: v for k, v in sorted(size_instances.items(), key=lambda item: item[1])}
    m = 0
    for path in size_instances:
        output[int(m % ngroups)].append(path)
        m += 1

    with open("tsp_group_{}_{}_{}.p".format(min_node, max_node, ngroups), "wb") as file:
        _pickle.dump(output, file)


@app.command()
def get_num_nodes_from_path(path: str):
    tmp = path.split("/")
    nums = re.findall(r'\d+', tmp[-1])
    return int(nums[0])


def nodes2edge(u, v):
    return min(u, v), max(u, v)


def distance_solution_cuts(solution: np.array, vars: List[int], coefs: List[float], rhs: float):
    return abs(sum([coefs[i] * solution[var_idx] for i, var_idx in enumerate(vars)]) - rhs) / np.sqrt(len(coefs))


if __name__ == "__main__":
    app()

