import _pickle
import os.path
import socket

import typer

import standard_solve
import ml_solve
from utils import get_num_nodes_from_path

app = typer.Typer()


@app.command()
def run_baseline_tsp_subtour(group: int, min_nodes=100, max_nodes=500, ngroups=8):
    frequents = [1, 10, 50, 100]
    problem_type = "tsp"
    cut_type = "subtour"
    with open("tsp_group_{}_{}_{}.p".format(min_nodes, max_nodes, ngroups), "rb") as file:
        instance_paths = _pickle.load(file)[group]
    print("Instance paths", instance_paths)

    for f in frequents:
        try:
            if not os.path.isdir("../results/tsplib"):
                os.makedirs("../results/tsplib")
        except FileExistsError:
            print("File exist")
        result_path = "../results/tsplib/{}.csv".format(f)
        for path in instance_paths:
            log_folder = "../results/tsplib/logs/{}".format(f)
            try:
                if not os.path.isdir(log_folder):
                    os.makedirs(log_folder)
            except FileExistsError:
                print("File exist")
            log_path = os.path.join(log_folder, "{}.log".format(path.split("/")[-1][:-4]))

            if os.path.isfile(log_path):
                continue

            print("Solve instance", path)
            standard_solve.solve(problem_type=problem_type, cut_type=cut_type, instance_path=path, use_callback=True,
                                 frequent=f,
                                 terminal_gap=0.01, result_path=result_path, display_log=0, log_path=log_path)


@app.command()
def solve_tsp_baseline(path: str, freq: int, result_path: str, display_log: int, log_folder):
    nNode = get_num_nodes_from_path(path)
    if nNode < 100 or nNode > 500:
        return

    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)
    log_path = os.path.join(log_folder, "{}.log".format(path.split("/")[-1][:-4]))

    if os.path.isfile(log_path):
        return

    print("Solving instance", path, freq)
    standard_solve.solve(problem_type="tsp", cut_type="subtour", instance_path=path, use_callback=True, frequent=freq,
                         terminal_gap=0.01, result_path=result_path, display_log=display_log, log_path=log_path)


@app.command()
def run_ML_tsp_subtour(group: int, min_nodes=100, max_nodes=500, ngroups=8):
    frequents = [1, 10, 50, 100]
    problem_type = "tsp"
    cut_type = "subtour"
    with open("tsp_group_{}_{}_{}.p".format(min_nodes, max_nodes, ngroups), "rb") as file:
        instance_paths = _pickle.load(file)[group]

    print("Instance paths", instance_paths)
    cut_detector_path = "../models/subtour_detector/GNN_mincut.pt"
    agent_root = "../logs"
    agent_folder = "CutEnv_gap_12_1_661057"
    agent_name = "best_model.pt"
    terminal_gap = 0.01
    display_log = False
    device = "cuda:0"

    for use_detector, use_agent in [(True, True), (True, False), (False, True), (False, False)]:
        for f in frequents:
            result_root = "../results/tsplib/{}/GNN_{}_RL_{}".format(socket.gethostname(), use_detector, use_agent)
            try:
                if not os.path.isdir(result_root):
                    os.makedirs(result_root)
            except FileExistsError:
                print("File exist")
            result_path = os.path.join(result_root, "{}.csv".format(f))
            for path in instance_paths:
                log_folder = os.path.join(result_root, "logs", str(f))
                try:
                    if not os.path.isdir(log_folder):
                        os.makedirs(log_folder)
                except FileExistsError:
                    print("File exist")
                log_path = os.path.join(log_folder, "{}.log".format(path.split("/")[-1][:-4]))

                if os.path.isfile(log_path):
                    continue

                print("Solve instance", path)
                ml_solve.solve(problem_type, cut_type, path, use_cut_detector=use_detector,
                               cut_detector_path=cut_detector_path, use_rl_agent=use_agent, agent_root=agent_root,
                               agent_folder=agent_folder, agent_name=agent_name, frequent=f,
                               terminal_gap=terminal_gap, result_path=result_path, display_log=display_log,
                               log_path=log_path, device=device)


@app.command()
def run_random_tsp_subtour(group: int, min_nodes=100, max_nodes=500, ngroups=8):
    frequents = [1, 10, 50, 100]
    problem_type = "tsp"
    cut_type = "subtour"
    with open("tsp_group_{}_{}_{}.p".format(min_nodes, max_nodes, ngroups), "rb") as file:
        instance_paths = _pickle.load(file)[group]
    print("Instance paths", instance_paths)

    result_root = "../results/tsplib/{}/random".format(socket.gethostname())
    try:
        if not os.path.isdir(result_root):
            os.makedirs(result_root)
    except FileExistsError:
        print("Result root exists")
    for f in frequents:
        result_path = os.path.join(result_root, "{}.csv".format(f))
        for path in instance_paths:
            log_folder = os.path.join(result_root, "logs", str(f))
            try:
                if not os.path.isdir(log_folder):
                    os.makedirs(log_folder)
            except FileExistsError:
                print("Log folder exists")
            log_path = os.path.join(log_folder, "{}.log".format(path.split("/")[-1][:-4]))

            if os.path.isfile(log_path):
                continue

            print("Solve instance", path)
            standard_solve.solve_random(problem_type=problem_type, cut_type=cut_type, instance_path=path,
                                        use_callback=True, frequent=f, terminal_gap=0.01, result_path=result_path,
                                        display_log=0, log_path=log_path)


if __name__ == "__main__":
    app()
