import json
import os
import sys
from typing import *

import torch
from stable_baselines3 import DQN

from agents.feature_extractors import EvalFeatureExtractor
from problems.problem_name import PROBLEM_NAME
from solvers.callbacks.state_extractor.state_extractor_name import STATE_EXTRACTOR_NAME
from solvers.solver_name import SOLVER_NAME
from solvers.callbacks.callback_name import CALLBACK_NAME
from agents.feature_extractor_name import FEATURE_EXTRACTOR_NAME


def create_features_extractor(extractor_config: Dict[str, Any], env_config: Dict[str, Any]):
    sup_feature_extractor = FEATURE_EXTRACTOR_NAME[
        extractor_config["sup_feature_extractor"]
    ](
        node_dim=env_config["sup_node_dim"],
        edge_dim=env_config["sup_edge_dim"],
        hidden_size=extractor_config["sup_hidden_size"],
        num_layers=extractor_config["sup_num_layers"],
        n_clusters=extractor_config["sup_n_clusters"],
        dropout=extractor_config["sup_dropout"],
    )

    ori_feature_extractor = FEATURE_EXTRACTOR_NAME[
        extractor_config["ori_feature_extractor"]
    ](
        node_dim=env_config["ori_node_dim"],
        edge_dim=env_config["ori_edge_dim"],
        hidden_size=extractor_config["ori_hidden_size"],
        num_layers=extractor_config["ori_num_layers"],
        n_clusters=extractor_config["ori_n_clusters"],
        dropout=extractor_config["ori_dropout"],
    )

    statistic_extractor = FEATURE_EXTRACTOR_NAME[extractor_config["statistic_extractor"]](
        input_size=env_config["statistic_dim"],
        hidden_sizes=extractor_config["statistic_hidden_sizes"],
        output_size=extractor_config["statistic_output_size"],
    )

    return EvalFeatureExtractor(sup_feature_extractor, ori_feature_extractor, statistic_extractor)


def convert_config(conf: Dict[Any, Any]):
    new_conf = {"env": {
        "space_config": {
            "instance_size": 200,
            "ori_nEdges": 3000,
            "ori_node_dim": conf["CutEnv"]["ori_node_dim"],
            "ori_edge_dim": conf["CutEnv"]["ori_edge_dim"],
            "statistic_dim": conf["CutEnv"]["static_dim"],
            "sup_nNodes": 200,
            "sup_nEdges": 500,
            "sup_node_dim": conf["CutEnv"]["node_dim"],
            "sup_edge_dim": conf["CutEnv"]["edge_dim"],
        },
        "state_extractor_class": "SubtourStateExtractor",
        "data_folder": "../data/tsp_instances/200",
        "k_nearest_neighbors": 10,
        "episode_config": {
            "limited_one_action": 2000,
            "time_limit": 3000,
            "reward_time_limit": -1000,
            "terminal_gap": 0.01,
            "reward_type": "time"
        },
        "initial_start_distance": 0.05,
        "final_start_distance": 1,
        "start_distance_fraction": 0.5,
        "total_train_steps": 1000000
    }, "extractor": {
        "sup_feature_extractor": "GNNGraphExtractor",
        "sup_hidden_size": 64,
        "sup_num_layers": 2,
        "sup_n_clusters": 2,
        "sup_dropout": 0,
        "ori_feature_extractor": "GINEGraphExtractor",
        "ori_hidden_size": 64,
        "ori_num_layers": 2,
        "ori_n_clusters": 2,
        "ori_dropout": 0,
        "statistic_extractor": "MLPOneLayer",
        "statistic_hidden_sizes": [64],
        "statistic_output_size": 64
    }}

    return new_conf


if __name__ == "__main__":
    # args = sys.argv
    # problem_type: str = args[1]
    # cut_type: str = args[2]
    # callback_type: str = args[3]
    # instance_path: str = args[4]
    # model_dir: str = args[5]
    # model_folder: str = args[6]
    # model_name: str = args[7]

    problem_type = "tsp"
    cut_type = "subtour"
    callback_type = "RLUserCallback"
    instance_path = "../data/tsplib/kroA200.tsp"
    model_dir = "../logs"
    model_folder = "CutEnv_gap_12_1_661057"
    model_name = "best_model.pt"

    config_path = os.path.join(model_dir, model_folder, "config.json")
    with open(config_path, "r") as file:
        config = json.load(file)
        if "env" not in config or "extractor" not in config:
            config = convert_config(config)

    prob = PROBLEM_NAME[problem_type](instance_path)
    solver = SOLVER_NAME[problem_type](prob, cut_type)

    state_extractor = STATE_EXTRACTOR_NAME[cut_type]["default"](solver.separator, padding=False,
                                                                config=config["env"]["space_config"])
    state_extractor.initialize_original_graph(prob, solver.edge2idx, k=config["env"]["k_nearest_neighbors"])

    model_path = os.path.join(model_dir, model_folder, model_name)
    if not torch.cuda.is_available():
        model = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        model = torch.load(model_path)
    features_extractor = create_features_extractor(extractor_config=config["extractor"],
                                                   env_config=config["env"]["space_config"])
    # print(features_extractor)
    # print(model["features_extractor"])
    features_extractor.load_state_dict(model["features_extractor"])
    agent = model["agent"]

    user_callback = solver.register_callback(CALLBACK_NAME[callback_type])
    user_cb_kwargs = {
        "state_extractor": state_extractor,
        "features_extractor": features_extractor,
        "agent": agent,
        "config": config["env"]["episode_config"],
    }
    user_callback.set_attribute(solver.separator, **user_cb_kwargs)

    solver.solve()
