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


if __name__ == "__main__":
    args = sys.argv
    problem_type: str = args[1]
    cut_type: str = args[2]
    callback_type: str = args[3]
    instance_path: str = args[4]
    model_dir: str = args[5]
    model_folder: str = args[6]
    model_name: str = args[7]

    features_extractor_path = os.path.join(model_dir, model_folder, "features_extractor_{}.pt".format(model_name))
    agent_path = os.path.join(model_dir, model_folder, "q_net_{}.pt".format(model_name))
    if not os.path.isfile(features_extractor_path):
        model_path = os.path.join(model_dir, model_folder, model_name)
        print(model_path)
        model = DQN.load(model_path)
        torch.save(model.q_net.features_extractor.state_dict(), features_extractor_path)
        torch.save(model.q_net.q_net, agent_path)

    config_path = os.path.join(model_dir, model_folder, "config.json")
    with open(config_path, "r") as file:
        config = json.load(file)

    prob = PROBLEM_NAME[problem_type](instance_path)
    solver = SOLVER_NAME[problem_type](prob)

    state_extractor = STATE_EXTRACTOR_NAME[cut_type](solver.separator, padding=False, config=config["env"])
    state_extractor.initialize_original_graph(prob, solver.edge2idx, k=config["env"]["k"])
    features_extractor = create_features_extractor(extractor_config=config["extractor"], env_config=config["env"])
    features_extractor.load_state_dict(torch.load(features_extractor_path))
    agent = torch.load(agent_path)

    user_callback = solver.register_callback(CALLBACK_NAME[callback_type])
    user_cb_kwargs = {
        "state_extractor": state_extractor,
        "features_extractor": features_extractor,
        "agent": agent,
        "config": config["env"],
    }
    user_callback.set_attribute(solver.separator, **user_cb_kwargs)

    solver.solve()

