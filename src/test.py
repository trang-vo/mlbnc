import json
import os
from glob import glob

import torch
from stable_baselines3 import DQN

from agents.feature_extractor_name import FEATURE_EXTRACTOR_NAME
from agents.feature_extractors import FeatureExtractor
from problems.tsp import TSPProblem
from solvers.callbacks.base import BaseUserCallback, PreprocessUserCallback
from solvers.tsp import TSPSolver
from solvers.maxcut import MaxcutSolver
from problems.maxcut import MaxcutProblem
from environments.base import BaseCutEnv

import numpy as np
from lescode.config import load_config, get_config


def write_model_parameters():
    model_path = "../logs/CycleEnv_8171628/model_614000_steps"
    config_path = "../logs/CycleEnv_8171628/config.json"
    with open(config_path, "r") as file:
        config = json.load(file)

    model_config = config["dqn"]
    extractor_config = config["extractor"]
    env_config = config["env"]

    device = torch.device("cpu")
    print("Device to train model", device)

    sup_feature_extractor = FEATURE_EXTRACTOR_NAME[
        extractor_config["sup_feature_extractor"]
    ](
        node_dim=env_config["sup_node_dim"],
        edge_dim=env_config["sup_edge_dim"],
        hidden_size=extractor_config["sup_hidden_size"],
        num_layers=extractor_config["sup_num_layers"],
        n_clusters=extractor_config["sup_n_clusters"],
        dropout=extractor_config["sup_dropout"],
        device=device,
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
        device=device,
    )

    statistic_extractor = FEATURE_EXTRACTOR_NAME[extractor_config["statistic_extractor"]](
        input_size=env_config["statistic_dim"],
        hidden_sizes=extractor_config["statistic_hidden_sizes"],
        output_size=extractor_config["statistic_output_size"],
        device=device,
    )

    policy_kwargs = dict(
        features_extractor_class=FeatureExtractor,
        features_extractor_kwargs={
            "sup_feature_extractor": sup_feature_extractor,
            "ori_feature_extractor": ori_feature_extractor,
            "statistic_extractor": statistic_extractor,
            "device": device,
        },
    )
    env = BaseCutEnv(ENV_CONFIG, problem_type="maxcut", cut_type="cycle", mode="train")

    model = DQN(
        "MultiInputPolicy", env, policy_kwargs=policy_kwargs
    )

    model.set_parameters(model_path)

    print(model.q_net.features_extractor.state_dict())
    print(model.q_net.q_net)

    torch.save(model.q_net.features_extractor.state_dict(),
               "../logs/CycleEnv_8171628/features_extractor_model_614000_steps.pt")
    torch.save(model.q_net.q_net, "../logs/CycleEnv_8171628/q_net_model_614000_steps.pt")

    print("finish")


if __name__ == "__main__":
    # instance_paths = glob("../data/tsp_instances/100/eval/C*.tsp")
    # instance_paths = ["../data/tsp_instances/100/train/C100_90.tsp"]
    # for path in instance_paths:
    #     print("Processing", path)
    #     prob = TSPProblem(path)
    #     solver = TSPSolver(prob, display_log=True)
    #     # solver.basic_solve(user_callback="SubtourUserCallback", user_cb_kwargs={"terminal_gap": 0.01})
    #     user_callback = solver.register_callback(PreprocessUserCallback)
    #     user_callback.set_attribute(solver.separator, terminal_gap=0, logger=solver.logger, frequent=1, skip_root=False)
    #     solver.solve()
    #     if user_callback.processed_nodes < 5:
    #         print("Remove", path)
            # os.remove(path)

    # prob = MaxcutProblem("../data/maxcut/rudy_all/pm1s_100.1")
    # solver = MaxcutSolver(prob)
    # solver.basic_solve(user_callback="CycleUserCallback", user_cb_kwargs={"origin_graph": prob.graph})

    # ENTRY_CONFIG = load_config(name="entry", path="configs/cycle.yaml").detail
    # ENV_CONFIG = load_config(name="env", path=ENTRY_CONFIG.env_config).detail
    # AGENT_CONFIG = load_config(name="agent", path=ENTRY_CONFIG.agent_config).detail
    #
    # env = BaseCutEnv(ENV_CONFIG, problem_type="maxcut", cut_type="cycle", mode="eval", result_path="../logs/result_eval.csv")
    # done = False
    # obs = env.reset(instance_path="../data/maxcut_instances/100/eval/pm100_10_1907.maxcut", display_log=True)
    # action = 0
    # total_reward = 0
    # while not done:
    #     obs, r, done, _ = env.step(action)
    #     total_reward += r
    #     action = 0
    # print(total_reward)



    # model_path = "../logs/CutEnv_gap_12_1_661057/model_976000_steps.zip"
    # model_path = "../logs/CycleEnv_8171628/model_614000_steps.zip"
    # model_path = "../logs/SubtourEnv_818544/model_1000_steps.zip"
    model_path = "../logs/SubtourEnv_824542/model_60_steps.zip"
    model = DQN.load(model_path)
    print("finish")

