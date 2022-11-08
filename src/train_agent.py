import datetime
import os
import re
import sys
from typing import *

from lescode.config import load_config
from lescode.namespace import asdict
from lescode.config import Config
from environments.base import BaseCutEnv, PriorCutEnv
from agents.base import DQNAgent


ENV_NAME = {
    "BaseCutEnv": BaseCutEnv,
    "PriorCutEnv": PriorCutEnv,
}


if __name__ == "__main__":
    args = sys.argv
    #args = ["", "tsp", "subtour" , "PriorCutEnv" , "../data/tsp_instances/C100_7"]
    problem_type: str = args[1]
    cut_type: str = args[2]
    env_name: str = args[3]

    ENTRY_CONFIG = load_config(name="entry", path="configs/{}.yaml".format(cut_type)).detail
    ENV_CONFIG = load_config(name="env", path=ENTRY_CONFIG.env_config).detail
    AGENT_CONFIG = load_config(name="agent", path=ENTRY_CONFIG.agent_config).detail
    # print(ENV_CONFIG)
    temp=asdict(ENV_CONFIG)
    cities_num=int(re.search(r'[0-9]{3}',args[4]).group())
    temp["instance_size"]=cities_num
    temp["ori_nEdges"]=cities_num*int(temp["k"])*2
    temp["sup_nNodes"]=cities_num
    temp["sup_nEdges"]=cities_num*3
    temp["data_folder"]=args[4]
    ENV_CONFIG=Config()
    ENV_CONFIG=temp
    # print(ENV_CONFIG)
    logdir = "../logs"
    now = datetime.datetime.now()
    t = "{}{}{}{}".format(now.month, now.day, now.hour, now.minute)

    folder = "{}Env_{}".format(cut_type.capitalize(), t)
    if not os.path.isdir(os.path.join(logdir, folder)):
        os.makedirs(os.path.join(logdir, folder))
    log_path = os.path.join(logdir, folder) + "/"

    env = ENV_NAME[env_name](ENV_CONFIG, problem_type=problem_type, cut_type=cut_type, mode="train")
    result_path = os.path.join(logdir, folder, "results_eval.csv")
    eval_env = ENV_NAME[env_name](ENV_CONFIG, problem_type=problem_type, cut_type=cut_type, mode="eval",
                                  result_path=result_path)

    agent = DQNAgent()
    agent.train(
        env,
        eval_env,
        ENV_CONFIG,
        AGENT_CONFIG.feature_extractor,
        AGENT_CONFIG.dqn,
        AGENT_CONFIG.learn,
        folder,
        pretrain_path=None,
        log_path=log_path,
    )
