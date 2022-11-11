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
from utils import config_single_instance

ENV_NAME = {
    "BaseCutEnv": BaseCutEnv,
    "PriorCutEnv": PriorCutEnv,
}


if __name__ == "__main__":
    args = sys.argv
    #args = ["", "tsp", "subtour" , "PriorCutEnv" , "../data/tsp_instances/C200_2"]
    problem_type: str = args[1]
    cut_type: str = args[2]
    env_name: str = args[3]
    #extra argument compared with master, indicating which instance folder this test would use, originally encoded in src/config/environment/subtour.yaml, attribute "data_folder: ../data/tsp_instances/200"
    #I search for the longest digit in "data_folder" to be the instance size, in the case above, 200
    data_folder: str = args[4]

    ENTRY_CONFIG = load_config(name="entry", path="configs/{}.yaml".format(cut_type)).detail
    #Adapt config to the instance
    ENV_CONFIG=config_single_instance(ENV_CONFIG=load_config(name="env", path=ENTRY_CONFIG.env_config).detail,instance_name=data_folder)
    AGENT_CONFIG = load_config(name="agent", path=ENTRY_CONFIG.agent_config).detail
    
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
