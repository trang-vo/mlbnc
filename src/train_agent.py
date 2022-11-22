import datetime
import os
import sys
from typing import *

from lescode.config import load_config
from stable_baselines3.common.monitor import Monitor
from gym import spaces

from environments.base import BaseCutEnv, PriorCutEnv, ArbitraryCutEnv, DistanceCutEnv, DistancePriorCutEnv
from agents.agent_name import AGENT_NAME

ENV_NAME = {
    "BaseCutEnv": BaseCutEnv,
    "PriorCutEnv": PriorCutEnv,
    "ArbitraryCutEnv": ArbitraryCutEnv,
    "DistanceCutEnv": DistanceCutEnv,
    "DistancePriorCutEnv": DistancePriorCutEnv,
}

if __name__ == "__main__":
    # args = sys.argv
    args = ["", "tsp", "subtour", "PriorCutEnv", "sac"]
    problem_type: str = args[1]
    cut_type: str = args[2]
    env_name: str = args[3]
    agent_class: str = args[4]

    ENTRY_CONFIG = load_config(name="entry", path="configs/{}.yaml".format(cut_type)).detail
    ENV_CONFIG = load_config(name="env", path=ENTRY_CONFIG.env_config).detail
    AGENT_CONFIG = load_config(name="agent", path=ENTRY_CONFIG.agent_config).detail

    logdir = "../logs"
    now = datetime.datetime.now()
    t = "{}{}{}{}".format(now.month, now.day, now.hour, now.minute)

    folder = "{}{}_{}".format(cut_type.capitalize(), env_name, t)
    if not os.path.isdir(os.path.join(logdir, folder)):
        os.makedirs(os.path.join(logdir, folder))
    log_path = os.path.join(logdir, folder) + "/"

    env = ENV_NAME[env_name](problem_type=problem_type, cut_type=cut_type, mode="train", **ENV_CONFIG)
    result_path = os.path.join(logdir, folder, "results_eval.csv")
    eval_env = ENV_NAME[env_name](problem_type=problem_type, cut_type=cut_type, mode="eval", **ENV_CONFIG,
                                  result_path=result_path)

    if agent_class.lower() == "sac":
        env.action_space = spaces.Box(low=0, high=1, shape=(2,))
        eval_env.action_space = spaces.Box(low=0, high=1, shape=(2,))

    agent_name = agent_class.upper() + "Agent"
    agent = AGENT_NAME[agent_name]()

    agent.train(
        env,
        eval_env,
        ENV_CONFIG,
        AGENT_CONFIG.feature_extractor,
        AGENT_CONFIG[agent_class],
        AGENT_CONFIG.learn,
        folder,
        pretrain_path=None,
        log_path=log_path,
    )
