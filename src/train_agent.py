import datetime
import os
import sys
from typing import *

from lescode.config import load_config

from environments.base import BaseCutEnv
from agents.base import DQNAgent

if __name__ == "__main__":
    args = sys.argv
    # args = ["", "maxcut", "cycle"]
    problem_type: str = args[1]
    cut_type: str = args[2]

    ENTRY_CONFIG = load_config(name="entry", path="configs/{}.yaml".format(cut_type)).detail
    ENV_CONFIG = load_config(name="env", path=ENTRY_CONFIG.env_config).detail
    AGENT_CONFIG = load_config(name="agent", path=ENTRY_CONFIG.agent_config).detail

    logdir = "../logs"
    now = datetime.datetime.now()
    t = "{}{}{}{}".format(now.month, now.day, now.hour, now.minute)

    folder = "{}Env_{}".format(cut_type.capitalize(), t)
    if not os.path.isdir(os.path.join(logdir, folder)):
        os.makedirs(os.path.join(logdir, folder))
    log_path = os.path.join(logdir, folder) + "/"

    env = BaseCutEnv(ENV_CONFIG, problem_type=problem_type, cut_type=cut_type, mode="train")
    result_path = os.path.join(logdir, folder, "results_eval.csv")
    eval_env = BaseCutEnv(ENV_CONFIG, problem_type=problem_type, cut_type=cut_type, mode="eval",
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
