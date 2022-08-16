import datetime
import os

from lescode.config import load_config, get_config

from environments.subtour import SubtourEnv
from environments.cycle import CycleEnv
from agents.base import DQNAgent

ENTRY_CONFIG = load_config(name="entry", path="configs/subtour.yaml").detail
ENV_CONFIG = load_config(name="env", path=ENTRY_CONFIG.env_config).detail
AGENT_CONFIG = load_config(name="agent", path=ENTRY_CONFIG.agent_config).detail


def train_cycle(folder):
    env = CycleEnv(ENV_CONFIG, mode="train")
    eval_env = CycleEnv(ENV_CONFIG, mode="eval")

    agent = DQNAgent()
    agent.train(
        env,
        eval_env,
        ENV_CONFIG,
        AGENT_CONFIG.feature_extractor,
        AGENT_CONFIG.dqn,
        AGENT_CONFIG.learn,
        folder,
    )


def train_subtour(logdir, folder):
    print(folder)
    env = SubtourEnv(ENV_CONFIG, mode="train")
    result_path = os.path.join(logdir, folder, "results_eval.csv")
    eval_env = SubtourEnv(ENV_CONFIG, mode="eval", result_path=result_path)

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
        log_path=logdir,
    )


if __name__ == "__main__":
    logdir = "../logs"
    now = datetime.datetime.now()
    t = "{}{}{}{}".format(now.month, now.day, now.hour, now.minute)

    folder = "SubtourEnv_{}".format(t)
    if not os.path.isdir(os.path.join(logdir, folder)):
        os.makedirs(os.path.join(logdir, folder))
    log_path = os.path.join(logdir, folder) + "/"

    train_subtour(logdir, folder)

    # result_path = "../logs/results_eval.txt"
    # env = SubtourEnv(ENV_CONFIG, mode="eval", result_path=result_path)
    # obs = env.reset()
    # done = False
    # while not done:
    #     action = 0
    #     obs, _, done, _ = env.step(action)
