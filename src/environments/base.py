import datetime
import os
import sys
from typing import *

import gym
from torch.multiprocessing import Queue


class BaseCutEnv(gym.Env):
    def __init__(self, config: Dict[str, Any], mode="train", result_path="") -> None:
        super().__init__()
        self.solver_proc = None
        self.action_queue = None
        self.state_queue = None
        self.done = False
        self.mode = mode
        self.config = config
        self.result_path = result_path

    def solve(self, steps=""):
        if self.mode == "train":
            self.solver.set_results_stream(None)
            self.solver.set_warning_stream(None)
            self.solver.set_error_stream(None)
            self.solver.set_log_stream(None)
            self.solver.parameters.timelimit.set(3600)
        elif self.mode == "eval":
            assert self.result_path != ""
            tmp = self.result_path.split("/")
            logdir = os.path.join(*tmp[:-1])
            if not os.path.isdir(os.path.join(logdir, "eval_log")):
                os.makedirs(os.path.join(logdir, "eval_log"))
            if len(steps) > 1:
                now = datetime.datetime.now()
                t = "{}{}{}{}".format(now.month, now.day, now.hour, now.minute)
            else:
                t = steps
            logfile = open(
                os.path.join(
                    logdir,
                    "eval_log",
                    "{}_{}.log".format(self.problem.graph.graph["name"], t),
                ),
                "w",
            )
            self.solver.set_results_stream(logfile)
            self.solver.set_warning_stream(logfile)
            self.solver.set_error_stream(logfile)
            self.solver.set_log_stream(logfile)
            self.solver.parameters.timelimit.set(900)
        elif self.mode == "test":
            self.solver.parameters.timelimit.set(1800)

        self.solver.solve()

    def reset(self):
        if self.solver_proc is not None:
            self.solver_proc.terminate()
            self.action_queue.close()
            self.state_queue.close()

        self.action_queue = Queue()
        self.state_queue = Queue()
        self.done = False

    def step(self, action: int):
        raise NotImplementedError
