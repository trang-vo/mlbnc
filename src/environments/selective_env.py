from torch.multiprocessing import Process,Queue
import numpy as np
import datetime
import os
from typing import *
import queue

from torch.multiprocessing import Queue, Process
import numpy as np

from solvers.solver_name import SOLVER_NAME
from solvers.callbacks.state_extractor.state_extractor_name import STATE_EXTRACTOR_NAME
from problems.problem_name import PROBLEM_NAME
from typing import Dict,Any
from .base import PriorCutEnv

class SelectiveEnv(PriorCutEnv):
    def __init__(self, config: Dict[str, Any], problem_type: str, cut_type: str, mode="train", result_path="") -> None:
        super().__init__(config, problem_type, cut_type, mode, result_path)
        self.cached_instance_path=None
        self.use_cached_instance=False
    def reset(self, instance_path=None, steps="", **kwargs):
        if self.solver_proc is not None:
            self.solver_proc.terminate()
            self.action_queue.close()
            self.state_queue.close()

        self.action_queue = Queue()
        self.state_queue = Queue()
        self.done = False
        
        if not self.use_cached_instance:
            while True:
                if instance_path is None:
                    if self.mode == "train":
                        i = np.random.randint(0, len(self.train_instances))
                        instance_path = os.path.join(
                            self.train_folder, self.train_instances[i]
                        )
                    elif self.mode == "eval":
                        i = np.random.randint(0, len(self.eval_instances))
                        instance_path = os.path.join(
                            self.eval_folder, self.eval_instances[i]
                        )

                self.problem = PROBLEM_NAME[self.problem_type](instance_path)
                if len(self.problem.graph.nodes) == self.init_config.instance_size:
                    break
            self.cached_instance_path=instance_path
            self.use_cached_instance=True
        else:
            instance_path=self.cached_instance_path
            self.problem = PROBLEM_NAME[self.problem_type](instance_path)
            if len(self.problem.graph.nodes) != self.init_config.instance_size:
                raise Exception("Using incorrect city size instance"
            )
            self.use_cached_instance=False

        print("Processing instance", instance_path)

        time_limit = 3600 if self.mode == "train" else 1800
        log_path = ""
        if self.mode == "eval":
            assert self.result_path != ""
            tmp = self.result_path.split("/")
            logdir = os.path.join(*tmp[:-1])
            if not os.path.isdir(os.path.join(logdir, "eval_log")):
                os.makedirs(os.path.join(logdir, "eval_log"))
            if steps == "":
                now = datetime.datetime.now()
                t = "{}{}{}{}".format(now.month, now.day, now.hour, now.minute)
            else:
                t = steps
            log_path = os.path.join(
                logdir,
                "eval_log",
                "{}_{}.log".format(self.problem.graph.graph["name"], t),
            )

        self.solver = SOLVER_NAME[self.problem_type](
            problem=self.problem,
            display_log=kwargs["display_log"] if "display_log" in kwargs else False,
            log_path=log_path,
            time_limit=time_limit,
        )
        state_extractor = STATE_EXTRACTOR_NAME[self.cut_type][self.config["state_extractor"]](self.solver.separator,
                                                                                              padding=True,
                                                                                              config=self.config)
        state_extractor.initialize_original_graph(self.problem, self.solver.edge2idx, k=self.config["k"])

        self.user_callback = self.solver.register_callback(self.user_callback_class)
        user_cb_kwargs = {
            "state_extractor": state_extractor,
            "state_queue": self.state_queue,
            "action_queue": self.action_queue,
            "env_mode": self.mode,
            "config": self.config,
            "logger": self.solver.logger,
        }
        self.user_callback.set_attribute(self.solver.separator, **user_cb_kwargs)

        print("Start solve instance", instance_path)
        self.solver_proc = Process(target=self.cplex_solve, args=())
        self.solver_proc.daemon = True
        self.solver_proc.start()
        try:
            obs, temp_reward, temp_done, temp_info = self.state_queue.get(timeout=5)
        except queue.Empty:
            if self.use_cached_instance:
                self.use_cached_instance=False
                print('solved_in_reset')
                return self.reset(instance_path,steps,**kwargs)
            else:
                raise Exception("Problem solved while reset, and this is the second trajectory")
        if obs is None:
            if self.use_cached_instance:
                self.use_cached_instance=False
                print('obs_is_none')
                return self.reset(instance_path,steps,**kwargs)
            else:
                raise Exception("Problem solved while reset, and this is the second trajectory")
        return obs
