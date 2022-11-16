import datetime
import os
import time
from typing import *
import queue

import gym
from gym import spaces
from torch.multiprocessing import Queue, Process
import numpy as np

from solvers.solver_name import SOLVER_NAME
from solvers.callbacks.state_extractor.state_extractor_name import STATE_EXTRACTOR_NAME
from solvers.base import CALLBACK_NAME
from problems.problem_name import PROBLEM_NAME
from config import EnvConfig


class BaseCutEnv(gym.Env):
    def __init__(
            self,
            config: Dict[str, Any],
            problem_type: str,
            cut_type: str,
            mode="train",
            result_path="",
    ) -> None:
        super().__init__()
        self.problem = None
        self.solver = None
        self.solver_proc = None
        self.action_queue = None
        self.state_queue = None
        self.done = False
        self.mode = mode
        self.config = config
        self.result_path = result_path

        self.problem_type = problem_type
        self.cut_type = cut_type
        self.user_callback = None
        self.last_state = None

        self.init_config = EnvConfig(config)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict(
            {
                "sup_node_feature": spaces.Box(
                    low=0.0,
                    high=self.init_config.sup_nNodes,
                    shape=(self.init_config.sup_nNodes, self.init_config.sup_node_dim),
                ),
                "sup_edge_index": spaces.Box(
                    low=0,
                    high=self.init_config.sup_nNodes,
                    shape=(2, self.init_config.sup_nEdges),
                ),
                "sup_edge_feature": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.init_config.sup_nEdges, self.init_config.sup_edge_dim),
                ),
                "sup_lens": spaces.Box(
                    low=0, high=self.init_config.sup_edge_dim, shape=(2,)
                ),
                "ori_node_feature": spaces.Box(
                    low=0.0,
                    high=self.init_config.instance_size,
                    shape=(
                        self.init_config.instance_size,
                        self.init_config.ori_node_dim,
                    ),
                ),
                "ori_edge_index": spaces.Box(
                    low=0,
                    high=self.init_config.instance_size,
                    shape=(2, self.init_config.ori_nEdges),
                ),
                "ori_edge_feature": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.init_config.ori_nEdges, self.init_config.ori_edge_dim),
                ),
                "ori_lens": spaces.Box(
                    low=0, high=self.init_config.ori_edge_dim, shape=(2,)
                ),
                "statistic": spaces.Box(
                    low=0.0, high=1e6, shape=(self.init_config.statistic_dim,)
                ),
            }
        )

        self.data_folder = config["data_folder"]
        self.train_folder = os.path.join(self.data_folder, "train")
        self.train_instances = os.listdir(self.train_folder)
        self.eval_folder = os.path.join(self.data_folder, "eval")
        self.eval_instances = os.listdir(self.eval_folder)

        self.user_callback_class = CALLBACK_NAME[config["user_callback"]]
        self.step_time=time.time()
        print(">>Initialize Timer start<<")

    def cplex_solve(self):
        self.solver.solve()
        if self.mode == "eval":
            print("Write evaluate results")
            with open(self.result_path, "a") as file:
                file.write(
                    "{},{:.2f},{},{},{}\n".format(self.problem.graph.graph["name"], self.user_callback.total_time,
                                                  self.user_callback.total_cuts,
                                                  self.user_callback.actions[0], self.user_callback.actions[1]))

    def reset(self, instance_path=None, steps="", **kwargs):
        reset_start=time.time()
        print(">>Reset start, before it it cost {:.2f}<<".format(reset_start-self.step_time))
        if self.solver_proc is not None:
            self.solver_proc.terminate()
            self.action_queue.close()
            self.state_queue.close()

        self.action_queue = Queue()
        self.state_queue = Queue()
        self.done = False

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
        process_instance_start=time.time()
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
        initialize_solver_start=time.time()
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

        obs, _, _, _ = self.state_queue.get()
        print("Reset total cost:{:.2f}\tProcess instance cost:{:.2f}\tInitialized solver cost:{:.2f}".format(time.time()-reset_start,process_instance_start-reset_start,time.time()-initialize_solver_start))
        return obs

    def step(self, action: int):
        self.action_queue.put(action)
        while self.solver_proc.is_alive():
            try:
                #Cplex start to solve
                inner_time_start=time.time()

                obs, reward, done, info = self.state_queue.get(timeout=5)
                self.last_state = obs
                # Wait to write results to file if mode is eval
                if self.mode == "eval" and done:
                    time.sleep(1)

                #Timer
                Total_cost=time.time()-self.step_time
                Cplex_cost=time.time()-inner_time_start
                print("Total cost: {:.2f}\tCplex cost: {:.2f}\tOther cost: {:.2f}\tCplex propotion: {:.2f} %".format(Total_cost,Cplex_cost,Total_cost-Cplex_cost,(Cplex_cost/Total_cost)*100))
                self.step_time=time.time()
                print(">>Timer start<<")

                return obs, reward, done, info
            except queue.Empty:
                print("Queue is empty")
        
        done = True
        reward = 0
        info = {"terminal_observation": self.last_state}
        if self.mode == "eval":
            time.sleep(1)
        return self.last_state, reward, done, info


class PriorCutEnv(BaseCutEnv):
    def __init__(
            self,
            config: Dict[str, Any],
            problem_type: str,
            cut_type: str,
            mode="train",
            result_path="",
    ) -> None:
        super(PriorCutEnv, self).__init__(config, problem_type, cut_type, mode, result_path)
        self.observation_space = spaces.Dict(
            {
                "sup_node_feature": spaces.Box(
                    low=0.0,
                    high=self.init_config.sup_nNodes,
                    shape=(self.init_config.sup_nNodes, self.init_config.sup_node_dim),
                ),
                "sup_edge_index": spaces.Box(
                    low=0,
                    high=self.init_config.sup_nNodes,
                    shape=(2, self.init_config.sup_nEdges),
                ),
                "sup_edge_feature": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.init_config.sup_nEdges, self.init_config.sup_edge_dim),
                ),
                "sup_lens": spaces.Box(
                    low=0, high=self.init_config.sup_edge_dim, shape=(2,)
                ),
                "ori_node_feature": spaces.Box(
                    low=0.0,
                    high=self.init_config.instance_size,
                    shape=(
                        self.init_config.instance_size,
                        self.init_config.ori_node_dim,
                    ),
                ),
                "ori_edge_index": spaces.Box(
                    low=0,
                    high=self.init_config.instance_size,
                    shape=(2, self.init_config.ori_nEdges),
                ),
                "ori_edge_feature": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.init_config.ori_nEdges, self.init_config.ori_edge_dim),
                ),
                "ori_lens": spaces.Box(
                    low=0, high=self.init_config.ori_edge_dim, shape=(2,)
                ),
                "statistic": spaces.Box(
                    low=0.0, high=1e6, shape=(self.init_config.statistic_dim,)
                ),
                "prior": spaces.Discrete(10),
            }
        )

    def reset(self, instance_path=None, steps="", **kwargs):
        return super(PriorCutEnv, self).reset(instance_path=None, steps="", **kwargs)

    def step(self, action: int):
        return super(PriorCutEnv, self).step(action)
