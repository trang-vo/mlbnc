from typing import *
import os
from time import time
import queue

from gym import spaces
import numpy as np
from torch.multiprocessing import Process

from .base import BaseCutEnv
from problems.maxcut import MaxcutProblem
from solvers.maxcut import MaxcutSolver, CALLBACK_NAME
from solvers.callbacks.state_extractor.cycle import CycleStateExtractor
from config import EnvConfig


class CycleEnv(BaseCutEnv):
    def __init__(self, config: Dict[str, Any], mode="train", result_path="") -> None:
        super().__init__(config, mode, result_path)
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

        self.user_callback = CALLBACK_NAME[config["user_callback"]]

    def reset(self, instance_path=None, steps=""):
        super().reset()

        while True:
            if instance_path is None:
                if self.mode == "train":
                    i = np.random.randint(0, len(self.train_instances))
                    instance_path = os.path.join(
                        self.train_folder, self.train_instances[i]
                    )
                elif self.mode == "eval":
                    i = np.random.randint(0, len(self.eval_folder))
                    instance_path = os.path.join(
                        self.eval_folder, self.eval_instances[i]
                    )

            self.problem = MaxcutProblem(instance_path)
            if len(self.problem.graph.nodes) == self.init_config.instance_size:
                break
        print("The number of edges", len(self.problem.graph.edges))
        print("Processing instance", instance_path)

        self.solver = MaxcutSolver(problem=self.problem)
        self.state_extractor = CycleStateExtractor(self.solver.separator, self.config)
        self.state_extractor.initialize_original_graph(
            self.problem, self.solver.edge2idx, k=self.config["k"]
        )

        self.userSubtour = self.solver.register_callback(self.user_callback)
        user_cb_kwargs = {
            "state_extractor": self.state_extractor,
            "state_queue": self.state_queue,
            "action_queue": self.action_queue,
            "env_mode": self.mode,
            "config": self.config,
        }
        self.userSubtour.set_attribute(self.solver.separator, **user_cb_kwargs)

        print("Start solve instance", instance_path)
        self.solver_proc = Process(target=self.solve, args=(steps,))
        self.solver_proc.daemon = True
        self.solver_proc.start()

        obs, _, _, _ = self.state_queue.get()

        return obs

    def step(self, action: int):
        self.action_queue.put(action)

        while self.solver_proc.is_alive():
            try:
                obs, reward, done, info = self.state_queue.get(timeout=5)
                self.last_state = obs
                return obs, reward, done, info
            except queue.Empty:
                print("Queue is empty")

        done = True
        info = {"terminal_observation": self.last_state}
        return self.last_state, reward, done, info
