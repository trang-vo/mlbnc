import os
import time
from abc import ABC
from typing import *
import queue

import gym
from gym import spaces
from torch.multiprocessing import Queue, Process
import numpy as np
from stable_baselines3.common.utils import get_linear_fn

from solvers.callbacks.base import BaseUserCallback
from solvers.solver_name import SOLVER_NAME
from solvers.callbacks.state_extractor.state_extractor_name import STATE_EXTRACTOR_NAME
from solvers.base import CALLBACK_NAME
from problems.problem_name import PROBLEM_NAME
from config import EnvConfig


class BaseCutEnv(gym.Env, ABC):
    def __init__(
            self,
            problem_type: str,
            cut_type: str,
            data_folder: str,
            space_config: Dict[str, Any],
            episode_config: Dict[str, Any],
            state_extractor_class: str,
            mode: str = "train",
            result_path: str = "",
            k_nearest_neighbors: int = 10
    ) -> None:
        self.problem_type = problem_type
        self.cut_type = cut_type
        self.space_config = space_config
        self.episode_config = episode_config
        self.state_extractor_class = state_extractor_class
        self.mode = mode
        self.result_path = result_path
        self.k_nearest_neighbors = k_nearest_neighbors

        self.problem = None
        self.solver = None
        self.solver_proc = None
        self.action_queue = None
        self.state_queue = None
        self.done = False
        self.total_time = 0
        self.user_callback = None
        self.last_state = None
        self.prior_buffer = False

        self.init_config = EnvConfig(space_config)
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

        self.data_folder = data_folder
        self.train_folder = os.path.join(self.data_folder, "train")
        self.train_instances = os.listdir(self.train_folder)
        self.eval_folder = os.path.join(self.data_folder, "eval")
        self.eval_instances = os.listdir(self.eval_folder)

        self.num_steps = 0
        self.user_callback_class = CALLBACK_NAME["EnvUserCallback"]

    def cplex_solve(self):
        self.solver.solve()
        if self.mode == "eval":
            with open(self.result_path, "a") as file:
                file.write(
                    "{},{:.2f},{},{},{}\n".format(self.problem.graph.graph["name"], self.user_callback.total_time,
                                                  self.user_callback.total_cuts,
                                                  self.user_callback.actions[0], self.user_callback.actions[1]))

    def get_instance_path(self):
        instance_path = None
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

        return instance_path

    def get_log_path_evaluation(self):
        tmp = self.result_path.split("/")
        logdir = os.path.join(*tmp[:-1])
        if not os.path.isdir(os.path.join(logdir, "eval_log")):
            os.makedirs(os.path.join(logdir, "eval_log"))
        log_path = os.path.join(
            logdir,
            "eval_log",
            "{}_{}.log".format(self.problem.graph.graph["name"], self.num_steps),
        )

        return log_path

    def create_mip_solver(self, **kwargs):
        log_path = ""
        if self.mode == "eval":
            assert self.result_path != ""
            log_path = self.get_log_path_evaluation()

        self.solver = SOLVER_NAME[self.problem_type](
            problem=self.problem,
            cut_type=self.cut_type,
            display_log=kwargs["display_log"] if "display_log" in kwargs else False,
            log_path=log_path,
            time_limit=3600 if self.mode == "train" else 1800,
        )
        state_extractor = STATE_EXTRACTOR_NAME[self.cut_type][self.state_extractor_class](self.solver.separator,
                                                                                          padding=True,
                                                                                          config=self.space_config)
        state_extractor.initialize_original_graph(self.problem, self.solver.edge2idx, k=self.k_nearest_neighbors)

        self.user_callback = self.solver.register_callback(self.user_callback_class)
        user_cb_kwargs = {
            "state_extractor": state_extractor,
            "state_queue": self.state_queue,
            "action_queue": self.action_queue,
            "env_mode": self.mode,
            "config": self.episode_config,
            "logger": self.solver.logger,
        }
        self.user_callback.set_attribute(self.solver.separator, **user_cb_kwargs)

    def set_mip_solver(self, instance_path=None, **kwargs):
        if self.solver_proc is not None:
            self.solver_proc.terminate()
            self.action_queue.close()
            self.state_queue.close()

        self.action_queue = Queue()
        self.state_queue = Queue()
        self.done = False

        while True:
            if instance_path is None:
                instance_path = self.get_instance_path()
            self.problem = PROBLEM_NAME[self.problem_type](instance_path)
            if len(self.problem.graph.nodes) == self.init_config.instance_size:
                break

        print("Processing instance", instance_path)
        self.create_mip_solver(**kwargs)

        print("Start solve instance", instance_path)
        self.solver_proc = Process(target=self.cplex_solve, args=())
        self.solver_proc.daemon = True
        self.solver_proc.start()

    def reset(self, instance_path=None, **kwargs):
        self.set_mip_solver(instance_path, **kwargs)

        while True:
            try:
                obs, _, done, _ = self.state_queue.get(timeout=5)
                return obs
            except queue.Empty:
                print("Waiting an initial state")
                if not self.solver_proc.is_alive():
                    self.set_mip_solver(**kwargs)

    def step(self, action: int):
        self.action_queue.put(action)

        while self.solver_proc.is_alive():
            try:
                obs, reward, done, info = self.state_queue.get(timeout=5)
                self.last_state = obs
                self.total_time = info["total_time"]
                # Wait to write results to file if mode is eval
                if self.mode == "eval" and done:
                    time.sleep(1)
                self.num_steps += 1
                return obs, reward, done, info
            except queue.Empty:
                print("Queue is empty")

        done = True
        reward = 0
        info = {"terminal_observation": self.last_state, "total_time": self.total_time}
        if self.mode == "eval":
            time.sleep(1)
        self.num_steps += 1
        return self.last_state, reward, done, info


class PriorCutEnv(BaseCutEnv):
    def __init__(
            self,
            problem_type: str,
            cut_type: str,
            data_folder: str,
            space_config: Dict[str, Any],
            episode_config: Dict[str, Any],
            state_extractor_class: str,
            mode: str = "train",
            result_path: str = "",
            k_nearest_neighbors: int = 10,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(problem_type, cut_type, data_folder, space_config, episode_config, state_extractor_class, mode,
                         result_path, k_nearest_neighbors)
        self.prior_buffer = True
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

    def cplex_solve(self):
        super(PriorCutEnv, self).cplex_solve()

    def get_instance_path(self):
        return super(PriorCutEnv, self).get_instance_path()

    def get_log_path_evaluation(self):
        return super(PriorCutEnv, self).get_log_path_evaluation()

    def create_mip_solver(self, **kwargs):
        super(PriorCutEnv, self).create_mip_solver(**kwargs)

    def set_mip_solver(self, instance_path=None, **kwargs):
        super(PriorCutEnv, self).set_mip_solver(instance_path, **kwargs)

    def reset(self, instance_path=None, steps="", **kwargs):
        return super(PriorCutEnv, self).reset(instance_path=None, steps="", **kwargs)

    def step(self, action: int):
        print("Action is", action)
        return super(PriorCutEnv, self).step(action)


class ArbitraryCutEnv(BaseCutEnv):
    def __init__(self, problem_type: str, cut_type: str, data_folder: str, space_config: Dict[str, Any],
                 episode_config: Dict[str, Any], state_extractor_class: str, mode: str = "train", result_path: str = "",
                 initial_start_distance: float = 1, final_start_distance: float = 1, start_distance_fraction: float = 0,
                 total_train_steps: int = 0, k_nearest_neighbors: int = 10) -> None:
        super().__init__(problem_type, cut_type, data_folder, space_config, episode_config, state_extractor_class, mode,
                         result_path, k_nearest_neighbors)
        if self.mode == "train":
            self.initial_start_distance = initial_start_distance
        elif self.mode == "eval":
            self.initial_start_distance = final_start_distance
        self.final_start_distance = final_start_distance
        self.start_distance_fraction = start_distance_fraction
        self.start_distance_schedule = get_linear_fn(self.initial_start_distance, self.final_start_distance,
                                                     self.start_distance_fraction)
        self.total_train_steps = total_train_steps
        if self.start_distance_fraction > 0 and self.total_train_steps == 0:
            raise "Need to provide the total time steps for training when setting the start distance"

        if self.start_distance_fraction == 0:
            self.user_callback_class = CALLBACK_NAME["EnvUserCallback"]
        else:
            self.user_callback_class = CALLBACK_NAME["ArbitraryStartEnvUserCallback"]

    def cplex_solve(self):
        super(ArbitraryCutEnv, self).cplex_solve()

    def get_instance_path(self):
        return super(ArbitraryCutEnv, self).get_instance_path()

    def get_log_path_evaluation(self):
        return super(ArbitraryCutEnv, self).get_log_path_evaluation()

    def create_mip_solver(self, **kwargs):
        log_path = ""
        if self.mode == "eval":
            assert self.result_path != ""
            log_path = self.get_log_path_evaluation()

        self.solver = SOLVER_NAME[self.problem_type](
            problem=self.problem,
            cut_type=self.cut_type,
            display_log=kwargs["display_log"] if "display_log" in kwargs else False,
            log_path=log_path,
            time_limit=3600 if self.mode == "train" else 1800,
        )
        state_extractor = STATE_EXTRACTOR_NAME[self.cut_type][self.state_extractor_class](self.solver.separator,
                                                                                          padding=True,
                                                                                          config=self.space_config)
        state_extractor.initialize_original_graph(self.problem, self.solver.edge2idx, k=self.k_nearest_neighbors)

        progress_remaining = 0
        if self.start_distance_fraction > 0 and self.total_train_steps > 0:
            progress_remaining = 1 - self.num_steps / self.total_train_steps

        self.user_callback = self.solver.register_callback(self.user_callback_class)
        user_cb_kwargs = {
            "state_extractor": state_extractor,
            "state_queue": self.state_queue,
            "action_queue": self.action_queue,
            "env_mode": self.mode,
            "config": self.episode_config,
            "logger": self.solver.logger,
            "start_criterion": (self.start_distance_schedule(progress_remaining), "gap")
        }
        self.user_callback.set_attribute(self.solver.separator, **user_cb_kwargs)

    def set_mip_solver(self, instance_path=None, **kwargs):
        super(ArbitraryCutEnv, self).set_mip_solver(instance_path, **kwargs)

    def reset(self, instance_path=None, **kwargs):
        return super(ArbitraryCutEnv, self).reset(instance_path, **kwargs)

    def step(self, action: int):
        return super(ArbitraryCutEnv, self).step(action)


class DistanceCutEnv(ArbitraryCutEnv):
    def __init__(self, problem_type: str, cut_type: str, data_folder: str, space_config: Dict[str, Any],
                 episode_config: Dict[str, Any], state_extractor_class: str, mode: str = "train", result_path: str = "",
                 initial_start_distance: float = 1, final_start_distance: float = 1, start_distance_fraction: float = 0,
                 total_train_steps: int = 0, k_nearest_neighbors: int = 10) -> None:
        super().__init__(problem_type, cut_type, data_folder, space_config, episode_config, state_extractor_class, mode,
                         result_path, initial_start_distance, final_start_distance, start_distance_fraction,
                         total_train_steps, k_nearest_neighbors)

    def cplex_solve(self):
        super().cplex_solve()

    def get_instance_path(self):
        return super().get_instance_path()

    def get_log_path_evaluation(self):
        return super().get_log_path_evaluation()

    def create_mip_solver(self, **kwargs):
        log_path = ""
        if self.mode == "eval":
            assert self.result_path != ""
            log_path = self.get_log_path_evaluation()

        self.solver = SOLVER_NAME[self.problem_type](
            problem=self.problem,
            cut_type=self.cut_type,
            display_log=kwargs["display_log"] if "display_log" in kwargs else False,
            log_path=log_path,
            time_limit=3600 if self.mode == "train" else 1800,
        )
        state_extractor = STATE_EXTRACTOR_NAME[self.cut_type][self.state_extractor_class](self.solver.separator,
                                                                                          padding=True,
                                                                                          config=self.space_config)
        state_extractor.initialize_original_graph(self.problem, self.solver.edge2idx, k=self.k_nearest_neighbors)

        progress_remaining = 0
        if self.start_distance_fraction > 0 and self.total_train_steps > 0:
            progress_remaining = 1 - self.num_steps / self.total_train_steps

        if not os.path.isfile("../results/solutions/{}.npy".format(self.problem.graph.name)):
            solver = SOLVER_NAME[self.problem_type](problem=self.problem, cut_type=self.cut_type, display_log=False,
                                                    log_path="../results/solutions/{}.log".format(
                                                        self.problem.graph.name))
            user_callback = solver.register_callback(BaseUserCallback)
            user_cb_kwargs = {"frequent": 10, "terminal_gap": 0.01, "logger": solver.logger}
            user_callback.set_attribute(solver.separator, **user_cb_kwargs)
            solver.solve()
            np.save("../results/solutions/{}.npy".format(self.problem.graph.name),
                    np.asarray(solver.solution.get_values()))

        optimal_solution = np.load("../results/solutions/{}.npy".format(self.problem.graph.name))
        self.user_callback = self.solver.register_callback(self.user_callback_class)
        user_cb_kwargs = {
            "state_extractor": state_extractor,
            "state_queue": self.state_queue,
            "action_queue": self.action_queue,
            "env_mode": self.mode,
            "config": self.episode_config,
            "logger": self.solver.logger,
            "start_criterion": (self.start_distance_schedule(progress_remaining), "gap"),
            "optimal_solution": optimal_solution
        }
        self.user_callback.set_attribute(self.solver.separator, **user_cb_kwargs)

    def set_mip_solver(self, instance_path=None, **kwargs):
        super().set_mip_solver(instance_path, **kwargs)

    def reset(self, instance_path=None, **kwargs):
        return super().reset(instance_path, **kwargs)

    def step(self, action: int):
        return super().step(action)


class DistancePriorCutEnv(DistanceCutEnv):
    def __init__(self, problem_type: str, cut_type: str, data_folder: str, space_config: Dict[str, Any],
                 episode_config: Dict[str, Any], state_extractor_class: str, mode: str = "train", result_path: str = "",
                 initial_start_distance: float = 1, final_start_distance: float = 1, start_distance_fraction: float = 0,
                 total_train_steps: int = 0, k_nearest_neighbors: int = 10) -> None:
        super().__init__(problem_type, cut_type, data_folder, space_config, episode_config, state_extractor_class, mode,
                         result_path, initial_start_distance, final_start_distance, start_distance_fraction,
                         total_train_steps, k_nearest_neighbors)
        self.prior_buffer = True
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

    def cplex_solve(self):
        super().cplex_solve()

    def get_instance_path(self):
        return super().get_instance_path()

    def get_log_path_evaluation(self):
        return super().get_log_path_evaluation()

    def create_mip_solver(self, **kwargs):
        super().create_mip_solver()

    def set_mip_solver(self, instance_path=None, **kwargs):
        super().set_mip_solver(instance_path, **kwargs)

    def reset(self, instance_path=None, **kwargs):
        return super().reset(instance_path, **kwargs)

    def step(self, action: int):
        return super().step(action)
