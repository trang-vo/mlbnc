from time import time
from typing import *

import numpy as np
import torch
from torch.multiprocessing.queue import Queue

from .separators.base import Separator
from .state_extractor.base import StateExtractor
from ..cplex_api import cplex, LazyConstraintCallback, UserCutCallback


class BaseLazyCallback(LazyConstraintCallback):
    def __call__(self):
        solution = np.asarray(self.get_values())

        support_graph = self.separator.create_support_graph(solution)
        constraints = self.separator.get_lazy_constraints(support_graph)
        for vars, coefs, sense, rhs in constraints:
            self.add(constraint=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)

    def set_attribute(self, separator: Separator, *args, **kwargs):
        self.separator = separator


class BaseUserCallback(UserCutCallback):
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if not self.is_after_cut_loop():
            return

        if self.get_num_nodes() % self.frequent != 0:
            return

        if self.get_MIP_relative_gap() < self.terminal_gap:
            return

        s = time()
        solution = np.asarray(self.get_values())
        support_graph = self.separator.create_support_graph(solution)
        cuts = self.separator.get_user_cuts(support_graph)

        for vars, coefs, sense, rhs in cuts:
            self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)
        self.total_cuts += len(cuts)
        msg = "At node {}, add {} user cuts in {:.4f}s, total cuts {}\n".format(
                self.get_num_nodes(), len(cuts), time() - s, self.total_cuts
            )
        if self.logger is not None:
            self.logger.write(msg)
        else:
            print(msg, end="")

    def set_attribute(self, separator: Separator, *args, **kwargs):
        self.separator: Separator = separator
        self.total_cuts = 0
        self.frequent = kwargs["frequent"] if "frequent" in kwargs else 1
        self.terminal_gap = kwargs["terminal_gap"] if "terminal_gap" in kwargs else 0
        self.logger = kwargs["logger"] if "logger" in kwargs else None


class PreprocessUserCallback(BaseUserCallback):
    def __call__(self, *args, **kwargs):
        if not self.is_after_cut_loop():
            return

        if self.get_num_nodes() % self.frequent != 0:
            return

        if self.get_MIP_relative_gap() < self.terminal_gap:
            return

        if self.skip_root and self.get_num_nodes() == 0:
            return

        s = time()
        solution = np.asarray(self.get_values())
        support_graph = self.separator.create_support_graph(solution)
        cuts = self.separator.get_user_cuts(support_graph)

        for vars, coefs, sense, rhs in cuts:
            self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)
        self.total_cuts += len(cuts)
        msg = "At node {}, add {} user cuts in {:.4f}s, total cuts {}\n".format(
            self.get_num_nodes(), len(cuts), time() - s, self.total_cuts
        )
        if self.logger is not None:
            self.logger.write(msg)
        else:
            print(msg, end="")

    def set_attribute(self, separator: Separator, *args, **kwargs):
        super(PreprocessUserCallback, self).set_attribute(separator, *args, **kwargs)
        self.skip_root = kwargs["skip_root"] if "skip_root" in kwargs else False


class EnvUserCallback(BaseUserCallback):
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if not self.is_after_cut_loop():
            return

        s = time()
        processed_leaves = self.get_num_nodes()

        # If the env mode is evaluated and the agent predicts only one action kind for more than K first leaves,
        # then early stop
        if (
            self.env_mode == "eval"
            and self.actions[1] * self.actions[0] == 0
            and processed_leaves > self.config["limited_one_action"]
        ):
            self.state_queue.put((None, -1e6, True, {}))
            self.abort()

        # Define the initial state for a episode, i.e., the initial state = the processed (add all possible cuts +
        # heuristics) root node
        if processed_leaves == 0:
            solution = np.asarray(self.get_values())
            support_graph = self.separator.create_support_graph(solution)
            cuts = self.separator.get_user_cuts(support_graph)

            for vars, coefs, sense, rhs in cuts:
                self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)

            ncuts = len(cuts)
            self.start = time()
            self.prev_cuts = len(cuts)
            self.prev_gap = min(self.prev_gap, self.get_MIP_relative_gap())
            self.prev_time = time()
            self.total_time += s - time()
            self.total_cuts += len(cuts)
            msg = "Node 0, add {} user cuts in {:.4f}s, total cuts {}\n".format(
                    ncuts, time() - s, self.total_cuts)
            if self.logger is not None and self.env_mode == "eval":
                self.logger.write(msg)
            else:
                print(msg, end="")

            return

        # Define the terminal state for an episode, i.e., the terminal state = the leaf which has gap less than 1%
        gap = min(self.prev_gap, self.get_MIP_relative_gap())
        if gap < self.config["terminal_gap"]:
            info = {
                "terminal_observation": self.last_state,
                "total_time": self.total_time,
            }
            self.state_queue.put((self.last_state, 0, True, info))
            self.abort()
            return

        # Extract state information
        state, support_graph = self.state_extractor.get_state_representation(self)
        self.last_state = state

        # Apply time limit for a episode
        if self.env_mode == "train" and processed_leaves > self.config["time_limit"]:
            info = {
                "TimeLimit.truncated": True,
                "terminal_observation": state,
                "total_time": self.total_time,
            }
            reward = self.config["reward_time_limit"]
            self.state_queue.put((state, reward, True, info))
            self.abort()
            return

        # Calculate the reward
        if self.prev_time is not None:
            self.reward = -(time() - self.prev_time)
            self.total_time += self.reward
        else:
            self.reward = 0

        done = False
        info = {"total_time": self.total_time}
        self.state_queue.put((state, self.reward, done, info))
        msg = "Node {}, add {} user cuts, gap {:.2f}, reward {:.2f}, total cuts {}, {}, total time {}\n".format(
            processed_leaves,
            self.prev_cuts,
            gap * 100 if gap < 1 else -1,
            self.reward,
            self.total_cuts,
            self.actions,
            self.total_time,
        )
        if self.logger is not None and self.env_mode == "eval":
            self.logger.write(msg)
        else:
            print(msg, end="")

        action = self.action_queue.get()
        self.actions[action] += 1
        self.prev_time = time()

        ncuts = -1
        if action == 1:
            s = time()
            cuts = self.separator.get_user_cuts(support_graph)
            for vars, coefs, sense, rhs in cuts:
                self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)

            ncuts = len(cuts)
            self.total_cuts += ncuts
            print("Time to solver separation problem", time() - s)
        self.prev_cuts = ncuts
        self.prev_gap = gap

    def set_attribute(self, separator, *args, **kwargs):
        super().set_attribute(separator, *args, **kwargs)
        self.state_extractor: StateExtractor = kwargs["state_extractor"]
        self.state_queue: Queue = kwargs["state_queue"]
        self.action_queue: Queue = kwargs["action_queue"]
        self.env_mode: str = kwargs["env_mode"]
        self.config: Dict[str, Any] = kwargs["config"]

        self.prev_cuts = 0
        self.prev_gap = 1
        self.prev_time = None
        self.total_cuts = 0
        self.total_time = 0
        self.actions = {0: 0, 1: 0}
        self.last_state = None


class RLUserCallback(BaseUserCallback):
    def __call__(self, *args, **kwargs):
        if not self.is_after_cut_loop():
            return

        s = time()
        processed_leaves = self.get_num_nodes()

        # Generate cuts at the root node
        if processed_leaves == 0:
            solution = np.asarray(self.get_values())
            support_graph = self.separator.create_support_graph(solution)
            cuts = self.separator.get_user_cuts(support_graph)

            for vars, coefs, sense, rhs in cuts:
                self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)

            ncuts = len(cuts)
            self.start = time()
            self.prev_cuts = len(cuts)
            self.prev_gap = min(self.prev_gap, self.get_MIP_relative_gap())
            self.prev_time = time()
            self.total_time += s - time()
            self.total_cuts += len(cuts)
            msg = "Node 0, add {} user cuts in {:.4f}s, total cuts {}\n".format(
                ncuts, time() - s, self.total_cuts)
            if self.logger is not None:
                self.logger.write(msg)
            else:
                print(msg, end="")
            return

        # Skip generate cuts when the MIP gap is less than 1%
        gap = min(self.prev_gap, self.get_MIP_relative_gap())
        if gap < self.config["terminal_gap"]:
            return

        # Extract state information
        state, support_graph = self.state_extractor.get_state_representation(self)

        msg = "Node {}, add {} user cuts, gap {:.2f}, total cuts {}, {}, total time {}\n".format(
            processed_leaves,
            self.prev_cuts,
            gap * 100 if gap < 1 else -1,
            self.total_cuts,
            self.actions,
            self.total_time,
        )
        if self.logger is not None:
            self.logger.write(msg)
        else:
            print(msg, end="")

        for key in state:
            if isinstance(state[key], np.ndarray):
                state[key] = torch.Tensor(np.asarray([state[key]]))

        state_feature = self.features_extractor(state)
        action = self.agent(state_feature)[0].argmax().tolist()
        self.actions[action] += 1

        ncuts = -1
        if action == 1:
            s = time()
            cuts = self.separator.get_user_cuts(support_graph)
            for vars, coefs, sense, rhs in cuts:
                self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)

            ncuts = len(cuts)
            self.total_cuts += ncuts
            print("Time to solver separation problem", time() - s)
        self.prev_cuts = ncuts
        self.prev_gap = gap

    def set_attribute(self, separator, *args, **kwargs):
        super().set_attribute(separator, *args, **kwargs)
        self.state_extractor: StateExtractor = kwargs["state_extractor"]
        self.features_extractor = kwargs["features_extractor"]
        self.agent = kwargs["agent"]
        self.config: Dict[str, Any] = kwargs["config"]

        self.prev_cuts = 0
        self.prev_gap = 1
        self.total_cuts = 0
        self.total_time = 0
        self.actions = {0: 0, 1: 0}
        self.last_state = None


