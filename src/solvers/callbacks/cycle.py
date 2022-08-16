from typing import *
from time import time

import numpy as np
from torch.multiprocessing import Queue

from .separators.cycle import CycleSeparator
from .state_extractor.cycle import CycleStateExtractor
from ..cplex_api import cplex, LazyConstraintCallback, UserCutCallback


class CycleLazyCallback(LazyConstraintCallback):
    def __call__(self):
        solution = np.asarray(self.get_values())

        s = time()
        support_graph = self.separator.create_support_graph(solution)
        cuts = self.separator.get_lazy_constraints(support_graph)
        for vars, coefs, rhs in cuts:
            self.add(constraint=cplex.SparsePair(vars, coefs), sense="L", rhs=rhs)
        print(
            "At node {}, add {} lazy constraints in {:.4f}s".format(
                self.get_num_nodes(), len(cuts), time() - s
            )
        )

    def set_attribute(self, separator: CycleSeparator, *args, **kwargs):
        self.separator = separator


class CycleUserCallback(UserCutCallback):
    def __call__(self):
        solution = np.asarray(self.get_values())

        s = time()
        support_graph = self.separator.create_support_graph(solution)
        cuts = self.separator.get_lazy_constraints(support_graph)
        for vars, coefs, rhs in cuts:
            self.add(cut=cplex.SparsePair(vars, coefs), sense="L", rhs=rhs)
        print(
            "At node {}, add {} user cuts in {:.4f}s".format(
                self.get_num_nodes(), len(cuts), time() - s
            )
        )

    def set_attribute(self, separator: CycleSeparator, **kwargs):
        self.separator: CycleSeparator = separator


class EnvCycleUserCallback(CycleUserCallback):
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.is_after_cut_loop() == False:
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

            for vars, coefs, rhs in cuts:
                self.add(cut=cplex.SparsePair(vars, coefs), sense="L", rhs=rhs)

            ncuts = len(cuts)
            self.start = time()
            self.prev_cuts = len(cuts)
            self.prev_gap = min(self.prev_gap, self.get_MIP_relative_gap())
            self.prev_time = time()
            self.total_time += s - time()
            self.total_cuts += len(cuts)
            print(
                "Node 0, add {} cycle cuts in {}s, total cuts {}".format(
                    ncuts, time() - s, self.total_cuts
                )
            )
            return

        # Define the terminal state for a episode, i.e., the terminal state = the leaf which has gap less than 1%
        gap = min(self.prev_gap, self.get_MIP_relative_gap())
        if gap < self.config["terminal_gap"]:
            info = {
                "terminal_observation": self.last_state,
                "total_time": self.total_time,
            }
            self.state_queue.put((self.last_state, 0, True, info))
            self.abort()

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

        # Calculate the reward
        if self.prev_time is not None:
            self.reward = -(time() - self.prev_time)
            self.total_time += self.reward
        else:
            self.reward = 0

        done = False
        info = {"total_time": self.total_time}
        self.state_queue.put((state, self.reward, done, info))
        print(
            "Node {}, add {} cycle cuts, gap {:.2f}, reward {:.2f}, total cuts {}, {}, total time {}".format(
                processed_leaves,
                self.prev_cuts,
                gap * 100 if gap < 1 else -1,
                self.reward,
                self.total_cuts,
                self.actions,
                self.total_time,
            )
        )

        action = self.action_queue.get()
        self.actions[action] += 1
        self.prev_time = time()

        ncuts = -1
        if action == 1:
            cuts = self.separator.get_user_cuts(support_graph)
            for vars, coefs, rhs in cuts:
                self.add(cut=cplex.SparsePair(vars, coefs), sense="L", rhs=rhs)

            ncuts = len(cuts)
            self.total_cuts += ncuts

        self.prev_cuts = ncuts
        self.prev_gap = gap

    def set_attribute(self, separator: CycleSeparator, *args, **kwargs):
        super().set_attribute(separator)
        self.state_extractor: CycleStateExtractor = kwargs["state_extractor"]
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
