from time import time
from typing import *

import numpy as np
from torch.multiprocessing.queue import Queue

from .state_extractor.base import StateExtractor
from ..cplex_api import cplex
from .base import BaseUserCallback


class RewardCalculator:
    def __init__(self, reward_type: str, *args, **kwargs):
        self.reward_type = reward_type

    def get_reward(self, callback) -> float:
        reward = 0
        if self.reward_type == "time":
            if callback.prev_time is not None:
                reward = -(time() - callback.prev_time)
            else:
                reward = 0
        elif self.reward_type == "reward_shaping":
            action_cost = -0.01
            reinforce_cuts = 0
            if callback.prev_cuts > 0:
                reinforce_cuts = callback.prev_cuts * 0.01
            elif callback.prev_cuts == 0:
                reinforce_cuts = -0.1
            reward = action_cost + reinforce_cuts

        return reward


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
        self.reward = self.reward_calculator.get_reward(self)
        self.total_time += -(time() - self.prev_time)
        self.total_reward += self.reward

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
        self.reward_calculator = RewardCalculator(reward_type=self.config["reward_type"])

        self.prev_cuts = -1
        self.prev_gap = 1
        self.prev_time = None
        self.total_cuts = 0
        self.total_time = 0
        self.total_reward = 0
        self.actions = {0: 0, 1: 0}
        self.last_state = None
