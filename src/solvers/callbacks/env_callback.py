from time import time
from typing import *

import numpy as np
from torch.multiprocessing.queue import Queue

from utils import distance_solution_cuts
from .state_extractor.base import StateExtractor
from ..cplex_api import cplex
from .base import BaseUserCallback
from constant import TOLERANCE


class RewardCalculator:
    def __init__(self, reward_type: str, *args, **kwargs):
        self.reward_type = reward_type

    def get_reward(self, callback, **kwargs) -> float:
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
        elif self.reward_type == "time_reward_shaping":
            if callback.prev_time is not None:
                reward = -(time() - callback.prev_time)
            else:
                reward = 0

            bonus = 0
            # if callback.prev_remain_nodes is not None:
            #     num_remaining_nodes = callback.get_num_remaining_nodes()
            #     diff_remaining_nodes = - (num_remaining_nodes - callback.prev_remain_nodes) / 10
            #     bonus += diff_remaining_nodes
            #     print("Bonus from diff remaining nodes", diff_remaining_nodes)

            if callback.prev_obj is not None:
                print(callback.has_incumbent(), callback.get_cutoff())
                diff_obj = (-callback.prev_obj + callback.get_objective_value()) / callback.get_best_objective_value()
                print("Bonus from diff obj", diff_obj)
                bonus += diff_obj

            if callback.prev_gap is not None:
                diff_gap = (callback.prev_gap - kwargs["gap"]) * 100
                print("Bonus from diff gap", diff_gap)
                bonus += diff_gap

            reward += bonus
        elif self.reward_type == "gap_reward_shaping":
            assert "gap" in kwargs, "Provide gap to compute reward"
            reward = (callback.prev_gap - kwargs["gap"]) * 100
            if abs(reward) < TOLERANCE:
                reward = -0.01
                if callback.prev_cuts == 0:
                    reward = -0.1
            rw_shaping = 0
            if callback.prev_cuts > 0:
                rw_shaping += callback.prev_cuts * 0.01
            if callback.prev_obj is not None:
                rw_shaping += abs(callback.prev_obj - callback.get_objective_value()) / callback.get_objective_value()
            reward += rw_shaping
        elif self.reward_type == "time_distance":
            if callback.prev_time is not None:
                reward = -(time() - callback.prev_time)
            else:
                reward = 0

            rw_shaping = 0
            bonus_dist = 0
            for vars, coefs, sense, rhs in callback.prev_list_cuts:
                dist = distance_solution_cuts(callback.optimal_solution, vars, coefs, rhs)
                bonus_dist += np.sign(0.1 - dist) * ((0.1 - dist) ** 2) * 10
            rw_shaping += bonus_dist
            print("Bonus from distance", bonus_dist)

            reward += rw_shaping
        elif self.reward_type == "relative_time_distance":
            if callback.prev_time is not None:
                reward = -(time() - callback.prev_time)
            else:
                reward = 0

            if len(callback.prev_list_cuts) > 0:
                time_find_a_cut = reward / len(callback.prev_list_cuts)
                distances = []
                for vars, coefs, sense, rhs in callback.prev_list_cuts:
                    dist = distance_solution_cuts(callback.optimal_solution, vars, coefs, rhs)
                    distances.append(dist)

                distances = np.asarray(distances)
                cut_cost = np.sum(np.where(distances > TOLERANCE, 1, 0)) * time_find_a_cut - np.mean(distances)
                reward = cut_cost / len(callback.prev_list_cuts)

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
            self.state_queue.put((None, -1e6, True, {"total_time": -1e+6}))
            self.abort()

        # Define the initial state for an episode, i.e., the initial state = the processed (add all possible cuts +
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
            self.prev_remain_nodes = self.get_num_remaining_nodes()
            self.prev_obj = self.get_objective_value()
            self.prev_cutoff = self.get_cutoff()
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
        self.reward = self.reward_calculator.get_reward(self, gap=gap)
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
        self.prev_remain_nodes = self.get_num_remaining_nodes()
        self.prev_obj = self.get_objective_value()
        self.prev_cutoff = self.get_cutoff()

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
        self.prev_remain_nodes = None
        self.prev_obj = None
        self.prev_cutoff = None

        self.total_cuts = 0
        self.total_time = 0
        self.total_reward = 0

        self.actions = {0: 0, 1: 0}
        self.last_state = None


class ArbitraryStartEnvUserCallback(EnvUserCallback):
    def __call__(self, *args, **kwargs):
        if not self.is_after_cut_loop():
            return

        # Define the terminal state for an episode, i.e., the terminal state = the leaf which has gap less than 1%
        gap = min(self.prev_gap, self.get_MIP_relative_gap())
        if gap < self.config["terminal_gap"]:
            if self.start_episode_flag:
                info = {
                    "terminal_observation": self.last_state,
                    "total_time": self.total_time,
                }
                self.state_queue.put((self.last_state, 0, True, info))
            self.abort()
            return

        self.check_start_condition()
        s = time()
        processed_leaves = self.get_num_nodes()

        # If the env mode is evaluated and the agent predicts only one action kind for more than K first leaves,
        # then early stop
        if (
                self.env_mode == "eval"
                and self.actions[1] * self.actions[0] == 0
                and processed_leaves > self.config["limited_one_action"]
        ):
            self.state_queue.put((None, -1e6, True, {"total_time": -1e+6}))
            self.abort()

        if not self.start_episode_flag:
            action = np.random.randint(0, 2)
            if action == 0 and processed_leaves != 0:
                return
            solution = np.asarray(self.get_values())
            support_graph = self.separator.create_support_graph(solution)
            cuts = self.separator.get_user_cuts(support_graph)

            for vars, coefs, sense, rhs in cuts:
                self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)

            self.start = time()
            self.prev_cuts = len(cuts)
            self.prev_gap = min(self.prev_gap, self.get_MIP_relative_gap())
            self.prev_time = time()
            self.prev_remain_nodes = self.get_num_remaining_nodes()
            self.prev_obj = self.get_objective_value()
            self.prev_cutoff = self.get_cutoff()
            self.prev_list_cuts = cuts
            self.total_time += s - time()
            self.total_cuts += len(cuts)
            gap = min(self.prev_gap, self.get_MIP_relative_gap())
            msg = "Node {}, add {} user cuts, gap {:.2f}, total cuts {}, {}, total time {}\n".format(
                processed_leaves,
                self.prev_cuts,
                gap * 100 if gap < 1 else -1,
                self.total_cuts,
                self.actions,
                self.total_time,
            )
            if self.logger is not None and self.env_mode == "eval":
                self.logger.write(msg)
            else:
                print(msg, end="")

            return

        # Extract state information
        state, support_graph = self.state_extractor.get_state_representation(self)
        self.last_state = state

        # Apply time limit for an episode
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
        self.reward = self.reward_calculator.get_reward(self, gap=gap)
        if self.prev_time is not None:
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
        cuts = []
        if action == 1:
            cuts = self.separator.get_user_cuts(support_graph)
            for vars, coefs, sense, rhs in cuts:
                self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)
            ncuts = len(cuts)
            self.total_cuts += ncuts
        self.prev_cuts = ncuts
        self.prev_list_cuts = cuts
        self.prev_gap = gap
        self.prev_remain_nodes = self.get_num_remaining_nodes()
        self.prev_obj = self.get_objective_value()
        self.prev_cutoff = self.get_cutoff()

    def set_attribute(self, separator, *args, **kwargs):
        super().set_attribute(separator, *args, **kwargs)
        assert "start_criterion" in kwargs, "Need to provide a criterion to determine the initial state"
        self.start_condition = kwargs["start_criterion"]
        self.start_episode_flag = False
        self.prev_list_cuts = []

        if self.config["reward_type"] in ["time_distance", "relative_time_distance"]:
            assert "optimal_solution" in kwargs, "Need to provide the optimal solution to compute reward"
            self.optimal_solution: np.array = kwargs["optimal_solution"]

    def check_start_condition(self):
        start_value, criterion = self.start_condition
        if self.start_episode_flag:
            return

        if criterion == "gap":
            if min(1, self.get_MIP_relative_gap()) <= start_value and self.get_num_nodes() > 0:
                self.start_episode_flag = True
                return
        elif criterion == "node_id":
            if self.get_num_nodes() >= start_value:
                self.start_episode_flag = True
                return

