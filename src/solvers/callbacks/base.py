import sys
from time import time
from typing import *

import numpy as np
import torch

from utils import distance_solution_cuts
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

        self.processed_nodes = self.get_num_nodes()
        if self.processed_nodes % self.frequent != 0:
            return

        if self.get_MIP_relative_gap() < self.terminal_gap:
            return

        s = time()
        self.actions[1] += 1
        solution = np.asarray(self.get_values())
        support_graph = self.separator.create_support_graph(solution)
        cuts = self.separator.get_user_cuts(support_graph)

        for vars, coefs, sense, rhs in cuts:
            self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)
        self.total_cuts += len(cuts)
        if len(cuts) > 0:
            self.portion_cuts["cuts"] += 1
        else:
            self.portion_cuts["other"] += 1
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
        self.logger = kwargs["logger"] if "logger" in kwargs else sys.stdout
        self.processed_nodes = 0
        self.actions = {0: 0, 1: 0}
        self.portion_cuts = {"cuts": 0, "other": 0}


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
        s = time()
        state, support_graph = self.state_extractor.get_state_representation(self)

        for key in state:
            if isinstance(state[key], np.ndarray):
                state[key] = torch.Tensor(np.asarray([state[key]]))

        with torch.no_grad():
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
        t = time() - s

        msg = "Node {}, add {} user cuts, gap {:.2f}, total cuts {}, {}, predict action in {:.4f}s\n".format(
            processed_leaves,
            self.prev_cuts,
            gap * 100 if gap < 1 else -1,
            self.total_cuts,
            self.actions,
            t,
        )
        if self.logger is not None:
            self.logger.write(msg)
        else:
            print(msg, end="")

    def set_attribute(self, separator, *args, **kwargs):
        super().set_attribute(separator, *args, **kwargs)
        self.state_extractor: StateExtractor = kwargs["state_extractor"]
        self.state_extractor.padding = False
        self.features_extractor = kwargs["features_extractor"]
        self.features_extractor.eval()
        self.agent = kwargs["agent"]
        self.agent.eval()
        self.config: Dict[str, Any] = kwargs["config"]

        self.prev_cuts = 0
        self.prev_gap = 1
        self.total_cuts = 0
        self.total_time = 0
        self.actions = {0: 0, 1: 0}
        self.last_state = None


class MLUserCallback(BaseUserCallback):
    def __call__(self, *args, **kwargs):
        if not self.is_after_cut_loop():
            return

        s = time()
        self.processed_nodes = self.get_num_nodes()

        # Generate cuts at the root node
        if self.processed_nodes == 0:
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
            self.logger.write(msg)
            return

        # Skip generate cuts when the MIP gap is less than 1%
        gap = min(self.prev_gap, self.get_MIP_relative_gap())
        if gap < self.config["terminal_gap"]:
            return

        if self.processed_nodes % self.frequent != 0:
            return

        if self.use_cut_detector and not self.use_rl_agent:
            s = time()
            solution = np.asarray(self.get_values())
            support_graph = self.separator.create_support_graph(solution=solution)
            support_rep = self.state_extractor.get_support_graph_representation(support_graph)
            support_batch = np.zeros(support_rep["sup_lens"][0])
            exist_cuts = self.cut_detector.predict(support_rep["sup_node_feature"], support_rep["sup_edge_index"],
                                                   support_rep["sup_edge_feature"], support_batch)
            self.logger.write("Predict cut existence in {:.4f}s\n".format(time() - s))
            if not exist_cuts:
                self.logger.write("Node {}, cut detector predicts no cuts\n".format(self.processed_nodes))
                self.actions[0] += 1
            else:
                cuts = self.separator.get_user_cuts(support_graph)
                for vars_, coefs, sense, rhs in cuts:
                    self.add(cut=cplex.SparsePair(vars_, coefs), sense=sense, rhs=rhs)
                self.total_cuts += len(cuts)
                self.prev_cuts = len(cuts)
                self.prev_gap = gap
                self.actions[1] += 1
                t = time() - s
                self.logger.write("Node {}, add {} cuts in {:.4f}s\n".format(self.processed_nodes, self.prev_cuts, t))
            return

        elif self.use_rl_agent:
            # Extract state information
            s = time()
            state, support_graph = self.state_extractor.get_state_representation(self)

            # predict the existence of cuts
            if self.use_cut_detector:
                support_batch = np.zeros(state["sup_lens"][0])
                exist_cuts = self.cut_detector.predict(state["sup_node_feature"], state["sup_edge_index"],
                                                       state["sup_edge_feature"], support_batch)
                if not exist_cuts:
                    self.logger.write("Node {}, predicts no cuts in {:.4f}s\n".format(self.processed_nodes, time() - s))
                    return

            for key in state:
                if isinstance(state[key], np.ndarray):
                    state[key] = torch.Tensor(np.asarray([state[key]]))

            with torch.no_grad():
                state_feature = self.features_extractor(state)
                action = self.agent(state_feature)[0].argmax().tolist()
                self.actions[action] += 1
                t_action = time() - s

                ncuts = -1
                if action == 1:
                    s = time()
                    cuts = self.separator.get_user_cuts(support_graph)
                    for vars, coefs, sense, rhs in cuts:
                        self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)

                    ncuts = len(cuts)
                    self.total_cuts += ncuts
                self.prev_cuts = ncuts
                self.prev_gap = gap
            t = time() - s

            msg = "Node {}, add {} user cuts in {:.4f}s, gap {:.2f}, total cuts {}, {}, predict action in {:.4f}s\n".format(
                self.processed_nodes,
                self.prev_cuts,
                t,
                gap * 100 if gap < 1 else -1,
                self.total_cuts,
                self.actions,
                t_action,
            )
            self.logger.write(msg)

        elif not self.use_cut_detector and not self.use_rl_agent:
            self.actions[1] += 1
            solution = np.asarray(self.get_values())
            support_graph = self.separator.create_support_graph(solution)
            cuts = self.separator.get_user_cuts(support_graph)

            for vars, coefs, sense, rhs in cuts:
                self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)
            self.total_cuts += len(cuts)
            msg = "At node {}, add {} user cuts in {:.4f}s, total cuts {}\n".format(
                self.get_num_nodes(), len(cuts), time() - s, self.total_cuts
            )
            self.logger.write(msg)

    def set_attribute(self, separator, *args, **kwargs):
        super().set_attribute(separator, *args, **kwargs)
        self.state_extractor: StateExtractor = kwargs["state_extractor"]
        self.state_extractor.padding = False
        self.config: Dict[str, Any] = kwargs["config"]

        self.use_cut_detector = False
        if "cut_detector" in kwargs:
            self.use_cut_detector = True
            self.cut_detector = kwargs["cut_detector"]
            # self.cut_detector.eval()

        self.use_rl_agent = False
        if "agent" in kwargs:
            self.use_rl_agent = True
            self.features_extractor = kwargs["features_extractor"]
            self.features_extractor.eval()
            self.agent = kwargs["agent"]
            self.agent.eval()

        self.prev_cuts = 0
        self.prev_gap = 1
        self.total_cuts = 0
        self.total_time = 0
        self.actions = {0: 0, 1: 0}
        self.last_state = None


class RandomUserCallback(BaseUserCallback):
    def __call__(self, *args, **kwargs):
        if not self.is_after_cut_loop():
            return

        self.processed_nodes = self.get_num_nodes()
        if self.processed_nodes % self.frequent != 0:
            return

        if self.get_MIP_relative_gap() < self.terminal_gap:
            return

        s = time()
        action = np.random.randint(0, 2)
        self.actions[action] += 1
        if action == 0:
            return
        solution = np.asarray(self.get_values())
        support_graph = self.separator.create_support_graph(solution)
        cuts = self.separator.get_user_cuts(support_graph)

        for vars, coefs, sense, rhs in cuts:
            self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)
        self.total_cuts += len(cuts)
        msg = "At node {}, add {} user cuts in {:.4f}s, total cuts {}\n".format(
            self.processed_nodes, len(cuts), time() - s, self.total_cuts
        )
        if self.logger is not None:
            self.logger.write(msg)
        else:
            print(msg, end="")

    def set_attribute(self, separator: Separator, *args, **kwargs):
        super().set_attribute(separator, *args, **kwargs)
        self.skip_root = kwargs["skip_root"] if "skip_root" in kwargs else False


class MiningUserCallback(BaseUserCallback):
    def __call__(self, *args, **kwargs):
        if not self.is_after_cut_loop():
            return

        if self.get_num_nodes() % self.frequent != 0:
            return

        if self.get_MIP_relative_gap() < self.terminal_gap:
            return

        s = time()
        solution = np.asarray(self.get_values())
        print("Distance current solution and optimal solution", np.linalg.norm(solution - self.optimal_solution))
        support_graph = self.separator.create_support_graph(solution)
        cuts = self.separator.get_user_cuts(support_graph)
        find_cuts_time = time() - s
        print("Time to find cuts", time() - s)

        distances = []
        for vars, coefs, sense, rhs in cuts:
            dist = distance_solution_cuts(self.optimal_solution, vars, coefs, rhs)
            distances.append(dist)

        self.total_cuts += len(cuts)
        if len(cuts) > 0:
            # print("Min distance: {}, max {}".format(min(distances), max(distances)))
            # print("Distances", distances)
            # print("Sum distance", sum([1 - dist for dist in distances]))
            # print("Average distance", sum(distances) / len(distances))
            # print("Diff", 0.3 - sum(distances) / len(distances))
            # print("Median", np.median(np.asarray(distances)))
            # assert 0.1 - min(distances) >= 0

            # bonus = self.max_bonus - np.median(np.asarray(distances))
            # for dist in distances:
            #     # bonus += np.sign(0.5 - dist) * ((0.5 - dist) ** 2)
            #     if dist < 1e-4:
            #         bonus += 1
            #         self.zero_cut += 1
            # else:
            #     bonus += (0.1 - dist) / 2
            # print("Bonus for cuts", bonus)
            # print("Another bonus", 10 * sum([np.sign(0.1 - dist) * (0.1 - dist) * (0.1 - dist) for dist in distances]))
            # if min(distances) < 1e-4 and self.get_num_nodes() > 0:
            # if True:
            for vars, coefs, sense, rhs in cuts:
                self.add(cut=cplex.SparsePair(vars, coefs), sense=sense, rhs=rhs)
            t = time() - s
            print("Time to find a cut", t / len(cuts))
            time_a_cut = t / len(cuts)
            reward = 0
            for dist in distances:
                if dist > 1e-4:
                    reward -= (time_a_cut + dist)
            reward = reward / len(cuts)
            msg = "At node {}, add {} user cuts in {:.4f}s, total cuts {}, pseudo reward {}\n".format(
                self.get_num_nodes(), len(cuts), t, self.total_cuts, reward
            )
            if self.logger is not None:
                self.logger.write(msg)
            else:
                print(msg, end="")
            if self.get_num_nodes() > 0:
                self.rewards.append(reward)
        # if self.get_num_nodes() == 0:
        #     self.state_time.append(find_cuts_time)
        # else:
        #     self.max_bonus = np.ceil(sum(self.state_time) / len(self.state_time) * 100) / 100
        #     print("Average finding cut time", self.max_bonus)

    def set_attribute(self, separator: Separator, *args, **kwargs):
        super().set_attribute(separator, *args, **kwargs)
        self.optimal_solution: np.ndarray = kwargs["optimal_solution"]
        self.zero_cut = 0
        self.rewards = []
        self.state_time = []
        self.max_bonus = 1
