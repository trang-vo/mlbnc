from time import time
import numpy as np
from lescode.config import Config
from lescode.namespace import asdict
def nodes2edge(u, v):
    return (min(u, v), max(u, v))

def config_instance_size(ENV_CONFIG,cities_num):
    temp=asdict(ENV_CONFIG)
    temp["instance_size"]=cities_num
    temp["ori_nEdges"]=cities_num*int(temp["k"])*2
    temp["sup_nNodes"]=cities_num
    temp["sup_nEdges"]=cities_num*3
    temp["data_folder"]="../data/tsp_instances/"+str(cities_num)
    ENV_CONFIG=Config()
    ENV_CONFIG=temp
    return ENV_CONFIG

class RewardCalculator:
    def __init__(self, reward_type: str, *args, **kwargs):
        self.reward_type = reward_type
        #please update
        if reward_type=="vectorized":
            self.vec_reward_num=len(self.assign_reward(0))

    def get_reward(self, callback):
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
        elif self.reward_type == "vectorized":
            gap = min(callback.prev_gap, callback.get_MIP_relative_gap())
            time_cost = -(time() - callback.prev_time)#should be negative
            gap_diff=gap-callback.prev_gap#should be negative
            obj_diff=callback.prev_obj_value - callback.get_objective_value()#should be negative
            action_cost = -0.01
            reinforce_cuts = 0
            if callback.prev_cuts > 0:
                reinforce_cuts = callback.prev_cuts * 0.01
            elif callback.prev_cuts == 0:
                reinforce_cuts = -0.1
            reward = np.array([time_cost,gap_diff,obj_diff,action_cost,reinforce_cuts])
        return reward

    def assign_reward(self,assigned_value):
        reward = 0
        if self.reward_type == "time":
            reward=assigned_value
        elif self.reward_type == "reward_shaping":
            reward=assigned_value
        elif self.reward_type == "vectorized":
            time_cost = 1e6
            gap_diff=assigned_value
            obj_diff=assigned_value
            action_cost = assigned_value
            reinforce_cuts = assigned_value
            reward = [time_cost,gap_diff,obj_diff,action_cost,reinforce_cuts]
        return reward