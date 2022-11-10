from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from typing import Any, Callable, List, Optional, Sequence, Type, Union
import numpy as np
import gym

class VecRewardDummyVecEnv(DummyVecEnv):
    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        super().__init__(env_fns)
        #override the original reward buf
        self.envs[0].reward_calculator
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)