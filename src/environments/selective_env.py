from torch.multiprocessing import Process
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from typing import Dict,Any
from .base import BaseCutEnv
class SelectiveEnv(BaseCutEnv):
    def __init__(self, config: Dict[str, Any], problem_type: str, cut_type: str, mode="train", result_path="") -> None:
        super().__init__(config, problem_type, cut_type, mode, result_path)