from typing import Any, Dict, Generator, List, Optional, Union
import numpy as np

from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class PriorDictReplayBuffer(DictReplayBuffer):
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        upper_bound = self.buffer_size if self.full else self.pos
        samples = self.observations["prior"].ravel()[:upper_bound]
        probabilities = samples / samples.sum()
        batch_inds = np.random.choice(range(len(samples)), batch_size, p=probabilities)
        return self._get_samples(batch_inds, env=env)

