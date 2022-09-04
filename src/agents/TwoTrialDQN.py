from stable_baselines3 import DQN
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, is_vectorized_observation, polyak_update
from stable_baselines3.dqn.policies import DQNPolicy

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from environments.base import BaseCutEnv

from .cache import DictTransitionCache

class TwoTrialDQN(DQN):
    def __init__(self, policy: Union[str, Type[DQNPolicy]], env: Union[GymEnv, str], learning_rate: Union[float, Schedule] = 0.0001, buffer_size: int = 1000000, learning_starts: int = 50000, batch_size: Optional[int] = 32, tau: float = 1, gamma: float = 0.99, train_freq: Union[int, Tuple[int, str]] = 4, gradient_steps: int = 1, replay_buffer_class: Optional[ReplayBuffer] = None, replay_buffer_kwargs: Optional[Dict[str, Any]] = None, optimize_memory_usage: bool = False, target_update_interval: int = 10000, exploration_fraction: float = 0.1, exploration_initial_eps: float = 1, exploration_final_eps: float = 0.05, max_grad_norm: float = 10, tensorboard_log: Optional[str] = None, create_eval_env: bool = False, policy_kwargs: Optional[Dict[str, Any]] = None, verbose: int = 0, seed: Optional[int] = None, device: Union[th.device, str] = "auto", _init_setup_model: bool = True):
        super().__init__(policy, env, learning_rate, buffer_size, learning_starts, batch_size, tau, gamma, train_freq, gradient_steps, replay_buffer_class, replay_buffer_kwargs, optimize_memory_usage, target_update_interval, exploration_fraction, exploration_initial_eps, exploration_final_eps, max_grad_norm, tensorboard_log, create_eval_env, policy_kwargs, verbose, seed, device, _init_setup_model)
    
    def collect_rollouts(
        self,
        env: BaseCutEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        self.train_mode=env.mode=="train" 
        #init caches
        if self.train_mode:
            self.cache=DictTransitionCache(100,env.observation_space,env.action_space,2,self.replay_buffer.handle_timeout_termination)

        episode_rewards, total_timesteps = [], []
        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            done = False
            episode_reward, episode_timesteps = 0.0, 0

            while not done:

                if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                action, buffer_action = self._sample_action(learning_starts, action_noise)

                # Rescale and perform action
                new_obs, reward, done, infos = env.step(action)

                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, num_collected_steps, num_collected_episodes, continue_training=False)

                self.cache.cacheSingle(new_obs, reward, done, infos)
                if self.cache.all_done():
                    self.cache.add_cache_to_buf(self)
                    
                    selected_trajectory_idx=self.cache.get_best_trajactory_index()
                    selected_episode_steps=self.cache.poses[selected_trajectory_idx]+1
                    selected_episode_rewards=self.cache.total_rewards[selected_trajectory_idx]
                    self.cache.clear()

                    self.num_timesteps += selected_episode_steps
                    episode_reward=selected_episode_rewards
                    episode_timesteps += selected_episode_steps
                    num_collected_steps += selected_episode_steps

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)

                # Store data in replay buffer (normalized action and unnormalized observation)
                self._store_transition(replay_buffer, buffer_action, new_obs, reward, done, infos)

                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self._on_step()

                if not should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
                    break

            if done:
                num_collected_episodes += 1
                self._episode_num += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

        mean_reward = np.mean(episode_rewards) if num_collected_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, num_collected_steps, num_collected_episodes, continue_training)

    def collect_rollouts(
        self,
        env: BaseCutEnv,#modified
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        assert isinstance(env, BaseCutEnv), "You must pass a BaseCutEnv"
        self.env_training=env.mode=="train"

        episode_rewards, total_timesteps = [], []
        num_collected_steps, num_collected_episodes = 0, 0
        #customed helper parameters

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            #done = False #we want to wait both trajectory done, so we use "done" no more
            temp_num_collected_steps=0#a self-defined counter, counter the total number of steps of both trajectory, only used to replace num_collected_steps to keep actor noise changing correctly
            episode_reward, episode_timesteps = 0.0, 0

            while not self.cache.all_done():

                if self.use_sde and self.sde_sample_freq > 0 and temp_num_collected_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                action, buffer_action = self._sample_action(learning_starts, action_noise)

                # Rescale and perform action
                new_obs, reward, done, infos = env.step(action)


                self.num_timesteps += 1
                episode_timesteps += 1
                num_collected_steps += 1

                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, num_collected_steps, num_collected_episodes, continue_training=False)

                episode_reward += reward
                #if is training
                
                #total time steps
                temp_num_collected_steps+=1
                self._store_transition_into_cache(buffer_action, new_obs, reward, done, infos)

                if self.cache.all_done():
                   
                    selected_trajectory_idx=self.cache.get_best_trajactory_index()
                    selected_episode_steps=self.cache.poses[selected_trajectory_idx]+1
                    selected_episode_rewards=self.cache.total_rewards[selected_trajectory_idx]
                    
                    #update original parameters
                    self.num_timesteps += selected_episode_steps#total steps from begining of this train
                    episode_reward=selected_episode_rewards#reward in this episode
                    episode_timesteps = selected_episode_steps#steps in this episode
                    num_collected_steps += selected_episode_steps#current steps in a rollout
                    
                    #do original things
                    for step in range(self.cache.poses[selected_trajectory_idx]):
                        self._update_info_buffer(self.cache.infos[step][selected_trajectory_idx], self.cache.dones[step][selected_trajectory_idx])
                        self._store_transition(replay_buffer,self.cache.actions[step][selected_trajectory_idx],)
                        self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
                        self._on_step()

                    self.cache.clear()
                    
                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)

                # Store data in replay buffer (normalized action and unnormalized observation)
                self._store_transition(replay_buffer, buffer_action, new_obs, reward, done, infos)

                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self._on_step()

                if not should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
                    break

            if done:
                num_collected_episodes += 1
                self._episode_num += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

        mean_reward = np.mean(episode_rewards) if num_collected_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, num_collected_steps, num_collected_episodes, continue_training)

    def _store_transition_into_cache(
        self,
        buffer_action: np.ndarray,
        new_obs: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        if done and infos[0].get("terminal_observation") is not None:
            next_obs = infos[0]["terminal_observation"]
            # VecNormalize normalizes the terminal observation
            if self._vec_normalize_env is not None:
                next_obs = self._vec_normalize_env.unnormalize_obs(next_obs)
        else:
            next_obs = new_obs_

        self.cache.cacheSingle(
            self._last_original_obs,
            next_obs,
            buffer_action,
            reward_,
            done,
            infos,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_