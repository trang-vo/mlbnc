import time
from stable_baselines3 import DQN
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq
from stable_baselines3.common.utils import safe_mean,should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv

from .cache import DictTransitionCache

class selectiveDQN(DQN):
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        #Must have mode attribute
        self.env_training=env.get_attr("mode")[0]=="train"

        if not self.env_training:
            return super(selectiveDQN,self).collect_rollouts(self,env, callback, train_freq, replay_buffer, action_noise, learning_starts, log_interval)
        else:
            return self.collect_selected_rollouts(env, callback, train_freq, replay_buffer, action_noise, learning_starts, log_interval)

    def collect_selected_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        episode_rewards, total_timesteps = [], []

        #current steps in a rollout, used to end the rollout
        num_collected_steps=0
        #current number of episodes in a rollout, used to end the rollout
        num_collected_episodes = 0
        #customed helper parameters
        if not hasattr(self,'corrected_start_time'):
            self.corrected_start_time=self.start_time
        self.cache=DictTransitionCache(3000,env.observation_space,env.action_space,2,self.replay_buffer.handle_timeout_termination)

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

                #total time steps (counting both trajectory, good and bad)
                temp_num_collected_steps+=1
                #add to cache insteaf of replaybuffer
                self._store_transition_into_cache(buffer_action, new_obs, reward, done, infos)
                #if two trajectory both end
                if self.cache.all_done():
                   
                    selected_trajectory_idx=self.cache.get_best_trajactory_index()
                    
                    # --update original parameters--
                    episode_timesteps=self.cache.poses[selected_trajectory_idx]#reward in this episode
                    episode_reward=self.cache.total_rewards[selected_trajectory_idx]#steps in this episode
                    
                    #do original things at each step for the better trejactory
                    for step in range(episode_timesteps):
                        # --callbacks here--
                        # Give access to local variables
                        callback.update_locals(locals())
                        # Only stop training if return value is False, not when it is None.
                        if callback.on_step() is False:
                            return RolloutReturn(0.0, num_collected_steps, num_collected_episodes, continue_training=False)
                        # --do original things--
                        #total steps from begining of this train, used to end the training, controll explore rate and determine when to update target function
                        self.num_timesteps += 1
                        num_collected_steps += 1
                        self._update_info_buffer(self.cache.infos[step][selected_trajectory_idx], self.cache.dones[step][selected_trajectory_idx])#update information of self.ep_info_buffer and self.ep_success_buffer used in _dump_log()
                        self.cache.add_one_to_buf(step,replay_buffer)#add one transition into replaybuffer
                        self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)#update the remaining steps/episode of this training, also used to controll the explore rate
                        self._on_step()#update explore rate and target network
                    
                if not should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
                    break

            if self.cache.all_done():
                num_collected_episodes += 1
                self._episode_num += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()#TODO:modify self.start_time
            #reset the cache
            self.cache.clear()

        mean_reward = np.mean(episode_rewards) if num_collected_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, num_collected_steps, num_collected_episodes, continue_training)

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        if not self.env_training:
            super()._dump_logs()
        else:
            time_elapsed = time.time() - self.corrected_start_time
            fps = int(self.num_timesteps / (time_elapsed + 1e-8))
            self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
            if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
            self.logger.record("time/fps", fps)
            self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            if self.use_sde:
                self.logger.record("train/std", (self.actor.get_std()).mean().item())

            if len(self.ep_success_buffer) > 0:
                self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
            # Pass the number of timesteps for tensorboard
            self.logger.dump(step=self.num_timesteps)

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

# safe parameters
# #almost correct
# self.num_episodes
# self.num_timesteps
# num_collected_steps
# num_collected_episodes
# #used to update at each step, now update once at the end of both epsiode
# episode_timesteps
# episode_reward
# #the counter that counts all trejactories
# temp_num_collected_steps