from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch as th
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from gym import spaces


class DictTransitionCache:
    def __init__(self,cache_size:int,observation_space: spaces.Space, action_space: spaces.Space,n_envs:int,handle_timeout_termination: bool = True):
        #basic information
        self.cache_size=cache_size
        self.n_envs=n_envs
        self.handle_timeout_termination=handle_timeout_termination
        self.observation_space=observation_space
        self.action_space=action_space
        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)
        #data
        self.observations = {
            key: np.zeros((self.cache_size, self.n_envs) + _obs_shape, dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }
        self.next_observations = {
            key: np.zeros((self.cache_size, self.n_envs) + _obs_shape, dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }
        self.actions = np.zeros((self.cache_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.cache_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.cache_size, self.n_envs), dtype=np.float32)
        self.infos={i:{j:[] for j in range(self.n_envs)} for i in range(self.cache_size)}
        self.timeouts = np.zeros((self.cache_size, self.n_envs), dtype=np.float32)
        #helper data
        self.poses=np.zeros((self.n_envs,),dtype=np.int8)
        self.total_rewards=np.zeros((self.n_envs,),dtype=np.float32)
        self.done_flags=np.zeros((self.n_envs,),dtype=np.bool8)
        self.current_env=0#start with env0

    def clear(self):
        self.poses=np.zeros((self.n_envs,),dtype=np.int8)
        self.total_rewards=np.zeros((self.n_envs,),dtype=np.float32)
        self.done_flags=np.zeros((self.n_envs,),dtype=np.bool8)
        self.infos={i:{j:[] for j in range(self.n_envs)} for i in range(self.cache_size)}

    def cacheMulti(self, obs: Dict[str, np.ndarray], next_obs: Dict[str, np.ndarray], action: np.ndarray, reward: np.ndarray, done: np.ndarray, infos: List[Dict[str, Any]]) -> None:

        # Should be activated in stable_baselines3 1.6.0
        #
        # for key in self.observations.keys():
        #     # Reshape needed when using multiple envs with discrete observations
        #     # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        #     if isinstance(self.observation_space.spaces[key], spaces.Discrete):
        #         obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])

        # for key in self.next_observations.keys():
        #     if isinstance(self.observation_space.spaces[key], spaces.Discrete):
        #         next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
        
        # # Same reshape, for actions
        # if isinstance(self.action_space, spaces.Discrete):
        #     action = action.reshape((self.n_envs, self.action_dim))

        for env in range(self.n_envs):
            #if this is a pause state, skip it
            if "pause" in infos[env]:
                continue

            #do original things
            for key in self.observations.keys():
                self.observations[key][self.poses[env]] = np.array(obs[key][env])
            for key in self.next_observations.keys():
                self.next_observations[key][self.poses[env]] = np.array(next_obs[key][env]).copy()
            self.actions[self.poses[env]] = np.array(action[env]).copy()
            self.rewards[self.poses[env]] = np.array(reward[env]).copy()
            self.dones[self.poses[env]] = np.array(done[env]).copy()
            self.infos[self.poses[env]] = {i:infos[i].copy() for i in range(self.n_envs)}

            if self.handle_timeout_termination:
                self.timeouts[self.poses[env]] = np.array([info.get("TimeLimit.truncated", False) for info in infos[env]])

            self.poses[env] += 1
            self.total_rewards[env]+=reward[env]

            if self.poses[env] == self.cache_size:
                raise Exception("Epsiode in env{} exceeds the cache size {}".format(env,self.cache_size))
            #check done
            if done[env]:
                if "terminal_observation" in infos[env]:
                    self.done_flags[env]=True
                else:
                    raise Exception("Env{} done but has no 'terminal_observation' feild in its info".format(env))

    def cacheSingle(self,obs: Dict[str, np.ndarray], next_obs: Dict[str, np.ndarray], action: np.ndarray, reward: np.ndarray, done: np.ndarray, infos: Dict[str, Any],to_env:int=None):
        #if to_env not assigned, count the env using current env
        if to_env==None:
            to_env=self.current_env
        elif to_env not in range(self.n_envs):
            raise Exception("Cannot cache to env{}, index out of range".format(to_env))

        # Should be activated in stable_baselines3 1.6.0
        #
        # for key in self.observations.keys():
        #     # Reshape needed when using multiple envs with discrete observations
        #     # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        #     if isinstance(self.observation_space.spaces[key], spaces.Discrete):
        #         obs[key] = np.array(obs[key]).reshape((1,) + self.obs_shape[key])

        # for key in self.next_observations.keys():
        #     if isinstance(self.observation_space.spaces[key], spaces.Discrete):
        #         next_obs[key] = np.array(next_obs[key]).reshape((1,) + self.obs_shape[key])
        
        # # Same reshape, for actions
        # if isinstance(self.action_space, spaces.Discrete):
        #     action = action.reshape((1, self.action_dim))

        env=to_env
        #if this is a pause state, skip it
        if "pause" in infos:
            return

        #do original things
        for key in self.observations.keys():
            self.observations[key][self.poses[env]][env] = np.array(obs[key])
        for key in self.next_observations.keys():
            self.next_observations[key][self.poses[env]][env] = np.array(next_obs[key]).copy()
        self.actions[self.poses[env]][env] = np.array(action).copy()
        self.rewards[self.poses[env]][env] = np.array(reward).copy()
        self.dones[self.poses[env]][env] = np.array(done).copy()
        self.infos[self.poses[env]][env]=infos.copy()

        if self.handle_timeout_termination:
            self.timeouts[self.poses[env]][env] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.poses[env] += 1
        self.total_rewards[env]+=reward

        if self.poses[env] == self.cache_size:
            raise Exception("Epsiode in env{} exceeds the cache size {}".format(env,self.cache_size))
        #check done
        if done:
            if not "terminal_observation" in infos:
                self.done_flags[env]=True
                #update the current_env
                self.current_env=(self.current_env+1)%self.n_envs
            else:
                raise Exception("Env{} done but has no 'terminal_observation' feild in its info".format(env))

    def all_done(self):
        """check if every environment has finished its episode"""
        return self.done_flags.all()

    def get_best_trajactory_index(self):
        if self.all_done():
            return self.total_rewards.argmax()
        else:
            raise Exception("Cannot get best trajactory index before all env done")

    def cycle_concat(len,start,end,source:np.ndarray,dest:np.ndarray):
        if np.shape(source[0])==np.shape(dest[0]):
            if start+len<=end:
                dest[start:start+len]=source[:len]
                return start+len
            else:
                len1=end-start
                len2=start+len-end
                dest[start:end]=source[:len1]
                dest[:len2]=source[len1:len]
                return len2
        else:
            raise Exception("Inconsist shape")

    def add_cache_to_buf(self,buf:DictReplayBuffer):
        idx=self.get_best_trajactory_index()
        trajectory_len=self.poses[idx,:]
        buf_pos=buf.pos
        buf_size=buf_size

        for key in self.observations.keys():
            self.cycle_concat(trajectory_len,buf_pos,buf_size,self.observations[key][idx,:],buf.observations[key])
        for key in self.next_observations.keys():
            self.cycle_concat(trajectory_len,buf_pos,buf_size,self.next_observations[key][idx,:],buf.next_observations[key])
        self.cycle_concat(trajectory_len,buf_pos,buf_size,self.actions[key][idx,:],buf.actions[key])
        self.cycle_concat(trajectory_len,buf_pos,buf_size,self.rewards[key][idx,:],buf.rewards[key])
        self.cycle_concat(trajectory_len,buf_pos,buf_size,self.dones[key][idx,:],buf.dones[key])
        if self.handle_timeout_termination:
            self.cycle_concat(trajectory_len,buf_pos,buf_size,self.timeouts[key][idx,:],buf.timeouts[key])

        if buf.pos+trajectory_len<=buf.buffer_size:
            end_pos=self.pos+trajectory_len
        else:
            end_pos=buf.pos+trajectory_len-buf.buffer_size
            buf.full=True
        buf.pos=end_pos
        
        if buf.pos==buf.buffer_size:
            buf.full=True
            buf.pos=0

        print("cache trajectory_len:{}, from env{}".format(trajectory_len,idx))

    def add_one_to_buf(self,i:int,buf:DictReplayBuffer):
        tmp_observations,tmp_next_observations,tmp_actions,tmp_rewards,tmp_dones,tmp_infos=self.at(i)

        for key in buf.observations.keys():
            buf.observations[key][buf.pos] = tmp_observations[key]

        for key in buf.next_observations.keys():
            buf.next_observations[key][buf.pos] = tmp_next_observations[key]

        buf.actions[buf.pos] = tmp_actions
        buf.rewards[buf.pos] = tmp_rewards
        buf.dones[buf.pos] = tmp_dones

        if buf.handle_timeout_termination:
            buf.timeouts[buf.pos] = np.array([info.get("TimeLimit.truncated", False) for info in tmp_infos])

        buf.pos += 1
        if buf.pos == buf.buffer_size:
            buf.full = True
            buf.pos = 0

    def at(self,i:int):
        idx=self.get_best_trajactory_index()
        if i not in range(self.poses[idx]):
            raise Exception("Use index out of range")
        return {key:self.observations[key][i][idx] for key in self.observations.keys()},{key:self.next_observations[key][i][idx] for key in self.next_observations.keys()},self.actions[i][idx],self.rewards[i][idx],self.dones[i][idx],self.infos[i][idx]

