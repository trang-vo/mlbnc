import json
import os
from time import time

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

#from stable_baselines3 import DQN
from modified_lib.dqn import DQN
from lescode.namespace import asdict
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.buffers import DictReplayBuffer
import torch

from .feature_extractor_name import FEATURE_EXTRACTOR_NAME
from .feature_extractors import FeatureExtractor
from .buffers import PriorDictReplayBuffer
from environments.base import BaseCutEnv, PriorCutEnv


class DumpLogsEveryNTimeSteps(BaseCallback):
    def __init__(self, n_steps=500, verbose=1):
        super(DumpLogsEveryNTimeSteps, self).__init__(verbose)
        self.check_freq = n_steps

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            self.model._dump_logs()
        return True


class SaveReplayBufferEveryNTimeSteps(BaseCallback):
    def __init__(self, save_dir: str, n_steps=10000, verbose=1):
        super(SaveReplayBufferEveryNTimeSteps, self).__init__(verbose)
        self.check_freq = n_steps
        self.save_dir = save_dir

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        if self.n_calls % self.check_freq == 0:
            save_path = os.path.join(self.save_dir, "buffer_{}_step.pkl".format(self.n_calls))
            self.model.save_replay_buffer(save_path)
        return True


class EvalCheckpointCallback(EvalCallback):
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Reset success rate buffer
            self._is_success_buffer = []
            
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )
            dump_start=time()
            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(
                episode_lengths
            )
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            if len(self.model.policy.q_net.q_value_recordings) > 0:
                eval_q_values = [torch.Tensor.numpy(i).reshape(-1) for i in self.model.policy.q_net.q_value_recordings]
                statistic_record=[torch.Tensor.numpy(i).reshape(-1) for i in self.model.policy.q_net.statistic_recordings]
                # print("episode_lengths: {}\teval_q_values_length: {}".format(episode_lengths[0],len(eval_q_values)))
                eval_q_values = np.array(eval_q_values[-(episode_lengths[0]):])
                statistic_record = np.array(statistic_record [-(episode_lengths[0]):])
                self.model.policy.q_net.q_value_recordings=[]
                self.model.policy.q_net.statistic_recordings=[]
                mean_q_values={"action0":np.mean(eval_q_values[:,0]),"action1":np.mean(eval_q_values[:,1])}
                # mean_statistics={
                #     "standarized_depth":np.mean(statistic_record[:,0]),
                #     "has_incumbent":np.mean(statistic_record[:,1]),
                #     "gap":np.mean(statistic_record[:,2]),
                #     "proportion_of_determined_edges":np.mean(statistic_record[:,3]),
                #     "cutoff_rate":np.mean(statistic_record[:,4]),
                #     "remain_leaf_proportion":np.mean(statistic_record[:,5]),
                #     "processed_leaf_proportion":np.mean(statistic_record[:,6]),
                #     "sum(lb)/cities":np.mean(statistic_record[:,7]),
                #     "1-sum(lb)/edges":np.mean(statistic_record[:,8]),
                #     "sum(ub)/edges":np.mean(statistic_record[:,9]),
                #     "1-sum(ub)/edges":np.mean(statistic_record[:,10]),
                #     "number_of_close_undetermined_edges/edges":np.mean(statistic_record[:,11]),
                #     }
                if self.verbose > 0:
                    print("Mean q_value: {}".format(mean_q_values))
                eval_log_path=os.path.join(self.logger.get_dir(),"Evaluation_Details.csv")
                if (not os.path.exists(eval_log_path)) or os.path.getsize(eval_log_path)==0:
                    with open(eval_log_path,mode='w',encoding="utf-8") as EvaluationDetails:
                        EvaluationDetails.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format("TS","EPL","Q0","Q1","standarized_depth","has_incumbent","gap","proportion_of_determined_edges","cutoff_rate","remain_leaf_proportion","processed_leaf_proportion","sum(lb)/cities","1-sum(lb)/edges","sum(ub)/edges","1-sum(ub)/edges","determined_edges/edges"))
                with open(eval_log_path,mode='a',encoding="utf-8") as EvaluationDetails:
                    # q_value_log.write("Evaluated at Timestep: {}\t with episode_length: {}\n".format(self.num_timesteps,episode_lengths[0]))
                    for i in range(episode_lengths[0]):
                        EvaluationDetails.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                            self.num_timesteps,
                            episode_lengths[0],
                            eval_q_values[i,0],eval_q_values[i,1],
                            statistic_record[i,0],statistic_record[i,1],statistic_record[i,2],statistic_record[i,3],statistic_record[i,4],statistic_record[i,5],statistic_record[i,6],statistic_record[i,7],statistic_record[i,8],statistic_record[i,9],statistic_record[i,10],statistic_record[i,11]
                            ))
                    print("Record q values in evaluation at {} training steps".format(self.num_timesteps))
                self.logger.record("eval/q_value_0", mean_q_values["action0"])
                self.logger.record("eval/q_value_1", mean_q_values["action1"])

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(
                "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
            )
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    model_path = os.path.join(self.best_model_save_path, "best_model")
                    torch.save({"features_extractor": self.model.q_net.features_extractor.state_dict(),
                                "agent": self.model.q_net.q_net}, model_path)
                self.best_mean_reward = mean_reward

            if mean_reward > -1000:
                if self.best_model_save_path is not None:
                    model_path = os.path.join(self.best_model_save_path, "model_{}_steps.pt".format(self.n_calls))
                    torch.save({"features_extractor": self.model.q_net.features_extractor.state_dict(),
                                "agent": self.model.q_net.q_net}, model_path)

                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()
            print("Dump log cost:{:.2f}".format(time()-dump_start))

        return True


def evaluate(env, model_path):
    model = DQN.load(model_path)

    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)


class DQNAgent:
    def __init__(self) -> None:
        self.model = None

    def train(
            self,
            env,
            eval_env,
            env_config,
            extractor_config,
            model_config,
            learn_config,
            model_folder,
            pretrain_path=None,
            log_path="../logs/",
    ):
        with open(log_path + "config.json", "w") as file:
            json.dump(
                {
                    "dqn": asdict(model_config),
                    "learn": asdict(learn_config),
                    "env": asdict(env_config),
                    "extractor": asdict(extractor_config),
                },
                file,
            )

        device = torch.device(model_config["device"])
        print("Device to train model", device)

        sup_feature_extractor = FEATURE_EXTRACTOR_NAME[
            extractor_config["sup_feature_extractor"]
        ](
            node_dim=env_config["sup_node_dim"],
            edge_dim=env_config["sup_edge_dim"],
            hidden_size=extractor_config["sup_hidden_size"],
            num_layers=extractor_config["sup_num_layers"],
            n_clusters=extractor_config["sup_n_clusters"],
            dropout=extractor_config["sup_dropout"],
            device=device,
        )

        ori_feature_extractor = FEATURE_EXTRACTOR_NAME[
            extractor_config["ori_feature_extractor"]
        ](
            node_dim=env_config["ori_node_dim"],
            edge_dim=env_config["ori_edge_dim"],
            hidden_size=extractor_config["ori_hidden_size"],
            num_layers=extractor_config["ori_num_layers"],
            n_clusters=extractor_config["ori_n_clusters"],
            dropout=extractor_config["ori_dropout"],
            device=device,
        )

        statistic_extractor = FEATURE_EXTRACTOR_NAME[extractor_config["statistic_extractor"]](
            input_size=env_config["statistic_dim"],
            hidden_sizes=extractor_config["statistic_hidden_sizes"],
            output_size=extractor_config["statistic_output_size"],
            device=device,
        )

        policy_kwargs = dict(
            features_extractor_class=FeatureExtractor,
            features_extractor_kwargs={
                "sup_feature_extractor": sup_feature_extractor,
                "ori_feature_extractor": ori_feature_extractor,
                "statistic_extractor": statistic_extractor,
                "device": device,
            },
        )

        replay_buffer_class = None
        if isinstance(env, BaseCutEnv):
            replay_buffer_class = DictReplayBuffer
        elif isinstance(env, PriorCutEnv):
            replay_buffer_class = PriorDictReplayBuffer

        self.model = DQN(
            "MultiInputPolicy", env, replay_buffer_class=replay_buffer_class, policy_kwargs=policy_kwargs,
            **model_config
        )

        if not pretrain_path:
            print("CREATE NEW MODEL")
        else:
            print("LOAD PRETRAIN MODEL")
            self.model.set_parameters(pretrain_path)

        log_callback = DumpLogsEveryNTimeSteps(n_steps=1000)
        save_buffer_callback = SaveReplayBufferEveryNTimeSteps(log_path, n_steps=model_config.buffer_size)
        eval_callback = EvalCheckpointCallback(
            eval_env,
            best_model_save_path=log_path,
            log_path=log_path,
            eval_freq=learn_config.eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=learn_config.n_eval_episodes,
        )
        logger = configure(log_path, ["stdout", "csv", "tensorboard"])

        self.model.set_logger(logger)
        self.model.learn(
            total_timesteps=learn_config.total_timesteps,
            log_interval=learn_config.log_interval,
            callback=[eval_callback, log_callback, save_buffer_callback],
        )
        self.model.save(log_path + model_folder + ".pt")
