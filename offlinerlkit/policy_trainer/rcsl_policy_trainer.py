import time
import os

import numpy as np
import torch
import gym
import gymnasium

from typing import Optional, Dict, List, Tuple, Union
from tqdm import tqdm
from collections import deque
from torch.utils.data import DataLoader

from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import BasePolicy, RcslPolicy, RcslGaussianPolicy, SimpleDiffusionPolicy
from offlinerlkit.utils.dataset import DictDataset


# rcsl policy trainer
class RcslPolicyTrainer:
    def __init__(
        self,
        policy: Union[RcslPolicy, RcslGaussianPolicy, SimpleDiffusionPolicy],
        eval_env: Union[gym.Env, gymnasium.Env],
        offline_dataset: Dict[str, np.ndarray],
        rollout_dataset: Optional[Dict[str, np.ndarray]],
        goal: float,
        logger: Logger,
        seed,
        # rollout_setting: Tuple[int, int, int],
        eval_env2: Optional[Union[gym.Env, gymnasium.Env]] = None,
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        offline_ratio: float = 0,
        eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        # dynamics_update_freq: int = 0,
        horizon: Optional[int] = None,
        num_workers = 1,
        has_terminal = False
        # device = 'cpu'
    ) -> None:
        '''
        offline_ratio = 0: rollout only, 1: offline only
        '''
        self.policy = policy
        self.eval_env = eval_env
        self.eval_env2 = eval_env2
        self.horizon = horizon
        self.offline_dataset = offline_dataset
        self.rollout_dataset = rollout_dataset
        self.goal = goal
        self.logger = logger
        # self.device = device

        # self._rollout_freq, self._rollout_batch_size, \
        #     self._rollout_length = rollout_setting
        # self._dynamics_update_freq = dynamics_update_freq

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._offline_ratio = offline_ratio
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler
        self.num_workers = num_workers
        self.env_seed = seed

        self.is_gymnasium_env = hasattr(self.eval_env, "get_true_observation")
        assert (not self.is_gymnasium_env) or (self.horizon is not None), "Horizon must be specified for Gymnasium env"
        self.has_terminal = has_terminal

    # def train(self) -> Dict[str, float]:
    #     start_time = time.time()

    #     num_timesteps = 0
    #     last_10_performance = deque(maxlen=10)

    #     # Version 1: Rollout only
    #     if self._offline_ratio == 0:
    #         data_loader = DataLoader(
    #             DictDataset(self.rollout_dataset),
    #             batch_size = self._batch_size,
    #             shuffle = True,
    #             pin_memory = True,
    #             num_workers = self.num_workers
    #         )
    #     elif self._offline_ratio == 1:
    #         data_loader = DataLoader(
    #             DictDataset(self.offline_dataset),
    #             batch_size = self._batch_size,
    #             shuffle = True,
    #             pin_memory = True,
    #             num_workers = self.num_workers
    #         )    
    #     else:
    #         raise NotImplementedError       

    #     # train loop
    #     eval_info = self._evaluate()
    #     ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
    #     print(f"Mean: {ep_reward_mean}, std: {ep_reward_std}")
    #     for e in range(1, self._epoch + 1):

    #         self.policy.train()

    #         pbar = tqdm(enumerate(data_loader), desc=f"Epoch #{e}/{self._epoch}")
    #         for it, batch in pbar:
    #             # Sample from both offline (offline data) and rollout (rollout data) according to offline_ratio
    #             # offline_sample_size = int(self._batch_size * self._offline_ratio)
    #             # rollout_sample_size = self._batch_size - offline_sample_size
    #             # offline_batch = self.offline_buffer.sample(batch_size=offline_sample_size)
    #             # rollout_batch = self.rollout_buffer.sample(batch_size=rollout_sample_size)
    #             # batch = {"offline": offline_batch, "rollout": rollout_batch}
    #             '''
    #             batch: dict with keys
    #                 'observations'
    #                 'next_observations'
    #                 'actions'
    #                 'terminals'
    #                 'rewards'
    #                 'rtgs'

    #             '''
    #             loss_dict = self.policy.learn(batch)
    #             pbar.set_postfix(**loss_dict)

    #             for k, v in loss_dict.items():
    #                 self.logger.logkv_mean(k, v)
                
    #             # update the dynamics if necessary
    #             # if 0 < self._dynamics_update_freq and (num_timesteps+1)%self._dynamics_update_freq == 0:
    #             #     dynamics_update_info = self.policy.update_dynamics(self.offline_buffer)
    #             #     for k, v in dynamics_update_info.items():
    #             #         self.logger.logkv_mean(k, v)
                
    #             num_timesteps += 1

    #         if self.lr_scheduler is not None:
    #             self.lr_scheduler.step()
            
    #         # evaluate current policy
    #         eval_info = self._evaluate()
    #         ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
    #         ep_reward_max, ep_reward_min = np.max(eval_info["eval/episode_reward"]), np.min(eval_info["eval/episode_reward"])
    #         ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])

    #         if not hasattr(self.eval_env, "get_normalized_score"): # gymnasium_env does not have normalized score
    #             last_10_performance.append(ep_reward_mean)
    #             self.logger.logkv("eval/episode_reward", ep_reward_mean)
    #             self.logger.logkv("eval/episode_reward_std", ep_reward_std)         
    #         else:       
    #             norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
    #             norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
    #             norm_ep_rew_max = self.eval_env.get_normalized_score(ep_reward_max) * 100
    #             norm_ep_rew_min = self.eval_env.get_normalized_score(ep_reward_min) * 100
    #             last_10_performance.append(norm_ep_rew_mean)
    #             self.logger.logkv("eval/normalized_episode_reward", norm_ep_rew_mean)
    #             self.logger.logkv("eval/normalized_episode_reward_std", norm_ep_rew_std)
    #             self.logger.logkv("eval/normalized_episode_reward_max", norm_ep_rew_max)
    #             self.logger.logkv("eval/normalized_episode_reward_min", norm_ep_rew_min)
    #         self.logger.logkv("eval/episode_length", ep_length_mean)
    #         self.logger.logkv("eval/episode_length_std", ep_length_std)

    #         if self.eval_env2 is not None:
    #             eval_info_no_fix = self._evaluate_no_fix_seed()
    #             ep_reward_mean_no_fix, ep_reward_std_no_fix = np.mean(eval_info_no_fix["eval/episode_reward"]), np.std(eval_info_no_fix["eval/episode_reward"])
    #             ep_length_mean_no_fix, ep_length_std_no_fix = np.mean(eval_info_no_fix["eval/episode_length"]), np.std(eval_info_no_fix["eval/episode_length"])
    #             if not hasattr(self.eval_env, "get_normalized_score"): # gymnasium_env does not have normalized score
    #                 last_10_performance.append(ep_reward_mean)
    #                 self.logger.logkv("eval/episode_reward_no_fix_seed", ep_reward_mean_no_fix)
    #                 self.logger.logkv("eval/episode_reward_std_no_fix_seed", ep_reward_std_no_fix)         
    #             else:       
    #                 norm_ep_rew_mean_no_fix = self.eval_env.get_normalized_score(ep_reward_mean_no_fix) * 100
    #                 norm_ep_rew_std_no_fix = self.eval_env.get_normalized_score(ep_reward_std_no_fix) * 100
    #                 last_10_performance.append(norm_ep_rew_mean)
    #                 self.logger.logkv("eval/normalized_episode_reward_no_fix_seed", norm_ep_rew_mean_no_fix)
    #                 self.logger.logkv("eval/normalized_episode_reward_std_no_fix_seed", norm_ep_rew_std_no_fix)
    #             self.logger.logkv("eval/episode_length_no_fix_seed", ep_length_mean_no_fix)
    #             self.logger.logkv("eval/episode_length_std_no_fix_seed", ep_length_std_no_fix)

    #         self.logger.set_timestep(num_timesteps)
    #         self.logger.dumpkvs(exclude=["dynamics_training_progress"])
        
    #         # save checkpoint
    #         torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy.pth"))

    #     self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
    #     torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))

    #     # self.policy.dynamics.save(self.logger.model_dir)
    #     self.logger.close()
    
    #     return {"last_10_performance": np.mean(last_10_performance)}

    # def _evaluate(self) -> Dict[str, List[float]]:
    #     # Pointmaze obs has different format, needs to be treated differently
    #     is_gymnasium_env = self.is_gymnasium_env

    #     self.eval_env.reset(seed=self.env_seed) # Fix seed
        
    #     self.policy.eval()
    #     if is_gymnasium_env:
    #         obs, _ = self.eval_env.reset()
    #         obs = self.eval_env.get_true_observation(obs)
    #     else:
    #         obs = self.eval_env.reset()
            

    #     eval_ep_info_buffer = []
    #     num_episodes = 0
    #     episode_reward, episode_length = 0, 0

    #     if not self.has_terminal: # pointmaze environment, don't use horizon
    #         while num_episodes < self._eval_episodes:
    #             rtg = torch.tensor([[self.goal]]).type(torch.float32)
    #             for timestep in range(self.horizon): # One epoch
    #                 # print(f"Timestep {timestep}, obs {obs}")
    #                 action = self.policy.select_action(obs.reshape(1, -1), rtg)
    #                 if hasattr(self.eval_env, "get_true_observation"): # gymnasium env 
    #                     next_obs, reward, terminal, _, _ = self.eval_env.step(action.flatten())
    #                 else:
    #                     next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
    #                 if is_gymnasium_env:
    #                     next_obs = self.eval_env.get_true_observation(next_obs)
    #                 # if num_episodes == 2 and timestep < 10:
    #                 #     print(f"Action {action}, next_obs {next_obs}, reward {reward}, rtg {rtg.item()}")
    #                 episode_reward += reward
    #                 # rtg = rtg - reward
    #                 episode_length += 1

    #                 obs = next_obs

    #                 # if terminal:
    #                 #     break # Stop current epoch
    #             eval_ep_info_buffer.append(
    #                 {"episode_reward": episode_reward, "episode_length": episode_length}
    #             )
    #             num_episodes +=1
    #             episode_reward, episode_length = 0, 0
    #             if is_gymnasium_env:
    #                 obs, _ = self.eval_env.reset()
    #                 obs = self.eval_env.get_true_observation(obs)
    #             else:
    #                 obs = self.eval_env.reset()
    #     else:
    #         rtg = torch.tensor([[self.goal]]).type(torch.float32)
    #         while num_episodes < self._eval_episodes:
    #             # print(f"Timestep {timestep}, obs {obs}")
    #             action = self.policy.select_action(obs.reshape(1, -1), rtg)
    #             if hasattr(self.eval_env, "get_true_observation"): # gymnasium env 
    #                 next_obs, reward, terminal, _, _ = self.eval_env.step(action.flatten())
    #             else:
    #                 next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
    #             if is_gymnasium_env:
    #                 next_obs = self.eval_env.get_true_observation(next_obs)
    #             episode_reward += reward
    #             rtg = rtg - reward
    #             episode_length += 1

    #             obs = next_obs

    #             if terminal: # Episode finishes
    #                 eval_ep_info_buffer.append(
    #                     {"episode_reward": episode_reward, "episode_length": episode_length}
    #                 )
    #                 num_episodes +=1
    #                 episode_reward, episode_length = 0, 0
    #                 if is_gymnasium_env:
    #                     obs, _ = self.eval_env.reset()
    #                     obs = self.eval_env.get_true_observation(obs)
    #                 else:
    #                     obs = self.eval_env.reset()
    #                 rtg = torch.tensor([[self.goal]]).type(torch.float32)
        
    #     return {
    #         "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
    #         "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
    #     }

    def train(self, holdout_ratio: float = 0.2) -> Dict[str, float]:
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = deque(maxlen=10)

        dataset = DictDataset(self.offline_dataset)

        # holdout_size = int(len(dataset) * holdout_ratio)
        # train_size = len(dataset) - holdout_size
        # train_dataset, holdout_dataset = torch.utils.data.random_split(dataset, [train_size, holdout_size], 
        #                                                                generator=torch.Generator().manual_seed(self.env_seed))
        data_loader = DataLoader(
            dataset,
            batch_size = self._batch_size,
            shuffle = True,
            pin_memory = True,
            num_workers = self.num_workers
        )
        # Version 1: Rollout only
        # if self._offline_ratio == 0:
        #     data_loader = DataLoader(
        #         DictDataset(self.rollout_dataset),
        #         batch_size = self._batch_size,
        #         shuffle = True,
        #         pin_memory = True,
        #         num_workers = self.num_workers
        #     )
        # elif self._offline_ratio == 1:
        #     data_loader = DataLoader(
        #         DictDataset(self.offline_dataset),
        #         batch_size = self._batch_size,
        #         shuffle = True,
        #         pin_memory = True,
        #         num_workers = self.num_workers
        #     )    
        # else:
        #     raise NotImplementedError       

        # train loop
        # eval_info = self._evaluate_vector()
        # ep_reward_mean, ep_reward_std = np.mean(eval_info["eval_vec/episode_reward"]), np.std(eval_info["eval_vec/episode_reward"])
        # print(f"Mean: {ep_reward_mean}, std: {ep_reward_std}")
        # eval_info = self._evaluate()
        # ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
        # print(f"Mean: {ep_reward_mean}, std: {ep_reward_std}")
        for e in range(1, self._epoch + 1):

            self.policy.train()

            pbar = tqdm(enumerate(data_loader), desc=f"Epoch #{e}/{self._epoch}")
            for it, batch in pbar:
                # Sample from both offline (offline data) and rollout (rollout data) according to offline_ratio
                # offline_sample_size = int(self._batch_size * self._offline_ratio)
                # rollout_sample_size = self._batch_size - offline_sample_size
                # offline_batch = self.offline_buffer.sample(batch_size=offline_sample_size)
                # rollout_batch = self.rollout_buffer.sample(batch_size=rollout_sample_size)
                # batch = {"offline": offline_batch, "rollout": rollout_batch}
                '''
                batch: dict with keys
                    'observations'
                    'next_observations'
                    'actions'
                    'terminals'
                    'rewards'
                    'rtgs'

                '''
                loss_dict = self.policy.learn(batch)
                pbar.set_postfix(**loss_dict)

                for k, v in loss_dict.items():
                    self.logger.logkv_mean(k, v)
                
                # update the dynamics if necessary
                # if 0 < self._dynamics_update_freq and (num_timesteps+1)%self._dynamics_update_freq == 0:
                #     dynamics_update_info = self.policy.update_dynamics(self.offline_buffer)
                #     for k, v in dynamics_update_info.items():
                #         self.logger.logkv_mean(k, v)
                
                num_timesteps += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Test validation loss
            # self.validate(holdout_dataset)
            
            # evaluate current policy
            # eval_vec_info = self._evaluate_vector()
            # ep_reward_mean_vec, ep_reward_std_vec = np.mean(eval_vec_info["eval_vec/episode_reward"]), np.std(eval_vec_info["eval_vec/episode_reward"])
            # ep_reward_max_vec, ep_reward_min_vec = np.max(eval_vec_info["eval_vec/episode_reward"]), np.min(eval_vec_info["eval_vec/episode_reward"])
            # ep_length_mean_vec, ep_length_std_vec = np.mean(eval_vec_info["eval_vec/episode_length"]), np.std(eval_vec_info["eval_vec/episode_length"])
            # pick_success_vec = np.mean(eval_vec_info["eval_vec/pick_success"])

            eval_info = self._evaluate()
            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_reward_max, ep_reward_min = np.max(eval_info["eval/episode_reward"]), np.min(eval_info["eval/episode_reward"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
            # pick_success = np.mean(eval_info["eval/pick_success"])

            if not hasattr(self.eval_env, "get_normalized_score"): # gymnasium_env does not have normalized score
                last_10_performance.append(ep_reward_mean)
                self.logger.logkv("eval/episode_reward", ep_reward_mean)
                self.logger.logkv("eval/episode_reward_std", ep_reward_std)  
                # self.logger.logkv("eval/pick_success", pick_success)    
                # self.logger.logkv("eval_vec/episode_reward", ep_reward_mean_vec)
                # self.logger.logkv("eval_vec/episode_reward_std", ep_reward_std_vec)   
                # self.logger.logkv("eval_vec/pick_success", pick_success_vec)
                    
            else:       
                norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
                norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
                norm_ep_rew_max = self.eval_env.get_normalized_score(ep_reward_max) * 100
                norm_ep_rew_min = self.eval_env.get_normalized_score(ep_reward_min) * 100
                last_10_performance.append(norm_ep_rew_mean)
                self.logger.logkv("eval/normalized_episode_reward", norm_ep_rew_mean)
                self.logger.logkv("eval/normalized_episode_reward_std", norm_ep_rew_std)
                self.logger.logkv("eval/normalized_episode_reward_max", norm_ep_rew_max)
                self.logger.logkv("eval/normalized_episode_reward_min", norm_ep_rew_min)
            self.logger.logkv("eval/episode_length", ep_length_mean)
            self.logger.logkv("eval/episode_length_std", ep_length_std)

            # if self.eval_env2 is not None:
            #     eval_info_no_fix = self._evaluate_no_fix_seed()
            #     ep_reward_mean_no_fix, ep_reward_std_no_fix = np.mean(eval_info_no_fix["eval/episode_reward"]), np.std(eval_info_no_fix["eval/episode_reward"])
            #     ep_length_mean_no_fix, ep_length_std_no_fix = np.mean(eval_info_no_fix["eval/episode_length"]), np.std(eval_info_no_fix["eval/episode_length"])
            #     if not hasattr(self.eval_env, "get_normalized_score"): # gymnasium_env does not have normalized score
            #         last_10_performance.append(ep_reward_mean)
            #         self.logger.logkv("eval/episode_reward_no_fix_seed", ep_reward_mean_no_fix)
            #         self.logger.logkv("eval/episode_reward_std_no_fix_seed", ep_reward_std_no_fix)         
            #     else:       
            #         norm_ep_rew_mean_no_fix = self.eval_env.get_normalized_score(ep_reward_mean_no_fix) * 100
            #         norm_ep_rew_std_no_fix = self.eval_env.get_normalized_score(ep_reward_std_no_fix) * 100
            #         last_10_performance.append(norm_ep_rew_mean)
            #         self.logger.logkv("eval/normalized_episode_reward_no_fix_seed", norm_ep_rew_mean_no_fix)
            #         self.logger.logkv("eval/normalized_episode_reward_std_no_fix_seed", norm_ep_rew_std_no_fix)
            #     self.logger.logkv("eval/episode_length_no_fix_seed", ep_length_mean_no_fix)
            #     self.logger.logkv("eval/episode_length_std_no_fix_seed", ep_length_std_no_fix)

            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs(exclude=["dynamics_training_progress"])
        
            # save checkpoint
            torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy.pth"))

        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))

        # self.policy.dynamics.save(self.logger.model_dir)
        self.logger.close()
    
        return {"last_10_performance": np.mean(last_10_performance)}

    def _evaluate(self) -> Dict[str, List[float]]:
        '''
        Always set desired rtg to 0
        '''
        # Pointmaze obs has different format, needs to be treated differently
        is_gymnasium_env = self.is_gymnasium_env

        self.eval_env.reset(seed=self.env_seed) # Fix seed
        
        self.policy.eval()
        if is_gymnasium_env:
            obs, _ = self.eval_env.reset()
            obs = self.eval_env.get_true_observation(obs)
        else:
            obs = self.eval_env.reset()
            

        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        if not self.has_terminal: # pointmaze environment, don't use horizon
            while num_episodes < self._eval_episodes:
                rtg = torch.tensor([[self.goal]]).type(torch.float32)
                pick_success = False
                for timestep in range(self.horizon): # One epoch
                    # print(f"Timestep {timestep}, obs {obs}")
                    action = self.policy.select_action(obs.reshape(1, -1), rtg)
                    if hasattr(self.eval_env, "get_true_observation"): # gymnasium env 
                        next_obs, reward, terminal, _, _ = self.eval_env.step(action.flatten())
                    else:
                        next_obs, reward, terminal, info = self.eval_env.step(action.flatten())
                    if is_gymnasium_env:
                        next_obs = self.eval_env.get_true_observation(next_obs)

                    if timestep == 161:
                        print(f"Timestep {timestep}, action {action}")
                    # if num_episodes == 2 and timestep < 10:
                    #     print(f"Action {action}, next_obs {next_obs}, reward {reward}, rtg {rtg.item()}")
                    episode_reward += reward
                    # No need to update return
                    rtg = rtg - reward
                    episode_length += 1

                    # if info['grasp_success_target']: # pick okay
                    #     # pick_success = True

                    obs = next_obs

                    # if terminal:
                    #     break # Stop current epoch
                # print(episode_reward)
                # episode_reward = 1 if episode_reward > 0 else 0 # Clip to 1
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length,
                    #  "pick_success": float(pick_success)
                    }
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                if is_gymnasium_env:
                    obs, _ = self.eval_env.reset()
                    obs = self.eval_env.get_true_observation(obs)
                else:
                    obs = self.eval_env.reset()
        else:
            rtg = torch.tensor([[self.goal]]).type(torch.float32)
            while num_episodes < self._eval_episodes:
                # print(f"Timestep {timestep}, obs {obs}")
                action = self.policy.select_action(obs.reshape(1, -1), rtg)
                if hasattr(self.eval_env, "get_true_observation"): # gymnasium env 
                    next_obs, reward, terminal, _, _ = self.eval_env.step(action.flatten())
                else:
                    next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
                if is_gymnasium_env:
                    next_obs = self.eval_env.get_true_observation(next_obs)
                episode_reward += reward
                # rtg = rtg - reward
                episode_length += 1

                obs = next_obs

                if terminal: # Episode finishes
                    # print(episode_reward)
                    episode_reward = 1 if episode_reward > 0 else 0 # Clip to 1
                    eval_ep_info_buffer.append(
                        {"episode_reward": episode_reward, "episode_length": episode_length}
                    )
                    episode_reward, episode_length = 0, 0
                    if is_gymnasium_env:
                        obs, _ = self.eval_env.reset()
                        obs = self.eval_env.get_true_observation(obs)
                    else:
                        obs = self.eval_env.reset()
                    rtg = torch.tensor([[self.goal]]).type(torch.float32)
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer],
            # "eval/pick_success": [ep_info["pick_success"] for ep_info in eval_ep_info_buffer]
        }
  
    def _evaluate_no_fix_seed(self) -> Dict[str, List[float]]:
        '''
        Use self.eval_env2, which does not fix seed in every epoch
        '''
        # Pointmaze obs has different format, needs to be treated differently
        is_gymnasium_env = self.is_gymnasium_env

        assert self.eval_env2 is not None
        eval_env = self.eval_env2
        
        self.policy.eval()
        if is_gymnasium_env:
            obs, _ = eval_env.reset()
            obs = eval_env.get_true_observation(obs)
        else:
            obs = eval_env.reset()
            

        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        if not self.has_terminal: # pointmaze environment, don't use horizon
            while num_episodes < self._eval_episodes:
                rtg = torch.tensor([[self.goal]]).type(torch.float32)
                for timestep in range(self.horizon): # One epoch
                    # print(f"Timestep {timestep}, obs {obs}")
                    action = self.policy.select_action(obs.reshape(1, -1), rtg)
                    if hasattr(eval_env, "get_true_observation"): # gymnasium env 
                        next_obs, reward, terminal, _, _ = eval_env.step(action.flatten())
                    else:
                        next_obs, reward, terminal, _ = eval_env.step(action.flatten())
                    if is_gymnasium_env:
                        next_obs = eval_env.get_true_observation(next_obs)
                    # if num_episodes == 2 and timestep < 10:
                    #     print(f"Action {action}, next_obs {next_obs}, reward {reward}, rtg {rtg.item()}")
                    episode_reward += reward
                    rtg = rtg - reward
                    episode_length += 1

                    obs = next_obs

                    # if terminal:
                    #     break # Stop current epoch
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                if is_gymnasium_env:
                    obs, _ = eval_env.reset()
                    obs = eval_env.get_true_observation(obs)
                else:
                    obs = eval_env.reset()
        else:
            rtg = torch.tensor([[self.goal]]).type(torch.float32)
            while num_episodes < self._eval_episodes:
                # print(f"Timestep {timestep}, obs {obs}")
                action = self.policy.select_action(obs.reshape(1, -1), rtg)
                if hasattr(eval_env, "get_true_observation"): # gymnasium env 
                    next_obs, reward, terminal, _, _ = eval_env.step(action.flatten())
                else:
                    next_obs, reward, terminal, _ = eval_env.step(action.flatten())
                if is_gymnasium_env:
                    next_obs = eval_env.get_true_observation(next_obs)
                episode_reward += reward
                rtg = rtg - reward
                episode_length += 1

                obs = next_obs

                if terminal: # Episode finishes
                    eval_ep_info_buffer.append(
                        {"episode_reward": episode_reward, "episode_length": episode_length}
                    )
                    num_episodes +=1
                    episode_reward, episode_length = 0, 0
                    if is_gymnasium_env:
                        obs, _ = eval_env.reset()
                        obs = eval_env.get_true_observation(obs)
                    else:
                        obs = eval_env.reset()
                    rtg = torch.tensor([[self.goal]]).type(torch.float32)
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }

class RcslPolicyTrainer_v2(RcslPolicyTrainer):
    def __init__(
        self, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

    def train(self, holdout_ratio: float = 0.2, last_eval = False) -> Dict[str, float]:
        '''
        last_eval: If True, only evaluates at the last epoch
        '''
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = deque(maxlen=10)

        dataset = DictDataset(self.offline_dataset)

        holdout_size = int(len(dataset) * holdout_ratio)
        train_size = len(dataset) - holdout_size
        train_dataset, holdout_dataset = torch.utils.data.random_split(dataset, [train_size, holdout_size], 
                                                                       generator=torch.Generator().manual_seed(self.env_seed))
        data_loader = DataLoader(
            train_dataset,
            batch_size = self._batch_size,
            shuffle = True,
            pin_memory = True,
            num_workers = self.num_workers
        )
        # Version 1: Rollout only
        # if self._offline_ratio == 0:
        #     data_loader = DataLoader(
        #         DictDataset(self.rollout_dataset),
        #         batch_size = self._batch_size,
        #         shuffle = True,
        #         pin_memory = True,
        #         num_workers = self.num_workers
        #     )
        # elif self._offline_ratio == 1:
        #     data_loader = DataLoader(
        #         DictDataset(self.offline_dataset),
        #         batch_size = self._batch_size,
        #         shuffle = True,
        #         pin_memory = True,
        #         num_workers = self.num_workers
        #     )    
        # else:
        #     raise NotImplementedError       

        # train loop
        # eval_info = self._evaluate_vector()
        # ep_reward_mean, ep_reward_std = np.mean(eval_info["eval_vec/episode_reward"]), np.std(eval_info["eval_vec/episode_reward"])
        # print(f"Mean: {ep_reward_mean}, std: {ep_reward_std}")
        # eval_info = self._evaluate()
        # ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
        # print(f"Mean: {ep_reward_mean}, std: {ep_reward_std}")

        # best_ep_reward_mean = ep_reward_mean
        best_ep_reward_mean = 1e10
        best_policy_dict = self.policy.state_dict()
        for e in range(1, self._epoch + 1):

            self.policy.train()

            pbar = tqdm(enumerate(data_loader), desc=f"Epoch #{e}/{self._epoch}")
            for it, batch in pbar:
                # Sample from both offline (offline data) and rollout (rollout data) according to offline_ratio
                # offline_sample_size = int(self._batch_size * self._offline_ratio)
                # rollout_sample_size = self._batch_size - offline_sample_size
                # offline_batch = self.offline_buffer.sample(batch_size=offline_sample_size)
                # rollout_batch = self.rollout_buffer.sample(batch_size=rollout_sample_size)
                # batch = {"offline": offline_batch, "rollout": rollout_batch}
                '''
                batch: dict with keys
                    'observations'
                    'next_observations'
                    'actions'
                    'terminals'
                    'rewards'
                    'rtgs'

                '''
                loss_dict = self.policy.learn(batch)
                pbar.set_postfix(**loss_dict)

                for k, v in loss_dict.items():
                    self.logger.logkv_mean(k, v)
                
                # update the dynamics if necessary
                # if 0 < self._dynamics_update_freq and (num_timesteps+1)%self._dynamics_update_freq == 0:
                #     dynamics_update_info = self.policy.update_dynamics(self.offline_buffer)
                #     for k, v in dynamics_update_info.items():
                #         self.logger.logkv_mean(k, v)
                
                num_timesteps += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Test validation loss
            self.validate(holdout_dataset)
            
            # evaluate current policy
            # eval_vec_info = self._evaluate_vector()
            # ep_reward_mean_vec, ep_reward_std_vec = np.mean(eval_vec_info["eval_vec/episode_reward"]), np.std(eval_vec_info["eval_vec/episode_reward"])
            # ep_reward_max_vec, ep_reward_min_vec = np.max(eval_vec_info["eval_vec/episode_reward"]), np.min(eval_vec_info["eval_vec/episode_reward"])
            # ep_length_mean_vec, ep_length_std_vec = np.mean(eval_vec_info["eval_vec/episode_length"]), np.std(eval_vec_info["eval_vec/episode_length"])
            # pick_success_vec = np.mean(eval_vec_info["eval_vec/pick_success"])

            if last_eval and e < self._epoch: # When last_eval is True, only evaluate on last epoch
                pass
            else:
                eval_info = self._evaluate()
                ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
                ep_reward_max, ep_reward_min = np.max(eval_info["eval/episode_reward"]), np.min(eval_info["eval/episode_reward"])
                ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
                pick_success = np.mean(eval_info["eval/pick_success"])

                if not hasattr(self.eval_env, "get_normalized_score"): # gymnasium_env does not have normalized score
                    last_10_performance.append(ep_reward_mean)
                    self.logger.logkv("eval/episode_reward", ep_reward_mean)
                    self.logger.logkv("eval/episode_reward_std", ep_reward_std)  
                    self.logger.logkv("eval/pick_success", pick_success)    
                    # self.logger.logkv("eval_vec/episode_reward", ep_reward_mean_vec)
                    # self.logger.logkv("eval_vec/episode_reward_std", ep_reward_std_vec)   
                    # self.logger.logkv("eval_vec/pick_success", pick_success_vec)
                        
                else:       
                    norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
                    norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
                    norm_ep_rew_max = self.eval_env.get_normalized_score(ep_reward_max) * 100
                    norm_ep_rew_min = self.eval_env.get_normalized_score(ep_reward_min) * 100
                    last_10_performance.append(norm_ep_rew_mean)
                    self.logger.logkv("eval/normalized_episode_reward", norm_ep_rew_mean)
                    self.logger.logkv("eval/normalized_episode_reward_std", norm_ep_rew_std)
                    self.logger.logkv("eval/normalized_episode_reward_max", norm_ep_rew_max)
                    self.logger.logkv("eval/normalized_episode_reward_min", norm_ep_rew_min)
                self.logger.logkv("eval/episode_length", ep_length_mean)
                self.logger.logkv("eval/episode_length_std", ep_length_std)

                # save checkpoint
                if ep_reward_mean >= best_ep_reward_mean:
                    best_ep_reward_mean = ep_reward_mean
                    best_policy_dict = self.policy.state_dict()
                    torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy_best.pth"))


            # if self.eval_env2 is not None:
            #     eval_info_no_fix = self._evaluate_no_fix_seed()
            #     ep_reward_mean_no_fix, ep_reward_std_no_fix = np.mean(eval_info_no_fix["eval/episode_reward"]), np.std(eval_info_no_fix["eval/episode_reward"])
            #     ep_length_mean_no_fix, ep_length_std_no_fix = np.mean(eval_info_no_fix["eval/episode_length"]), np.std(eval_info_no_fix["eval/episode_length"])
            #     if not hasattr(self.eval_env, "get_normalized_score"): # gymnasium_env does not have normalized score
            #         last_10_performance.append(ep_reward_mean)
            #         self.logger.logkv("eval/episode_reward_no_fix_seed", ep_reward_mean_no_fix)
            #         self.logger.logkv("eval/episode_reward_std_no_fix_seed", ep_reward_std_no_fix)         
            #     else:       
            #         norm_ep_rew_mean_no_fix = self.eval_env.get_normalized_score(ep_reward_mean_no_fix) * 100
            #         norm_ep_rew_std_no_fix = self.eval_env.get_normalized_score(ep_reward_std_no_fix) * 100
            #         last_10_performance.append(norm_ep_rew_mean)
            #         self.logger.logkv("eval/normalized_episode_reward_no_fix_seed", norm_ep_rew_mean_no_fix)
            #         self.logger.logkv("eval/normalized_episode_reward_std_no_fix_seed", norm_ep_rew_std_no_fix)
            #     self.logger.logkv("eval/episode_length_no_fix_seed", ep_length_mean_no_fix)
            #     self.logger.logkv("eval/episode_length_std_no_fix_seed", ep_length_std_no_fix)

            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs(exclude=["dynamics_training_progress"])
        

        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))

        # load best policy
        # self.policy.load_state_dict(best_policy_dict)
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy_final.pth"))
        eval_info = self._evaluate()
        ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
        print(f"Mean: {ep_reward_mean}, std: {ep_reward_std}")


        # self.policy.dynamics.save(self.logger.model_dir)
        self.logger.close()
    
        return {"last_10_performance": np.mean(last_10_performance)}

    def _evaluate(self, eval_episodes: int = -1) -> Dict[str, List[float]]:
        '''
        Always set desired rtg to 0
        '''
        # Pointmaze obs has different format, needs to be treated differently
        if eval_episodes == -1:
            real_eval_episodes = self._eval_episodes
        else:
            real_eval_episodes = eval_episodes
        is_gymnasium_env = self.is_gymnasium_env

        self.eval_env.reset(seed=self.env_seed) # Fix seed
        
        self.policy.eval()
        if is_gymnasium_env:
            obs, _ = self.eval_env.reset()
            obs = self.eval_env.get_true_observation(obs)
        else:
            obs = self.eval_env.reset()
            

        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        if not self.has_terminal: # pointmaze environment, don't use horizon
            while num_episodes < real_eval_episodes:
                rtg = torch.tensor([[self.goal]]).type(torch.float32)
                pick_success = False
                for timestep in range(self.horizon): # One epoch
                    # print(f"Timestep {timestep}, obs {obs}")
                    action = self.policy.select_action(obs.reshape(1, -1), rtg)
                    if hasattr(self.eval_env, "get_true_observation"): # gymnasium env 
                        next_obs, reward, terminal, _, _ = self.eval_env.step(action.flatten())
                    else:
                        next_obs, reward, terminal, info = self.eval_env.step(action.flatten())
                    if is_gymnasium_env:
                        next_obs = self.eval_env.get_true_observation(next_obs)
                    # if num_episodes == 2 and timestep < 10:
                    #     print(f"Action {action}, next_obs {next_obs}, reward {reward}, rtg {rtg.item()}")
                    episode_reward += reward
                    # No need to update return
                    rtg = rtg - reward
                    episode_length += 1

                    if info['grasp_success_target']: # pick okay
                        pick_success = True

                    obs = next_obs

                    # if terminal:
                    #     break # Stop current epoch
                # print(episode_reward)
                episode_reward = 1 if episode_reward > 0 else 0 # Clip to 1
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length,
                     "pick_success": float(pick_success)}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                if is_gymnasium_env:
                    obs, _ = self.eval_env.reset()
                    obs = self.eval_env.get_true_observation(obs)
                else:
                    obs = self.eval_env.reset()
        else:
            rtg = torch.tensor([[self.goal]]).type(torch.float32)
            while num_episodes < self._eval_episodes:
                # print(f"Timestep {timestep}, obs {obs}")
                action = self.policy.select_action(obs.reshape(1, -1), rtg)
                if hasattr(self.eval_env, "get_true_observation"): # gymnasium env 
                    next_obs, reward, terminal, _, _ = self.eval_env.step(action.flatten())
                else:
                    next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
                if is_gymnasium_env:
                    next_obs = self.eval_env.get_true_observation(next_obs)
                episode_reward += reward
                # rtg = rtg - reward
                episode_length += 1

                obs = next_obs

                if terminal: # Episode finishes
                    # print(episode_reward)
                    episode_reward = 1 if episode_reward > 0 else 0 # Clip to 1
                    eval_ep_info_buffer.append(
                        {"episode_reward": episode_reward, "episode_length": episode_length}
                    )
                    episode_reward, episode_length = 0, 0
                    if is_gymnasium_env:
                        obs, _ = self.eval_env.reset()
                        obs = self.eval_env.get_true_observation(obs)
                    else:
                        obs = self.eval_env.reset()
                    rtg = torch.tensor([[self.goal]]).type(torch.float32)
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer],
            "eval/pick_success": [ep_info["pick_success"] for ep_info in eval_ep_info_buffer]
        }
    
    def _evaluate_vector(self) -> Dict[str, List[float]]:
        '''
        Use self.eval_env2, vectorised evaluation
        WARNING: not supported in roboverse!!!
        '''
        # Pointmaze obs has different format, needs to be treated differently
        is_gymnasium_env = self.is_gymnasium_env

        assert self.eval_env2 is not None
        eval_env = self.eval_env2
        
        self.policy.eval()
        if is_gymnasium_env:
            obs, _ = eval_env.reset()
            obs = eval_env.get_true_observation(obs)
        else:
            obs = eval_env.reset()

        batch = obs.shape[0]
            

        eval_ep_info_buffer = []
        # num_episodes = 0
        episode_reward, episode_length = np.zeros(batch), np.zeros(batch, dtype = np.int64)

        # if not self.has_terminal: # pointmaze environment, don't use horizon
        # rtg = torch.tensor([[self.goal]]).type(torch.float32)
        rtg = self.goal * torch.ones((batch, 1), dtype=torch.float32)
        pick_success = [False for _ in range(batch)]
        for timestep in range(self.horizon): # One epoch
            # print(f"Timestep {timestep}, obs {obs}")
            # print(f"Batch: {batch}")s
            action = self.policy.select_action(obs, rtg)
            if hasattr(eval_env, "get_true_observation"): # gymnasium env 
                next_obs, reward, terminal, _, _ = eval_env.step(action)
            else:
                next_obs, reward, terminal, info = eval_env.step(action)

            if is_gymnasium_env:
                next_obs = eval_env.get_true_observation(next_obs)
            # if num_episodes == 2 and timestep < 10:
            #     print(f"Action {action}, next_obs {next_obs}, reward {reward}, rtg {rtg.item()}")
            # reward shape: (batch, )
            episode_reward += reward
            rtg = rtg - reward[:,None]
            episode_length += 1

            for idx, single_info in enumerate(info):
                if single_info["grasp_success_target"]:
                    pick_success[idx] = True

            obs = next_obs

            # if terminal:
            #     break # Stop current epoch
        episode_reward = (episode_reward >= 1).astype(np.float32)
        # eval_ep_info_buffer.append(
        #     {"episode_reward": episode_reward, "episode_length": episode_length}
        # )
        eval_ep_info_buffer = [{"episode_reward": episode_reward[e], "episode_length": episode_length[e], "pick_success": float(pick_success[e])} for e in range(len(episode_reward))]
        # num_episodes +=1
        episode_reward, episode_length = np.zeros(self._eval_episodes), np.zeros(self._eval_episodes, dtype = np.int64)
        if is_gymnasium_env:
            obs, _ = eval_env.reset()
            obs = eval_env.get_true_observation(obs)
        else:
            obs = eval_env.reset()
        # else:
        #     rtg = torch.tensor([[self.goal]]).type(torch.float32)
        #     while num_episodes < self._eval_episodes:
        #         # print(f"Timestep {timestep}, obs {obs}")
        #         action = self.policy.select_action(obs.reshape(1, -1), rtg)
        #         if hasattr(eval_env, "get_true_observation"): # gymnasium env 
        #             next_obs, reward, terminal, _, _ = eval_env.step(action.flatten())
        #         else:
        #             next_obs, reward, terminal, _ = eval_env.step(action.flatten())
        #         if is_gymnasium_env:
        #             next_obs = eval_env.get_true_observation(next_obs)
        #         episode_reward += reward
        #         # rtg = rtg - reward
        #         episode_length += 1

        #         obs = next_obs

        #         if terminal: # Episode finishes
        #             episode_reward = 1 if episode_reward > 0 else 0 # Clip to 1
        #             eval_ep_info_buffer.append(
        #                 {"episode_reward": episode_reward, "episode_length": episode_length}
        #             )
        #             num_episodes +=1
        #             episode_reward, episode_length = 0, 0
        #             if is_gymnasium_env:
        #                 obs, _ = eval_env.reset()
        #                 obs = eval_env.get_true_observation(obs)
        #             else:
        #                 obs = eval_env.reset()
        #             rtg = torch.tensor([[self.goal]]).type(torch.float32)
        
        return {
            "eval_vec/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval_vec/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer],
            "eval_vec/pick_success": [ep_info["pick_success"] for ep_info in eval_ep_info_buffer]
        }

    @ torch.no_grad()
    def validate(self, holdout_dataset: torch.utils.data.Dataset) -> List[float]:
        data_loader = DataLoader(
            holdout_dataset,
            batch_size = self._batch_size,
            shuffle = True,
            pin_memory = True,
            num_workers = self.num_workers
        )
        self.policy.eval()

        pbar = tqdm(enumerate(data_loader), total=len(data_loader))
        for it, batch in pbar:
            # Sample from both offline (offline data) and rollout (rollout data) according to offline_ratio
            # offline_sample_size = int(self._batch_size * self._offline_ratio)
            # rollout_sample_size = self._batch_size - offline_sample_size
            # offline_batch = self.offline_buffer.sample(batch_size=offline_sample_size)
            # rollout_batch = self.rollout_buffer.sample(batch_size=rollout_sample_size)
            # batch = {"offline": offline_batch, "rollout": rollout_batch}
            '''
            batch: dict with keys
                'observations'
                'next_observations'
                'actions'
                'terminals'
                'rewards'
                'rtgs'
            '''
            loss_dict = self.policy.validate(batch)
            # pbar.set_postfix(**loss_dict)

            for k, v in loss_dict.items():
                self.logger.logkv_mean(k, v)