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

class DiffusionPolicyTrainer:
    def __init__(
        self,
        policy: Union[RcslPolicy, RcslGaussianPolicy, SimpleDiffusionPolicy],
        offline_dataset: Dict[str, np.ndarray],
        logger: Logger,
        seed,
        # rollout_setting: Tuple[int, int, int],
        epoch: int = 25,
        # step_per_epoch: int = 1000,
        batch_size: int = 256,
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
        self.horizon = horizon
        self.offline_dataset = offline_dataset
        self.logger = logger
        # self.device = device

        # self._rollout_freq, self._rollout_batch_size, \
        #     self._rollout_length = rollout_setting
        # self._dynamics_update_freq = dynamics_update_freq

        self._epoch = epoch
        self._batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.num_workers = num_workers
        self.env_seed = seed

        # self.is_gymnasium_env = hasattr(self.eval_env, "get_true_observation")
        # assert (not self.is_gymnasium_env) or (self.horizon is not None), "Horizon must be specified for Gymnasium env"
        self.has_terminal = has_terminal

    def train(self) -> Dict[str, float]:
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = deque(maxlen=10)

        data_loader = DataLoader(
            DictDataset(self.offline_dataset),
            batch_size = self._batch_size,
            shuffle = True,
            pin_memory = True,
            num_workers = self.num_workers
        )       

        # train loop
        # self._evaluate()
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
            
            # evaluate current policy

            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs(exclude=["dynamics_training_progress"])
        
            # save checkpoint
            torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy.pth"))

        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))

        # self.policy.dynamics.save(self.logger.model_dir)
        self.logger.close()
    
        return {"last_10_performance": np.mean(last_10_performance)}

