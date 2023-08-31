# Modified from combo.py

# TODO: Implement classic RCSL policy from actor_module and dist_module. Follow COMBO class

import numpy as np
import torch
import torch.nn as nn
import gym

from torch.nn import functional as F
from typing import Dict, Union, Tuple, Optional
from collections import defaultdict
from offlinerlkit.policy import BasePolicy, DiffusionBC
from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.modules.rcsl_module import RcslModule


class RcslPolicy(BasePolicy):
    """
    wrapped rcsl policy
    """

    def __init__(
        self,
        dynamics: Optional[BaseDynamics],
        rollout_policy: Optional[DiffusionBC],
        rcsl: RcslModule,
        rcsl_optim: torch.optim.Optimizer,
        device: Union[str, torch.device]
        # action_space: gym.spaces.Space,
        # tau: float = 0.005,
        # gamma: float  = 0.99,
        # alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        # cql_weight: float = 1.0,
        # temperature: float = 1.0,
        # max_q_backup: bool = False,
        # deterministic_backup: bool = True,
        # with_lagrange: bool = True,
        # lagrange_threshold: float = 10.0,
        # cql_alpha_lr: float = 1e-4,
        # num_repeart_actions:int = 10,
        # uniform_rollout: bool = False,
        # rho_s: str = "mix"
    ) -> None:
        super().__init__()

        self.dynamics = dynamics
        self.rollout_policy = rollout_policy

        self.rcsl = rcsl      
        self.rcsl_optim = rcsl_optim

        self.device = device
        # self._uniform_rollout = uniform_rollout
        # self._rho_s = rho_s

    def rollout(
        self,
        init_obss: np.ndarray,
        rollout_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        '''
        Sample a batch of trajectories at the same time.
        Output rollout_transitions contain keys:
        obss,
        next_obss,
        actions
        rewards, (N,1)
        rtgs, (N,1)
        traj_idxs, (N)
        '''

        num_transitions = 0
        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)
        valid_idxs = np.arange(init_obss.shape[0]) # maintain current valid trajectory indexes
        returns = np.zeros(init_obss.shape[0]) # maintain return of each trajectory
        acc_returns = np.zeros(init_obss.shape[0]) # maintain accumulated return of each valid trajectory

        # rollout
        observations = init_obss

        frozen_noise = self.rollout_policy.sample_init_noise(init_obss.shape[0])
        for _ in range(rollout_length):
            actions = self.rollout_policy.select_action(observations, frozen_noise)
            next_observations, rewards, terminals, info = self.dynamics.step(observations, actions)
            rollout_transitions["obss"].append(observations)
            rollout_transitions["next_obss"].append(next_observations)
            rollout_transitions["actions"].append(actions)
            rollout_transitions["rewards"].append(rewards)
            rollout_transitions["terminals"].append(terminals)
            rollout_transitions["traj_idxs"].append(valid_idxs)
            rollout_transitions["acc_rets"].append(acc_returns)

            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.flatten())

            # print(returns[valid_idxs].shape, rewards.shape)
            returns[valid_idxs] = returns[valid_idxs] + rewards.flatten() # Update return (for valid idxs only)
            acc_returns = acc_returns + rewards.flatten()

            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations[nonterm_mask] # Only keep trajs that have not terminated
            valid_idxs = valid_idxs[nonterm_mask] # update unterminated traj indexs
            acc_returns = acc_returns[nonterm_mask] # Only keep acc_ret of trajs that have not terminated
            frozen_noise = frozen_noise[nonterm_mask]
        
        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.concatenate(v, axis=0)

        # Compute rtgs
        traj_idxs = rollout_transitions["traj_idxs"]
        rtgs = returns[traj_idxs] - rollout_transitions["acc_rets"]
        rollout_transitions["rtgs"] = rtgs[..., None] # (N,1)

        return rollout_transitions, \
            {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean(), "returns": returns}
    
    # One batch update
    def learn(self, batch: Dict) -> Dict[str, float]:
        # real_batch, fake_batch = batch["real"], batch["fake"]
        # # Mix data from real (offline) and fake (rollout)
        # mix_batch = {k: torch.cat([real_batch[k], fake_batch[k]], 0) for k in real_batch.keys()}

        # obss, actions, next_obss, rewards, terminals = mix_batch["observations"], mix_batch["actions"], \
        #     mix_batch["next_observations"], mix_batch["rewards"], mix_batch["terminals"]
        obss, actions, rtgs = batch["observations"], batch["actions"], batch["rtgs"]
        batch_size = obss.shape[0]

        pred_actions = self.rcsl.forward(obss, rtgs)
        # Average over batch and dim, sum over ensembles.
        loss = torch.pow(pred_actions - actions.to(pred_actions.device), 2).mean() # MSE error

        # Compute Cross Entropy loss
        # Need to compute by ourselves. Because dist.log_prob does not contain correct gradient
        # log_probs = dist.log_prob(actions.to(self.device))
        # print(f"log_probs shape: {log_probs.shape}")
        # loss = -log_probs.mean()

        self.rcsl_optim.zero_grad()
        loss.backward()
        self.rcsl_optim.step()

        result =  {
            "loss": loss.item(),
        }
        
        return result
    
    def select_action(self, obs: np.ndarray, rtg: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            action = self.rcsl.forward(obs, rtg)
            # print(f"Dist mean, var: {dist.mean}, {dist.scale}")
        return action.cpu().numpy()
    
    def train(self) -> None:
        self.rcsl.train()

    def eval(self) -> None:
        self.rcsl.eval()