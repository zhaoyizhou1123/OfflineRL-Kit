import os
import numpy as np
import torch
import torch.nn as nn

from typing import Callable, List, Tuple, Dict, Optional
from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.logger import Logger
from offlinerlkit.modules import AutoregressiveDynamicsModel

class AutoregressiveDynamics(BaseDynamics):
    def __init__(
        self,
        model: AutoregressiveDynamicsModel,
        optim: torch.optim.Optimizer,
        scaler: StandardScaler,
    ) -> None:
        super().__init__(model, optim)
        self.scaler = scaler

    @ torch.no_grad()
    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        '''
        Return:
            reward (B,1) (if obs has batch)
            terminal (B,1)
        '''
        "imagine single forward step"
        obs_act = np.concatenate([obs, action], axis=-1)
        # print(obs_act.shape)
        obs_act = self.scaler.transform(obs_act)
        # print(obs_act.shape)

        next_predict = self.model(obs_act) # (batch, obs_dim + 1) [delta_ob, reward]
        next_delta_obs = next_predict[:, :-1].cpu().numpy()
        next_obs = next_delta_obs + obs
        reward = next_predict[:, -1].cpu().numpy()
        terminal = np.array([False for _ in range(reward.shape[0])])

        # mean, logvar = self.model(obs_act)
        # mean = mean.cpu().numpy()
        # logvar = logvar.cpu().numpy()
        # mean[..., :-1] += obs
        # std = np.sqrt(np.exp(logvar))

        # ensemble_samples = (mean + np.random.normal(size=mean.shape) * std).astype(np.float32)

        # # choose one model from ensemble
        # num_models, batch_size, _ = ensemble_samples.shape
        # model_idxs = self.model.random_elite_idxs(batch_size)
        # samples = ensemble_samples[model_idxs, np.arange(batch_size)]
        
        # next_obs = samples[..., :-1]
        # reward = samples[..., -1:]
        # terminal = self.terminal_fn(obs, action, next_obs)
        # info = {}
        # info["raw_reward"] = reward

        # if self._penalty_coef:
        #     if self._uncertainty_mode == "aleatoric":
        #         penalty = np.amax(np.linalg.norm(std, axis=2), axis=0)
        #     elif self._uncertainty_mode == "pairwise-diff":
        #         next_obses_mean = mean[..., :-1]
        #         next_obs_mean = np.mean(next_obses_mean, axis=0)
        #         diff = next_obses_mean - next_obs_mean
        #         penalty = np.amax(np.linalg.norm(diff, axis=2), axis=0)
        #     elif self._uncertainty_mode == "ensemble_std":
        #         next_obses_mean = mean[..., :-1]
        #         penalty = np.sqrt(next_obses_mean.var(0).mean(1))
        #     else:
        #         raise ValueError
        #     penalty = np.expand_dims(penalty, 1).astype(np.float32)
        #     assert penalty.shape == reward.shape
        #     reward = reward - self._penalty_coef * penalty
        #     info["penalty"] = penalty
        
        return next_obs, reward, terminal, {}
    
    @ torch.no_grad()
    # def sample_next_obss(
    #     self,
    #     obs: torch.Tensor,
    #     action: torch.Tensor,
    #     num_samples: int
    # ) -> torch.Tensor:
    #     obs_act = torch.cat([obs, action], dim=-1)
    #     obs_act = self.scaler.transform_tensor(obs_act)
    #     mean, logvar = self.model(obs_act)
    #     mean[..., :-1] += obs
    #     std = torch.sqrt(torch.exp(logvar))

    #     mean = mean[self.model.elites.data.cpu().numpy()]
    #     std = std[self.model.elites.data.cpu().numpy()]

    #     samples = torch.stack([mean + torch.randn_like(std) * std for i in range(num_samples)], 0)
    #     next_obss = samples[..., :-1]
    #     return next_obss

    def format_samples_for_training(self, data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        obss = data["observations"]
        actions = data["actions"]
        next_obss = data["next_observations"]
        rewards = data["rewards"]
        rewards = rewards.reshape(rewards.shape[0], -1)
        delta_obss = next_obss - obss
        inputs = np.concatenate((obss, actions), axis=-1)
        targets = np.concatenate((delta_obss, rewards), axis=-1)
        if 'weights' in data:
            weights = data['weights']
            weights = weights.reshape(weights.shape[0], -1) # (N,1)
        else:
            weights = None
        return inputs, targets, weights

    def train(
        self,
        data: Dict,
        logger: Logger,
        max_epochs: Optional[float] = None,
        max_epochs_since_update: int = 5,
        batch_size: int = 256,
        holdout_ratio: float = 0.2,
        # logvar_loss_coef: float = 0.01
    ) -> None:
        inputs, targets, weights = self.format_samples_for_training(data)
        data_size = inputs.shape[0]
        holdout_size = min(int(data_size * holdout_ratio), 1000)
        train_size = data_size - holdout_size
        train_splits, holdout_splits = torch.utils.data.random_split(range(data_size), (train_size, holdout_size))
        train_inputs, train_targets = inputs[train_splits.indices], targets[train_splits.indices]
        holdout_inputs, holdout_targets = inputs[holdout_splits.indices], targets[holdout_splits.indices]
        if weights is not None:
            train_weights, holdout_weights = weights[train_splits.indices], weights[holdout_splits.indices]
        else: 
            train_weights, holdout_weights = None, None

        self.scaler.fit(train_inputs) # I didn't implement weighted loss here.
        train_inputs = self.scaler.transform(train_inputs) 
        holdout_inputs = self.scaler.transform(holdout_inputs)
        holdout_loss = 1e10

        # data_idxes = np.random.randint(train_size, size=[train_size]) # (N)
        data_idxes = np.arange(train_size)
        np.random.shuffle(data_idxes)
        # def shuffle_rows(arr):
        #     idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
        #     return arr[np.arange(arr.shape[0])[:, None], idxes]

        epoch = 0
        cnt = 0
        logger.log("Training dynamics:")
        while True:
            epoch += 1
            if train_weights is not None:
                train_loss = self.learn(train_inputs[data_idxes], train_targets[data_idxes], train_weights[data_idxes], batch_size)
            else:
                train_loss = self.learn(train_inputs[data_idxes], train_targets[data_idxes], None, batch_size)
            new_holdout_loss = self.validate(holdout_inputs, holdout_targets, holdout_weights)
            # holdout_loss = (np.sort(new_holdout_loss)[:self.model.num_elites]).mean()
            logger.logkv("loss/dynamics_train_loss", train_loss)
            logger.logkv("loss/dynamics_holdout_loss", new_holdout_loss)
            logger.set_timestep(epoch)
            logger.dumpkvs(exclude=["policy_training_progress"])

            # shuffle data for each base learner
            # data_idxes = shuffle_rows(data_idxes)
            np.random.shuffle(data_idxes)

            indexes = []
            # for i, new_loss, old_loss in zip(range(len(holdout_losses)), new_holdout_losses, holdout_losses):
            #     improvement = (old_loss - new_loss) / old_loss
            #     if improvement > 0.01:
            #         indexes.append(i)
            #         holdout_losses[i] = new_loss
            improvement = (holdout_loss - new_holdout_loss) / abs(holdout_loss)
            if improvement > 0.01:
                holdout_loss = new_holdout_loss
                # self.model.update_save(indexes)
                cnt = 0
            else:
                cnt += 1
            
            if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
                break

        # indexes = self.select_elites(holdout_losses)
        # self.model.set_elites(indexes)
        # self.model.load_save()
        self.save(logger.model_dir)
        self.model.eval()
        logger.log(f"holdout loss: {holdout_loss}")
    
    def learn(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        weights: Optional[np.ndarray],
        batch_size: int = 256,
    ) -> float:
        '''
        inputs, targets: (N, dim). N is sampled with replacement
        weights: None / (N, 1)
        '''
        self.model.train()
        assert inputs.ndim == 2, f"{inputs.shape}"
        train_size = inputs.shape[0]
        losses = []

        for batch_num in range(int(np.ceil(train_size / batch_size))):
            inputs_batch = inputs[batch_num * batch_size:(batch_num + 1) * batch_size]
            inputs_batch = torch.as_tensor(inputs_batch).type(torch.float32).to(self.model.device)
            targets_batch = targets[batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = torch.as_tensor(targets_batch).type(torch.float32).to(self.model.device)
            if weights is not None:
                weights_batch = weights[batch_num * batch_size:(batch_num + 1) * batch_size]
                weights_batch = torch.as_tensor(weights_batch).type(torch.float32).to(self.model.device)
            else:
                weights_batch is None
            
            # mean, logvar = self.model(inputs_batch)
            # inv_var = torch.exp(-logvar)
            # # Average over batch and dim, sum over ensembles.
            # mse_loss_inv = (torch.pow(mean - targets_batch, 2) * inv_var).mean(dim=(1, 2)) # MLE for Gaussian
            # var_loss = logvar.mean(dim=(1, 2))
            # loss = mse_loss_inv.sum() + var_loss.sum()
            # loss = loss + self.model.get_decay_loss()
            # loss = loss + logvar_loss_coef * self.model.max_logvar.sum() - logvar_loss_coef * self.model.min_logvar.sum()
            loss = self.model.fit(inputs_batch, targets_batch, weights_batch)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses.append(loss.item())
        return np.mean(losses)
    
    @ torch.no_grad()
    def validate(self, inputs: np.ndarray, targets: np.ndarray, weights: Optional[np.ndarray]) -> float:
        # self.model.eval()
        # targets = torch.as_tensor(targets).to(self.model.device)
        # mean, _ = self.model(inputs)
        # loss = ((mean - targets) ** 2).mean(dim=(1, 2))
        # val_loss = list(loss.cpu().numpy())
        inputs = torch.as_tensor(inputs).type(torch.float32).to(self.model.device)
        targets = torch.as_tensor(targets).type(torch.float32).to(self.model.device)
        if weights is not None:
            weights = torch.as_tensor(weights).type(torch.float32).to(self.model.device)
        else:
            weights = None
        val_loss = self.model.fit(inputs, targets, weights)
        return val_loss.item()
    
    # def select_elites(self, metrics: List) -> List[int]:
    #     pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
    #     pairs = sorted(pairs, key=lambda x: x[0])
    #     elites = [pairs[i][1] for i in range(self.model.num_elites)]
    #     return elites

    def save(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(save_path, "dynamics.pth"))
        self.scaler.save_scaler(save_path)
    
    def load(self, load_path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
        self.scaler.load_scaler(load_path)
