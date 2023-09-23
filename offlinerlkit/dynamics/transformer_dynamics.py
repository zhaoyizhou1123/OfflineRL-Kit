import os
import numpy as np
import torch
import torch.nn as nn

from typing import Callable, List, Tuple, Dict, Optional
from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.logger import Logger
from offlinerlkit.modules import TransformerDynamicsModel, RewardDynamicsModel

class TransformerDynamics(BaseDynamics):
    def __init__(
        self,
        model: TransformerDynamicsModel,
        r_model: RewardDynamicsModel, 
        optim: torch.optim.Optimizer,
        optim_r: torch.optim.Optimizer,
        # scaler: StandardScaler,
    ) -> None:
        super().__init__(model, optim)
        # self.scaler = scaler
        self.optim_r = optim_r
        self.r_model = r_model

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
        next_obs, _ = self.model.sample(obs, action) # (batch, obs_dim + 1) [reward, obs]
        reward = self.r_model.forward(obs_act)

        next_obs = next_obs.cpu().numpy()
        reward = reward.cpu().numpy()

        terminal = np.array([False for _ in range(reward.shape[0])])
        
        return next_obs, reward, terminal, {}

    def format_samples_for_training(self, data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        obss = data["observations"]
        actions = data["actions"]
        next_obss = data["next_observations"]
        rewards = data["rewards"]
        rewards = rewards.reshape(rewards.shape[0], -1)
        # delta_obss = next_obss - obss
        inputs = np.concatenate((obss, actions), axis=-1)
        # targets = np.concatenate((delta_obss, rewards), axis=-1)
        # targets = np.concatenate((next_obss, rewards), axis=-1)
        targets = np.concatenate((rewards, next_obss), axis=-1) # estimate reward first
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
        self.train_obs(
            data,
            logger,
            max_epochs,
            max_epochs_since_update,
            batch_size,
            holdout_ratio,
        )
        self.train_r(
            data,
            logger,
            max_epochs,
            max_epochs_since_update,
            batch_size,
            holdout_ratio,
        )

    def train_obs(
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

        # self.scaler.fit(train_inputs) # I didn't implement weighted loss here.
        # train_inputs = self.scaler.transform(train_inputs) 
        # holdout_inputs = self.scaler.transform(holdout_inputs)
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
            # if improvement > 0.01:
            #     holdout_loss = new_holdout_loss
            #     # self.model.update_save(indexes)
            #     cnt = 0
            # else:
            #     cnt += 1
            
            if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
                break

        # indexes = self.select_elites(holdout_losses)
        # self.model.set_elites(indexes)
        # self.model.load_save()
        self.save(logger.model_dir)
        self.model.eval()
        logger.log(f"holdout loss: {holdout_loss}")

    def train_r(
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

        # self.scaler.fit(train_inputs) # I didn't implement weighted loss here.
        # train_inputs = self.scaler.transform(train_inputs) 
        # holdout_inputs = self.scaler.transform(holdout_inputs)
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
        best_r_state_dict = self.r_model.state_dict()
        while True:
            epoch += 1
            if train_weights is not None:
                train_loss = self.learn_r(train_inputs[data_idxes], train_targets[data_idxes], train_weights[data_idxes], batch_size)
            else:
                train_loss = self.learn_r(train_inputs[data_idxes], train_targets[data_idxes], None, batch_size)
            new_holdout_loss = self.validate_r(holdout_inputs, holdout_targets, holdout_weights)
            # holdout_loss = (np.sort(new_holdout_loss)[:self.model.num_elites]).mean()
            logger.logkv("loss/reward_train_loss", train_loss)
            logger.logkv("loss/reward_holdout_loss", new_holdout_loss)
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
            if epoch >= 70: # Consider validation loss after some epochs.
                improvement = (holdout_loss - new_holdout_loss) / abs(holdout_loss)
                if improvement > 0.01:
                    holdout_loss = new_holdout_loss
                    best_r_state_dict = self.r_model.state_dict()
                    # self.model.update_save(indexes)
                    cnt = 0
                else:
                    cnt += 1
            
            if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
                break

        # indexes = self.select_elites(holdout_losses)
        # self.model.set_elites(indexes)
        # self.model.load_save()

        # Get the best state dict
        self.r_model.load_state_dict(best_r_state_dict)
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
        # losses_r = []

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
            # loss_r = self.model.fit(inputs_batch, targets_batch, weights_batch)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # self.optim_r.zero_grad()
            # loss_r.backward()
            # self.optim_r.step()

            losses.append(loss.item())
            # losses_r.append(loss_r.item())
        return np.mean(losses)
    
    def learn_r(
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
        self.r_model.train()
        assert inputs.ndim == 2, f"{inputs.shape}"
        train_size = inputs.shape[0]
        # losses = []
        losses_r = []

        for batch_num in range(int(np.ceil(train_size / batch_size))):
            inputs_batch = inputs[batch_num * batch_size:(batch_num + 1) * batch_size]
            inputs_batch = torch.as_tensor(inputs_batch).type(torch.float32).to(self.r_model.device)
            targets_batch = targets[batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = torch.as_tensor(targets_batch).type(torch.float32).to(self.r_model.device)
            if weights is not None:
                weights_batch = weights[batch_num * batch_size:(batch_num + 1) * batch_size]
                weights_batch = torch.as_tensor(weights_batch).type(torch.float32).to(self.r_model.device)
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
            # loss = self.model.fit(inputs_batch, targets_batch, weights_batch)
            loss_r = self.r_model.fit_r(inputs_batch, targets_batch, weights_batch)

            # self.optim.zero_grad()
            # loss.backward()
            # self.optim.step()

            self.optim_r.zero_grad()
            loss_r.backward()
            self.optim_r.step()

            # losses.append(loss.item())
            losses_r.append(loss_r.item())
        return np.mean(losses_r)
    
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

    @ torch.no_grad()
    def validate_r(self, inputs: np.ndarray, targets: np.ndarray, weights: Optional[np.ndarray]) -> float:
        # self.model.eval()
        # targets = torch.as_tensor(targets).to(self.model.device)
        # mean, _ = self.model(inputs)
        # loss = ((mean - targets) ** 2).mean(dim=(1, 2))
        # val_loss = list(loss.cpu().numpy())
        inputs = torch.as_tensor(inputs).type(torch.float32).to(self.r_model.device)
        targets = torch.as_tensor(targets).type(torch.float32).to(self.r_model.device)
        if weights is not None:
            weights = torch.as_tensor(weights).type(torch.float32).to(self.r_model.device)
        else:
            weights = None
        val_loss_r = self.r_model.fit_r(inputs, targets, weights)
        return val_loss_r.item()
    
    # def select_elites(self, metrics: List) -> List[int]:
    #     pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
    #     pairs = sorted(pairs, key=lambda x: x[0])
    #     elites = [pairs[i][1] for i in range(self.model.num_elites)]
    #     return elites

    def save(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(save_path, "dynamics.pth"))
        torch.save(self.r_model.state_dict(), os.path.join(save_path, "r_dynamics.pth"))
        # self.scaler.save_scaler(save_path)
    
    def load(self, load_path: str, load_type = 'all') -> None:
        '''
        load_type: 'all', 'obs', 'r'
        '''
        if load_type == 'all':
            self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
            self.r_model.load_state_dict(torch.load(os.path.join(load_path, "r_dynamics.pth"), map_location=self.r_model.device))
        elif load_type == 'obs':
            self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
        elif load_type == 'r':
            self.r_model.load_state_dict(torch.load(os.path.join(load_path, "r_dynamics.pth"), map_location=self.r_model.device))
        else:
            raise NotImplementedError
        # self.scaler.load_scaler(load_path)