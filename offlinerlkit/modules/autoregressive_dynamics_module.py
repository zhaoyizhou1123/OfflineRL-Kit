import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Union, List, Tuple
import numpy as np
from offlinerlkit.modules.dynamics_module import soft_clamp


class AutoregressiveDynamicsModel(nn.Module):
    def __init__(self, 
                 obs_dim: int, 
                 act_dim: int, 
                 hidden_dims: Union[List[int], Tuple[int]],
                 r_hidden_dims: Union[List[int], Tuple[int]] = [200,200],
                #  num_ensemble: int = 7,
                #  num_elites: int = 5,
                 with_reward: bool = True, 
                 device: str = "cpu"
                 ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        # self.num_ensemble = num_ensemble
        # self.num_elites = num_elites

        # self.register_parameter(
        #     "elites",
        #     nn.Parameter(torch.tensor(list(range(0, self.num_elites))), requires_grad=False)
        # )

        # dimension that needs to predict
        # if with_reward:
        #     self.predict_dim = self.obs_dim + 1 
        # else:
        #     self.predict_dim = self.obs_dim
        self.predict_dim = self.obs_dim

        self.register_parameter(
            "max_logstd",
            nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)
        )
        self.register_parameter(
            "min_logstd",
            nn.Parameter(torch.ones(1) * -10, requires_grad=True)
        )
        self.register_parameter(
            "max_logstd_r",
            nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)
        )
        self.register_parameter(
            "min_logstd_r",
            nn.Parameter(torch.ones(1) * -10, requires_grad=True)
        )
        # self.max_logstd = torch.ones(1) * 0.5
        # self.max_logstd = self.max_logstd.to(self.device)
        # self.min_logstd = torch.ones(1) * -10
        # self.min_logstd = self.min_logstd.to(self.device)

        # Input is obs + act + obs + one-hot for the predicted dimension
        # Output is the mean and standard deviation of the predicted dimension
        input_dim = obs_dim + act_dim + self.predict_dim + self.predict_dim
        all_dims = [input_dim] + list(hidden_dims) + [2]
        self.model = nn.ModuleList()
        for in_dim, out_dim in zip(all_dims[:-1], all_dims[1:]):
            self.model.append(nn.Linear(in_dim, out_dim))
            self.model.append(nn.LeakyReLU())

        # reward_model
        # r_input_dim = obs_dim + act_dim + 1 + 1
        r_input_dim = obs_dim + 1 + 1 # no action
        r_all_dims = [r_input_dim] + list(r_hidden_dims) + [2]
        self.r_model = nn.ModuleList()
        for in_dim, out_dim in zip(r_all_dims[:-1], r_all_dims[1:]):
            self.r_model.append(nn.Linear(in_dim, out_dim))
            self.r_model.append(nn.LeakyReLU())

        self.to(self.device)
    def forward(self, obs_act: np.ndarray):
        '''
        Return:
            [reward, obs]
        '''
        batch_size = obs_act.shape[0]
        obs_act = torch.as_tensor(obs_act).type(torch.float32).to(self.device)
        # print(obs_act.shape, self.predict_dim)

        # Initialize [next_obs,reward] to zeros
        next_predict = torch.zeros((batch_size, self.predict_dim), device=self.device)

        # One-hot encoding for all dimensions
        one_hot_all = torch.eye(self.predict_dim, device=self.device)

        # Predict each dimension autoregressively
        for i in range(self.predict_dim):
            one_hot = one_hot_all[i][None, :].repeat(batch_size, 1) # (batch, predict_dim)
            # print(obs_act.shape, next_predict.shape, one_hot.shape)
            x = torch.cat([obs_act, next_predict, one_hot], dim=1)
            for layer in self.model:
                # print(x.shape)
                x = layer(x)
            mean, logstd = torch.chunk(x, 2, dim=-1)
            # if logstd.exp() == 0:
            #     print(f"Sample directly from mean")
            #     next_dim = mean
            # else:
            logstd = soft_clamp(logstd, self.min_logstd, self.max_logstd)
            # print(logstd)
            dist = Normal(mean, logstd.exp()) # make scale nonzeros
            next_dim = dist.sample()
            # print(f"Next dim: {next_dim.shape}, mean: {mean.shape}, logstd: {logstd.shape}")
            # next_dim = dist.sample()
            next_predict = torch.cat([next_predict[:, :i], next_dim, next_predict[:, i + 1 :]], dim=1) # (b, obs_dim)

        # reward prediction
        # Initialize [next_obs,reward] to zeros
        next_r_predict = torch.zeros((batch_size, 1), device=self.device)

        # One-hot encoding for all dimensions
        r_one_hot_all = torch.eye(1, device=self.device)

        # Predict each dimension autoregressively
        for i in range(1):
            r_one_hot = r_one_hot_all[i][None, :].repeat(batch_size, 1) # (batch, predict_dim)
            # print(obs_act.shape, next_predict.shape, one_hot.shape)
            x = torch.cat([obs_act[:, :self.obs_dim], next_r_predict, r_one_hot], dim=1) # (obs,0,0)
            for layer in self.r_model:
                # print(x.shape)
                x = layer(x)
            mean, logstd = torch.chunk(x, 2, dim=-1)
            # if logstd.exp() == 0:
            #     print(f"Sample directly from mean")
            #     next_dim = mean
            # else:
            logstd = soft_clamp(logstd, self.min_logstd_r, self.max_logstd_r)
            # print(logstd)
            dist = Normal(mean, logstd.exp()) # make scale nonzeros
            next_dim = dist.sample()
            # print(f"Next dim: {next_dim.shape}, mean: {mean.shape}, logstd: {logstd.shape}")
            # next_dim = dist.sample()
            next_r_predict = torch.cat([next_r_predict[:, :i], next_dim, next_r_predict[:, i + 1 :]], dim=1) # (batch, 1)

        next_full_predict = torch.cat([next_r_predict, next_predict], dim = 1)
        return next_full_predict

    def fit(self, obs_act, next_predict, weights = None):
        '''
        Args:
            next_predict: [r, next_obs]
        Return:
            loss (num_ensembles), averaged loss of each model
        '''
        assert obs_act.shape[-1] == self.obs_dim + self.act_dim
        # assert next_predict.shape[-1] == self.predict_dim, f"{next_predict.shape}"
        batch_size = obs_act.shape[0]

        next_predict_obs = next_predict[:, 1:]
        next_predict_r = next_predict[:, :1]

        # Generate all the one-hot vectors, expand by repeat
        one_hot_all = torch.eye(self.predict_dim, device=obs_act.device)
        one_hot_full = one_hot_all.repeat_interleave(batch_size, dim=0)

        # Repeat next_predict by predict_dim times and mask by one-hot encoding
        mask = (
            torch.tril(torch.ones((self.predict_dim, self.predict_dim), device=obs_act.device))
            - one_hot_all
        )  # lower trig - diag
        mask_full = mask.repeat_interleave(batch_size, dim=0)
        next_predict_obs_full = next_predict_obs.repeat(self.predict_dim, 1)
        next_predict_obs_masked = next_predict_obs_full * mask_full

        # Repeat obs by predict_dim times
        # obs_full = obs.repeat(self.predict_dim, 1) # (batch * predict_dim, obs_dim)
        # act_full = act.repeat(self.predict_dim, 1)
        obs_act_full = obs_act.repeat(self.predict_dim, 1)

        # Concatenate everything to get input
        # input_full = torch.cat([obs_full, act_full, next_predict_masked, one_hot_full], dim=1)
        input_full = torch.cat([obs_act_full, next_predict_obs_masked, one_hot_full], dim=1)

        # Use the one-hot vector as boolean mask to get target
        target = next_predict_obs_full[one_hot_full.bool()].unsqueeze(1)

        # Forward through model and compute loss
        x = input_full
        for layer in self.model:
            x = layer(x)
        mean, logstd = torch.chunk(x, 2, dim=-1)
        logstd = soft_clamp(logstd, self.min_logstd, self.max_logstd)
        dist = Normal(mean, logstd.exp())
        loss = -dist.log_prob(target)
        assert loss.dim() == 2, f"{loss.shape}"
        if weights is None:
            loss = loss.mean()
        else:
            loss = loss.reshape(loss.shape[0], -1) # (batch * act_dim, 1)
            weights = weights.reshape(weights.shape[0], -1) # (batch, 1)
            weights = weights.repeat(self.predict_dim, 1) # (batch * act_dim, 1)
            loss = torch.sum(loss * weights) / (torch.sum(weights) * loss.shape[-1])

        return loss



    # def set_elites(self, indexes: List[int]) -> None:
    #     assert len(indexes) <= self.num_ensemble and max(indexes) < self.num_ensemble
    #     self.register_parameter('elites', nn.Parameter(torch.tensor(indexes), requires_grad=False))
    
    # def random_elite_idxs(self, batch_size: int) -> np.ndarray:
    #     idxs = np.random.choice(self.elites.data.cpu().numpy(), size=batch_size)
    #     return idxs

    def fit_r(self, obs_act, next_predict, weights = None):
        assert obs_act.shape[-1] == self.obs_dim + self.act_dim
        # assert next_predict.shape[-1] == self.predict_dim, f"{next_predict.shape}"
        batch_size = obs_act.shape[0]

        next_predict_obs = next_predict[:, 1:]
        next_predict_r = next_predict[:, :1]
        # reward model
        # Generate all the one-hot vectors, expand by repeat
        one_hot_r_all = torch.eye(1, device=obs_act.device)
        one_hot_r_full = one_hot_r_all.repeat_interleave(batch_size, dim=0) # (batch, 1)

        # Repeat next_predict by predict_dim times and mask by one-hot encoding
        mask_r = (
            torch.tril(torch.ones((1, 1), device=obs_act.device))
            - one_hot_r_all
        )  # lower trig - diag
        mask_r_full = mask_r.repeat_interleave(batch_size, dim=0)
        next_predict_r_full = next_predict_r.repeat(1, 1) # (batch, 1)
        next_predict_r_masked = next_predict_r_full * mask_r_full
        # print(f"Full {next_predict_r_full.shape}, r {next_predict_r.shape}")

        # Repeat obs by predict_dim times
        # obs_full = obs.repeat(self.predict_dim, 1) # (batch * predict_dim, obs_dim)
        # act_full = act.repeat(self.predict_dim, 1)
        obs_act_r_full = obs_act[:, :self.obs_dim].repeat(1, 1) # only obs

        # Concatenate everything to get input
        # input_full = torch.cat([obs_full, act_full, next_predict_masked, one_hot_full], dim=1)
        input_r_full = torch.cat([obs_act_r_full, next_predict_r_masked, one_hot_r_full], dim=1)

        # Use the one-hot vector as boolean mask to get target
        target_r = next_predict_r_full[one_hot_r_full.bool()].unsqueeze(1)
        # print(f"one hot {one_hot_r_full.shape}, target {target_r.shape}")

        # Forward through model and compute loss
        x_r = input_r_full
        # print(f"input_r_full: {input_r_full.shape}")
        for layer in self.r_model:
            x_r = layer(x_r)
        mean_r, logstd_r = torch.chunk(x_r, 2, dim=-1)
        # print(f"mean {mean_r.shape}, logstd {logstd_r.shape}")
        logstd_r = soft_clamp(logstd_r, self.min_logstd_r, self.max_logstd_r)
        dist_r = Normal(mean_r, logstd_r.exp())
        loss_r = -dist_r.log_prob(target_r)
        assert loss_r.dim() == 2, f"{loss_r.shape}"

        if weights is None:
            loss_r = loss_r.mean()
        else:
            loss_r = loss_r.reshape(loss_r.shape[0], -1) # (batch * (obs_dim+1), 1)
            weights = weights.reshape(weights.shape[0], -1) # (batch, 1)
            weights = weights.repeat(1, 1) # (batch * (obs_dim+1), 1)
            loss_r = torch.sum(loss_r * weights) / (torch.sum(weights) * loss_r.shape[-1])
        return loss_r

if __name__ == "__main__":
    model = AutoregressiveDynamicsModel(10, 5, [32, 32])
    obs = torch.randn(32, 10)
    act = torch.randn(32, 5)
    next_obs = torch.randn(32, 10)

    # Test forward
    next_obs_pred = model(obs, act)
    print(next_obs_pred.shape)

    # Test fit
    loss = model.fit(obs, act, next_obs)
    print(loss)
