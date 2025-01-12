import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict
import numpy as np


class AutoregressivePolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dims, lr, device):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Input is obs + act + one-hot for the predicted dimension
        # Output is the mean and standard deviation of the predicted dimension
        input_dim = obs_dim + act_dim + act_dim
        all_dims = [input_dim] + hidden_dims + [2]
        self.model = nn.ModuleList()
        for in_dim, out_dim in zip(all_dims[:-1], all_dims[1:]):
            self.model.append(nn.Linear(in_dim, out_dim))
            self.model.append(nn.LeakyReLU())

        self.rcsl_optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = device
        self.model = self.model.to(self.device)

    def forward(self, obs):
        batch_size = obs.size(0)

        # Initialize action to zeros
        act = torch.zeros((batch_size, self.act_dim), device=obs.device)

        # One-hot encoding for all dimensions
        one_hot_all = torch.eye(self.act_dim, device=obs.device)

        # Predict each dimension autoregressively
        for i in range(self.act_dim):
            one_hot = one_hot_all[i][None, :].repeat(batch_size, 1)
            x = torch.cat([obs, act, one_hot], dim=1)
            for layer in self.model:
                x = layer(x)
            mean, logstd = torch.chunk(x, 2, dim=-1)
            # assert logstd.exp() > 0, logstd

            # logstd might be too small
            if logstd.exp() == 0:
                next_dim = mean
            else:
                dist = Normal(mean, logstd.exp())
                next_dim = dist.sample()
            act = torch.cat([act[:, :i], next_dim, act[:, i + 1 :]], dim=1)

        return act

    def select_action(self, obs: np.ndarray, rtg: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
            # print(obs.shape)
            action = self.forward(obs)
            # print(f"Dist mean, var: {dist.mean}, {dist.scale}")
        return action.cpu().numpy()

    def fit(self, obs, act):
        batch_size = obs.size(0)

        # Generate all the one-hot vectors, expand by repeat
        one_hot_all = torch.eye(self.act_dim, device=obs.device)
        one_hot_full = one_hot_all.repeat_interleave(batch_size, dim=0)

        # Repeat act by act_dim times and mask by one-hot encoding
        mask = (
            torch.tril(torch.ones((self.act_dim, self.act_dim), device=obs.device))
            - one_hot_all
        )  # lower trig - diag
        mask_full = mask.repeat_interleave(batch_size, dim=0)
        act_full = act.repeat(self.act_dim, 1)
        act_masked = act_full * mask_full

        # Repeat obs by act_dim times
        obs_full = obs.repeat(self.act_dim, 1)

        # Concatenate everything to get input
        input_full = torch.cat([obs_full, act_masked, one_hot_full], dim=1)

        # Use the one-hot vector as boolean mask to get target
        target = act_full[one_hot_full.bool()].unsqueeze(1)

        # Forward through model and compute loss
        x = input_full
        for layer in self.model:
            x = layer(x)
        mean, logstd = torch.chunk(x, 2, dim=-1)
        dist = Normal(mean, logstd.exp())
        loss = -dist.log_prob(target).mean()
        return loss
    
    def learn(self, batch: Dict) -> Dict[str, float]:
        # real_batch, fake_batch = batch["real"], batch["fake"]
        # # Mix data from real (offline) and fake (rollout)
        # mix_batch = {k: torch.cat([real_batch[k], fake_batch[k]], 0) for k in real_batch.keys()}

        # obss, actions, next_obss, rewards, terminals = mix_batch["observations"], mix_batch["actions"], \
        #     mix_batch["next_observations"], mix_batch["rewards"], mix_batch["terminals"]
        obss, actions, rtgs = batch["observations"], batch["actions"], batch["rtgs"]
        obss = obss.to(self.device)
        actions = actions.to(self.device)
        loss = self.fit(obss, actions)

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


if __name__ == "__main__":
    model = AutoregressivePolicy(10, 5, [32, 32])
    obs = torch.randn(32, 10)
    act = torch.randn(32, 5)

    # Test forward
    act_pred = model(obs)
    print(act_pred.shape)

    # Test fit
    loss = model.fit(obs, act)
    print(loss)
