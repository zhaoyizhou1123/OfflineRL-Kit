import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict

class ConditionEncoder(nn.Module):
    def __init__(self, cond_shape_dict):
        super().__init__()
        self.cond_encoders = nn.ModuleDict()
        self.cond_dim = 0
        for name, shape in cond_shape_dict.items():
            if len(shape) == 1:
                self.cond_encoders[name] = nn.Identity()
                self.cond_dim += shape[0]
            else:
                raise NotImplementedError

    def forward(self, cond_dict):
        return torch.cat(
            [self.cond_encoders[k](v) for k, v in cond_dict.items()], dim=-1
        )


class Discretizer:
    def __init__(self, min_value, max_value, bins):
        self.min = min_value
        self.max = max_value
        self.range = max_value - min_value
        self.step_size = self.range / bins

    def discretize(self, x):
        ind = ((x - self.min) / self.step_size).long()
        return ind

    def reconstruct(self, ind):
        x = (ind + 0.5) * self.step_size + self.min
        return x


class ConditionalAutoregressiveModel:
    def __init__(
        self,
        dim, # action dim
        cond_shape_dict, # example cond_dict. {"obs": (obs_dim,), "feat": (feature_dim,)
        output_min,
        output_max,
        output_bins,
        device,
        lr=1e-4,
        weight_decay=1e-6,
    ):
        self.dim = dim
        self.device = device

        # Discretizer
        self.discretizer = Discretizer(output_min, output_max, output_bins)

        # Condition encoders
        self.cond_encoder = ConditionEncoder(cond_shape_dict).to(self.device)

        # Model
        input_dim = self.cond_encoder.cond_dim + dim * 2
        hidden_dim = min(256, output_bins * 2)

        # net output: (output_bins)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_bins),
        ).to(self.device)

        # Optimizer
        self.model_params = list(self.net.parameters()) + list(
            self.cond_encoder.parameters()
        )
        self.optimizer = torch.optim.Adam(
            self.model_params, lr=lr, weight_decay=weight_decay
        )

        # One-hot encoding for all dimensions
        self.one_hot_all = torch.eye(self.dim, device=self.device) # diag (action_dim, action_dim)

    def learn(self, x, cond_dict, weights = None):
        # Infer batch size
        batch_size = x.size(0)

        # Encode condition
        cond = self.cond_encoder(cond_dict)

        # Repeat condition by input_dim times
        cond_full = cond.repeat(self.dim, 1)

        # Generate all the one-hot vectors, expand by repeat
        one_hot_full = self.one_hot_all.repeat_interleave(batch_size, dim=0) # (action_dim * batch, action_dim)

        # Repeat x by input_dim times and mask by one-hot encoding
        mask = (
            torch.tril(torch.ones((self.dim, self.dim), device=cond.device))
            - self.one_hot_all
        )  # lower trig - diag
        mask_full = mask.repeat_interleave(batch_size, dim=0)
        x_full = x.repeat(self.dim, 1) # (batch * action_dim, action_dim) ?
        x_masked = x_full * mask_full

        # Concatenate everything to get input
        input_full = torch.cat([cond_full, x_masked, one_hot_full], dim=1)

        # Use the one-hot vector as boolean mask to get target
        target = self.discretizer.discretize(x_full[one_hot_full.bool()]) # [actions[:,0],actions[:,1],...,actions[:,action_dim-1]]

        # Forward through model and compute loss
        logits = self.net(input_full)
        loss = F.cross_entropy(logits, target, reduction='none') # (batch * action,)
        loss = loss.reshape(loss.shape[0], -1) # (batch*action, 1)
        # assert loss.dim() == 2, f"{loss.shape}"
        if weights is not None:
            weights = weights.reshape(weights.shape[0], -1) # (batch, 1)
            weights = weights.repeat(self.act_dim, 1) # (batch * action_dim, 1)
            loss = torch.sum(loss * weights) / (torch.sum(weights) * loss.shape[-1])
        else:
            loss = loss.mean()

        # Step optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        result =  {
            "loss": loss.item(),
        }
        
        return result

    @ torch.no_grad()
    def validate(self, x, cond_dict, weights = None):
        # Infer batch size
        batch_size = x.size(0)

        # Encode condition
        cond = self.cond_encoder(cond_dict)

        # Repeat condition by input_dim times
        cond_full = cond.repeat(self.dim, 1)

        # Generate all the one-hot vectors, expand by repeat
        one_hot_full = self.one_hot_all.repeat_interleave(batch_size, dim=0) # (action_dim * batch, action_dim)

        # Repeat x by input_dim times and mask by one-hot encoding
        mask = (
            torch.tril(torch.ones((self.dim, self.dim), device=cond.device))
            - self.one_hot_all
        )  # lower trig - diag
        mask_full = mask.repeat_interleave(batch_size, dim=0)
        x_full = x.repeat(self.dim, 1) # (batch * action_dim, action_dim) ?
        x_masked = x_full * mask_full

        # Concatenate everything to get input
        input_full = torch.cat([cond_full, x_masked, one_hot_full], dim=1)

        # Use the one-hot vector as boolean mask to get target
        target = self.discretizer.discretize(x_full[one_hot_full.bool()]) # [actions[:,0],actions[:,1],...,actions[:,action_dim-1]]

        # Forward through model and compute loss
        logits = self.net(input_full)
        loss = F.cross_entropy(logits, target, reduction='none') # (batch * action,)
        loss = loss.reshape(loss.shape[0], -1) # (batch*action, 1)
        # assert loss.dim() == 2, f"{loss.shape}"
        if weights is not None:
            weights = weights.reshape(weights.shape[0], -1) # (batch, 1)
            weights = weights.repeat(self.act_dim, 1) # (batch * action_dim, 1)
            loss = torch.sum(loss * weights) / (torch.sum(weights) * loss.shape[-1])
        else:
            loss = loss.mean()

        # Step optimizer
        result =  {
            "holdout_loss": loss.item(),
        }
        
        return result

    def sample(self, cond_dict):
        # Encode condition
        cond = self.cond_encoder(cond_dict)

        # Infer batch size
        batch_size = cond.size(0)

        # Initialize sample to zeros
        sample = torch.zeros((batch_size, self.dim), device=cond.device)

        # Predict each dimension autoregressively
        for i in range(self.dim):
            # Retrieve one-hot vector for current dimension
            curr_one_hot = self.one_hot_all[i][None, :].repeat(batch_size, 1)

            # Predict current dimension
            curr_input = torch.cat([cond, sample, curr_one_hot], dim=1)
            logits = self.net(curr_input)
            dist = Categorical(logits=logits)
            next_dim = self.discretizer.reconstruct(dist.sample().unsqueeze(1))

            # Update sample
            sample = torch.cat([sample[:, :i], next_dim, sample[:, i + 1 :]], dim=1)

        return sample
    
    def state_dict(self):
        return {
            "net": self.net.state_dict(),
            "cond_encoder": self.cond_encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict["net"])
        self.cond_encoder.load_state_dict(state_dict["cond_encoder"])
        self.optimizer.load_state_dict(state_dict["optimizer"])


class AutoregressiveCategoricalPolicy(ConditionalAutoregressiveModel):
    def __init__(
        self,
        obs_dim,
        act_dim,
        feature_dim,
        output_min,
        output_max,
        output_bins,
        device,
        **kwargs,
    ):
        super().__init__(
            dim=act_dim,
            cond_shape_dict={"obs": (obs_dim,), "feat": (feature_dim,)},
            output_min=output_min,
            output_max=output_max,
            output_bins=output_bins,
            device=device,
            **kwargs,
        )

    def learn(self, batch: Dict):
        obss = batch['observations'].type(torch.float32).to(self.device)
        actions = batch['actions'].type(torch.float32).to(self.device)
        rtgs = batch['rtgs']
        rtgs = rtgs.reshape(rtgs.shape[0], -1).type(torch.float32).to(self.device)
        if 'weights' in batch:
            weights = batch['weights'].type(torch.float32).to(self.device) # (batch, )
        else:
            weights = None
        return super().learn(actions, {"obs": obss, "feat": rtgs}, weights)

    def validate(self, batch: Dict):
        '''
        Update one batch
        '''
        obss = batch['observations'].type(torch.float32).to(self.device)
        actions = batch['actions'].type(torch.float32).to(self.device)
        rtgs = batch['rtgs']
        rtgs = rtgs.reshape(rtgs.shape[0], -1).type(torch.float32).to(self.device)
        if 'weights' in batch:
            weights = batch['weights'].type(torch.float32).to(self.device) # (batch, )
        else:
            weights = None

        return super().validate(actions, {"obs": obss, "feat": rtgs}, weights)

    def select_action(self, obs, feat):
        obs = torch.as_tensor(obs, dtype = torch.float32, device = self.device)
        feat = torch.as_tensor(feat, dtype = torch.float32, device = self.device)

        with torch.no_grad():
            action = super().sample({"obs": obs, "feat": feat})
        return action.cpu().numpy()
    
    def train(self) -> None:
        self.net.train()
        self.cond_encoder.train()

    def eval(self) -> None:
        self.net.eval()
        self.cond_encoder.eval()


if __name__ == "__main__":
    model = AutoregressiveCategoricalPolicy(10, 5, 1, -1, 1, 100, "cpu")
    obs = torch.rand(32, 10)
    act = torch.rand(32, 5)
    ret = torch.zeros(32, 1)

    # Test forward
    act_pred = model.sample(obs, ret)
    print(act_pred.shape)

    # Test fit
    loss = model.train(obs, ret, act)
    print(loss)