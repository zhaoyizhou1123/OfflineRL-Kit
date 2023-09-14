# Modified from Chuning's code

import os
import torch
import torch.nn as nn
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from torch.utils.data.dataloader import DataLoader
from typing import Optional

from offlinerlkit.nets.unet import ConditionalUnet1D
from offlinerlkit.utils.diffusion_logger import Logger


class DiffusionBC:
    def __init__(self, config, dataset, logger: Logger):
        '''
        config: Contains attributes
        - act_dim
        - obs_dim
        - spectral_norm, bool. Default false
        - num_epochs
        - num_diffusion_iters
        - batch_size
        - num_workers = 1
        - save_ckpt_freq: Periods to save model checkpoint

        '''
        self.c = config
        self.dataset = dataset
        self.logger = logger
        # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        self.device = config.device

        self.dataloader = DataLoader(self.dataset, shuffle=True, pin_memory=True,
                    batch_size= self.c.batch_size,
                    num_workers= self.c.num_workers)

        # Noise prediction net
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.c.act_dim,
            global_cond_dim=self.c.obs_dim,
            # kernel_size=1,
            # n_groups=1,
        ).to(self.device)

        if self.c.spectral_norm:
            self.noise_pred_net.apply(
                lambda x: nn.utils.spectral_norm(x)
                if isinstance(x, nn.Linear) or isinstance(x, nn.Conv1d)
                else x
            )

        self.optimizer = torch.optim.AdamW(
            params=self.noise_pred_net.parameters(), lr=1e-4, weight_decay=1e-6
        )

        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=len(self.dataloader) * self.c.num_epochs,
        )

        # Noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.c.num_diffusion_iters,
            # The choise of beta schedule has big impact on performance
            # We found squared cosine works the best
            beta_schedule="squaredcos_cap_v2",
            # Clip output to [-1,1] to improve stability
            clip_sample=True,
            # Our network predicts noise (instead of denoised action)
            prediction_type="epsilon",
        )

        # Exponential moving average
        self.ema = EMAModel(model=self.noise_pred_net, power=0.75)

        self.start_epoch = 0 # Starting epoch number, will be modified after loading checkpoint

    def train(self):
        loaded = self.load_checkpoint(final=False) # Load checkpoint
        if loaded:
            print(f"Loaded checkpoint. Starting from epoch {self.start_epoch}")
        else:
            print(f"No checkpoint found, starting from epoch 0")
        for epoch in range(self.start_epoch, self.c.num_epochs):
            for batch in self.dataloader:
                # (B, obs_horizon * obs_dim)
                obs = batch["obs"].to(self.device)
                # (B, pred_horizon, act_dim)
                action = batch["action"].to(self.device).unsqueeze(1)

                # Sample noise to add to actions
                noise = torch.randn(action.shape, device=self.device)

                # Sample a diffusion iteration for each data point
                batch_size = obs.shape[0]
                timesteps = torch.randint(
                    low=0,
                    high=self.noise_scheduler.config.num_train_timesteps,
                    size=(batch_size,),
                    device=self.device,
                ).long()

                # Add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = self.noise_scheduler.add_noise(action, noise, timesteps).type(torch.float32)

                # Predict the noise residual
                noise_pred = self.noise_pred_net(
                    noisy_actions, timesteps, global_cond=obs
                )

                # L2 loss
                loss = nn.functional.mse_loss(noise_pred, noise)

                # Step optimizer
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Step lr scheduler every batch
                # this is different from standard pytorch behavior
                self.lr_scheduler.step()

                # Update Exponential Moving Average of the model weights
                self.ema.step(self.noise_pred_net)

                # Logging
                self.logger.record_mean("train/loss", loss.item())

            # Dump logger
            self.logger.record("train/epoch", epoch, exclude=['tensorboard'])
            self.logger.dump(step = epoch)

            if (epoch + 1) % self.c.save_ckpt_freq == 0:
                print(f"Saving checkpoint of epoch {epoch}")
                self.save_checkpoint(epoch = epoch)

        # Weights of the EMA model is used for inference
        self.ema_noise_pred_net = self.ema.averaged_model

    def sample_init_noise(self, batch: int=1):
        return torch.randn((batch, 1, self.c.act_dim), device=self.device)

    def select_action(self, obs, init_noise=None):
        '''
        obs: np.array (obs_dim) or (B, obs_dim)
        init_noise: (B,1,self.c.act_dim), according to sample_init_noise()

        Return:
            action: np.array (act_dim) or (B, act_dim), according to obs shape
        '''
        has_batch = (obs.ndim == 2) # whether there is batch

        # (B, obs_horizon * obs_dim)
        obs_cond = torch.from_numpy(obs).to(self.device, dtype=torch.float32)
        obs_cond = obs_cond.reshape(-1, obs_cond.shape[-1]) # (B,obs_dim)

        # Initialize action
        if init_noise is not None:
            act_pred = init_noise.clone()
        else:
            act_pred = self.sample_init_noise()

        # Initialize scheduler
        self.noise_scheduler.set_timesteps(self.c.num_diffusion_iters)

        for k in self.noise_scheduler.timesteps:
            # Predict noise
            noise_pred = self.ema_noise_pred_net(
                sample=act_pred, timestep=k, global_cond=obs_cond
            )

            # Inverse diffusion step (remove noise)
            act_pred = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=act_pred
            ).prev_sample

        if not has_batch:
            return act_pred.detach().cpu().numpy()[0, 0]
        else:
            return act_pred.detach().cpu().numpy()[:,0,:]

    def save_checkpoint(self, epoch: Optional[int]):
        '''
        epoch: Current training epoch. If None, save as final model
        '''
        if epoch is None: # Default: last epoch
            epoch = self.c.num_epochs - 1
            final = True # Indicate whether we are saving final model or checkpoint
            ckpt_name = "models.pt"
        else:
            final = False
            ckpt_name = "checkpoint.pt"
        # Save checkpoint
        if final:
            info_dict = {'epoch': epoch,
                         'model_state_dict': self.ema_noise_pred_net.state_dict()}
        else:
            info_dict = {'epoch': epoch,
                         'model_state_dict': self.noise_pred_net.state_dict(),
                        }            
        torch.save(
            info_dict,
            os.path.join(self.logger.dir, ckpt_name),
        )

    def load_checkpoint(self, final=False) -> bool:
        '''
        final: If True, load final model. Else load checkpoint model
        Return: If True, load succeed, else load failed
        '''
        # Load checkpoint
        if self.c.path is None:
            return False
        
        if final: # Load from config diffusion path
            ckpt_path = os.path.join(self.c.path, "models.pt")
        else:
            ckpt_path = os.path.join(self.c.path, "checkpoint.pt")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.start_epoch = checkpoint['epoch'] + 1 # Start from the next epoch
            state_dict = checkpoint['model_state_dict']
            if final: # Load ema_noise_pred_net
                self.ema_noise_pred_net = self.noise_pred_net
                self.ema_noise_pred_net.load_state_dict(state_dict)
            else: # Load noise_pred_net
                self.noise_pred_net.load_state_dict(state_dict)
            return True
        else:
            return False
