import argparse
import os
import sys
import random

import gym
import d4rl

import numpy as np
import torch

import pickle
from copy import deepcopy
from typing import Dict, Tuple
import roboverse
import matplotlib.pyplot as plt


# import __init__
from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, EnsembleDynamicsModel, RcslModule
from offlinerlkit.dynamics import BaseDynamics, EnsembleDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn
from offlinerlkit.utils.load_dataset import qlearning_dataset, traj_rtg_datasets
from offlinerlkit.utils.config import Config
from offlinerlkit.utils.dataset import ObsActDataset
from offlinerlkit.utils.pickplace_utils import SimpleObsWrapper, get_pickplace_dataset, get_true_obs
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.utils.diffusion_logger import setup_logger
from offlinerlkit.policy_trainer import RcslPolicyTrainer, RcslPolicyTrainer_v2
from offlinerlkit.utils.trajectory import Trajectory
from offlinerlkit.utils.none_or_str import none_or_str
from offlinerlkit.policy import DiffusionBC, RcslPolicy, SimpleDiffusionPolicy
from offlinerlkit.env.linearq import Linearq

# from rvs.policies import RvS


from pointmaze.utils.trajectory import Trajs2Dict


"""
suggested hypers

halfcheetah-medium-v2: rollout-length=5, cql-weight=0.5
hopper-medium-v2: rollout-length=5, cql-weight=5.0
walker2d-medium-v2: rollout-length=1, cql-weight=5.0
halfcheetah-medium-replay-v2: rollout-length=5, cql-weight=0.5
hopper-medium-replay-v2: rollout-length=5, cql-weight=0.5
walker2d-medium-replay-v2: rollout-length=1, cql-weight=0.5
halfcheetah-medium-expert-v2: rollout-length=5, cql-weight=5.0
hopper-medium-expert-v2: rollout-length=5, cql-weight=5.0
walker2d-medium-expert-v2: rollout-length=1, cql-weight=5.0
"""


def get_args():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--algo-name", type=str, default="rcsl_diffusion")
    parser.add_argument("--task", type=str, default="pickplace", help="maze") # Self-constructed environment
    parser.add_argument('--debug',action='store_true', help='Print debuuging info if true')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers, align with cpu number")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # env config (pickplace)
    parser.add_argument('--data_dir', type=str, default='./dataset')
    parser.add_argument('--horizon', type=int, default=40, help="max path length for pickplace")

    # env config (maze)
    # parser.add_argument('--maze_config_file', type=str, default='./pointmaze/config/maze2_simple_moredata.json')
    # parser.add_argument('--data_file', type=str, default='./pointmaze/dataset/maze2_smds_acc.dat')
    # parser.add_argument('--render', action='store_true')
    # parser.add_argument('--log_to_wandb',action='store_true', help='Set up wandb')
    # parser.add_argument('--tb_path', type=str, default=None, help="./logs/stitch/, Folder to tensorboard logs" )
    # parser.add_argument('--env_type', type=str, default='pointmaze', help='pointmaze or ?')

    parser.add_argument("--dynamics-lr", type=float, default=1e-3)
    parser.add_argument("--dynamics-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--dynamics-weight-decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
    parser.add_argument("--n-ensemble", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    # # parser.add_argument("--rollout-freq", type=int, default=1000)
    # parser.add_argument("--rollout-batch-size", type=int, default=50000)
    # parser.add_argument("--rollout-length", type=int, default=5)
    # parser.add_argument("--model-retain-epochs", type=int, default=5)
    # parser.add_argument("--real-ratio", type=float, default=0.5)
    parser.add_argument("--load-dynamics-path", type=none_or_str, default=None)

    # Behavior policy (diffusion)
    parser.add_argument("--behavior_epoch", type=int, default=1000)
    parser.add_argument("--num_diffusion_iters", type=int, default=10, help="Number of diffusion steps")
    parser.add_argument('--behavior_batch', type=int, default=256)
    parser.add_argument('--load_diffusion_path', type=none_or_str, default=None, help = "path to .pth file")
    parser.add_argument('--diffusion_seed', type=str, default='0', help="Distinguish runs for diffusion policy, not random seed")
    parser.add_argument('--task_weight', type=float, default=1.)

    # Rollout 
    parser.add_argument('--rollout_ckpt_path', type=none_or_str, default=None, help="./checkpoint/maze2_smd_stable, file path, used to load/store rollout trajs" )
    parser.add_argument('--rollout_epochs', type=int, default=1000, help="Max number of epochs to rollout the policy")
    parser.add_argument('--num_need_traj', type=int, default=100, help="Needed valid trajs in rollout")
    parser.add_argument("--rollout-batch", type=int, default=256, help="Number of trajs to be sampled at one time")

    # RCSL policy (mlp)
    parser.add_argument("--rcsl-hidden-dims", type=int, nargs='*', default=[20])
    parser.add_argument("--rcsl-lr", type=float, default=1e-3)
    parser.add_argument("--rcsl-batch", type=int, default=256)
    parser.add_argument("--rcsl-epoch", type=int, default=30)
    parser.add_argument("--rcsl-step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--fix_eval_seed", action="store_true", help="True to fix the seed for every eval")

    parser.add_argument("--batch-size", type=int, default=256)

    return parser.parse_args()

def plot(frame, filename):
    frame = frame.reshape(48,3,48)
    frame = np.transpose(frame, (1,2,0))
    frame = np.transpose(frame, (1,2,0))
    frame = (frame * 255).astype(np.ubyte)
    print(type(frame), frame.shape)
    # plt.plot(frame)
    # plt.savefig(frame, filename)
    plt.imsave(filename, frame)

def train(args=get_args()):
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # log
    # log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args))
    # # key: output file name, value: output handler type
    # output_config = {
    #     "consoleout_backup": "stdout",
    #     "policy_training_progress": "csv",
    #     "dynamics_training_progress": "csv",
    #     "tb": "tensorboard"
    # }
    # logger = Logger(log_dirs, output_config)
    # logger.log_hyperparameters(vars(args))

    # create env and dataset
    if args.task == 'pickplace':
        # render_mode = 'human' if args.render else None
        # env = Linearq(size_param=args.env_param)
        # env2 = Linearq(size_param=args.env_param)
        # # dataset = qlearning_dataset(env, get_rtg=True)
        # dataset, init_obss_dataset, max_offline_return = traj_rtg_datasets(env)
        # obs_space = env.observation_space
        # args.obs_shape = (1,)
        # print(args.obs_shape)
        env = roboverse.make('Widow250PickTray-v0')
        # env = SimpleObsWrapper(env)
        # v_env = gym.vector.SyncVectorEnv([lambda: SimpleObsWrapper(roboverse.make('Widow250PickTray-v0')) for t in range(args.eval_episodes)])
        # env2 = roboverse.make('Widow250PickTray-v0')
        # env2 = SimpleObsWrapper(env2)
        env_wrapped = SimpleObsWrapper(env)
        obs_space = env_wrapped.observation_space
        args.obs_shape = obs_space.shape
        obs_dim = np.prod(args.obs_shape)
        args.action_shape = env_wrapped.action_space.shape
        action_dim = np.prod(args.action_shape)

        dataset, init_obss = get_pickplace_dataset(args.data_dir, task_weight=args.task_weight)
    else:
        raise NotImplementedError

    
    # elif args.task == 'maze': # self-constructed
        
    #     from pointmaze.envs.create_maze_dataset import create_env_dataset
    #     point_maze = create_env_dataset(args)
    #     env = point_maze.env_cls()
    #     trajs = point_maze.dataset[0] # first object is trajs
    #     dataset = Trajs2Dict(trajs)

    #     # Add a get_true_observation method for Env
    #     def get_true_observation(obs):
    #         '''
    #         obs, obs received from pointmaze Env. Dict.
    #         '''
    #         return obs['observation']
    
    #     setattr(env, 'get_true_observation', get_true_observation)

    #     obs_space = env.observation_space['observation']
    #     args.obs_shape = env.observation_space['observation'].shape
    # else:
    #     render_mode = 'human' if args.render else None
    #     env = gym.make(args.task, render_mode = render_mode)
    #     env2 = gym.make(args.task, render_mode = render_mode)
    #     # dataset = qlearning_dataset(env, get_rtg=True)
    #     dataset, init_obss_dataset, max_offline_return = traj_rtg_datasets(env)
    #     obs_space = env.observation_space
    #     args.obs_shape = env.observation_space.shape
    # obs_dim = np.prod(args.obs_shape)
    # args.action_dim = np.prod(env.action_space.shape)
    # obs_dim = int(1)
    # args.action_dim = int(1)
    # print(obs_dim, args.action_dim)
    # args.max_action = env.action_space.high[0]

    # seed
    env.reset()
    # env2.reset(seed = args.seed)

    # rcsl_backbone = MLP(input_dim=obs_dim+1, hidden_dims=args.rcsl_hidden_dims, output_dim=args.action_dim)

    # rcsl_module = RcslModule(rcsl_backbone, args.device)
    # rcsl_optim = torch.optim.Adam(rcsl_module.parameters(), lr=args.rcsl_lr)

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(rcsl_optim, args.rcsl_epoch)

    # rcsl_policy = RcslPolicy(
    #     dynamics = None,
    #     rollout_policy = None,
    #     rcsl = rcsl_module,
    #     rcsl_optim = rcsl_optim,
    #     device = args.device
    # )

    diffusion_policy = SimpleDiffusionPolicy(
        obs_shape = args.obs_shape,
        act_shape= args.action_shape,
        feature_dim = 1,
        num_training_steps = args.behavior_epoch,
        num_diffusion_steps = args.num_diffusion_iters,
        device = args.device
    )

    if args.load_diffusion_path is not None:
        with open(args.load_diffusion_path, 'rb') as f:
            state_dict = torch.load(f, map_location= args.device)
        diffusion_policy.load_state_dict(state_dict)


    diff_lr_scheduler = diffusion_policy.get_lr_scheduler()

    # create buffer
    # offline_buffer = ReplayBuffer(
    #     buffer_size=len(dataset["observations"]),
    #     obs_shape=args.obs_shape,
    #     obs_dtype=np.float32,
    #     action_dim=args.action_dim,
    #     action_dtype=np.float32,
    #     device=args.device
    # )
    # offline_buffer.load_dataset(dataset)

    # train

    # Creat policy trainer
    # rcsl_log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args))
    # # key: output file name, value: output handler type
    # rcsl_output_config = {
    #     "consoleout_backup": "stdout",
    #     "policy_training_progress": "csv",
    #     "dynamics_training_progress": "csv",
    #     "tb": "tensorboard"
    # }
    # rcsl_logger = Logger(rcsl_log_dirs, rcsl_output_config)
    # rcsl_logger.log_hyperparameters(vars(args))

    def _evaluate(eval_episodes: int = 1):
        '''
        Always set desired rtg to 0
        '''
        # Pointmaze obs has different format, needs to be treated differently
        # if eval_episodes == -1:
        #     real_eval_episodes = self._eval_episodes
        # else:
        real_eval_episodes = eval_episodes
        is_gymnasium_env = False

        env.reset() # Fix seed
        
        diffusion_policy.eval()
        if is_gymnasium_env:
            obs, _ = env.reset()
            obs = env.get_true_observation(obs)
        else:
            obs = env.reset()

        

        frame = obs['image']
        plot(frame, 'init.png')

        obs = get_true_obs(obs)
            

        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0
        
        picked = False
        placed = True
        while num_episodes < real_eval_episodes:
            rtg = torch.tensor([[1]]).type(torch.float32)
            pick_success = False
            for timestep in range(40): # One epoch
                # print(f"Timestep {timestep}, obs {obs}")
                action = diffusion_policy.select_action(obs.reshape(1, -1), rtg)
                if hasattr(env, "get_true_observation"): # gymnasium env 
                    next_obs, reward, terminal, _, _ = env.step(action.flatten())
                else:
                    next_obs, reward, terminal, info = env.step(action.flatten())

                if not picked and info['grasp_success'] :
                    picked = True
                    frame = next_obs['image']
                    plot(frame)
                elif not placed and info['place_success']:
                    placed = True
                    frame = next_obs['image']
                    plot(frame)
                if is_gymnasium_env:
                    next_obs = env.get_true_observation(next_obs)
                # if num_episodes == 2 and timestep < 10:
                #     print(f"Action {action}, next_obs {next_obs}, reward {reward}, rtg {rtg.item()}")
                episode_reward += reward
                # No need to update return
                rtg = rtg - reward
                episode_length += 1

                if info['grasp_success_target']: # pick okay
                    pick_success = True

                obs = get_true_obs(next_obs)

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
                obs, _ = env.reset()
                obs = env.get_true_observation(obs)
            else:
                obs = env.reset()
    
    # print(f"Start evaluate")
    # policy_trainer.train(last_eval=True)
    _evaluate()
    # result = policy_trainer._evaluate()
    # print(result)


if __name__ == "__main__":
    train()