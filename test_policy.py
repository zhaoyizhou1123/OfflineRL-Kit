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
from typing import Dict, Tuple, List
import roboverse
from collections import defaultdict


# import __init__
from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, EnsembleDynamicsModel, RcslModule
from offlinerlkit.dynamics import BaseDynamics, EnsembleDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn
from offlinerlkit.utils.load_dataset import qlearning_dataset, traj_rtg_datasets
from offlinerlkit.utils.config import Config
from offlinerlkit.utils.dataset import ObsActDataset
from offlinerlkit.utils.pickplace_utils import SimpleObsWrapper, get_pickplace_dataset
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.utils.diffusion_logger import setup_logger
from offlinerlkit.policy_trainer import RcslPolicyTrainer, DiffusionPolicyTrainer
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
    parser.add_argument("--algo-name", type=str, default="test_dyn")
    parser.add_argument("--task", type=str, default="pickplace", help="pickplace") # Self-constructed environment
    parser.add_argument("--dataset", type=none_or_str, default=None, help="../D4RL/dataset/halfcheetah/output.hdf5") # Self-constructed environment
    parser.add_argument('--debug',action='store_true', help='Print debuuging info if true')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1, help="Dataloader workers, align with cpu number")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # env config (pickplace)
    parser.add_argument('--data_dir', type=str, default='./dataset')
    parser.add_argument('--horizon', type=int, default=40, help="max path length for pickplace")

    # env config (maze)
    # parser.add_argument('--maze_config_file', type=str, default='./pointmaze/config/maze2_simple_moredata.json')
    # parser.add_argument('--data_file', type=str, default='./pointmaze/dataset/maze2_smds_acc.dat')
    # parser.add_argument('--render', action='store_true')
    # # parser.add_argument('--log_to_wandb',action='store_true', help='Set up wandb')
    # # parser.add_argument('--tb_path', type=str, default=None, help="./logs/stitch/, Folder to tensorboard logs" )
    # # parser.add_argument('--env_type', type=str, default='pointmaze', help='pointmaze or ?')
    # parser.add_argument('--algo', type=str, default='stitch-mlp', help="rcsl-mlp, rcsl-dt or stitch-mlp, stitch-dt")
    # parser.add_argument('--horizon', type=int, default=1000, help="horizon for pointmaze, or max timesteps for d4rl")

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
    parser.add_argument("--behavior_epoch", type=int, default=30)
    parser.add_argument("--num_diffusion_iters", type=int, default=10, help="Number of diffusion steps")
    parser.add_argument('--behavior_batch', type=int, default=256)
    parser.add_argument('--load_diffusion_path', type=none_or_str, default=None)
    parser.add_argument('--diffusion_seed', type=str, default='0', help="Distinguish runs for diffusion policy, not random seed")

    # Rollout 
    parser.add_argument('--rollout_ckpt_path', type=none_or_str, default=None, help="./checkpoint/maze2_smd_stable, file path, used to load/store rollout trajs" )
    parser.add_argument('--rollout_epochs', type=int, default=100, help="Max number of epochs to rollout the policy")
    parser.add_argument('--num_need_traj', type=int, default=10000, help="Needed valid trajs in rollout")
    parser.add_argument("--rollout-batch", type=int, default=256, help="Number of trajs to be sampled at one time")

    # RCSL policy (mlp)
    parser.add_argument("--rcsl-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--rcsl-lr", type=float, default=1e-3)
    parser.add_argument("--rcsl-batch", type=int, default=256)
    parser.add_argument("--rcsl-epoch", type=int, default=50)
    parser.add_argument("--rcsl-step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=100)

    parser.add_argument("--batch-size", type=int, default=256)

    return parser.parse_args()

def rollout(
    init_obss: np.ndarray,
    dynamics: EnsembleDynamicsModel,
    rollout_policy: SimpleDiffusionPolicy,
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

    # frozen_noise = rollout_policy.sample_init_noise(init_obss.shape[0])
    goal = np.zeros((init_obss.shape[0],1), dtype = np.float32)
    for _ in range(rollout_length):
        actions = rollout_policy.select_action(observations, goal)
        next_observations, rewards, terminals, info = dynamics.step(observations, actions)
        rollout_transitions["observations"].append(observations)
        rollout_transitions["next_observations"].append(next_observations)
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
        goal = goal[nonterm_mask]
    
    for k, v in rollout_transitions.items():
        rollout_transitions[k] = np.concatenate(v, axis=0)

    # Compute rtgs. Sparse reward only has 0/1 rtg
    returns = (returns >= 1).astype(np.float32) # return >=1 means success, 1; otherwise 0
    traj_idxs = rollout_transitions["traj_idxs"]
    # rtgs = returns[traj_idxs] - rollout_transitions["acc_rets"]
    rtgs = returns[traj_idxs] 
    rollout_transitions["rtgs"] = rtgs[..., None] # (N,1)

    return rollout_transitions, \
        {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean(), "returns": returns}

def train(args=get_args()):
    print(args)

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
        env = SimpleObsWrapper(env)
        # env2 = roboverse.make('Widow250PickTray-v0')
        # env2 = SimpleObsWrapper(env2)
        obs_space = env.observation_space
        args.obs_shape = obs_space.shape
        obs_dim = np.prod(args.obs_shape)
        args.action_shape = env.action_space.shape
        action_dim = np.prod(args.action_shape)
        # args.max_action = env.action_space.high[0]
        # print(args.action_dim, type(args.action_dim))

        dataset, init_obss_dataset = get_pickplace_dataset(args.data_dir)
    else:
        raise NotImplementedError


    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.reset(seed = args.seed)

    # print(f"dynamics_hidden_dims = {args.dynamics_hidden_dims}")
    # log

    rcsl_backbone = MLP(input_dim=obs_dim+1, hidden_dims=args.rcsl_hidden_dims, output_dim=action_dim)

    rcsl_module = RcslModule(rcsl_backbone, args.device)
    rcsl_optim = torch.optim.Adam(rcsl_module.parameters(), lr=args.rcsl_lr)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(rcsl_optim, args.rcsl_epoch)

    rcsl_policy = RcslPolicy(
        dynamics = None,
        rollout_policy = None,
        rcsl = rcsl_module,
        rcsl_optim = rcsl_optim,
        device = args.device
    )

    model_path = "logs/pickplace/test_dyn/rcsl/timestamp_23-0915-054910&0/checkpoint/policy.pth"
    with open(model_path, "rb") as f:
        state_dict = torch.load(model_path, map_location=args.device)
    rcsl_policy.load_state_dict(state_dict)

    def evaluate() -> Dict[str, List[float]]:
        # Pointmaze obs has different format, needs to be treated differently
        is_gymnasium_env = False

        # env.reset(seed=self.env_seed) # Fix seed
        
        rcsl_policy.eval()
        if is_gymnasium_env:
            obs, _ = env.reset()
            obs = env.get_true_observation(obs)
        else:
            obs = env.reset()
            

        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < args.eval_episodes:
            rtg = torch.tensor([[1.]]).type(torch.float32)
            for timestep in range(args.horizon): # One epoch
                # print(f"Timestep {timestep}, obs {obs}")
                action = rcsl_policy.select_action(obs.reshape(1, -1), rtg)
                if hasattr(env, "get_true_observation"): # gymnasium env 
                    next_obs, reward, terminal, _, _ = env.step(action.flatten())
                else:
                    next_obs, reward, terminal, info = env.step(action.flatten())
                if is_gymnasium_env:
                    next_obs = env.get_true_observation(next_obs)
                print(f"Step {timestep}, reward {reward}")
                print(info)
                print("------------")
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
                obs, _ = env.reset()
                obs = env.get_true_observation(obs)
            else:
                obs = env.reset()
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }

    evaluate()
    


if __name__ == "__main__":
    train()