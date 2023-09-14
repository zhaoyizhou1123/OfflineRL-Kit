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
from typing import Dict, Tuple, Optional
from offlinerlkit.utils.cumsum import discount_cumsum
import collections


# import __init__
from offlinerlkit.nets import MLP
from offlinerlkit.modules import RcslGaussianModule, DiagGaussian
# from offlinerlkit.utils.load_dataset import traj_rtg_datasets
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.utils.diffusion_logger import setup_logger
from offlinerlkit.policy_trainer import RcslPolicyTrainer
from offlinerlkit.utils.trajectory import Trajectory
from offlinerlkit.utils.none_or_str import none_or_str
from offlinerlkit.policy import DiffusionBC, RcslGaussianPolicy

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
    parser.add_argument("--algo-name", type=str, default="rcsl_gauss")
    parser.add_argument("--task", type=str, default="hopper-medium-expert-v2", help="maze") # Self-constructed environment
    parser.add_argument("--dataset", type=none_or_str, default=None, help="../D4RL/dataset/halfcheetah/output.hdf5") # Self-constructed environment
    parser.add_argument('--debug',action='store_true', help='Print debuuging info if true')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1, help="Dataloader workers, align with cpu number")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # env config (maze)
    parser.add_argument('--maze_config_file', type=str, default='./pointmaze/config/maze2_simple_moredata.json')
    parser.add_argument('--data_file', type=str, default='./pointmaze/dataset/maze2_smds_acc.dat')
    parser.add_argument('--render', action='store_true')
    # parser.add_argument('--log_to_wandb',action='store_true', help='Set up wandb')
    # parser.add_argument('--tb_path', type=str, default=None, help="./logs/stitch/, Folder to tensorboard logs" )
    # parser.add_argument('--env_type', type=str, default='pointmaze', help='pointmaze or ?')
    parser.add_argument('--algo', type=str, default='stitch-mlp', help="rcsl-mlp, rcsl-dt or stitch-mlp, stitch-dt")
    parser.add_argument('--horizon', type=int, default=1000, help="horizon for pointmaze, or max timesteps for d4rl")


    
    # parser.add_argument("--actor-lr", type=float, default=1e-4)
    # parser.add_argument("--critic-lr", type=float, default=3e-4)
    # parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256, 256])
    # parser.add_argument("--gamma", type=float, default=0.99)
    # parser.add_argument("--tau", type=float, default=0.005)
    # parser.add_argument("--alpha", type=float, default=0.2)
    # parser.add_argument("--auto-alpha", default=True)
    # parser.add_argument("--target-entropy", type=int, default=None)
    # parser.add_argument("--alpha-lr", type=float, default=1e-4)

    # parser.add_argument("--cql-weight", type=float, default=5.0)
    # parser.add_argument("--temperature", type=float, default=1.0)
    # parser.add_argument("--max-q-backup", type=bool, default=False)
    # parser.add_argument("--deterministic-backup", type=bool, default=True)
    # parser.add_argument("--with-lagrange", type=bool, default=False)
    # parser.add_argument("--lagrange-threshold", type=float, default=10.0)
    # parser.add_argument("--cql-alpha-lr", type=float, default=3e-4)
    # parser.add_argument("--num-repeat-actions", type=int, default=10)
    # parser.add_argument("--uniform-rollout", type=bool, default=False)
    # parser.add_argument("--rho-s", type=str, default="mix", choices=["model", "mix"])

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
    parser.add_argument("--behavior_epoch", type=int, default=50)
    parser.add_argument("--num_diffusion_iters", type=int, default=10, help="Number of diffusion steps")
    parser.add_argument('--behavior_batch', type=int, default=256)
    parser.add_argument('--load_diffusion_path', type=none_or_str, default=None)
    parser.add_argument('--diffusion_seed', type=str, default='0', help="Distinguish runs for diffusion policy, not random seed")

    # Rollout 
    parser.add_argument('--rollout_ckpt_path', type=none_or_str, default=None, help="./checkpoint/maze2_smd_stable, file path, used to load/store rollout trajs" )
    parser.add_argument('--rollout_epochs', type=int, default=1000, help="Max number of epochs to rollout the policy")
    parser.add_argument('--num_need_traj', type=int, default=100, help="Needed valid trajs in rollout")
    parser.add_argument("--rollout-batch", type=int, default=256, help="Number of trajs to be sampled at one time")

    # RCSL policy (mlp)
    parser.add_argument("--rcsl-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--rcsl-lr", type=float, default=1e-3)
    parser.add_argument("--rcsl-batch", type=int, default=256)
    parser.add_argument("--rcsl-epoch", type=int, default=50)
    parser.add_argument("--rcsl-step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--fix_eval_seed", action="store_true", help="True to fix the seed for every eval")

    parser.add_argument("--batch-size", type=int, default=256)

    return parser.parse_args()

def traj_rtg_datasets(env, data_path: Optional[str] = None):
    '''
    Download all datasets needed for experiments, and re-combine them as trajectory datasets
    Throw away the last uncompleted trajectory

    Args:
        data_dir: path to store dataset file

    Return:
        dataset: Dict,
        initial_obss: np.ndarray
        max_return: float
    '''
    dataset = env.get_dataset()

    N = dataset['rewards'].shape[0] # number of data (s,a,r)
    data_ = collections.defaultdict(list)

    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    paths = []
    # obs_ = []
    # next_obs_ = []
    # action_ = []
    # reward_ = []
    # done_ = []
    # rtg_ = []

    for i in range(N): # Loop through data points

        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == 1000-1)
        for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
            data_[k].append(dataset[k][i])
            
        # obs_.append(dataset['observations'][i].astype(np.float32))
        # next_obs_.append(dataset['next_observations'][i].astype(np.float32))
        # action_.append(dataset['actions'][i].astype(np.float32))
        # reward_.append(dataset['rewards'][i].astype(np.float32))
        # done_.append(bool(dataset['terminals'][i]))

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            # Update rtg
            rtg_traj = discount_cumsum(np.array(data_['rewards']))
            episode_data['rtgs'] = rtg_traj
            # rtg_ += rtg_traj

            paths.append(episode_data)
            data_ = collections.defaultdict(list)

        episode_step += 1

    init_obss = np.array([p['observations'][0] for p in paths]).astype(np.float32)

    returns = np.array([np.sum(p['rewards']) for p in paths])
    num_samples = np.sum([p['rewards'].shape[0] for p in paths])
    print(f'Number of samples collected: {num_samples}')
    print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

    if data_path is not None:
        with open(data_path, 'wb') as f:
            pickle.dump(paths, f)

    # print(f"N={N},len(obs_)={len(obs_)},len(reward_)={len(reward_)},len(rtg_)={len(rtg_)}!")
    # assert len(obs_) == len(rtg_), f"Got {len(obs_)} obss, but {len(rtg_)} rtgs!"

    # Concatenate paths into one dataset
    full_dataset = {}
    for k in ['observations', 'next_observations', 'actions', 'rewards', 'rtgs', 'terminals']:
        full_dataset[k] = np.concatenate([p[k] for p in paths], axis=0)

    return full_dataset, paths, init_obss, np.max(returns)

def train(task:str, policy_path:str, args=get_args()):
    print(args)

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
    if task == 'maze': # self-constructed
        
        from pointmaze.envs.create_maze_dataset import create_env_dataset
        point_maze = create_env_dataset(args)
        env = point_maze.env_cls()
        trajs = point_maze.dataset[0] # first object is trajs
        dataset = Trajs2Dict(trajs)

        # Add a get_true_observation method for Env
        def get_true_observation(obs):
            '''
            obs, obs received from pointmaze Env. Dict.
            '''
            return obs['observation']
    
        setattr(env, 'get_true_observation', get_true_observation)

        obs_space = env.observation_space['observation']
        args.obs_shape = env.observation_space['observation'].shape
    else:
        render_mode = 'human' if args.render else None
        env = gym.make(task, render_mode = render_mode)
        env2 = gym.make(task, render_mode = render_mode)
        # dataset = qlearning_dataset(env, get_rtg=True)
        dataset, trajs, init_obss_dataset, max_offline_return = traj_rtg_datasets(env)
        obs_space = env.observation_space
        args.obs_shape = env.observation_space.shape
    obs_dim = np.prod(args.obs_shape)
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.reset(seed = args.seed)
    env2.reset(seed = args.seed)

    # print(args.action_dim)
    rcsl_backbone = MLP(input_dim=obs_dim+1, hidden_dims=args.rcsl_hidden_dims, output_dim=args.action_dim)
    dist = DiagGaussian(
        latent_dim=getattr(rcsl_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )

    rcsl_module = RcslGaussianModule(rcsl_backbone, dist, args.device)
    rcsl_optim = torch.optim.Adam(rcsl_module.parameters(), lr=args.rcsl_lr)

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(rcsl_optim, args.rcsl_epoch)

    rcsl_policy = RcslGaussianPolicy(
        dynamics = None,
        rollout_policy = None,
        rcsl = rcsl_module,
        rcsl_optim = rcsl_optim,
        device = args.device
    )

    state_dict = torch.load(policy_path, map_location='cuda')
    print(f"Loaded policy")
    rcsl_policy.load_state_dict(state_dict)
    print(f"Loaded state dict")
    

    # create buffer
    offline_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    offline_buffer.load_dataset(dataset)

    def evaluate(repeat=10, random = False, truncate = 10, goal_mul = 1., threshold = 0.):
        # Pointmaze obs has different format, needs to be treated differently
        is_gymnasium_env = False

        env.reset(seed=0) # Fix seed
        
        rcsl_policy.eval()
        if is_gymnasium_env:
            obs, _ = env.reset()
            obs = env.get_true_observation(obs)
        else:
            obs = env.reset()
            

        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        if is_gymnasium_env: # pointmaze environment, don't use horizon
            while num_episodes < repeat:
                rtg = torch.tensor([[max_offline_return]]).type(torch.float32) * goal_mul
                for timestep in range(args.horizon): # One epoch
                    # print(f"Timestep {timestep}, obs {obs}")
                    action = rcsl_policy.select_action(obs.reshape(1, -1), rtg)
                    if hasattr(env, "get_true_observation"): # gymnasium env 
                        next_obs, reward, terminal, _, _ = env.step(action.flatten())
                    else:
                        next_obs, reward, terminal, _ = env.step(action.flatten())
                    if is_gymnasium_env:
                        next_obs = env.get_true_observation(next_obs)
                    if num_episodes == 2 and timestep < 10:
                        print(f"Action {action}, next_obs {next_obs}, reward {reward}, rtg {rtg.item()}")
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
        else:
            rtg = torch.tensor([[max_offline_return]]).type(torch.float32) * goal_mul
            while num_episodes < repeat:
                # env.render()
                # print(f"Timestep {timestep}, obs {obs}")

                if random:
                    action = env.action_space.sample()
                else:
                    action = rcsl_policy.select_action(obs.reshape(1, -1), rtg) 
                    # action += np.random.normal(0,2,action.shape)
                if hasattr(env, "get_true_observation"): # gymnasium env 
                    next_obs, reward, terminal, _, _ = env.step(action.flatten())
                else:
                    next_obs, reward, terminal, _ = env.step(action.flatten())
                if is_gymnasium_env:
                    next_obs = env.get_true_observation(next_obs)
                episode_reward += reward
                rtg = rtg - reward
                episode_length += 1

                obs = next_obs

                if terminal or episode_length >= truncate: # Episode finishes
                    if episode_reward > threshold:
                        print(f"Get total return {episode_reward}")
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
                    rtg = torch.tensor([[max_offline_return]]).type(torch.float32) * goal_mul
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }

    truncate = int(200)

    rets = np.array([env.get_normalized_score(traj['rtgs'][0]) * 100 for traj in trajs])
    # print(np.max(rets))
    threshold = 42
    large_rets = rets[rets>threshold]
    large_num = len(large_rets)
    ratio_large = len(large_rets) / len(rets)
    print(f"Large ratio: {ratio_large}")

    rtgs_trunc = []
    for traj in trajs:
        rtg_trunc = traj['rtgs'][0] - traj['rtgs'][truncate]
        # print(rtg_trunc)
        rtgs_trunc.append(rtg_trunc)
    
    ref_rtg_trunc = sorted(rtgs_trunc)[-large_num]
    print(f"{large_num}-th largest return in first {truncate} step : {ref_rtg_trunc}")

    evaluate(random=False, truncate=truncate, goal_mul=1, repeat=400, threshold = ref_rtg_trunc)


if __name__ == "__main__":
    train('halfcheetah-medium-v2', "logs/halfcheetah-medium-v2/rcsl_gauss/rcsl/timestamp_23-0912-084125&0/checkpoint/policy.pth")