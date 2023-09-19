import argparse
import os
import sys
import random

import numpy as np
import torch

import pickle
from copy import deepcopy
from typing import Dict, Tuple
import roboverse


# import __init__
from offlinerlkit.nets import MLP
from offlinerlkit.modules import RcslGaussianModule, DiagGaussian
from offlinerlkit.utils.load_dataset import traj_rtg_datasets
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.utils.diffusion_logger import setup_logger
from offlinerlkit.policy_trainer import RcslPolicyTrainer_v2
from offlinerlkit.utils.trajectory import Trajectory
from offlinerlkit.utils.none_or_str import none_or_str
from offlinerlkit.policy import DiffusionBC, AutoregressivePolicy
from offlinerlkit.utils.pickplace_utils import SimpleObsWrapper, get_pickplace_dataset, set_weight_dict, merge_dataset

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
    parser.add_argument("--algo-name", type=str, default="regress_gauss")
    parser.add_argument("--task", type=str, default="pickplace", help="pickplace_easy") # Self-constructed environment
    parser.add_argument("--dataset", type=none_or_str, default=None, help="../D4RL/dataset/halfcheetah/output.hdf5") # Self-constructed environment
    parser.add_argument('--debug',action='store_true', help='Print debuuging info if true')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers, align with cpu number")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # env config (pickplace)
    parser.add_argument('--data_dir', type=str, default='./dataset')
    parser.add_argument('--horizon', type=int, default=40, help="max path length for pickplace")

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
    parser.add_argument('--task_weight', type=float, default=1.)

    # Rollout 
    parser.add_argument('--rollout_ckpt_path', type=none_or_str, default=None, help="./checkpoint/maze2_smd_stable, file path, used to load/store rollout trajs" )
    parser.add_argument('--rollout_epochs', type=int, default=1000, help="Max number of epochs to rollout the policy")
    parser.add_argument('--num_need_traj', type=int, default=100, help="Needed valid trajs in rollout")
    parser.add_argument("--rollout-batch", type=int, default=256, help="Number of trajs to be sampled at one time")

    # RCSL policy (mlp)
    parser.add_argument("--rcsl-hidden-dims", type=int, nargs='*', default=[1024, 1024, 1024, 1024])
    parser.add_argument("--rcsl-lr", type=float, default=1e-3)
    parser.add_argument("--rcsl-batch", type=int, default=256)
    parser.add_argument("--rcsl-epoch", type=int, default=50)
    parser.add_argument("--rcsl-step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--fix_eval_seed", action="store_true", help="True to fix the seed for every eval")

    parser.add_argument("--batch-size", type=int, default=256)

    return parser.parse_args()

def train(args=get_args()):
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
    if args.task == 'pickplace' or args.task == 'pickplace_easy':
        # render_mode = 'human' if args.render else None
        # env = Linearq(size_param=args.env_param)
        # env2 = Linearq(size_param=args.env_param)
        # # dataset = qlearning_dataset(env, get_rtg=True)
        # dataset, init_obss_dataset, max_offline_return = traj_rtg_datasets(env)
        # obs_space = env.observation_space
        # args.obs_shape = (1,)
        # print(args.obs_shape)
        if args.task == 'pickplace':
            env = roboverse.make('Widow250PickTray-v0')
            env = SimpleObsWrapper(env)
            # v_env = gym.vector.SyncVectorEnv([lambda: SimpleObsWrapper(roboverse.make('Widow250PickTray-v0')), t ) for t in range(args.rollout_batch)])
        else:
            print(f"Env: easy")
            env = roboverse.make('Widow250PickTrayEasy-v0')
            env = SimpleObsWrapper(env)
            # v_env = gym.vector.SyncVectorEnv([lambda: reset_multi(SimpleObsWrapper(roboverse.make('Widow250PickTrayEasy-v0')), t ) for t in range(args.rollout_batch)])
        obs_space = env.observation_space
        args.obs_shape = obs_space.shape
        obs_dim = np.prod(args.obs_shape)
        args.action_shape = env.action_space.shape
        action_dim = np.prod(args.action_shape)
        # args.max_action = env.action_space.high[0]
        # print(args.action_dim, type(args.action_dim))

        if args.rollout_ckpt_path is not None:
            print(f"Will load rollout trajectories from dir {args.rollout_ckpt_path}")
            # os.makedirs(args.rollout_ckpt_path, exist_ok=True)
            data_path = os.path.join(args.rollout_ckpt_path, "rollout.dat")
            ckpt_dict = pickle.load(open(data_path,"rb")) # checkpoint in dict type
            rollout_data_all = ckpt_dict['data'] # should be dict
            num_traj_all = ckpt_dict['num_traj']
            returns_all = ckpt_dict['return']
            last_epoch = ckpt_dict['epoch']
            # trajs = ckpt_dict
            print(f"Loaded checkpoint. Already have {num_traj_all} valid trajectories, last epoch {last_epoch}.")
            # set_weight_dict(rollout_data_all, 5.)
            # task_dataset, _ = get_pickplace_dataset(args.data_dir, task_weight=1., set_type='task')
            # dataset = merge_dataset([rollout_data_all, task_dataset])
            dataset = rollout_data_all
        else:
            dataset, init_obss_dataset = get_pickplace_dataset(args.data_dir, task_weight=args.task_weight)

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.reset(seed = args.seed)
    # env2.reset(seed = args.seed)

    rcsl_policy = AutoregressivePolicy(
        obs_dim=obs_dim,
        act_dim = action_dim,
        hidden_dims=args.rcsl_hidden_dims,
        lr = args.rcsl_lr,
        device = args.device
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(rcsl_policy.rcsl_optim, args.rcsl_epoch)
    

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
    task_name = args.task
    rcsl_log_dirs = make_log_dirs(task_name, args.algo_name, args.seed, vars(args), part='rcsl_regress', record_params=['eval_episodes', 'task_weight'])
    # key: output file name, value: output handler type
    rcsl_output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    rcsl_logger = Logger(rcsl_log_dirs, rcsl_output_config)
    rcsl_logger.log_hyperparameters(vars(args))

    policy_trainer = RcslPolicyTrainer_v2(
        policy = rcsl_policy,
        eval_env = env,
        offline_dataset = dataset,
        rollout_dataset = None,
        goal = np.mean(returns_all), # no return
        logger = rcsl_logger,
        seed = args.seed,
        epoch = args.rcsl_epoch,
        step_per_epoch = args.rcsl_step_per_epoch,
        batch_size = args.rcsl_batch,
        offline_ratio = 1.,
        lr_scheduler = lr_scheduler,
        horizon = args.horizon,
        num_workers = args.num_workers,
        eval_episodes = args.eval_episodes
        # device = args.device
    )
    print(f"Goal: {np.mean(returns_all)}")
    
    policy_trainer.train()


if __name__ == "__main__":
    train()