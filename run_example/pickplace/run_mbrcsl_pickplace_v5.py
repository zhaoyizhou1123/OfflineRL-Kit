# Autoregressive transformer Dynamics_v2, AutoregressivePolicy

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
from collections import defaultdict


# import __init__
from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, TransformerDynamicsModel_v2
from offlinerlkit.dynamics import BaseDynamics, TransformerDynamics_v2
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn
from offlinerlkit.utils.load_dataset import qlearning_dataset, traj_rtg_datasets
from offlinerlkit.utils.config import Config
from offlinerlkit.utils.dataset import ObsActDataset
from offlinerlkit.utils.pickplace_utils import SimpleObsWrapper, get_pickplace_dataset
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.utils.diffusion_logger import setup_logger
from offlinerlkit.policy_trainer import RcslPolicyTrainer, DiffusionPolicyTrainer, RcslPolicyTrainer_v2
from offlinerlkit.utils.trajectory import Trajectory
from offlinerlkit.utils.none_or_str import none_or_str
from offlinerlkit.policy import DiffusionBC, RcslPolicy, SimpleDiffusionPolicy, AutoregressivePolicy
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
    parser.add_argument("--algo-name", type=str, default="mbrcsl_double_regress")
    parser.add_argument("--task", type=str, default="pickplace_easy", help="pickplace, pickplace_easy") # Self-constructed environment
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

    # transformer_autoregressive dynamics
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=32)

    # Behavior policy (diffusion)
    parser.add_argument("--behavior_epoch", type=int, default=30)
    parser.add_argument("--num_diffusion_iters", type=int, default=5, help="Number of diffusion steps")
    parser.add_argument('--behavior_batch', type=int, default=256)
    parser.add_argument('--load_diffusion_path', type=none_or_str, default=None)
    parser.add_argument('--diffusion_seed', type=str, default='0', help="Distinguish runs for diffusion policy, not random seed")
    parser.add_argument('--task_weight', type=float, default=1.0, help="Weight on task data when training diffusion policy")
    parser.add_argument('--sample_ratio', type=float, default=1.0, help="Use (sample_ratio * num_total_data) data to train diffusion policy")
    

    # Rollout 
    parser.add_argument('--rollout_ckpt_path', type=none_or_str, default=None, help="./checkpoint/maze2_smd_stable, file path, used to load/store rollout trajs" )
    parser.add_argument('--rollout_epochs', type=int, default=100, help="Max number of epochs to rollout the policy")
    parser.add_argument('--num_need_traj', type=int, default=1000, help="Needed valid trajs in rollout")
    parser.add_argument("--rollout-batch", type=int, default=256, help="Number of trajs to be sampled at one time")

    # RCSL policy (mlp)
    parser.add_argument("--rcsl-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--rcsl-lr", type=float, default=1e-3)
    parser.add_argument("--rcsl-batch", type=int, default=256)
    parser.add_argument("--rcsl-epoch", type=int, default=50)
    parser.add_argument("--rcsl-step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=20)

    parser.add_argument("--batch-size", type=int, default=256)

    return parser.parse_args()

def rollout_simple(
    init_obss: np.ndarray,
    dynamics: TransformerDynamicsModel_v2,
    rollout_policy: SimpleDiffusionPolicy,
    rollout_length: int
) -> Tuple[Dict[str, np.ndarray], Dict]:
    '''
    Only serves for non-terminal cases
    Sample a batch of trajectories at the same time.
    Output rollout_transitions contain keys:
    obss,
    next_obss,
    actions
    rewards, (N,1)
    rtgs, (N,1)
    traj_idxs, (N)
    '''
    # print(f"Original: {init_obss.shape}")
    # init_obss = np.array([[ 7.35553260e-01,  2.25970593e-01, -3.63337679e-01, -2.54800799e-03,
    # -4.09931962e-02,  9.99155758e-01, -9.14593738e-04,  6.00122578e-01,
    # 1.54955496e-01, -1.73008145e-01,  2.81431906e-04,  7.06825125e-01,
    # 7.07388213e-01,  2.81656111e-04,  3.60000000e-02, -3.60000000e-02,
    # 1.00000000e+00]])
    # # print(f"Initial obs: {init_obss.shape}")

    num_transitions = 0
    rewards_arr = np.array([])
    rollout_transitions = defaultdict(list)
    batch_size = init_obss.shape[0]
    valid_idxs = np.arange(init_obss.shape[0]) # maintain current valid trajectory indexes
    returns = np.zeros(init_obss.shape[0]) # maintain return of each trajectory
    acc_returns = np.zeros(init_obss.shape[0]) # maintain accumulated return of each valid trajectory
    max_rewards = np.zeros(init_obss.shape[0]) # maintain max reward seen in trajectory
    rewards_full = np.zeros((init_obss.shape[0], rollout_length)) # full rewards (batch, H)

    # rollout
    observations = init_obss

    # frozen_noise = rollout_policy.sample_init_noise(init_obss.shape[0])
    goal = np.zeros((init_obss.shape[0],1), dtype = np.float32)
    for t in range(rollout_length):
        actions = rollout_policy.select_action(observations, goal)
        next_observations, rewards, terminals, info = dynamics.step(observations, actions)
        # rewards = rewards.clip(0,2) # set rewards in [0,2]
        rollout_transitions["observations"].append(observations)
        rollout_transitions["next_observations"].append(next_observations)
        rollout_transitions["actions"].append(actions)
        rollout_transitions["rewards"].append(rewards)
        rollout_transitions["terminals"].append(terminals)
        rollout_transitions["traj_idxs"].append(valid_idxs)
        rollout_transitions["acc_rets"].append(acc_returns)

        rewards = rewards.reshape(batch_size) # (B)
        rewards_full[:, t] = rewards


        num_transitions += len(observations)
        rewards_arr = np.append(rewards_arr, rewards.flatten())

        # print(returns[valid_idxs].shape, rewards.shape)
        # returns[valid_idxs] = returns[valid_idxs] + rewards.flatten() # Update return (for valid idxs only)
        returns = returns + rewards.flatten() # Update return (for valid idxs only)
        # max_rewards[valid_idxs] = np.maximum(max_rewards[valid_idxs], rewards.flatten()) # Update max reward
        max_rewards = np.maximum(max_rewards, rewards.flatten()) # Update max reward
        acc_returns = acc_returns + rewards.flatten()

        # nonterm_mask = (~terminals).flatten()
        # if nonterm_mask.sum() == 0:
        #     break

        # observations = next_observations[nonterm_mask] # Only keep trajs that have not terminated
        observations = deepcopy(next_observations)
        # valid_idxs = valid_idxs[nonterm_mask] # update unterminated traj indexs
        # acc_returns = acc_returns[nonterm_mask] # Only keep acc_ret of trajs that have not terminated
        # goal = goal[nonterm_mask]
    
    for k, v in rollout_transitions.items():
        rollout_transitions[k] = np.concatenate(v, axis=0)

    # Compute rtgs.Keep dense return
    # returns = (returns >= 1).astype(np.float32) # return >=1 means success, 1; otherwise 0
    traj_idxs = rollout_transitions["traj_idxs"]
    rtgs = returns[traj_idxs] - rollout_transitions["acc_rets"]
    # rtgs = returns[traj_idxs] 
    rollout_transitions["rtgs"] = rtgs[..., None] # (N,1)

    # print(f"Get predicted last 3 rewards: {rewards_full[:, -3:]}")

    return rollout_transitions, \
        {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean(), "returns": returns, "max_rewards": max_rewards, "rewards_full": rewards_full}


def train(args=get_args()):
    print(args)

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    # env.reset(seed = args.seed)

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

        # offline_dataset, init_obss_dataset = get_pickplace_dataset(args.data_dir, task_weight=args.task_weight)
        diff_dataset, _ = get_pickplace_dataset(args.data_dir, sample_ratio =args.sample_ratio, task_weight=args.task_weight)
        dyn_dataset, init_obss_dataset = get_pickplace_dataset(args.data_dir)
        # args.max_action = env.action_space.high[0]
        # print(args.action_dim, type(args.action_dim

    env.reset(seed=args.seed)

    # print(f"dynamics_hidden_dims = {args.dynamics_hidden_dims}")
    # log
    # log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args), part = "dynamics", record_params=['eval_episodes', 'task_weight', 'sample_ratio'])
    # print(f"Logging dynamics to {log_dirs}")
    # # key: output file name, value: output handler type
    # output_config = {
    #     "consoleout_backup": "stdout",
    #     "policy_training_progress": "csv",
    #     "dynamics_training_progress": "csv",
    #     "tb": "tensorboard"
    # }
    # logger = Logger(log_dirs, output_config)
    # logger.log_hyperparameters(vars(args))

    # dynamics_model = EnsembleDynamicsModel(
    #     obs_dim=obs_dim,
    #     action_dim=action_dim,
    #     hidden_dims=args.dynamics_hidden_dims,
    #     num_ensemble=args.n_ensemble,
    #     num_elites=args.n_elites,
    #     weight_decays=args.dynamics_weight_decay,
    #     device=args.device
    # )
    # dynamics_optim = torch.optim.Adam(
    #     dynamics_model.parameters(),
    #     lr=args.dynamics_lr
    # )
    # scaler = StandardScaler()
    # termination_fn = get_termination_fn(task=args.task)
    # dynamics = EnsembleDynamics(
    #     dynamics_model,
    #     dynamics_optim,
    #     scaler,
    #     termination_fn
    # )

    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args), part = "dynamics_regress", record_params=['sample_ratio', 'task_weight'])
    # key: output file name, value: output handler type
    print(f"Logging dynamics to {log_dirs}")
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    dynamics_model = TransformerDynamicsModel_v2(
        obs_dim=obs_dim,
        act_dim=action_dim,
        obs_min = -1,
        obs_max = 1,
        act_min = -1,
        act_max = 1,
        r_min = 0,
        r_max = 1,
        ckpt_dir = logger.checkpoint_dir,
        # hidden_dims=args.dynamics_hidden_dims,
        # r_hidden_dims=args.dynamics_hidden_dims,
        device = args.device,
        n_layer = args.n_layer,
        n_head = args.n_head,
        n_embd = args.n_embd
    )
    
    dynamics_optim = dynamics_model.configure_optimizer(
        lr = args.dynamics_lr,
        weight_decay= 0. ,
        betas = (0.9, 0.999)
    )
    dynamics = TransformerDynamics_v2(
        dynamics_model,
        # reward_model,
        dynamics_optim,
        # r_optim
        # scaler,
        # termination_fn
    )

    # create rollout policy
    diffusion_policy = SimpleDiffusionPolicy(
        obs_shape = args.obs_shape,
        act_shape= args.action_shape,
        feature_dim = 1,
        num_training_steps = args.behavior_epoch,
        num_diffusion_steps = args.num_diffusion_iters,
        device = args.device
    )

    diff_lr_scheduler = diffusion_policy.get_lr_scheduler()

    diff_log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args), part="diffusion", record_params=['sample_ratio','task_weight'])
    print(f"Logging diffusion to {diff_log_dirs}")
    # key: output file name, value: output handler type
    diff_output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    diff_logger = Logger(diff_log_dirs, diff_output_config)
    diff_logger.log_hyperparameters(vars(args))

    diff_policy_trainer = DiffusionPolicyTrainer(
        policy = diffusion_policy,
        offline_dataset = diff_dataset,
        logger = diff_logger,
        seed = args.seed,
        epoch = args.behavior_epoch,
        batch_size = args.behavior_batch,
        lr_scheduler = diff_lr_scheduler,
        horizon = args.horizon,
        num_workers = args.num_workers,
        has_terminal = False,
        # device = args.device
    )
    

    # # create buffer
    # offline_buffer = ReplayBuffer(
    #     buffer_size=len(dataset["observations"]),
    #     obs_shape=args.obs_shape,
    #     obs_dtype=np.float32,
    #     action_dim=action_dim,
    #     action_dtype=np.float32,
    #     device=args.device
    # )
    # offline_buffer.load_dataset(dataset)


    # Training helper functions
    def get_dynamics():
        '''
        Load or train dynamics model
        '''
        if args.load_dynamics_path:
            print(f"Load dynamics from {args.load_dynamics_path}")
            dynamics.load(args.load_dynamics_path)
        else: 
            print(f"Train dynamics")
            dynamics.train(dyn_dataset, logger, max_epochs_since_update=5, max_epochs=80)
        
    # Finish get_rollout_policy
    def get_rollout_policy():
        '''
        Load or train rollout policy

        Return:
            rollout policy
        '''
        if args.load_diffusion_path is not None:
            print(f"Load behavior policy from {args.load_diffusion_path}")
            with open(args.load_diffusion_path, 'rb') as f:
                state_dict = torch.load(f, map_location= args.device)
            diffusion_policy.load_state_dict(state_dict)
        else:
            print(f"Train diffusion behavior policy")
            diff_policy_trainer.train() # save checkpoint periodically
            # diffusion_policy.save_checkpoint(epoch=None) # Save final model

    def get_rollout_trajs(logger: Logger, threshold = 0.9) -> Tuple[Dict[str, np.ndarray], float]:
        '''
        Rollout trajectories or load existing trajectories.
        If rollout, call `get_rollout_policy()` and `get_dynamics()` first to get rollout policy and dynamics

        Return:
            rollout trajectories
        '''
        '''
        diffusion behavior policy rollout

        - threshold: only keep trajs with ret > [threshold] (valid). Usually the max return in dataset
        - args.num_need_traj: number of valid trajectories needed. End rollout when get enough trajs
        - args.rollout_epochs: maximum rollout epoch. Should be large
        '''
        device = args.device
        num_need_traj = args.num_need_traj

        rollout_data_all = None # Initialize rollout_dataset as nothing
        num_traj_all = 0 # Initialize total number of rollout trajs
        start_epoch = 0 # Default starting epoch
        returns_all = []
        if args.rollout_ckpt_path is not None:
            print(f"Will save rollout trajectories to dir {args.rollout_ckpt_path}")
            os.makedirs(args.rollout_ckpt_path, exist_ok=True)
            data_path = os.path.join(args.rollout_ckpt_path, "rollout.dat")
            if os.path.exists(data_path): # Load ckpt_data
                ckpt_dict = pickle.load(open(data_path,"rb")) # checkpoint in dict type
                rollout_data_all = ckpt_dict['data'] # should be dict
                num_traj_all = ckpt_dict['num_traj']
                returns_all = ckpt_dict['return']
                start_epoch = ckpt_dict['epoch'] + 1
                # trajs = ckpt_dict
                print(f"Loaded checkpoint. Already have {num_traj_all} valid trajectories, start from epoch {start_epoch}.")

                if num_traj_all >= num_need_traj:
                    print(f"Checkpoint trajectories are enough. Skip rollout procedure.")
                    return rollout_data_all, max(returns_all)
        # Still need training, get dynamics and rollout policy
        get_dynamics()
        get_rollout_policy()

        with torch.no_grad():
            for epoch in range(start_epoch, args.rollout_epochs):
                batch_indexs = np.random.randint(0, init_obss_dataset.shape[0], size=args.rollout_batch)
                init_obss = init_obss_dataset[batch_indexs]
                # rollout_data, rollout_info = rollout(init_obss, dynamics, diffusion_policy, args.horizon)
                rollout_data, rollout_info = rollout_simple(init_obss, dynamics, diffusion_policy, args.horizon)
                    # print(pred_state)

                # Only keep trajs with returns > threshold
                returns = rollout_info['returns']
                # print(f"Get returns: {returns}")
                # assert returns.shape[0] == args.rollout_batch
                # max_rewards = rollout_info['max_rewards']
                rewards_full = rollout_info['rewards_full']
                min_last_rewards = np.min(rewards_full[:, -3:], axis = -1) # (B,), final steps must be large
                max_last_rewards = np.max(rewards_full[:, -3:], axis = -1)
                max_cond = np.logical_and(max_last_rewards > 0.9, max_last_rewards < 2)
                min_cond = min_last_rewards > 0.7
                valid_cond = np.logical_and(max_cond, min_cond)
                valid_trajs = np.arange(args.rollout_batch)[valid_cond] # np.array, indexs of all valid trajs

                valid_data_idxs = [rollout_data['traj_idxs'][i] in valid_trajs for i in range(rollout_data['traj_idxs'].shape[0])]
                for k in rollout_data:
                    rollout_data[k] = rollout_data[k][valid_data_idxs]

                # Add rollout_data to rollout_data_all
                if rollout_data_all is None: # No trajs collected
                    rollout_data_all = deepcopy(rollout_data)
                else:
                    for k in rollout_data:
                        rollout_data_all[k] = np.concatenate([rollout_data_all[k], rollout_data[k]], axis=0)
                
                num_traj_all += len(valid_trajs)
                returns_all += list(returns[valid_trajs])

                # print(f"Eval forward: states {states.shape}, actions {actions.shape}")
                print(f"-----------\nEpoch {epoch}, get {len(valid_trajs)} new trajs")
                logger.logkv("Epoch", epoch)
                logger.logkv("num_new_trajs", len(valid_trajs))
                logger.logkv("num_total_trajs", num_traj_all)
                logger.dumpkvs()

                # if args.rollout_ckpt_path is not None: # save periodically
                save_path = os.path.join(logger.checkpoint_dir, "rollout.dat")
                pickle.dump({'epoch': epoch, 
                                'data': rollout_data_all,
                                'num_traj': num_traj_all,
                                'return': returns_all}, open(save_path, "wb"))            

                if num_traj_all >= num_need_traj: # Get enough trajs, quit rollout
                    print(f"End rollout. Total epochs used: {epoch+1}")
                    break
            
        return rollout_data_all, max(returns_all)

    rollout_save_dir = make_log_dirs(args.task, args.algo_name, args.seed, vars(args), part="rollout", record_params=['sample_ratio','task_weight'])
    print(f"Logging diffusion rollout to {rollout_save_dir}")
    rollout_logger = Logger(rollout_save_dir, {"consoleout_backup": "stdout"})
    rollout_logger.log_hyperparameters(vars(args))
    rollout_dataset, max_offline_return = get_rollout_trajs(rollout_logger)
    # print(rollout_dataset.keys())

    # train

    rcsl_policy = AutoregressivePolicy(
        obs_dim=obs_dim,
        act_dim = action_dim,
        hidden_dims=args.rcsl_hidden_dims,
        lr = args.rcsl_lr,
        device = args.device
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(rcsl_policy.rcsl_optim, args.rcsl_epoch)
    
    task_name = args.task
    rcsl_log_dirs = make_log_dirs(task_name, args.algo_name, args.seed, vars(args), part='rcsl_regress', record_params=['sample_ratio','task_weight'])
    # key: output file name, value: output handler type
    print(f"Logging autoregressive gaussian rcsl to {rcsl_log_dirs}")
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
        offline_dataset = rollout_dataset,
        rollout_dataset = None,
        goal = max_offline_return, # AutoregressivePolicy is not return-conditioned
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

    policy_trainer.train(last_eval=True)


if __name__ == "__main__":
    train()