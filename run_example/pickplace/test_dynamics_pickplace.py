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


# import __init__
from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, EnsembleDynamicsModel, RcslGaussianModule as RcslModule
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
    parser.add_argument('--task_weight', type=float, default=1.5)
    parser.add_argument('--sample_ratio', type=float, default=1.0)
    

    # Rollout 
    parser.add_argument('--rollout_ckpt_path', type=none_or_str, default=None, help="./checkpoint/maze2_smd_stable, file path, used to load/store rollout trajs" )
    parser.add_argument('--rollout_epochs', type=int, default=20, help="Max number of epochs to rollout the policy")
    parser.add_argument('--num_need_traj', type=int, default=100, help="Needed valid trajs in rollout")
    parser.add_argument("--rollout-batch", type=int, default=256, help="Number of trajs to be sampled at one time")

    # RCSL policy (mlp)
    parser.add_argument("--rcsl-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--rcsl-lr", type=float, default=1e-3)
    parser.add_argument("--rcsl-batch", type=int, default=256)
    parser.add_argument("--rcsl-epoch", type=int, default=50)
    parser.add_argument("--rcsl-step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)

    parser.add_argument("--batch-size", type=int, default=256)

    return parser.parse_args()

def train(args=get_args()):
    print(args)

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

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

        diff_dataset, _ = get_pickplace_dataset(args.data_dir, sample_ratio =args.sample_ratio, task_weight=args.task_weight)
        dyn_dataset, init_obss = get_pickplace_dataset(args.data_dir)
    else:
        raise NotImplementedError

    # print(f"dynamics_hidden_dims = {args.dynamics_hidden_dims}")
    # log
    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args), part = "dynamics", record_params=['sample_ratio', 'task_weight'])
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    dynamics_model = EnsembleDynamicsModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=args.dynamics_hidden_dims,
        num_ensemble=args.n_ensemble,
        num_elites=args.n_elites,
        weight_decays=args.dynamics_weight_decay,
        device=args.device
    )
    dynamics_optim = torch.optim.Adam(
        dynamics_model.parameters(),
        lr=args.dynamics_lr
    )
    scaler = StandardScaler()
    termination_fn = get_termination_fn(task=args.task)
    dynamics = EnsembleDynamics(
        dynamics_model,
        dynamics_optim,
        scaler,
        termination_fn
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

    diff_log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args), part="diffusion", record_params=['sample_ratio', 'task_weight'])
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

    # rcsl_backbone = MLP(input_dim=obs_dim+1, hidden_dims=args.rcsl_hidden_dims)
    # dist = TanhDiagGaussian(
    #     latent_dim=getattr(rcsl_backbone, "output_dim"),
    #     output_dim=args.action_dim,
    #     unbounded=True,
    #     conditioned_sigma=True
    # )

    # rcsl_module = RcslModule(rcsl_backbone, dist, args.device)
    # rcsl_optim = torch.optim.Adam(rcsl_module.parameters(), lr=args.rcsl_lr)

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(rcsl_optim, args.rcsl_epoch)

    # rcsl_policy = RcslPolicy(
    #     dynamics,
    #     diffusion_policy,
    #     rcsl_module,
    #     rcsl_optim,
    #     device = args.device
    # )
    

    # create buffer
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
            dynamics.train(dyn_dataset, logger, max_epochs_since_update=5)
        
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

    get_dynamics()
    get_rollout_policy()

    def test_rollout(logger: Logger, use_pred = False):
        '''
        use_pred: If True, use predicted state each round
        '''
        if use_pred:
            # print("Using predicted observation to select action")
            logger.log("Using predicted observation to select action")
        else:
            # print("Using true observation to select action")
            logger.log("Using true observation to select action")
        device = args.device
        num_need_traj = args.num_need_traj

        trajs = [] # Initialize valid rollout trajs. If there is checkpoint, first load checkpoint
        start_epoch = 0 # Default starting epoch
        # if args.rollout_ckpt_path is not None:
        #     print(f"Will save rollout trajectories to dir {args.rollout_ckpt_path}")
        #     os.makedirs(args.rollout_ckpt_path, exist_ok=True)
        #     data_path = os.path.join(args.rollout_ckpt_path, "rollout.dat")
        #     if os.path.exists(data_path): # Load ckpt_data
        #         ckpt_dict = pickle.load(open(data_path,"rb")) # checkpoint in dict type
        #         trajs = ckpt_dict['trajs']
        #         start_epoch = ckpt_dict['epoch'] + 1
        #         # trajs = ckpt_dict
        #         print(f"Loaded checkpoint. Already have {len(trajs)} valid trajectories, start from epoch {start_epoch}.")
        #         if len(trajs) >= num_need_traj:
        #             print(f"Checkpoint trajectories are enough. Skip rollout procedure.")
        #             return trajs

        valid_cnt = 0
        pred_cnt = 0
        pick_cnt = 0
        with torch.no_grad():
            for epoch in range(start_epoch, args.rollout_epochs):
                # batch_indexs = np.random.randint(0, init_obss_dataset.shape[0], size=1)
                # init_obss = init_obss_dataset[batch_indexs]
                true_state = env.reset() # (state_dim)
                pred_state = true_state[None, :]
                # print(pred_state.shape)
                    # print(pred_state)

                # print(f"Eval forward: states {states.shape}, actions {actions.shape}")
                print(f"-----------\nEpoch {epoch}")

                pred_ret = 0
                true_ret = 0

                observations_ = []
                actions_ = []
                next_observations_ = []
                pred_rewards_ = []
                achieved_rets_ = [] # The total reward that has achieved, used to compute rtg
                # frozen_noise = diffusion_policy.sample_init_noise() # noise for action sampling
                goal = np.array([[0]], dtype=np.float32)
                pick = False
                place = False
                pred_place = False
                for h in range(args.horizon):
                    timestep = torch.tensor(h).to(device) # scalar
                    observations_.append(deepcopy(pred_state))
                    achieved_rets_.append(deepcopy(pred_ret))

                    # support_actions, support_probs = behavior_model(pred_state.unsqueeze(0).to(device)) # (1, n_support, action_dim), (1,n_support)
                    if use_pred:
                        action = diffusion_policy.select_action(pred_state, goal)
                    else:
                        action = diffusion_policy.select_action(true_state[None, :], goal)
                    # sample_idx = torch.multinomial(support_probs, num_samples=1).squeeze() # scalar
                    # action = support_actions[0,sample_idx,:] # (action_dim)
                    # action = sample_from_supports(support_actions.squeeze(0), support_probs.squeeze(0)).detach().cpu().numpy()
                    # print(action)
                    # print(pred_state.shape, action.shape)
                    if use_pred:
                        pred_next_state, pred_reward, _, _ = dynamics.step(pred_state, action) # (state_dim), (1)
                    else:
                        pred_next_state, pred_reward, _, _ = dynamics.step(true_state[None, :], action) # (state_dim), (1)
                    true_next_state, true_reward, terminated ,info = env.step(action.squeeze(0))
                    # pred_next_state = pred_next_state.squeeze(0) # (state_dim)
                    # pred_reward = pred_reward.squeeze() # scalar
                    # print(pred_next_state.shape, pred_reward.shape)
                    # Observe next states, rewards,
                    # next_state, reward, terminated, _, _ = env.step(action) # array (state_dim), scalar
                    # if hasattr(env, 'get_true_observation'): # For pointmaze
                    #     next_state = env.get_true_observation(next_state)
                    # next_state = torch.from_numpy(next_state) # (state_dim)
                    actions_.append(deepcopy(action))
                    # next_observations_.append(deepcopy(pred_next_state))
                    # rewards_.append(deepcopy(pred_reward))
                    print("-----------------------")
                    print(f"Step {h}, action {action}")
                    print(f"True reward {true_reward}, Predicted reward {pred_reward}")
                    print(info)
                    print(f"True state {true_next_state}, Predicted state {pred_next_state}")
                    print(f"State difference {np.linalg.norm(pred_next_state - true_next_state)} \n")
                    print("-----------------------\n")
                    # Calculate return
                    # ret += reward
                    pred_ret += pred_reward
                    true_ret += true_reward
                    pred_rewards_.append(pred_reward)
                    if info['grasp_success_target']:
                        pick = True
                    if info['place_success_target']:
                        place = True
                    
                    # Update states, actions, rtgs, timesteps
                    # true_state = next_state # (state_dim)
                    pred_state = pred_next_state.reshape(pred_state.shape)
                    true_state = true_next_state.reshape(true_state.shape)

                    # Update timesteps

                    if terminated: # Already reached goal, the rest steps get reward 1, break
                    #     ret += args.horizon - 1 - h
                    #     pred_ret += args.horizon - 1 -h
                        # print(f"Epoch {epoch}, true total return {true_ret}")
                        break
                    
                # print(f"Epoch {epoch}, true total return {true_ret}")
                if true_ret >= 1:
                    valid_cnt += 1

                # if max(pred_rewards_) >= 0.8:
                if min(pred_rewards_[-3:]) >= 0.9:
                    pred_cnt += 1
                    pred_place = True

                if pick:
                    pick_cnt += 1
                # print(f"Epoch {epoch}, True success {place}, predicted success {pred_place}, pick success {pick}")
                logger.log(f"Epoch {epoch}, True success {place}, predicted success {pred_place}, pick success {pick}")
        
        # print(f"Collected {valid_cnt} out of {args.rollout_epochs-start_epoch} trajectories, predicted {pred_cnt}, pick success {pick_cnt}")
        logger.log(f"Collected {valid_cnt} out of {args.rollout_epochs-start_epoch} trajectories, predicted {pred_cnt}, pick success {pick_cnt}")
        return trajs

    def test_rollout_onestep():
        '''
        pred_state is true_prev_state + one step prediction
        '''
        # if use_pred:
        #     print("Using predicted observation to select action")
        # else:
        #     print("Using true observation to select action")
        print("Using one-step predicted observation to select action")
        device = args.device
        num_need_traj = args.num_need_traj

        trajs = [] # Initialize valid rollout trajs. If there is checkpoint, first load checkpoint
        start_epoch = 0 # Default starting epoch
        # if args.rollout_ckpt_path is not None:
        #     print(f"Will save rollout trajectories to dir {args.rollout_ckpt_path}")
        #     os.makedirs(args.rollout_ckpt_path, exist_ok=True)
        #     data_path = os.path.join(args.rollout_ckpt_path, "rollout.dat")
        #     if os.path.exists(data_path): # Load ckpt_data
        #         ckpt_dict = pickle.load(open(data_path,"rb")) # checkpoint in dict type
        #         trajs = ckpt_dict['trajs']
        #         start_epoch = ckpt_dict['epoch'] + 1
        #         # trajs = ckpt_dict
        #         print(f"Loaded checkpoint. Already have {len(trajs)} valid trajectories, start from epoch {start_epoch}.")
        #         if len(trajs) >= num_need_traj:
        #             print(f"Checkpoint trajectories are enough. Skip rollout procedure.")
        #             return trajs

        valid_cnt = 0
        pred_cnt = 0
        with torch.no_grad():
            for epoch in range(start_epoch, args.rollout_epochs):
                # batch_indexs = np.random.randint(0, init_obss_dataset.shape[0], size=1)
                # init_obss = init_obss_dataset[batch_indexs]
                true_state = env.reset() # (state_dim)
                pred_state = true_state[None, :]
                # print(pred_state.shape)
                    # print(pred_state)

                # print(f"Eval forward: states {states.shape}, actions {actions.shape}")
                print(f"-----------\nEpoch {epoch}")

                pred_ret = 0
                true_ret = 0

                observations_ = []
                actions_ = []
                next_observations_ = []
                pred_rewards_ = []
                achieved_rets_ = [] # The total reward that has achieved, used to compute rtg
                # frozen_noise = diffusion_policy.sample_init_noise() # noise for action sampling
                goal = np.array([[0]], dtype=np.float32)
                for h in range(args.horizon):
                    timestep = torch.tensor(h).to(device) # scalar
                    observations_.append(deepcopy(pred_state))
                    achieved_rets_.append(deepcopy(pred_ret))

                    # support_actions, support_probs = behavior_model(pred_state.unsqueeze(0).to(device)) # (1, n_support, action_dim), (1,n_support)
                    # if use_pred:
                    action = diffusion_policy.select_action(pred_state, goal)
                    # else:
                    #     action = diffusion_policy.select_action(true_state[None, :], goal)
                    # sample_idx = torch.multinomial(support_probs, num_samples=1).squeeze() # scalar
                    # action = support_actions[0,sample_idx,:] # (action_dim)
                    # action = sample_from_supports(support_actions.squeeze(0), support_probs.squeeze(0)).detach().cpu().numpy()
                    # print(action)
                    # print(pred_state.shape, action.shape)
                    # if use_pred:
                    #     pred_next_state, pred_reward, _, _ = dynamics.step(pred_state, action) # (state_dim), (1)
                    # else:
                    # One step prediction
                    pred_next_state, pred_reward, _, _ = dynamics.step(true_state[None, :], action) # (state_dim), (1)
                    true_next_state, true_reward, terminated ,info = env.step(action.squeeze(0))
                    # pred_next_state = pred_next_state.squeeze(0) # (state_dim)
                    # pred_reward = pred_reward.squeeze() # scalar
                    # print(pred_next_state.shape, pred_reward.shape)
                    # Observe next states, rewards,
                    # next_state, reward, terminated, _, _ = env.step(action) # array (state_dim), scalar
                    # if hasattr(env, 'get_true_observation'): # For pointmaze
                    #     next_state = env.get_true_observation(next_state)
                    # next_state = torch.from_numpy(next_state) # (state_dim)
                    actions_.append(deepcopy(action))
                    # next_observations_.append(deepcopy(pred_next_state))
                    # rewards_.append(deepcopy(pred_reward))
                    print("-----------------------")
                    print(f"Step {h}, action {action}")
                    print(f"True reward {true_reward}, Predicted reward {pred_reward}")
                    print(info)
                    print(f"True state {true_next_state}, Predicted state {pred_next_state}")
                    print(f"State difference {np.linalg.norm(pred_next_state - true_next_state)} \n")
                    print("-----------------------\n")
                    # Calculate return
                    # ret += reward
                    pred_ret += pred_reward
                    true_ret += true_reward
                    pred_rewards_.append(pred_reward)
                    
                    # Update states, actions, rtgs, timesteps
                    # true_state = next_state # (state_dim)
                    pred_state = pred_next_state.reshape(pred_state.shape)
                    true_state = true_next_state.reshape(true_state.shape)

                    # Update timesteps

                    if terminated: # Already reached goal, the rest steps get reward 1, break
                    #     ret += args.horizon - 1 - h
                    #     pred_ret += args.horizon - 1 -h
                        # print(f"Epoch {epoch}, true total return {true_ret}")
                        break
                    
                # print(f"Epoch {epoch}, true total return {true_ret}")
                if true_ret >= 1:
                    valid_cnt += 1

                if max(pred_rewards_) >= 0.8:
                    pred_cnt += 1
                print(f"Epoch {epoch}, true total return {true_ret}, predicted {pred_ret}, max predicted reward {max(pred_rewards_)}")
        
        print(f"Collected {valid_cnt} out of {args.rollout_epochs-start_epoch} trajectories, predicted {pred_cnt}")
        return trajs
    # train

    rollout_save_dir = make_log_dirs(args.task, args.algo_name, args.seed, vars(args), part="rollout", record_params=['sample_ratio', 'task_weight'])
    print(f"Logging diffusion rollout to {rollout_save_dir}")
    rollout_logger = Logger(rollout_save_dir, {"consoleout_backup": "stdout"})
    rollout_logger.log_hyperparameters(vars(args))

    # Get rollout_trajs
    env.reset(seed=args.seed)
    test_rollout(rollout_logger, True)
    env.reset(seed=args.seed)
    test_rollout( rollout_logger, False)
    # env.reset(seed=args.seed)
    # test_rollout_onestep()


if __name__ == "__main__":
    train()