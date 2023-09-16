import __init__
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


# import __init__
from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, EnsembleDynamicsModel, RcslModule
from offlinerlkit.dynamics import BaseDynamics, EnsembleDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn
from offlinerlkit.utils.load_dataset import qlearning_dataset, traj_rtg_datasets
from offlinerlkit.utils.config import Config
from offlinerlkit.utils.dataset import ObsActDataset
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.utils.diffusion_logger import setup_logger
from offlinerlkit.policy_trainer import RcslPolicyTrainer
from offlinerlkit.utils.trajectory import Trajectory
from offlinerlkit.utils.none_or_str import none_or_str
from offlinerlkit.policy import DiffusionBC, RcslPolicy

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
    parser.add_argument("--algo-name", type=str, default="mbrcsl")
    parser.add_argument("--task", type=str, default="halfcheetah-medium-v2", help="maze") # Self-constructed environment
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
    parser.add_argument("--behavior_epoch", type=int, default=25)
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

    parser.add_argument("--batch-size", type=int, default=256)

    return parser.parse_args()

def rollout_diffusion(args, dynamics: BaseDynamics, diffusion_policy: DiffusionBC, dynamics_dataset: ReplayBuffer, threshold, save_ckpt_freq = 10):
    '''
    diffusion behavior policy rollout

    - threshold: only keep trajs with ret > [threshold] (valid). Usually the max return in dataset
    - args.num_need_traj: number of valid trajectories needed. End rollout when get enough trajs
    - args.rollout_epochs: maximum rollout epoch. Should be large
    '''
    device = args.device
    num_need_traj = args.num_need_traj

    trajs = [] # Initialize valid rollout trajs. If there is checkpoint, first load checkpoint
    start_epoch = 0 # Default starting epoch
    if args.rollout_ckpt_path is not None:
        print(f"Will save rollout trajectories to dir {args.rollout_ckpt_path}")
        os.makedirs(args.rollout_ckpt_path, exist_ok=True)
        data_path = os.path.join(args.rollout_ckpt_path, "rollout.dat")
        if os.path.exists(data_path): # Load ckpt_data
            ckpt_dict = pickle.load(open(data_path,"rb")) # checkpoint in dict type
            trajs = ckpt_dict['trajs']
            start_epoch = ckpt_dict['epoch'] + 1
            # trajs = ckpt_dict
            print(f"Loaded checkpoint. Already have {len(trajs)} valid trajectories, start from epoch {start_epoch}.")
            if len(trajs) >= num_need_traj:
                print(f"Checkpoint trajectories are enough. Skip rollout procedure.")
                return trajs


    with torch.no_grad():
        for epoch in range(start_epoch, args.rollout_epochs):
            pred_state = dynamics_dataset.sample_init_obs().detach().cpu().numpy() # np.array
                # print(pred_state)

            # print(f"Eval forward: states {states.shape}, actions {actions.shape}")
            print(f"-----------\nEpoch {epoch}")

            pred_ret = 0

            observations_ = []
            actions_ = []
            next_observations_ = []
            rewards_ = []
            achieved_rets_ = [] # The total reward that has achieved, used to compute rtg
            frozen_noise = diffusion_policy.sample_init_noise() # noise for action sampling
            for h in range(args.horizon):
                timestep = torch.tensor(h).to(device) # scalar
                observations_.append(deepcopy(pred_state))
                achieved_rets_.append(deepcopy(pred_ret))

                # support_actions, support_probs = behavior_model(pred_state.unsqueeze(0).to(device)) # (1, n_support, action_dim), (1,n_support)
                action = diffusion_policy.select_action(pred_state, frozen_noise)
                # sample_idx = torch.multinomial(support_probs, num_samples=1).squeeze() # scalar
                # action = support_actions[0,sample_idx,:] # (action_dim)
                # action = sample_from_supports(support_actions.squeeze(0), support_probs.squeeze(0)).detach().cpu().numpy()
                # print(action)
                pred_next_state, pred_reward, _, _ = dynamics.step(pred_state, action) # (1,state_dim), (1,1)
                pred_next_state = pred_next_state.squeeze(0) # (state_dim)
                pred_reward = pred_reward.squeeze() # scalar
                # print(pred_next_state.shape, pred_reward.shape)
                # Observe next states, rewards,
                # next_state, reward, terminated, _, _ = env.step(action) # array (state_dim), scalar
                # if hasattr(env, 'get_true_observation'): # For pointmaze
                #     next_state = env.get_true_observation(next_state)
                # next_state = torch.from_numpy(next_state) # (state_dim)
                actions_.append(deepcopy(action))
                next_observations_.append(deepcopy(pred_next_state))
                rewards_.append(deepcopy(pred_reward))
                if args.debug:
                    print(f"Step {h}, action {action}")
                    print(f"Predicted reward {pred_reward}")
                    print(f"Predicted state {pred_next_state}\n")
                # Calculate return
                # ret += reward
                pred_ret += pred_reward
                
                # Update states, actions, rtgs, timesteps
                # true_state = next_state # (state_dim)
                pred_state = pred_next_state

                # Update timesteps

                # if terminated: # Already reached goal, the rest steps get reward 1, break
                #     ret += args.horizon - 1 - h
                #     pred_ret += args.horizon - 1 -h
                #     break
            print(f"Epoch {epoch}, predicted total return {pred_ret}")
            if pred_ret > threshold: # Add a new trajectory
                returns_ = [pred_ret - achieved for achieved in achieved_rets_]

                # Add new traj
                trajs.append(Trajectory(observations = observations_, 
                                        actions = actions_, 
                                        next_observations = next_observations_,
                                        rewards = rewards_, 
                                        returns = returns_, 
                                        timesteps = list(range(args.horizon)), 
                                        terminals = None, 
                                        timeouts = None))
                print(f"Get new trajectory # {len(trajs)}")
                if len(trajs) >= num_need_traj: # Get enough trajs, quit rollout
                    print(f"End rollout. Total trajs sampled: {epoch+1}")
                    break
            if (epoch + 1) %  save_ckpt_freq == 0 and args.rollout_ckpt_path is not None: # save periodically
                print(f"Epoch {epoch}, save trajs checkpoint, collected {len(trajs)} trajectories")
                pickle.dump({'epoch': epoch, 'trajs': trajs}, open(data_path, "wb"))            
    if args.rollout_ckpt_path is not None:
        pickle.dump({'epoch': epoch, 'trajs': trajs}, open(data_path, "wb"))
        
    return trajs

def train(args=get_args()):
    print(args)

    # log
    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args), part='dynamics')
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    # create env and dataset
    if args.task == 'maze': # self-constructed
        
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
        env = gym.make(args.task)
        # dataset = qlearning_dataset(env, get_rtg=True)
        dataset, init_obss_dataset, max_offline_return = traj_rtg_datasets(env)
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

    # if hasattr(env, 'seed'):
    #     env.seed(args.seed)
    # else:
    #     env


    # # create policy model
    # actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    # critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    # critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    # dist = TanhDiagGaussian(
    #     latent_dim=getattr(actor_backbone, "output_dim"),
    #     output_dim=args.action_dim,
    #     unbounded=True,
    #     conditioned_sigma=True
    # )
    # actor = ActorProb(actor_backbone, dist, args.device)
    # critic1 = Critic(critic1_backbone, args.device)
    # critic2 = Critic(critic2_backbone, args.device)
    # actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    # critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    # critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    # if args.auto_alpha:
    #     target_entropy = args.target_entropy if args.target_entropy \
    #         else -np.prod(env.action_space.shape)
    #     args.target_entropy = target_entropy
    #     log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
    #     alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
    #     alpha = (target_entropy, log_alpha, alpha_optim)
    # else:
    #     alpha = args.alpha

    # create dynamics
    # load_dynamics_model = True if args.load_dynamics_path else False

    # print(f"dynamics_hidden_dims = {args.dynamics_hidden_dims}")
    dynamics_model = EnsembleDynamicsModel(
        obs_dim=obs_dim,
        action_dim=args.action_dim,
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

    # # create policy
    # policy = COMBOPolicy(
    #     dynamics,
    #     actor,
    #     critic1,
    #     critic2,
    #     actor_optim,
    #     critic1_optim,
    #     critic2_optim,
    #     action_space=env.action_space,
    #     tau=args.tau,
    #     gamma=args.gamma,
    #     alpha=alpha,
    #     cql_weight=args.cql_weight,
    #     temperature=args.temperature,
    #     max_q_backup=args.max_q_backup,
    #     deterministic_backup=args.deterministic_backup,
    #     with_lagrange=args.with_lagrange,
    #     lagrange_threshold=args.lagrange_threshold,
    #     cql_alpha_lr=args.cql_alpha_lr,
    #     num_repeart_actions=args.num_repeat_actions,
    #     uniform_rollout=args.uniform_rollout,
    #     rho_s=args.rho_s
    # )

    # create rollout policy
    conf = Config(
        obs_dim = obs_dim,
        act_dim = args.action_dim,
        spectral_norm = False,
        num_epochs = args.behavior_epoch,
        num_diffusion_iters = args.num_diffusion_iters,
        batch_size = args.behavior_batch,
        num_workers = args.num_workers,
        path = args.load_diffusion_path,
        save_ckpt_freq = 5,
        device = args.device
    )
    state_action_dataset = ObsActDataset(dataset)
    diff_logger_conf = Config(
        algo = args.algo_name,
        env_id = args.task,
        expr_name = 'diffusion',
        seed = args.diffusion_seed,
    )
    diffusion_logger = setup_logger(diff_logger_conf)
    diffusion_policy = DiffusionBC(conf, state_action_dataset, diffusion_logger)

    # create output return-conditioned policy. This is Gaussian-style

    # rcsl_policy = RvS(
    #     observation_space=obs_space,
    #     action_space=env.action_space,
    #     hidden_size=args.rcsl_hidden_size,
    #     depth=args.rcsl_depth,
    #     learning_rate=args.rcsl_lr,
    #     batch_size=args.rcsl_batch,
    #     dropout_p=0.,
    #     reward_conditioning=True,
    #     env_name=args.task
    # )
    rcsl_backbone = MLP(input_dim=obs_dim+1, hidden_dims=args.rcsl_hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(rcsl_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )

    rcsl_module = RcslModule(rcsl_backbone, dist, args.device)
    rcsl_optim = torch.optim.Adam(rcsl_module.parameters(), lr=args.rcsl_lr)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(rcsl_optim, args.rcsl_epoch)

    rcsl_policy = RcslPolicy(
        dynamics,
        diffusion_policy,
        rcsl_module,
        rcsl_optim,
        device = args.device
    )
    

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
    # fake_buffer = ReplayBuffer(
    #     buffer_size=args.rollout_batch_size*args.rollout_length*args.model_retain_epochs,
    #     obs_shape=args.obs_shape,
    #     obs_dtype=np.float32,
    #     action_dim=args.action_dim,
    #     action_dtype=np.float32,
    #     device=args.device
    # )


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
            dynamics.train(offline_buffer.sample_all(), logger, max_epochs_since_update=5)
        
    # Finish get_rollout_policy
    def get_rollout_policy():
        '''
        Load or train rollout policy

        Return:
            rollout policy
        '''
        # if args.load_diffusion_path is None:
        print(f"Train diffusion behavior policy")
        diffusion_policy.train() # save checkpoint periodically
        diffusion_policy.save_checkpoint(epoch=None) # Save final model
        # else:
        #     print(f"Load diffusion behavior policy")
        #     diffusion_policy.load_checkpoint(final=True)
                # config for both training and logger setup
        # conf = TrainerConfig(obs_dim = obs_dim,
        #                      act_dim = action_dim,
        #                      spectral_norm = False,
        #                      num_epochs = args.behavior_epoch,
        #                      num_diffusion_iters = args.num_diffusion_iters,
        #                      batch_size = args.behavior_batch,
        #                      algo = args.behavior_type,
        #                      env_id = args.env_type,
        #                      expr_name = 'default',
        #                      seed = args.diffusion_seed,
        #                      load = args.load_diffusion,
        #                      save_ckpt_freq = 5,
        #                      device = args.device)
        
        # state_action_dataset = ObsActDataset(trajs)
        # # Configure logger
        # diffusion_logger = setup_logger(conf)
        # print(f"Setup logger")

        # from maze.algos.stitch_rcsl.training.trainer_diffusion import DiffusionBC
        # diffusion_policy = DiffusionBC(conf, state_action_dataset, diffusion_logger)

        # if not args.load_diffusion:
        #     print(f"Train diffusion behavior policy")
        #     diffusion_policy.train() # save checkpoint periodically
        #     diffusion_policy.save_checkpoint(epoch=None) # Save final model
        # else:
        #     print(f"Load diffusion behavior policy")
        #     diffusion_policy.load_checkpoint(final=True)
        pass 

    # Finish get_rollout_trajs
    def get_rollout_trajs(threshold) -> Tuple[Dict[str, np.ndarray], float]:
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

        rollout_data_all = [] # Initialize valid rollout trajs. If there is checkpoint, first load checkpoint
        num_traj_all = 0 # Initialize total number of rollout trajs
        start_epoch = 0 # Default starting epoch
        returns_all = []
        if args.rollout_ckpt_path is not None:
            print(f"Will save rollout trajectories to dir {args.rollout_ckpt_path}")
            os.makedirs(args.rollout_ckpt_path, exist_ok=True)
            data_path = os.path.join(args.rollout_ckpt_path, "rollout.dat")
            if os.path.exists(data_path): # Load ckpt_data
                ckpt_dict = pickle.load(open(data_path,"rb")) # checkpoint in dict type
                rollout_data_all = ckpt_dict['data']
                num_traj_all = ckpt_dict['num_traj']
                returns_all = ckpt_dict['return']
                start_epoch = ckpt_dict['epoch'] + 1
                # trajs = ckpt_dict
                print(f"Loaded checkpoint. Already have {num_traj_all} valid trajectories, start from epoch {start_epoch}.")
                if num_traj_all >= num_need_traj:
                    print(f"Checkpoint trajectories are enough. Skip rollout procedure.")
                    return rollout_data_all
        # Still need training, get dynamics and rollout policy
        get_dynamics()
        get_rollout_policy()

        with torch.no_grad():
            for epoch in range(start_epoch, args.rollout_epochs):
                batch_indexs = np.random.randint(0, init_obss_dataset.shape[0], size=args.rollout_batch)
                init_obss = init_obss_dataset[batch_indexs]
                rollout_data, rollout_info = rcsl_policy.rollout(init_obss, args.horizon)
                    # print(pred_state)

                # Only keep trajs with retursn > threshold
                returns = rollout_info['returns']
                print(f"Get returns: {returns}")
                assert returns.shape[0] == args.rollout_batch
                valid_trajs = np.arange(args.rollout_batch)[rollout_info['returns'] > threshold] # np.array, indexs of all valid trajs

                valid_data_idxs = [rollout_data['traj_idxs'][i] in valid_trajs for i in range(rollout_data['traj_idxs'].shape[0])]
                for k in rollout_data:
                    rollout_data[k] = rollout_data[k][valid_data_idxs]

                # Add rollout_data to rollout_data_all
                for k in rollout_data_all:
                    rollout_data_all[k] = np.concatenate([rollout_data_all[k], rollout_data[k]], axis=0)
                
                num_traj_all += len(valid_trajs)
                returns_all += returns[valid_trajs]

                # print(f"Eval forward: states {states.shape}, actions {actions.shape}")
                print(f"-----------\nEpoch {epoch}, get {len(valid_trajs)} new trajs")

                if args.rollout_ckpt_path is not None: # save periodically
                    pickle.dump({'epoch': epoch, 
                                 'data': rollout_data_all,
                                 'num_traj': num_traj_all,
                                 'return': returns_all}, open(data_path, "wb"))            

                if num_traj_all >= num_need_traj: # Get enough trajs, quit rollout
                    print(f"End rollout. Total trajs sampled: {epoch+1}")
                    break
            
        return rollout_data_all, max(returns_all)

    # train

    # Get rollout_trajs
    rollout_dataset, max_rollout_return = get_rollout_trajs(max_offline_return)


    # policy_trainer = MBPolicyTrainer(
    #     policy=policy,
    #     eval_env=env,
    #     offline_buffer=offline_buffer,
    #     fake_buffer=fake_buffer,
    #     logger=logger,
    #     rollout_setting=(args.rollout_freq, args.rollout_batch_size, args.rollout_length),
    #     epoch=args.epoch,
    #     step_per_epoch=args.step_per_epoch, # step per epoch is changed to horizon
    #     batch_size=args.batch_size,
    #     real_ratio=args.real_ratio,
    #     eval_episodes=args.eval_episodes,
    #     lr_scheduler=lr_scheduler,
    #     horizon = args.horizon,
    # )

    # Creat policy trainer
    rcsl_log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args), part='rcsl')
    # key: output file name, value: output handler type
    rcsl_output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    rcsl_logger = Logger(rcsl_log_dirs, rcsl_output_config)
    rcsl_logger.log_hyperparameters(vars(args))

    policy_trainer = RcslPolicyTrainer(
        policy = rcsl_policy,
        eval_env = env,
        offline_dataset = dataset,
        rollout_dataset = rollout_dataset,
        goal = max_rollout_return,
        logger = rcsl_logger,
        epoch = args.rcsl_epoch,
        step_per_epoch = args.rcsl_step_per_epoch,
        batch_size = args.rcsl_batch,
        lr_scheduler = lr_scheduler,
        horizon = args.horizon,
        num_workers = args.num_workers,
        # device = args.device
    )
    
    policy_trainer.train()


if __name__ == "__main__":
    train()