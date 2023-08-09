import __init__

from pointmaze.envs.point_maze import PointMaze
# from offlinerlkit.utils.trajectory import show_trajectory
from pointmaze.utils.maze_utils import set_map_cell
import gymnasium
# import gym
# import d4rl

import argparse
import numpy as np
import json
import os
import pickle

def create_env_dataset(args):
    '''
    Create env and dataset (if not created)
    '''
    # if not hasattr(args, 'maze_config_file'):
    #     print(f"Current file: {__file__}")
    #     args.maze_config_file = './dataset/maze.dat'
    maze_config = json.load(open(args.maze_config_file, 'r'))
    maze = maze_config["maze"]
    map = maze['map']  

    start = maze['start']
    goal = maze['goal']

    # alt_start = [3,1]
    # mid_point = [3,4]

    # alt_goal = [1,7]

    # n_trajs = int(1e6) // args.horizon
    # # n_trajs = 6

    # repeat = n_trajs // 4

    # sample_starts = [np.array(start), np.array(alt_start), np.array(start), np.array(alt_start)]
    # sample_goals = [np.array(alt_goal), np.array(goal), np.array(mid_point), np.array(goal)]
    # sample_repeats = [repeat, repeat, repeat, n_trajs - 3 * repeat]
    # sample_end_randoms = [False, False, False, True]
    sample_args = maze_config["sample_args"]

    # for i in range(repeat): # a suboptimal path
    #     sample_starts.append(np.array(start))
    #     sample_goals.append(np.array(alt_goal))
    # for i in range(repeat, repeat * 2): # teach path mid->goal
    #     sample_starts.append(np.array(alt_start))
    #     sample_goals.append(np.array(goal))    
    # for i in range(repeat * 2, n_trajs): # teach path start->mid
    #     sample_starts.append(np.array(start))
    #     sample_goals.append(np.array(mid_point))    

    # print(sample_starts)
    # print(sample_goals)

    print(f"Create point maze")
    point_maze = PointMaze(data_path = args.data_file, 
                        horizon = args.horizon,
                        maze_map = map,
                        start = np.array(start),
                        goal = np.array(goal),
                        sample_args = sample_args,
                        debug=args.debug,
                        render=args.render)   
    
    return point_maze

def create_env(args):
    '''
    Create env(if not created)
    '''
    maze_config = json.load(open(args.maze_config_file, 'r'))
    maze = maze_config["maze"]
    map = maze['map']  

    start = maze['start']
    goal = maze['goal']

    target_map = set_map_cell(map, goal, 'g')
    target_map = set_map_cell(target_map, start, 'r')

    # print(sample_starts)
    # print(sample_goals)
    render_mode = "human" if args.render else "None"
    # render_mode = None

    env = gymnasium.make('PointMaze_UMazeDense-v3', 
             maze_map = target_map, 
             continuing_task = False,
             max_episode_steps=args.horizon,
             render_mode=render_mode)
    # env = gym.make('maze2d-umaze-dense-v1')
    # env = gym.make('PointMaze_UMazeDense-v3', 
    #          maze_map = target_map, 
    #          continuing_task = False,
    #          max_episode_steps=args.horizon)
    
    return env

def load_dataset(data_path):
    '''
    Try to load dataset from daa_path. If fails, return none
    '''
    if data_path is not None and os.path.exists(data_path):
        # print('Dataset file found. Loading existing trajectories.')
        with open(data_path, 'rb') as file:
            dataset = pickle.load(file) # Dataset may not be trajs, might contain other infos
        return dataset
    else:
        return None

# print("Trajectory 0")
# show_trajectory(trajs[0])

# print("Trajectory -1")
# show_trajectory(trajs[-1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default=None,help='./dataset/maze2_smdacc_.dat')
    parser.add_argument('--horizon', type=int, default=300)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--maze_config_file', type=str, default='./config/maze2_simple_moredata.json')

    args = parser.parse_args()
    print(args)

    create_env(args)

