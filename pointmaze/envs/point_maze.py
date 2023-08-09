'''
Point maze environment
'''

from pointmaze.envs.base import BaseOfflineEnv
from pointmaze.utils.maze_utils import set_map_cell
from pointmaze.samplers.maze_sampler import MazeSampler
import gymnasium
import pickle

class PointMaze(BaseOfflineEnv):
    def __init__(self, data_path, horizon, maze_map, start, goal, 
                 sample_args,
                 debug=False, render=False):
        '''
        data_path: path to dataset
        maze_map: list(list), basic map of the game, only specifies 0, 1. R and G are specified by start and goal.
        start, array (2,), int, the start point of the game we want to learn
        goal, array (2,), int, the goal point of the game we want to learn
        horizon: horizon of every trajectory
        n_trajs: number of trajectories for dataset
        sample_starts: list, list of starts for sampling
        sample_goals: list, list of goals for sampling
        '''

        # Create a env_cls, a function that returns the Env class
        # if map_path is None:
        #     env_cls = lambda : gym.make('PointMaze_UMaze-v3')
        # else:
        #     with open(map_path, 'rb') as f:
        #         maze_map = pickle.load(f) # list(list)

        self.MAZE_MAP = maze_map
        target_map = set_map_cell(self.MAZE_MAP, goal, 'g')
        target_map = set_map_cell(target_map, start, 'r')

        render_mode = "human" if render else "None"
        # render_mode = None
        env_cls = lambda : gymnasium.make('PointMaze_UMazeDense-v3', 
                                    maze_map = target_map, 
                                    continuing_task = False,
                                    max_episode_steps=self.horizon,
                                    render_mode=render_mode)
        
        sampler = MazeSampler(horizon=horizon,
                              maze_map=self.MAZE_MAP,
                              target_start=start,
                              target_goal=goal,
                              debug=debug,
                              render=render)
        
        # sample_args = {'starts': sample_starts, 'goals': sample_goals, 'repeats': sample_repeats, 'randoms': sample_end_randoms}
        
        super().__init__(data_path, env_cls, horizon,
                         sampler = sampler, sample_args=sample_args)


    # def get_map(self, map_name: str):
    #     '''
    #     Get the map according to map_name
    #     '''