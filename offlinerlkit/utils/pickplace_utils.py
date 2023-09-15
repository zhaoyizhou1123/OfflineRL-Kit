# Helper functions for pick place environment

from typing import Dict, Optional, Union
import numpy as np
import gym
from gym.spaces import Box
import os

def get_true_obs(obs: Dict[str, np.ndarray]) -> np.ndarray :
    '''
    Convert pick place observation to observation needed for training.
    our obs = object_position + object_orientation + state
    Question: Do we need to normalize state?
    '''
    obj_pos = obs['object_position']
    obj_ori = obs['object_orientation']
    state = obs['state']
    return np.concatenate([obj_pos, obj_ori, state], axis = 0)

class SimpleObsWrapper(gym.ObservationWrapper):
    '''
    Wrap pick place environment to return desired obs
    '''
    def __init__(self, env):
        super().__init__(env)
        # Get observation space
        tmp_obs = env.reset()

        tmp_true_obs = get_true_obs(tmp_obs)
        low = env.observation_space['state'].low[0]
        high = env.observation_space['state'].high[0]
        self.observation_space = Box(shape = tmp_true_obs.shape, low = low, high = high)

    def observation(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        return get_true_obs(observation)

    def reset(self, seed = None):
        if seed is not None:
            self.env.seed(seed)
        return self.observation(self.env.reset())

def get_pickplace_dataset(data_dir: str) -> Dict:
    '''
    Concatenate prior_data and task_data
    '''
    with open(os.path.join(data_dir, 'pickplace_prior.npy'), "rb") as fp:
        prior_data = np.load(fp, allow_pickle=True)
    with open(os.path.join(data_dir, 'pickplace_task.npy'), "rb") as ft:
        task_data = np.load(ft, allow_pickle=True)
    full_data = np.concatenate([prior_data, task_data], axis=0) # list of dict
    keys = ['observations', 'actions', 'rewards', 'next_observations', 'terminals']
    dict_data  = {}
    for key in keys:
        values = []
        for d in full_data: # trajectory, dict of lists
            value_list = d[key] # list
            if key == 'observations' or key == 'next_observations':
                # print(get_true_obs(value_list[0]))
                values += [get_true_obs(obs) for obs in value_list] # element is list
            else:
                values += value_list # element is list
        values = np.asarray(values)
        dict_data[key] = values
    # dict_data = flatten_dict(full_data, keys)
    rtgs = np.zeros_like(dict_data['rewards']) # no return
    dict_data['rtgs'] = rtgs
    return dict_data

# From https://github.com/avisingh599/cog/blob/master/rlkit/data_management/obs_dict_replay_buffer.py
def flatten_n(xs):
    xs = np.asarray(xs)
    return xs.reshape((xs.shape[0], -1))


def flatten_dict(dicts, keys):
    """
    Turns list of dicts into dict of np arrays
    """
    return {
        key: flatten_n([d[key] for d in dicts])
        for key in keys
    }

if __name__ == '__main__':
    import roboverse
    env = roboverse.make('Widow250PickTray-v0')
    env = SimpleObsWrapper(env)
    dict_data = get_pickplace_dataset("./dataset")
    for k,v in dict_data.items():
        print(k)
        print(v.shape)
    # dic = {1: np.array([1,2])}
    # dicts = [dic, dic, dic]
    # print(flatten_dict(dicts,[1]))