# Toy environment to show Bellman completeness
import gym
from gym import spaces
import numpy as np
from typing import Dict, List, Union
from copy import deepcopy

class Linearq(gym.Env):
    # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size_param: int =10):
        '''
        size_param: state space size = horizon = 3(size_param+1)
        '''
        # self.size = size  # The size of the square grid
        # self.window_size = 512  # The size of the PyGame window
        self.size_param = size_param
        self.state_space_size = 3 * (size_param + 1)
        self.horizon = self.state_space_size

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Discrete(self.state_space_size)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(2)

        # self._state = int(0)
        # self._timestep = int(0)

    def reset(self, seed=None, options=None):
        self._state = int(0)
        self._timestep = int(0)
        return np.array([self._state], dtype = np.float32)
    
    def step(self, a: Union[float, np.ndarray]):
        if type(a) is np.ndarray:
            a = float(a.squeeze())
        if a < 0:
            a_mod = 0
        elif a > 1:
            a_mod = 1
        else:
            a_mod = round(a)
        
        next_s = self._get_next_s(self._state, a_mod)
        
        q_s_a = self._get_q(self._state, a_mod)
        q_next_s_0 = self._get_q(next_s, 0)
        q_next_s_1 = self._get_q(next_s, 1)

        reward = q_s_a - max(q_next_s_0, q_next_s_1)
        self._state = next_s
        self._timestep += 1
        terminated = self._timestep >= self.horizon

        return np.array([next_s], dtype = np.float32), reward, terminated, None

    def get_dataset(self, *args, **kwargs) -> Dict:
        '''
        Contain keys:
        - observations
        - actions
        - rewards
        - next_observations
        - terminals
        - timeouts
        '''
        obss = []
        actions = []
        rs = []
        next_obss = []

        # optimal only
        optimal_repeat = self.state_space_size
        for _ in range(3*optimal_repeat):
            s = self.reset()
            for _ in range(self.horizon):
                a = self._get_optimal_a(s)
                next_s, r, _, _ = self.step(a)
                # obss.append(np.array([s], dtype=np.float32))
                obss.append(s)
                actions.append(np.array([a], dtype=np.float32))
                rs.append(r)
                # next_obss.append(np.array([next_s], dtype=np.float32))
                next_obss.append(next_s)

                s = next_s
        # Change action 
        for epoch in range(self.horizon):
            s = self.reset()
            for t in range(self.horizon):
                a = self._get_optimal_a(s)
                if t == epoch:
                    a = 1 - a # flip action
                next_s, r, _, _ = self.step(a)
                # obss.append(np.array([s], dtype=np.float32))
                obss.append(s)
                actions.append(np.array([a], dtype=np.float32))
                rs.append(r)
                # next_obss.append(np.array([next_s], dtype=np.float32))
                next_obss.append(next_s)

                s = next_s

        terminals = [False for _ in range(len(obss))]
        timeouts = [False for _ in range(len(obss))]
        for i in range(len(obss)):
            if (i+1) % self.horizon == 0:
                timeouts[i] = True

        return {
            'observations': np.array(obss),
            'actions': np.array(actions),
            'next_observations': np.array(next_obss),
            'rewards': np.array(rs),
            'terminals': terminals,
            'timeouts': timeouts
        }





    def _get_q(self, s: int, a: int):
        '''
        Compute Q function 
        '''
        if a == 0:
            return 2 * self._relu(-s + 2*self.size_param + 1)
        elif a == 1:
            return self._relu(-s + 3*self.size_param + 1.5)
        else:
            raise NotImplementedError

    def _get_optimal_a(self, s: int) -> int:
        '''
        Compute optimal action
        '''
        q_s_0 = self._get_q(s, 0)
        q_s_1 = self._get_q(s,1)
        return 0 if q_s_0 > q_s_1 else 1
        
    def _get_next_s(self, s: int, a: int) -> int:
        assert s in self.observation_space and a in self.action_space, f"{s},{a}" 

        if a == 0:
            if s <= self.size_param:
                return s + 1
            elif s >= 2 * self.size_param + 1:
                return 3 * self.size_param + 2
            elif s % 2 == 0: # [u+1, 2u] and even
                return 3 * self.size_param + 2
            else: # [u+1, 2u] and odd
                return 3 * self.size_param + 1
        else: # a==1
            if s == 3 * self.size_param + 2:
                return 3 * self.size_param + 2
            elif s >= self.size_param + 1: # [u+1, 3u+1]
                return s + 1
            elif s % 2 == 0: # [0,u] and even
                return 3 * self.size_param + 2
            else: # [0,u] and odd
                return 3 * self.size_param + 1


    def _relu(self, x) -> float:
        if x < 0:
            return 0
        else:
            return x

class Linearq_v2(gym.Env):
    # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size_param: int =10):
        '''
        size_param: state space size = horizon = 3*size_param+1
        '''
        # self.size = size  # The size of the square grid
        # self.window_size = 512  # The size of the PyGame window
        self.size_param = size_param
        self.state_space_size = 3 * size_param + 1
        self.horizon = self.state_space_size

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Discrete(self.state_space_size)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(2)

        # self._state = int(0)
        # self._timestep = int(0)

    def reset(self, seed=None, options=None):
        self._state = int(0)
        self._timestep = int(0)
        return np.array([self._state], dtype = np.float32)
    
    def step(self, a: Union[float, np.ndarray]):
        if type(a) is np.ndarray:
            a = float(a.squeeze())
        if a < 0:
            a_mod = 0
        elif a > 1:
            a_mod = 1
        else:
            a_mod = round(a)
        
        next_s = self._get_next_s(self._state, a_mod)
        
        q_s_a = self._get_q(self._state, a_mod)
        q_next_s_0 = self._get_q(next_s, 0)
        q_next_s_1 = self._get_q(next_s, 1)

        reward = q_s_a - max(q_next_s_0, q_next_s_1)
        self._state = next_s
        self._timestep += 1
        terminated = self._timestep >= self.horizon

        return np.array([next_s], dtype = np.float32), reward, terminated, None

    def get_dataset(self, *args, **kwargs) -> Dict:
        '''
        Contain keys:
        - observations
        - actions
        - rewards
        - next_observations
        - terminals
        - timeouts
        '''
        obss = []
        actions = []
        rs = []
        next_obss = []

        # optimal only
        optimal_repeat = self.state_space_size
        for _ in range(optimal_repeat):
            s = self.reset()
            for _ in range(self.horizon):
                a = self._get_optimal_a(s)
                next_s, r, _, _ = self.step(a)
                # obss.append(np.array([s], dtype=np.float32))
                obss.append(s)
                actions.append(np.array([a], dtype=np.float32))
                rs.append(r)
                # next_obss.append(np.array([next_s], dtype=np.float32))
                next_obss.append(next_s)

                s = next_s
        # Change action 
        # for epoch in range(self.horizon):
        #     s = self.reset()
        #     for t in range(self.horizon):
        #         a = self._get_optimal_a(s)
        #         if t == epoch:
        #             a = 1 - a # flip action
        #         next_s, r, _, _ = self.step(a)
        #         # obss.append(np.array([s], dtype=np.float32))
        #         obss.append(s)
        #         actions.append(np.array([a], dtype=np.float32))
        #         rs.append(r)
        #         # next_obss.append(np.array([next_s], dtype=np.float32))
        #         next_obss.append(next_s)

        #         s = next_s

        terminals = [False for _ in range(len(obss))]
        timeouts = [False for _ in range(len(obss))]
        for i in range(len(obss)):
            if (i+1) % self.horizon == 0:
                timeouts[i] = True

        return {
            'observations': np.array(obss),
            'actions': np.array(actions),
            'next_observations': np.array(next_obss),
            'rewards': np.array(rs),
            'terminals': terminals,
            'timeouts': timeouts
        }





    def _get_q(self, s: int, a: int):
        '''
        Compute Q function 
        '''
        if a == 0:
            return 2 * self._relu(-s + 2*self.size_param)
        elif a == 1:
            return self._relu(-s + 3*self.size_param)
        else:
            raise NotImplementedError

    def _get_optimal_a(self, s: int) -> int:
        '''
        Compute optimal action
        '''
        q_s_0 = self._get_q(s, 0)
        q_s_1 = self._get_q(s,1)
        return 0 if q_s_0 >= q_s_1 else 1
        
    def _get_next_s(self, s: int, a: int) -> int:
        assert s in self.observation_space and a in self.action_space, f"{s},{a}" 

        if a == 0:
            if s <= self.size_param:
                return s + 1
            elif s >= 2 * self.size_param + 1:
                return 3 * self.size_param
            elif s % 2 == 1: # [u+1, 2u] and odd
                return 3 * self.size_param - 1
            else: # [u+1, 2u] and even
                return 3 * self.size_param
        else: # a==1
            if s == 3 * self.size_param:
                return 3 * self.size_param
            elif s >= self.size_param: # [u, 3u-1]
                return s + 1
            elif s % 2 == 1: # [0,u-1] and odd
                return 3 * self.size_param
            else: # [0,u-1] and even
                return 3 * self.size_param - 1


    def _relu(self, x) -> float:
        if x < 0:
            return 0
        else:
            return x