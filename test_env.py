import gym
import numpy as np
import roboverse
import random
from offlinerlkit.utils.pickplace_utils import SimpleObsWrapper

env = roboverse.make('Widow250PickTray-v0')
env = SimpleObsWrapper(env)

# random.seed(0)
# np.random.seed(0)
obs = env.reset(seed=0)
print(obs)
obs = env.reset()
print(obs)
obs = env.reset(seed=0)
print(obs)