import gym
# import d4rl # Import required to register environments, you may need to also import the submodule
import gymnasium


# Create the environment
# env = gymnasium.make('PointMaze_UMazeDense-v3')
env = gym.make("maze2d-umaze-v1")

# d4rl abides by the OpenAI gym interface
obs = env.reset()
env.step(env.action_space.sample())

# Each task is associated with a dataset
# dataset contains observations, actions, rewards, terminals, and infos
dataset = env.get_dataset()
print(dataset['observations'].shape) # An N x dim_observation Numpy array of observations