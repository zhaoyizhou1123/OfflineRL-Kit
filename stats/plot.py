import numpy as np
import matplotlib.pyplot as plt
from offlinerlkit.env.linearq import Linearq

# def plot():

env = Linearq(size_param=4)
num_states = env.state_space_size

# a = 0
qf0 = []
for s in range(num_states):
    qf = env._get_q(s,0)
    qf0.append(qf)
qf0 = np.array(qf)
s_arr = np.arange(num_states)

plt.plot(s_arr, qf0)
plt.save("linearq.png")



