import numpy as np
import matplotlib.pyplot as plt
from offlinerlkit.env.linearq import Linearq

# def plot():

env = Linearq(size_param=4)
num_states = env.state_space_size

s_arr = np.arange(num_states)

# a = 0
qf0 = []
for s in range(num_states):
    qf = env._get_q(s,0)
    qf0.append(qf)
qf0 = np.array(qf0)

qf1 = []
for s in range(num_states):
    qf = env._get_q(s,1)
    qf1.append(qf)
qf1 = np.array(qf1)

plt.plot(s_arr, qf0, color='r', marker='o', label="$Q(s,a=0)$")
plt.plot(s_arr, qf1, color='b', marker='s', label="$Q(s,a=1)$")
plt.xlabel("state")
plt.ylabel("$Q$-function")
plt.legend()
plt.savefig("linearq.png")



