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

# rewards
r0 = []
for s in range(num_states):
    qf = env._get_q(s,0)
    next_s = env._get_next_s(s, 0)
    q_next_s_0 = env._get_q(next_s, 0)
    q_next_s_1 = env._get_q(next_s, 1)
    reward = qf - max(q_next_s_0, q_next_s_1)
    r0.append(reward)

r1 = []
for s in range(num_states):
    qf = env._get_q(s,1)
    next_s = env._get_next_s(s, 1)
    q_next_s_0 = env._get_q(next_s, 0)
    q_next_s_1 = env._get_q(next_s, 1)
    reward = qf - max(q_next_s_0, q_next_s_1)
    r1.append(reward)

plt.subplot(2,1,1)
plt.plot(s_arr, qf0, color='r', marker='o', label="$Q(s,a=0)$")
plt.plot(s_arr, qf1, color='b', marker='s', linestyle='dashed', label="$Q(s,a=1)$")
plt.ylabel("$Q$-function", fontsize=16)
plt.legend(fontsize=16)

plt.subplot(2,1,2)
plt.plot(s_arr, r0, color='r', marker='o', label="$r(s,a=0)$")
plt.plot(s_arr, r1, color='b', marker='s', linestyle='dashed', label="$r(s,a=1)$")
plt.ylabel("reward", fontsize=16)
plt.xlabel("state", fontsize=18)
plt.yticks([0,2,4,6,8,10,12])
plt.legend(fontsize=16)
plt.savefig("linearq_full.png")



