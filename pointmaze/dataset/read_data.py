import os
import sys

# print(f"test: {os.getcwd()}")
sys.path.append(os.getcwd()+"/../..")
sys.path.append(os.getcwd()+"/..")

import pickle
# from maze.utils.trajectory import show_trajectory
import numpy as np


def read_offline_data():
    data_file = "maze2_smds_acc.dat"
    with open(data_file, "rb") as f:
        trajs, horizon, map, start, goal = pickle.load(f)
    print(f"Horizon: {horizon}")
    print(f"Map: {map}")
    print(f"Start: {start}; Goal: {goal}")
    print(f"Trajectory number: {len(trajs)}")
    # for idx, traj in enumerate(trajs):
    #     print(f'--------------------\nTrajectory {idx}')
    #     show_trajectory(traj, timesteps = [0])
    rets = [traj.returns[0] for traj in trajs]
    rets.sort(reverse=True)
    top_num=100
    top_rets = rets[0:top_num]
    print(f"Top {top_num} rets: max {top_rets[0]}, min {top_rets[-1]}, avg {sum(top_rets) / len(top_rets)}")

    num_show_trajs = 10
    num_total_trajs = len(trajs)

    for _ in range(num_show_trajs):
        traj_idx = np.random.randint(0,num_total_trajs)
        traj = trajs[traj_idx]
        obss = traj.observations
        acts = traj.actions
        print(f"Start action: {acts[0]}, End: {obss[-1][0:2]}. Return: {traj.returns[0]}")

def read_rollout_data():
    ckpt_dict = pickle.load(open("./checkpoint/maze2_smds_accdyn/rollout.dat", "rb"))
    epoch = ckpt_dict['epoch']
    trajs = ckpt_dict['trajs']
    print(f"Epoch {epoch}")
    print(f"Collected {len(trajs)} trajectories")
    print(f"{trajs[-2].returns[0]}")

# read_rollout_data()
read_offline_data()