import pandas as pd
import os

# file_path = "../backup/linearq_newenv/ql_0.75expert_fromdttoy/timestamp_23-0924-195432&param16-arch4/record/policy_training_progress.csv"

def read_ql(choose_r_line = 300):
    root = '../backup/linearq_newenv/ql_0.75expert_fromdttoy'
    dirs = os.listdir(root)
    for dir in dirs:
        # abs_dir = os.path.abspath(dir)
        with open(os.path.join(root, dir, 'record', 'policy_training_progress.csv'), 'r') as f:
            data = pd.read_csv(f)
            # print(data['eval/episode_reward'])
        rewards = data['eval/episode_reward']
        rewards = list(rewards)
        arg = dir.split('&')[1] # param*-arch*
        param = arg.split('-')[0] # param*
        param = int(param[5:])
        gap = 4*param+2 - rewards[choose_r_line]
        if gap != 1:
            print(dir)
            print(param)
            print(f"Gap: {gap}")

def read_rcsl(choose_r_line = -1):
    root = '../backup/linearq_newenv/rcsl_0.75expert'
    dirs = os.listdir(root)
    for dir in dirs:
        # abs_dir = os.path.abspath(dir)
        with open(os.path.join(root, dir, 'record', 'policy_training_progress.csv'), 'r') as f:
            data = pd.read_csv(f)
            # print(data['eval/episode_reward'])
        rewards = data['eval/episode_reward']
        rewards = list(rewards)
        arg = dir.split('&')[1] # [param]-[arch]
        param = arg.split('-')[0] # [param]
        param = int(param)
        gap = 4*param+2 - rewards[choose_r_line]
        if gap != 0:
            print(dir)
            print(param)
            print(f"Gap: {gap}")

read_rcsl()