import pandas as pd
import numpy as np
import argparse

from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('data', type=float, default = None, nargs='*')

args = parser.parse_args()
stats =args.data

def stats_rcsl(debug=False):
    data_per_exp = int(50)
    for depth in [2,4]:
        for width in ['128','256','512','1024','2048']:
            arch = (width+'-') * depth
            arch = arch[:-1]
            file = f"./backup/stitch-cql-20230828-ratio0/arch{arch}.out"
            with open(file,"r") as f:
                lines = f.readlines()
            data_lines = []
            for line in lines:
                if line[0] == 'a': # average return
                    line = line.split()
                    data = float(line[-1])
                    data_lines.append(data)

            stats = [data_lines[data_per_exp-1], data_lines[2*data_per_exp-1], data_lines[3*data_per_exp-1], data_lines[4*data_per_exp-1]]

            if debug:
                print(f"Stats: {stats}")


            print(f"{'CQL':<18}{arch:<20}{np.mean(stats):10.4f}{np.std(stats):19.4f}")

def stats_manual(stats):
    print(f"{stats}")
    print(f"{np.mean(stats):.4f} +- {np.std(stats):.4f}")

# debug=True
# print(f"Algo{'':>14}Arch{'':>19}Mean{'':>15}Std{'':>15}")
# stats_mlp(True)
# stats_rcsl_mlp()
# stats_cql()
# stats_mlp_gaussian2()
stats_manual(stats)
