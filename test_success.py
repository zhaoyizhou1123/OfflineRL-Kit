import numpy as np
import argparse
from typing import List

tray_xy = np.array([0.5,0.25])
tray_z_threshold = -0.32
tray_dist_threshold = 0.04

def test_success(obj: List[float] ):
    x,y,z = obj[0], obj[1], obj[2]
    height_success = z < tray_z_threshold
    obj_xy = np.array([x,y])
    distance = np.linalg.norm(obj_xy - tray_xy)
    dist_success = distance < tray_dist_threshold
    print(f"distance: {distance}")
    print(f"Height success: {height_success}, dist_success: {dist_success}, success: {height_success and dist_success}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("obj", type=float, nargs=3)

    args = parser.parse_args()
    obj = args.obj
    print(f"Object pos: {obj}")
    test_success(obj)