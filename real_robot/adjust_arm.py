import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import argparse
import pickle as pkl
import time
import numpy as np

from real_robot.robot_interface import RobotInterface
from autolab_core import RigidTransform
from frankapy import FrankaArm


def adjust_arm(robot, direction, amount=0.002, duration=1):

    # +X: up, -X: down, +Y: right, -Y: left
    direction_map = {'up': [amount, 0, 0],
                     'down': [-amount, 0, 0],
                     'right': [0, amount, 0],
                     'left': [0, -amount, 0]}
    assert direction in direction_map, f"Invalid direction: {direction}"
    translation = direction_map[direction]

    curr_pose = robot.fa.get_pose()
    translation_in_base = curr_pose.rotation @ np.array(translation)
    new_translation = curr_pose.translation + translation_in_base
    target_pose = RigidTransform(translation=new_translation, rotation=curr_pose.rotation, from_frame='franka_tool', to_frame='world')

    fa = robot.fa
    fa: FrankaArm # type annotation
    fa.goto_pose(tool_pose=target_pose, duration=duration, use_impedance=False)
    robot.stop_skill()
    print(f"Target pose: {target_pose}")
    print(f"Current pose: {robot.fa.get_pose()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset', default=False, action='store_true')
    parser.add_argument('--direction', type=str, default='up', choices=['up', 'down', 'left', 'right'])
    parser.add_argument('--amount', type=float, default=0.002)
    parser.add_argument('--duration', type=float, default=1)
    args = parser.parse_args()

    robot = RobotInterface(robot_num=2)
    if args.reset:
        robot.reset_arm(home_gripper=True)

    adjust_arm(robot, direction=args.direction, amount=args.amount, duration=args.duration)