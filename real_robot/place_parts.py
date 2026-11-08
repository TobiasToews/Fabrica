'''
Lay out the parts without a fixture or template: the robot shows where each part goes.
For every part the robot picks (in reverse pick order, as stored in place_poses.json) it moves above the pickup spot, lowers the open gripper to the
pickup pose and waits; put the part flat on the board between the fingers in its planned resting pose, press Enter, and the
robot lifts off and moves on. Afterwards run real_robot/run.py without --place-at-pickup.
Needs <log-dir>/fixture/place_poses.json from planning/run_place_poses.py (fabrica env).
'''
import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import argparse
import json

from real_robot.robot_interface import RobotInterface


def place_parts(log_dir, parts=None, robot_num=1):
    with open(os.path.join(log_dir, 'fixture', 'place_poses.json'), 'r') as fp:
        info = json.load(fp)
    if not info['checked']:
        print('[place_parts] WARNING: planning/run_place_poses.py found colliding moves in this file')
    poses = [p for p in info['poses'] if parts is None or p['part'] in parts]
    print(f'[place_parts] parts in order: {[p["part"] for p in poses]} (part held by the human is not shown)')

    robot = RobotInterface(robot_num=robot_num)
    robot.reset_arm(home_gripper=True)
    for pose in poses:
        part = pose['part']
        input(f'[place_parts] Next: part {part}. Keep the board clear under the gripper, press Enter to move there.')
        robot.goto_joints(pose['q_above'], duration=5)
        robot.goto_gripper(pose['open_ratio'])
        robot.goto_joints(pose['q_pick'], duration=3)
        input(f'[place_parts] Put part {part} flat on the board between the fingers (resting pose of the plan), then press Enter.')
        robot.goto_gripper(pose['open_ratio_retract'])
        robot.goto_joints(pose['q_above'], duration=3)
    robot.goto_joints(info['rest_q'], duration=5)
    print('[place_parts] all parts placed; now run real_robot/run.py (without --place-at-pickup)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, required=True, help='e.g. logs/exp_human_normal_flat/human_normal')
    parser.add_argument('--parts', type=str, nargs='+', default=None, help='only these parts (e.g. to re-place one that moved)')
    args = parser.parse_args()
    place_parts(args.log_dir, args.parts)
