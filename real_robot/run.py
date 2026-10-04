import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import argparse
import pickle as pkl
import time
import numpy as np
import json

from real_robot.robot_interface import RobotInterface, load_config, load_rl_policy


prompt = '''
You are assisting a robot in aligning a grasped part for insertion using visual feedback from a camera mounted on the robot's wrist.

Task:
- The part is grasped by the robot and can move in four directions: ["up", "down", "left", "right"], each by 2 mm in the camera frame.
- The goal is to move the part to align it precisely with the hole for insertion.

Instructions:
- Carefully observe the video frames. Focus only on the position of the part relative to the hole.
- Determine the single best action to move the part to align with the hole.
- Focus only on spatial cues: Is the part too far left, right, above, or below the hole?

Response format:
{
"action": "right",
"reason": "The part is too far left relative to the hole and needs to move right to align."
}

Only output the single best action based on spatial cues. If the part is already aligned, output "hold".
What is the best action to move the part to align with the hole?
'''


def human_hold_step(d, i):
    """Emulate the hold arm in single-arm mode: tell the human what the hold arm would do and wait."""
    _, body_type, _, active_part, task = d[i]
    if body_type != 'arm' or task == 'init':
        return # gripper and init steps are covered by the arm instructions
    if task == 'transport' and active_part is not None:
        msg = f'Place part {active_part} from the fixture at the assembly spot in its final pose and hold it steady.'
    elif task == 'switch':
        msg = 'Re-grip the assembly where the hold gripper grasps next and keep holding it steady.'
    elif any(s[0] == 'hold' and s[3] is not None for s in d[:i]): # transport to rest after holding
        msg = 'Release the assembly and take your hand out of the workspace.'
    else:
        return # transport to pickup, nothing to do before placing the part
    input(f'[Human hold arm] {msg} Press Enter when done.')


def run_traj(fn, dt=0.1, reset=True, start_from=0, checkpoint_path=None, residual=False,
             auto=False, force=60, vlm=False, video_dir=None, single_arm=False):

    if vlm:
        from real_robot.vision.camera import Camera
        from real_robot.vision.gemini import GeminiVLM
        from real_robot.adjust_arm import adjust_arm
        camera = Camera()
        gemini = GeminiVLM(json_output=True)
        assert os.path.exists(video_dir)

    d = pkl.load(open(fn, 'rb')) # list of ['move/hold', 'arm/gripper', path, active_part, task]
    if single_arm:
        robot_left = None # hold arm is emulated by a human
        robot_right = RobotInterface(robot_num=1, residual=residual)
    else:
        robot_left = RobotInterface(robot_num=1)
        robot_right = RobotInterface(robot_num=2, residual=residual)
    robots = [robot for robot in [robot_left, robot_right] if robot is not None]
    if reset:
        for robot in robots:
            robot.reset_arm(home_gripper=True)

    if checkpoint_path is None:
        policy = None
    else:
        policy = load_rl_policy(checkpoint_path, device='cuda')

    # whether need to re-initialize a trajectory
    done_init_left = start_from == 0 # no need to re-initialize if start from 0
    done_init_right = start_from == 0
    
    global_pos_shift = np.zeros(3)

    for i in range(start_from, len(d)):
        print(f"========= Step {i+1}/{len(d)}, {('human' if single_arm else 'left') if d[i][0] == 'hold' else 'right'} arm, {d[i][1]} =========")
        new_dt = dt

        # NOTE: This kind of looks sus, maybe needs revisiting while testing
        if single_arm and d[i][0] == "hold":
            human_hold_step(d, i)
            continue

        if d[i][1] == "arm":
            if not auto:
                input("Ready?")
            
        # select arm
        if d[i][0] == "hold":
            robot = robot_left
        else:
            robot = robot_right
        # stop the current skill
        robot.stop_skill()
        # select motion
        if d[i][1] == "gripper":
            if d[i][4] == "close":
                grasp = True
                dist = d[i][2] - 1/8.
            else:
                grasp = False
                dist = d[i][2]
                time.sleep(1)
            robot.goto_gripper(dist, grasp=grasp, grasp_force=force)
        else:
            repeat = True
            repeat_idx = 0
            last_action = "init"
            past_actions = []
            while repeat:
                repeat_idx += 1
                record_path = os.path.join(video_dir, f"step_{i+1}", f"trial_{repeat_idx}_{last_action}.mp4")

                if not done_init_left and d[i][0] == "hold":
                    # go to the first joint pose in a trajectory
                    # in order to support trajectory resume from arbitrary step
                    robot.goto_joints(d[i][2][0], duration=5, ignore_virtual_walls=True)
                    done_init_left = True
                if not done_init_right and d[i][0] != "hold":
                    robot.goto_joints(d[i][2][0], duration=5, ignore_virtual_walls=True)
                    done_init_right = True
                
                if vlm:
                    for past_action in past_actions:
                        adjust_arm(robot, direction=past_action, amount=0.002, duration=1)
                        print(f"[Robot] Action completed.")

                if d[i][4] == 'assembly' and d[i][0] == 'move':
                    pos_diff = np.zeros(3)
                    
                    # start rl server
                    real_config = load_config()
                    robot.init_rl_publisher(real_config)
                    if vlm:
                        os.makedirs(os.path.join(video_dir, f"step_{i+1}"), exist_ok=True)

                    while True:
                        if policy is not None:
                            policy_type = 'r'
                        else:
                            policy_type = 'o'
                        if policy_type == 'o':
                            policy_success = robot.execute_openloop_policy(real_config=real_config, goal_joints=d[i][2][-1])
                            break
                        elif policy_type == 'r':
                            if vlm:
                                camera.start_video_recording(record_path)
                            policy_success = robot.execute_rl_policy(policy=policy, real_config=real_config, goal_joints=d[i][2][-1], global_pos_shift=global_pos_shift + pos_diff)
                            if vlm:
                                camera.stop_video_recording()
                            break
                        else:
                            print('Incorrect input, try again')

                    if auto:
                        if policy_success:
                            repeat = False
                        else:
                            done_init_right = False
                    else:
                        while True:
                            repeat_choice = input('Repeat? (y/n)')
                            if repeat_choice == 'n':
                                repeat = False
                                break
                            elif repeat_choice == 'y':
                                done_init_right = False

                                if vlm:
                                    vlm_choice = input('Use VLM? (y/n)')
                                    if vlm_choice == 'y':
                                        keyframes = gemini.get_keyframes_from_file(record_path, num_frames=10)
                                        query_list = [prompt] + keyframes
                                        response = gemini.generate_content(query_list, temperature=0.1)
                                        response = json.loads(response)
                                        action = response['action'].strip().lower()
                                        print(f"[Gemini] Response generated: {response}")
                                        last_action = action
                                        past_actions.append(action)

                                        response_path = os.path.join(video_dir, f"step_{i+1}", f"trial_{repeat_idx}_{last_action}.json")
                                        with open(response_path, 'w') as f:
                                            json.dump(response, f, indent=4)

                                break
                            else:
                                print('Incorrect input, try again')

                else:
                    # start trajectory server and go to the first joint pose
                    steps = len(d[i][2])
                    T = new_dt * (steps + 3)
                    if i > 0 and d[i-1][1] == 'gripper' and d[i-1][4] == 'open':
                        robot.goto_joints(d[i][2][0], duration=1, ignore_virtual_walls=True)
                    robot.init_joint_pose_publisher(T, d[i][2][0])

                    # start publishing trajectory
                    for j in range(1, steps):
                        joint_pose = d[i][2][j]
                        robot.publish_traj(joint_pose, new_dt)

                    repeat = False
                    
            desired_joint = d[i][2][-1]
            robot.stop_skill()
            robot.goto_joints(desired_joint, duration=1, ignore_virtual_walls=True)
            gt_joint = robot.fa.get_joints()

            if d[i][4] == 'assembly' and d[i][0] == 'move':
                pose_desired = robot.calculate_tool_pose(desired_joint)
                pose_gt = robot.calculate_tool_pose(gt_joint)
                global_pos_shift = pose_gt.translation - pose_desired.translation
                global_pos_shift[2] = 0 # only consider horizontal shift
                print(f"global pos shift: {global_pos_shift}")

    for robot in robots:
        robot.fa.goto_gripper(0.08, grasp=False)
        robot.fa.home_gripper()
    for robot in robots:
        robot.stop_skill()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn', type=str, default="motion.pkl")
    parser.add_argument('--dt', type=float, default=0.03333)
    parser.add_argument('--no-reset', default=False, action='store_true')
    parser.add_argument('--start-from', type=int, default=0)
    parser.add_argument('--checkpoint-path', type=str, default=None)
    parser.add_argument('--residual', default=False, action='store_true')
    parser.add_argument('--auto', default=False, action='store_true') # for automatic run without input
    parser.add_argument('--force', type=int, default=60, help='grasping force')
    parser.add_argument('--vlm', default=False, action='store_true')
    parser.add_argument('--video-dir', type=str, default=None)
    parser.add_argument('--single-arm', default=False, action='store_true', help='only the move arm is a real robot (robot 1), the hold arm is emulated by a human')
    args = parser.parse_args()

    if args.no_reset:
        if args.start_from == 0:
            print('Warning: starting from 0 without reset! Corrected!') # start from 0 has to reset
            args.no_reset = False
        else:
            if input(f'Warning: starting from {args.start_from} without reset! Continue? (y/n)') == 'n':
                exit()

    run_traj(fn=args.fn, dt=args.dt, reset=not args.no_reset, start_from=args.start_from, checkpoint_path=args.checkpoint_path,
        residual=args.residual, auto=args.auto, force=args.force, vlm=args.vlm, video_dir=args.video_dir, single_arm=args.single_arm)
