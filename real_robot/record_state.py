'''
Record what the real robot reports while real_robot/run.py runs (second terminal, isaacgym env + ROS sourced):
joint angles q, commanded q_d, velocities, measured and external joint torques, the TCP pose O_T_EE (base frame, m), gripper width, time. Ctrl+C saves the file.

python real_robot/record_state.py --out recordings/run.npz
Compare with the plan: python real_robot/compare_recording.py --rec recordings/run.npz --fn <log-dir>/motion.pkl (fabrica env)
'''
import os
import argparse
import threading
import numpy as np
import rospy
from franka_interface_msgs.msg import RobotState
from sensor_msgs.msg import JointState
from franka_interface_msgs.msg import SensorDataGroup
from frankapy.proto import JointPositionSensorMessage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--robot-num', type=int, default=1)
    parser.add_argument('--hz', type=float, default=50.0, help='samples per second')
    args = parser.parse_args()

    rospy.init_node('record_state', anonymous=True)
    lock = threading.Lock()
    latest = {'state': None, 'width': np.nan}

    def on_state(msg):
        with lock: latest['state'] = msg
    def on_gripper(msg):
        with lock: latest['width'] = msg.position[0] + msg.position[1]
    rospy.Subscriber(f'/robot_state_publisher_node_{args.robot_num}/robot_state', RobotState, on_state, queue_size=1)
    rospy.Subscriber(f'/franka_gripper_{args.robot_num}/joint_states', JointState, on_gripper, queue_size=1)

    # the joint goals real_robot/run.py streams to franka-interface (robot 1: franka_ros_interface/sensor)
    commands = {'cmd_t': [], 'cmd_id': [], 'cmd_q': []}
    def on_command(msg):
        if not msg.has_trajectory_generator_sensor_data: return
        proto = JointPositionSensorMessage()
        try:
            proto.ParseFromString(bytes(msg.trajectoryGeneratorSensorData.sensorData))
        except Exception:
            return
        if len(proto.joints) != 7: return
        with lock:
            commands['cmd_t'].append(rospy.get_time()); commands['cmd_id'].append(proto.id); commands['cmd_q'].append(list(proto.joints))
    rospy.Subscriber('franka_ros_interface/sensor' if args.robot_num == 1 else 'franka_ros_interface/sensor_2', SensorDataGroup, on_command, queue_size=1000)

    data = {'t': [], 'q': [], 'q_d': [], 'dq': [], 'tau_J': [], 'tau_ext': [], 'O_T_EE': [], 'width': [], 'robot_mode': []}
    rate = rospy.Rate(args.hz)
    print(f'[record_state] recording at {args.hz} Hz, Ctrl+C to stop and save {args.out}')
    try:
        while not rospy.is_shutdown():
            with lock:
                state, width = latest['state'], latest['width']
            if state is not None:
                data['t'].append(rospy.get_time())
                data['q'].append(list(state.q))
                data['q_d'].append(list(state.q_d))
                data['dq'].append(list(state.dq))
                data['tau_J'].append(list(state.tau_J)) # measured joint torques
                data['tau_ext'].append(list(state.tau_ext_hat_filtered)) # estimated external torques (contact, cables, ...)
                data['O_T_EE'].append(np.array(state.O_T_EE).reshape(4, 4).T) # column-major in the message
                data['width'].append(width)
                data['robot_mode'].append(state.robot_mode)
            rate.sleep()
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        pass
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with lock:
        np.savez(args.out, **{k: np.array(v) for k, v in {**data, **commands}.items()})
    print(f'[record_state] saved {len(data["t"])} samples and {len(commands["cmd_t"])} streamed joint goals to {args.out}')


if __name__ == '__main__':
    main()
