"""
Who sent the commands to the robot? Lists, for a rosbag of the command topics, every sending ROS node per topic with its number
of messages and the time span, and warns when a topic has more than one sender at the same time.

Record while real_robot/run.py runs (second terminal, ROS sourced), stop with Ctrl+C:
  rosbag record -O recordings/senders.bag /franka_ros_interface/sensor \
      /execute_skill_action_server_node_1/execute_skill/goal /execute_skill_action_server_node_1/execute_skill/cancel \
      /franka_gripper_1/grasp/goal /franka_gripper_1/move/goal /franka_gripper_1/homing/goal /franka_gripper_1/stop/goal
Then (system Python with ROS sourced; the conda envs lack rosbag's dependencies):
  /usr/bin/python3 real_robot/bag_senders.py recordings/senders.bag
"""

import sys
import rosbag


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    senders = {}  # (topic, node) -> [count, first time, last time]
    with rosbag.Bag(sys.argv[1]) as bag:
        t0 = bag.get_start_time()
        for topic, _, t, header in bag.read_messages(
            raw=True, return_connection_header=True
        ):
            node = header.get('callerid', b'?')
            node = node.decode() if isinstance(node, bytes) else node
            entry = senders.setdefault(
                (topic, node), [0, t.to_sec(), t.to_sec()]
            )
            entry[0] += 1
            entry[2] = t.to_sec()
    if not senders:
        sys.exit('no messages in the bag')

    print(
        f'{"topic":62} {"sending node":40} {"messages":>8}  first .. last [s after start]'
    )
    for (topic, node), (count, first, last) in sorted(senders.items()):
        print(
            f'{topic:62} {node:40} {count:8d}  {first - t0:7.1f} .. {last - t0:7.1f}'
        )

    clean = True
    for topic in sorted({topic for topic, _ in senders}):
        spans = [
            (node, v[1], v[2])
            for (tp, node), v in senders.items()
            if tp == topic
        ]
        for i, (node_a, first_a, last_a) in enumerate(spans):
            for node_b, first_b, last_b in spans[i + 1 :]:
                if first_a <= last_b and first_b <= last_a:
                    clean = False
                    print(
                        f'WARNING: {topic} has two senders in overlapping time spans: {node_a} and {node_b}'
                    )
    if clean:
        print('No topic had two senders at the same time.')


if __name__ == '__main__':
    main()
