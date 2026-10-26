import time
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.utils.visualization_utils import init_visualization, log_visualization_data, shutdown_visualization

robot_config = SO101FollowerConfig(
    port="/dev/tty.usbmodem5AB90687491",
    id="my_follower_arm",
    cameras={
        "wrist": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
        "top": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30)
    }
)

teleop_config = SO101LeaderConfig(
    port="/dev/tty.usbmodem5AB90689011",
    id="my_leader_arm",
)

init_visualization("rerun", session_name="teleoperation")  # pass "foxglove" to stream to Foxglove instead

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)
robot.connect()
teleop_device.connect()

TARGET_HZ = 30
TIME_PER_FRAME = 1.0 / TARGET_HZ

while True:
    start_time = time.perf_counter()

    observation = robot.get_observation()
    action = teleop_device.get_action()
    robot.send_action(action)
    log_visualization_data("rerun", observation=observation, action=action)

    elapsed_time = time.perf_counter() - start_time
    sleep_time = TIME_PER_FRAME - elapsed_time
    if sleep_time > 0:
        time.sleep(sleep_time)