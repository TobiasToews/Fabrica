'''
Show the initial setup of a planned assembly as a still scene: every part at its pickup pose on the board
(lying flat for plans from planning/run_pickup_gen.py, in the fixture otherwise), the fixture if there is one,
and both arms at rest. Works for any <log-dir> that has fixture/pickup.json (run_fixture_gen.py or run_pickup_gen.py).

python rendering/show_initial_setup.py --assembly-dir assets/fabrica/human_normal --log-dir logs/exp_human_normal_flat/human_normal

Viewer: drag to rotate, scroll to zoom, V prints the camera pose (pass it back with --camera-pos/--camera-lookat), close the window to quit.
'''
import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import json
import pickle
import numpy as np
import redmax_py as redmax
import trimesh

from assets.load import load_part_ids
from assets.transform import get_transform_matrix
from planning.robot.geometry import get_gripper_finger_states
from planning.robot.util_arm import get_arm_chain
from planning.robot.sim_string import get_arm_joints
from planning.robot.workcell import get_board_dx
from planning.run_motion_plan import OPEN_RATIO_REST
from rendering.render_motion_plan import create_assembly_dualarm_fixture_xml, get_camera_option

COLOR_NAMES = ['red', 'yellow', 'blue', 'purple', 'green'] # planning/robot/sim_string.py get_color(part_id)


def print_legend(assembly_dir, fixture_dir):
    with open(os.path.join(fixture_dir, 'pickup.json'), 'r') as fp:
        pickup = json.load(fp)
    flat_path = os.path.join(fixture_dir, 'flat_pickup.json')
    flat = json.load(open(flat_path, 'r')) if os.path.exists(flat_path) else {}
    dx = get_board_dx()
    print(f'[show_initial_setup] {"in the fixture" if os.path.exists(os.path.join(fixture_dir, "fixture.obj")) else "lying on the board (no fixture)"}; '
          f'board holes every {dx} cm, at (k + 0.5) * {dx} from the board center; footprints in board coordinates (x towards the move arm)')
    for part_id in load_part_ids(assembly_dir):
        mesh = trimesh.load_mesh(os.path.join(assembly_dir, f'{part_id}.obj'))
        mesh.apply_transform(get_transform_matrix(pickup[part_id]))
        (x0, y0, _), (x1, y1, z1) = mesh.bounds
        line = f'  part {part_id:>2} ({COLOR_NAMES[int(part_id) % len(COLOR_NAMES)]:>6}): x {x0:6.1f}..{x1:6.1f} cm, y {y0:6.1f}..{y1:6.1f} cm, {z1:4.1f} cm tall'
        if part_id in flat:
            info = flat[part_id]
            who = 'move arm, which brings it to the assembly spot' if info['arm'] == 'place' else f'{info["arm"]} arm' # 'place': the first part in the fallback of run_pickup_gen.py
            line += f', picked by the {who} ({info["tilt"]:.0f} deg from vertical, tips over at {info["tip_angle"]:.0f} deg)'
        print(line)


def show_initial_setup(assembly_dir, log_dir, camera_lookat=None, camera_pos=None):
    fixture_dir = os.path.join(log_dir, 'fixture')
    if not os.path.exists(os.path.join(fixture_dir, 'pickup.json')):
        print(f'[show_initial_setup] {os.path.join(fixture_dir, "pickup.json")} not found, run fixture or pickup generation first')
        return
    with open(os.path.join(log_dir, 'grasps.pkl'), 'rb') as fp:
        grasps = pickle.load(fp)
    arm_type, gripper_type, has_ft_sensor = grasps['arm'], grasps['gripper'], grasps['ft_sensor']

    arm_chain_move = get_arm_chain(arm_type, 'move')
    arm_chain_hold = get_arm_chain(arm_type, 'hold')
    xml_string = create_assembly_dualarm_fixture_xml(assembly_dir, fixture_dir, arm_chain_move, arm_chain_hold, gripper_type, has_ft_sensor)
    sim = redmax.Simulation(xml_string, os.path.join(project_base_dir, 'assets'))
    if camera_lookat is not None: sim.viewer_options.camera_lookat = camera_lookat
    if camera_pos is not None: sim.viewer_options.camera_pos = camera_pos

    # arms at rest, grippers half open (same as the first frame of rendering/render_motion_plan.py)
    for suffix, arm_chain in [('move', arm_chain_move), ('hold', arm_chain_hold)]:
        for joint_name, joint_state in zip(get_arm_joints(arm_type), arm_chain.rest_q):
            sim.set_joint_q_init(f'{joint_name}_{suffix}', np.array([joint_state]))
        for finger_name, finger_state in get_gripper_finger_states(gripper_type, OPEN_RATIO_REST, suffix=suffix).items():
            sim.set_joint_q_init(finger_name, np.array(finger_state))
    sim.reset(backward_flag=False)

    print_legend(assembly_dir, fixture_dir)

    q = sim.get_q()
    sim.set_state_his([q, q], [np.zeros_like(q), np.zeros_like(q)]) # still scene
    sim.viewer_options.loop = True
    sim.viewer_options.infinite = True # stays open until the window is closed
    sim.replay()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--assembly-dir', type=str, required=True, help='e.g. assets/fabrica/human_normal')
    parser.add_argument('--log-dir', type=str, required=True, help='e.g. logs/exp_human_normal_flat/human_normal')
    parser.add_argument('--camera-option', type=int, default=0, help='0: close, 1: far (as in render_motion_plan.py)')
    parser.add_argument('--camera-lookat', type=float, nargs=3, default=None)
    parser.add_argument('--camera-pos', type=float, nargs=3, default=None)
    args = parser.parse_args()

    camera_lookat, camera_pos = get_camera_option(args.camera_option)
    if args.camera_lookat is not None: camera_lookat = args.camera_lookat
    if args.camera_pos is not None: camera_pos = args.camera_pos
    show_initial_setup(args.assembly_dir, args.log_dir, camera_lookat, camera_pos)
