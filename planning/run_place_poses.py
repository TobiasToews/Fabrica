"""
Poses for real_robot/place_parts.py: the robot shows where each part it picks has to lie (no fixture, no template).
For every part the move arm picks from the board, in REVERSE pick order (the part picked last is placed first, so whenever a
part is shown exactly the parts lie on the board that also lie there when the robot picks it during the run):
    q_above: gripper LIFT cm above the pickup pose, same orientation
    q_pick:  the planned pickup pose (gripper open as before the grasp, opened wider before lifting off the part where needed)
The robot moves between these with joint-space moves (frankapy goto_joints), so every move is checked here for collisions
with the desk, the parts that already lie on the board, and franka-interface's virtual walls.
Writes <log-dir>/fixture/place_poses.json
"""

import os

os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)
sys.path.append(project_base_dir)

import json
import pickle
import numpy as np
import trimesh

from planning.config import RETRACT_OPEN_RATIO
from planning.robot.motion_plan_arm import ArmMotionPlanner
from planning.robot.util_arm import get_arm_chain, get_virtual_wall_margin

LIFT = 12.0  # cm above the pickup pose
N_INTERP = 50  # collision checks per joint-space move


def get_pickups(motion):
    """(part, q_pick (7 active joints), open_ratio) for every part the move arm picks, in plan order"""
    pickups = []
    for i, (arm, body, path, _, task) in enumerate(motion):
        if (
            arm == 'move'
            and body == 'gripper'
            and task == 'close'
            and i + 1 < len(motion)
            and motion[i + 1][3] is not None
        ):
            assert (
                motion[i - 1][1] == 'arm'
                and motion[i - 2][1] == 'gripper'
                and motion[i - 2][4] == 'open'
            )
            pickups.append(
                (
                    motion[i + 1][3],
                    np.asarray(motion[i - 1][2][-1]),
                    float(motion[i - 2][2]),
                )
            )
    return pickups


def load_layout_meshes(fixture_dir):
    scene = trimesh.load(
        os.path.join(fixture_dir, 'pickup_layout.glb'), force='scene'
    )
    meshes = {}
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        if not node.endswith('.obj'):
            continue  # the board box of the layout
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        meshes[node[: -len('.obj')]] = mesh
    return meshes


def check_move(planner, q_from, q_to, still_meshes, open_ratio, buffered):
    """True if the joint-space move q_from -> q_to (active) is collision free and inside the virtual walls"""
    _, _, _, collision_fn = planner.get_fns(
        None, None, still_meshes, open_ratio, buffered=buffered
    )
    for t in np.linspace(0, 1, N_INTERP):
        if collision_fn(q_from + t * (q_to - q_from)):
            return False
    return True


def run_place_poses(log_dir, lift=LIFT):
    with open(os.path.join(log_dir, 'motion.pkl'), 'rb') as fp:
        motion = pickle.load(fp)
    with open(os.path.join(log_dir, 'grasps.pkl'), 'rb') as fp:
        grasps = pickle.load(fp)
    assert grasps['arm'] == 'panda'
    fixture_dir = os.path.join(log_dir, 'fixture')
    part_meshes = load_layout_meshes(fixture_dir)

    arm_chain = get_arm_chain('panda', 'move')
    planner = ArmMotionPlanner(
        arm_chain,
        grasps['gripper'],
        grasps['ft_sensor']['move'],
        virtual_walls=True,
    )
    rest_q = np.array(arm_chain.rest_q)

    poses = []
    placed = []
    q_prev = rest_q
    ok = True
    for part_id, q_pick, open_ratio in reversed(
        get_pickups(motion)
    ):  # placed already = picked later in the run
        fk = arm_chain.forward_kinematics(arm_chain.active_to_full(q_pick))
        q_above_full = planner.inverse_kinematics(
            fk[:3, 3] + np.array([0, 0, lift]),
            fk[:3, :3],
            q_init=arm_chain.active_to_full(q_pick),
        )
        if q_above_full is None:
            print(
                f'[run_place_poses] part {part_id}: no IK {lift} cm above the pickup pose'
            )
            return False
        q_above = arm_chain.active_from_full(q_above_full)

        others = [part_meshes[p] for p in placed]
        checks = {
            'to above': check_move(
                planner, q_prev, q_above, others, open_ratio, buffered=True
            ),
            'down': check_move(
                planner, q_above, q_pick, others, open_ratio, buffered=False
            ),
        }
        # lift off the part: the planned width, or wider if the fingers would scrape the part (tilted gripper lifted straight up)
        open_ratio_retract = None
        for ratio in [
            open_ratio,
            open_ratio + RETRACT_OPEN_RATIO,
            open_ratio + 2 * RETRACT_OPEN_RATIO,
            1.0,
        ]:
            ratio = min(ratio, 1.0)
            if check_move(
                planner,
                q_pick,
                q_above,
                others + [part_meshes[part_id]],
                ratio,
                buffered=False,
            ):
                open_ratio_retract = ratio
                break
        checks['up'] = open_ratio_retract is not None
        if open_ratio_retract is None:
            open_ratio_retract = open_ratio
        margin = min(
            get_virtual_wall_margin(arm_chain, arm_chain.active_to_full(q))
            for q in [q_above, q_pick]
        )
        print(
            f'[run_place_poses] part {part_id}: '
            + ', '.join(
                f'{k} {"ok" if v else "COLLISION"}' for k, v in checks.items()
            )
            + f' (open {open_ratio:.2f} -> {open_ratio_retract:.2f}), virtual wall margin {margin:.1f} cm'
        )
        ok = ok and all(checks.values())
        poses.append(
            {
                'part': part_id,
                'q_above': q_above.tolist(),
                'q_pick': q_pick.tolist(),
                'open_ratio': open_ratio,
                'open_ratio_retract': open_ratio_retract,
            }
        )
        placed.append(part_id)
        q_prev = q_above

    back = check_move(
        planner,
        q_prev,
        rest_q,
        [part_meshes[p] for p in placed],
        0.5,
        buffered=True,
    )
    print(f'[run_place_poses] back to rest: {"ok" if back else "COLLISION"}')
    ok = ok and back

    with open(os.path.join(fixture_dir, 'place_poses.json'), 'w') as fp:
        json.dump(
            {
                'lift': lift,
                'rest_q': rest_q.tolist(),
                'poses': poses,
                'checked': ok,
            },
            fp,
            indent=4,
        )
    print(
        f'[run_place_poses] saved {os.path.join(fixture_dir, "place_poses.json")} ({"all moves checked" if ok else "WARNING: some moves collide"})'
    )
    return ok


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument(
        '--lift',
        type=float,
        default=LIFT,
        help='[cm] height above the pickup pose',
    )
    args = parser.parse_args()

    from planning.robot.workcell import use_log_workcell

    use_log_workcell(
        args.log_dir
    )  # the place of the assembly this log is planned for
    sys.exit(0 if run_place_poses(args.log_dir, args.lift) else 1)
