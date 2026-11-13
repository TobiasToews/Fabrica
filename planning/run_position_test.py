"""
Test of the flat pipeline at many places: the assembly and the pickup area are moved over the part of the board that both
arms reach, and planning/run_planning_flat.py has to give a verified plan at each pair of places.

  1. places: N_ASSEMBLY places of the assembly and, for each of them, N_PICKUP places of the pickup area, drawn at random
     (--seed) from the area both arms reach (reachable), with the assembly zone on the board and clear of the pickup area
  2. for every place of the assembly: the precedence plan moved there (as run_workspace_search.py does), grasps generated once
     (they depend on the place of the assembly, not on the pickup area)
  3. for every pair of places: the rest of the pipeline in a log of its own, --num-parallel at a time
  4. report: logs/<exp>/position_test_<assembly>.json, and a table on the console

A run that was interrupted can be started again with the same arguments: places that have a result are not planned again.

python planning/run_position_test.py --exp-name exp_beam_pos --assembly beam --precedence-from logs/exp_beam_flat_run1/beam
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
import shutil
import subprocess
import time
import numpy as np

from planning.config import (
    ASSEMBLY_ZONE_MARGIN,
    MIN_SIGMA,
    MIN_JOINT_LIMIT_MARGIN,
)
from planning.robot.motion_plan_arm import (
    ArmMotionPlanner,
    VIRTUAL_WALL_SAFETY,
)
from planning.robot.panda_condition import joint_condition
from planning.robot.util_arm import get_virtual_wall_margin
from planning.robot.workcell import (
    get_default_assembly_center,
    set_log_assembly_center,
    use_log_workcell,
)
from planning.run_workspace_search import shift_precedence, make_planners

N_ASSEMBLY = 10  # places of the assembly
N_PICKUP = 3  # places of the pickup area per place of the assembly
REACH_HEIGHT = 15.0  # cm, the flange above the table in the reach test (hand pointing straight down)
REACH_YAWS = 8  # rotations of the hand about the vertical tried at a point
PICKUP_RADIUS = 7.5  # cm, the parts are laid out around the pickup place: the circle of this radius has to be reachable
MIN_SEPARATION = 7.5  # cm between two places of the assembly, and between two pickup places of the same assembly
MIN_CORNERS = 4  # corners of the assembly's footprint that have to be reachable, besides its middle
N_SPARE = 4  # further places of the assembly, used if the real arm cannot build the assembly at one of the others (a part has no working grasp there)
PLACE_STEP = 2.5  # cm, places are drawn on the grid of the board holes
AREA_X = (
    -27.5,
    27.5,
)  # cm, where places are drawn (the board is 62.5 x 92.5 cm)
AREA_Y = (-42.5, 42.5)
DOWN = np.array(
    [[1, 0, 0], [0, -1, 0], [0, 0, -1.0]]
)  # flange orientation with the fingers pointing down


def reachable(xy, planners):
    """
    True if both arms reach the point (x, y) of the board: the hand pointing straight down, the flange REACH_HEIGHT above the
    table, in at least one of REACH_YAWS rotations of the hand about the vertical.
    Move arm (the real robot): its pose must keep the rules that do not depend on parts: virtual walls, singularity, joint
    limits (values of planning/plan_checks.py). Hold arm: an arm pose exists.
    """
    position = np.array([xy[0], xy[1], REACH_HEIGHT])
    for name in ['move', 'hold']:
        planner = planners[name]
        chain = planner.arm_chain
        rest = chain.active_to_full(chain.rest_q)
        ok = False
        for yaw in np.linspace(0, 2 * np.pi, REACH_YAWS, endpoint=False):
            c, s = np.cos(yaw), np.sin(yaw)
            q = planner.inverse_kinematics(
                position,
                np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]) @ DOWN,
                q_init=rest,
                n_restart=2,
            )
            if q is None:
                continue
            if name == 'move':
                sigma, margin, _ = joint_condition(chain.active_from_full(q))
                if (
                    get_virtual_wall_margin(chain, q) < VIRTUAL_WALL_SAFETY
                    or sigma < MIN_SIGMA
                    or margin < MIN_JOINT_LIMIT_MARGIN
                ):
                    continue
            ok = True
            break
        if not ok:
            return False
    return True


def sample_places(
    final_bounds,
    default_center,
    reachable_fn,
    n_assembly=N_ASSEMBLY,
    n_pickup=N_PICKUP,
    seed=0,
    board_half_size=(31.25, 46.25),
    area_x=AREA_X,
    area_y=AREA_Y,
    pickup_radius=PICKUP_RADIUS,
    min_separation=MIN_SEPARATION,
    step=PLACE_STEP,
    max_draws=20000,
    min_corners=MIN_CORNERS,
    n_spare=0,
):
    """
    Places for the test, drawn at random on the grid of the board holes. final_bounds: (2, 2) lower and upper xy of the assembly
    at the default place default_center; reachable_fn(xy) -> bool (see reachable; its answers are remembered per point).
    A place [x, y] of the assembly centre is kept if
      - its zone (the footprint plus ASSEMBLY_ZONE_MARGIN) stays 1 cm inside the board,
      - the centre of the footprint and at least min_corners of its four corners are reachable (4: the whole assembly stands
        in the area both arms reach. With fewer, a part at the far side can be out of reach of the move arm: human_normal at
        (-7.5, -30) had no grasp for part 4, all behind the virtual wall),
      - it is min_separation away from the places kept before.
    A pickup place for it is kept if
      - the circle of pickup_radius around it lies on the board (1 cm inside) and does not reach into the assembly zone,
      - its centre is reachable (the layout then looks for spots around it where each pickup passes the checks),
      - it is min_separation away from the other pickup places of this assembly.
    Returns [{'assembly': [x, y], 'footprint_centre': [x, y], 'corners_reachable': 0..4, 'pickup': [[x, y], ...]}] with n_assembly
    entries of n_pickup pickup places each; fewer if max_draws random points do not give that many.
    n_spare: further places after those (the first n_assembly are the same with and without them), for the case that the real
    arm cannot build the assembly at one of the places (a part without a working grasp there).
    """
    rng = np.random.default_rng(seed)
    final_bounds, default_center = (
        np.asarray(final_bounds, dtype=float),
        np.asarray(default_center, dtype=float)[:2],
    )
    board = np.asarray(board_half_size, dtype=float) - 1.0
    known = {}

    def reach(point):
        key = (round(float(point[0]), 3), round(float(point[1]), 3))
        if key not in known:
            known[key] = bool(reachable_fn(np.array(key)))
        return known[key]

    def draw():
        return (
            np.array(
                [
                    rng.integers(
                        round(area_x[0] / step), round(area_x[1] / step) + 1
                    ),
                    rng.integers(
                        round(area_y[0] / step), round(area_y[1] / step) + 1
                    ),
                ]
            )
            * step
        )

    places, draws = [], 0
    while len(places) < n_assembly + n_spare and draws < max_draws:
        draws += 1
        center = draw()
        lower, upper = (
            final_bounds[0] + (center - default_center),
            final_bounds[1] + (center - default_center),
        )
        zone = (lower - ASSEMBLY_ZONE_MARGIN, upper + ASSEMBLY_ZONE_MARGIN)
        if np.any(zone[0] < -board) or np.any(zone[1] > board):
            continue
        if any(
            np.linalg.norm(center - np.array(p['assembly'])) < min_separation
            for p in places
        ):
            continue
        middle = (lower + upper) / 2
        if not reach(middle):
            continue
        corners = [
            lower,
            upper,
            np.array([lower[0], upper[1]]),
            np.array([upper[0], lower[1]]),
        ]
        corners_reachable = int(sum(reach(corner) for corner in corners))
        if corners_reachable < min_corners:
            continue
        pickups, tries = [], 0
        while len(pickups) < n_pickup and tries < max_draws // n_assembly:
            tries += 1
            pickup = draw()
            if np.any(np.abs(pickup) + pickup_radius > board):
                continue
            nearest = np.clip(
                pickup, zone[0], zone[1]
            )  # the point of the zone closest to the pickup place
            if np.linalg.norm(pickup - nearest) < pickup_radius:
                continue
            if any(
                np.linalg.norm(pickup - np.array(p)) < min_separation
                for p in pickups
            ):
                continue
            if not reach(pickup):
                continue
            pickups.append([float(pickup[0]), float(pickup[1])])
        if len(pickups) == n_pickup:
            places.append(
                {
                    'assembly': [float(center[0]), float(center[1])],
                    'footprint_centre': [float(middle[0]), float(middle[1])],
                    'corners_reachable': corners_reachable,
                    'pickup': pickups,
                }
            )
    return places


def result_of(log_dir, console_path):
    """what a planning run left behind: {'verified', 'attempts', 'banned_grasps', 'banned_spots', 'place_poses', 'first_part', 'reason'}"""
    result = {
        'verified': False,
        'attempts': 0,
        'banned_grasps': 0,
        'banned_spots': 0,
        'place_poses': None,
        'first_part': None,
        'reason': None,
    }
    workcell = os.path.join(log_dir, 'workcell.json')
    if os.path.exists(workcell):
        with open(workcell, 'r') as fp:
            info = json.load(fp)
        result.update(
            verified=bool(info.get('verified')),
            banned_grasps=len(info.get('banned_move_grasps', [])),
            banned_spots=len(info.get('banned_spots', [])),
            place_poses=info.get('place_poses'),
        )
    flat_pickup = os.path.join(log_dir, 'fixture', 'flat_pickup.json')
    if os.path.exists(
        flat_pickup
    ):  # who brings the first part to the assembly spot: 'hold' (hold arm) or 'place' (the move arm, fallback)
        with open(flat_pickup, 'r') as fp:
            result['first_part'] = next(
                (
                    info['arm']
                    for info in json.load(fp).values()
                    if info['arm'] != 'move'
                ),
                None,
            )
    if os.path.exists(console_path):
        with open(console_path, 'r', errors='replace') as fp:
            lines = [line.strip() for line in fp]
        result['attempts'] = sum(
            '[run_planning_flat] sequence attempt' in line for line in lines
        )
        if not result['verified']:
            telling = [
                line
                for line in lines
                if any(
                    tag in line
                    for tag in [
                        'FAILED for',
                        'no sequence left',
                        'cannot be brought',
                        'place poses for',
                        'no clean plan',
                        'the final check failed',
                        'no plan where the assembly stands',
                        'failed without naming',
                        'not caused by a move grasp',
                        'Traceback',
                        'is planned for',
                    ]
                )
            ]
            result['reason'] = (
                telling[-1][:400]
                if telling
                else (lines[-1][:400] if lines else 'no output')
            )
    return result


def grasp_state(log_dir, console_path):
    """
    'ok' if every part the move arm has to insert got a move grasp at this place of the assembly, else why not: the parts
    and what the grasp generation printed about them (run_grasp_arm_gen.py: "no usable move grasp; ... fail: ...")
    """
    from planning.run_planning_flat import parts_without_move_grasp

    missing = parts_without_move_grasp(log_dir)
    if not missing:
        return 'ok'
    why = []
    if os.path.exists(console_path):
        with open(console_path, 'r', errors='replace') as fp:
            lines = [line.strip() for line in fp]
        for part in missing:
            why += [
                line[
                    line.index('[generate_grasps]')
                    + len('[generate_grasps] ') :
                ][:500]
                for line in lines
                if f'[generate_grasps] part {part}: no usable move grasp'
                in line
            ][:1]
    return f'no move grasp for part {", ".join(missing)}' + (
        ': ' + ' | '.join(why) if why else ''
    )


def run_position_test(
    exp_name,
    assembly,
    assembly_root='fabrica',
    precedence_from=None,
    n_assembly=N_ASSEMBLY,
    n_pickup=N_PICKUP,
    seed=0,
    num_proc=12,
    num_proc_grasp=50,
    num_parallel=4,
    places_only=False,
    min_corners=MIN_CORNERS,
    n_spare=N_SPARE,
    min_separation=MIN_SEPARATION,
):
    from planning.run_pickup_gen import load_final_part_cfg, BOARD_HALF_SIZE

    assembly_dir = os.path.join(
        project_base_dir, 'assets', assembly_root, assembly
    )
    default_center = get_default_assembly_center('panda')
    py = sys.executable
    ArmMotionPlanner.VISUALIZE = False

    # the precedence plan at the default place (made once, moved to every place)
    reference_dir = os.path.join(project_base_dir, 'logs', exp_name, assembly)
    os.makedirs(reference_dir, exist_ok=True)
    if use_log_workcell(reference_dir) is not None:
        print(
            f'[run_position_test] {reference_dir} is planned for another place: use a new --exp-name'
        )
        return False
    precedence_path = os.path.join(reference_dir, 'precedence.pkl')
    if not os.path.exists(precedence_path):
        if precedence_from is not None:
            if os.path.exists(
                os.path.join(precedence_from, 'assembly_center.json')
            ):
                print(
                    f'[run_position_test] {precedence_from} is planned for another place than the default'
                )
                return False
            shutil.copy(
                os.path.join(precedence_from, 'precedence.pkl'),
                precedence_path,
            )
            with open(os.path.join(precedence_from, 'stats.json'), 'r') as fp:
                stats = {'preced_plan': json.load(fp)['preced_plan']}
            with open(os.path.join(reference_dir, 'stats.json'), 'w') as fp:
                json.dump(stats, fp)
        elif subprocess.call(
            [
                py,
                'planning/run_preced_plan.py',
                '--assembly-dir',
                os.path.join('assets', assembly_root, assembly),
                '--log-dir',
                reference_dir,
                '--num-proc',
                str(num_proc),
                '--arm',
                'panda',
            ],
            cwd=project_base_dir,
        ) != 0 or not os.path.exists(precedence_path):
            print('[run_position_test] precedence planning failed')
            return False
    with open(precedence_path, 'rb') as fp:
        G_preced = pickle.load(fp)
    with open(os.path.join(reference_dir, 'stats.json'), 'r') as fp:
        stats_preced = {'preced_plan': json.load(fp)['preced_plan']}

    report_path = os.path.join(
        project_base_dir, 'logs', exp_name, f'position_test_{assembly}.json'
    )
    if os.path.exists(
        report_path
    ):  # the places of an earlier call are used again
        with open(report_path, 'r') as fp:
            report = json.load(fp)
        print(f'[run_position_test] using the places of {report_path}')
    else:
        final_meshes = load_final_part_cfg(assembly_dir, 'panda')['mesh']
        bounds = np.array([mesh.bounds for mesh in final_meshes.values()])
        final_bounds = np.array(
            [bounds[:, 0, :2].min(axis=0), bounds[:, 1, :2].max(axis=0)]
        )
        planners = make_planners('panda')
        np.random.seed(seed)  # the IK restarts of the reach test
        start = time.time()
        places = sample_places(
            final_bounds,
            default_center,
            lambda xy: reachable(xy, planners),
            n_assembly,
            n_pickup,
            seed,
            BOARD_HALF_SIZE,
            min_corners=min_corners,
            n_spare=n_spare,
            min_separation=min_separation,
        )
        print(
            f'[run_position_test] {len(places)} places of the assembly with {n_pickup} pickup places each drawn in {time.time() - start:.0f} s (seed {seed}, at least {min_corners} corners of the '
            f'footprint reachable, {min_separation} cm apart; {n_assembly} are wanted, the others are spare)'
        )
        report = {
            'assembly': assembly,
            'seed': seed,
            'min_corners': min_corners,
            'min_separation': min_separation,
            'n_assembly': n_assembly,
            'footprint_at_default': final_bounds.tolist(),
            'default_center': [float(v) for v in default_center[:2]],
            'places': places,
            'runs': {},
        }
        with open(report_path, 'w') as fp:
            json.dump(report, fp, indent=2)
    for i, place in enumerate(report['places']):
        print(
            f'    a{i}: assembly centre ({place["assembly"][0]:+6.1f}, {place["assembly"][1]:+6.1f}), middle of its footprint ({place["footprint_centre"][0]:+6.1f}, {place["footprint_centre"][1]:+6.1f}), '
            f'{place["corners_reachable"]} of 4 corners reachable; pickup areas around '
            + ', '.join(
                f'({p[0]:+6.1f}, {p[1]:+6.1f})' for p in place['pickup']
            )
        )
    if places_only:
        return True
    n_assembly = report.get('n_assembly', n_assembly)
    if len(report['places']) < n_assembly:
        print(
            f'[run_position_test] only {len(report["places"])} of {n_assembly} places found in the area both arms reach'
        )

    def save_report():
        with open(report_path, 'w') as fp:
            json.dump(report, fp, indent=2)

    # grasps for the places of the assembly, one after the other (the grasp generation uses all cores), until n_assembly places
    # have grasps for every part. place['grasps']: 'ok', or why the real arm cannot build the assembly there (grasp_state):
    # such a place is not in the area the arm can work in for this assembly, whatever the reach test with the empty hand said
    for i, place in enumerate(report['places']):
        if (
            sum(p.get('grasps') == 'ok' for p in report['places'])
            >= n_assembly
        ):
            break
        if place.get('grasps') is not None:
            continue
        exp_place = f'{exp_name}_a{i}'
        log_dir = os.path.join(project_base_dir, 'logs', exp_place, assembly)
        console = os.path.join(
            project_base_dir, 'logs', exp_place, 'console.log'
        )
        start = time.time()
        if not os.path.exists(os.path.join(log_dir, 'grasps.pkl')):
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, 'precedence.pkl'), 'wb') as fp:
                pickle.dump(
                    shift_precedence(
                        G_preced,
                        np.asarray(place['assembly']) - default_center[:2],
                    ),
                    fp,
                )
            with open(os.path.join(log_dir, 'stats.json'), 'w') as fp:
                json.dump(stats_preced, fp)
            set_log_assembly_center(log_dir, place['assembly'])
            with open(console, 'w') as fp:
                subprocess.call(
                    [
                        py,
                        '-u',
                        'planning/run_planning_flat.py',
                        '--exp-name',
                        exp_place,
                        '--assembly',
                        assembly,
                        '--assembly-root',
                        assembly_root,
                        '--num-proc',
                        str(num_proc),
                        '--num-proc-grasp',
                        str(num_proc_grasp),
                        '--assembly-center',
                        str(place['assembly'][0]),
                        str(place['assembly'][1]),
                        '--grasps-only',
                    ],
                    cwd=project_base_dir,
                    stdout=fp,
                    stderr=subprocess.STDOUT,
                )
        place['grasps'] = (
            grasp_state(log_dir, console)
            if os.path.exists(os.path.join(log_dir, 'grasps.pkl'))
            else f'the grasp generation failed, see {console}'
        )
        save_report()
        print(
            f'[run_position_test] a{i}: grasps at ({place["assembly"][0]:+.1f}, {place["assembly"][1]:+.1f}) after {time.time() - start:.0f} s: {place["grasps"]}',
            flush=True,
        )
    n_ok = sum(p.get('grasps') == 'ok' for p in report['places'])
    if n_ok < n_assembly:
        print(
            f'[run_position_test] only {n_ok} of the {len(report["places"])} places drawn have grasps for every part ({n_assembly} were wanted)'
        )

    # the rest of the pipeline for every pair of places, num_parallel at a time
    jobs = []
    for i, place in enumerate(report['places']):
        if place.get('grasps') != 'ok':
            continue
        source = os.path.join(
            project_base_dir, 'logs', f'{exp_name}_a{i}', assembly
        )
        for j, pickup in enumerate(place['pickup']):
            name = f'a{i}_p{j}'
            if name in report['runs']:
                continue
            exp_pair = f'{exp_name}_{name}'
            log_dir = os.path.join(
                project_base_dir, 'logs', exp_pair, assembly
            )
            if not os.path.exists(log_dir):
                os.makedirs(os.path.dirname(log_dir), exist_ok=True)
                shutil.copytree(source, log_dir)
            cmd = [
                py,
                '-u',
                'planning/run_planning_flat.py',
                '--exp-name',
                exp_pair,
                '--assembly',
                assembly,
                '--assembly-root',
                assembly_root,
                '--num-proc',
                str(num_proc),
                '--num-proc-grasp',
                str(num_proc_grasp),
                '--assembly-center',
                str(place['assembly'][0]),
                str(place['assembly'][1]),
                '--pickup-center',
                str(pickup[0]),
                str(pickup[1]),
            ]
            jobs.append(
                (
                    name,
                    place['assembly'],
                    pickup,
                    cmd,
                    log_dir,
                    os.path.join(
                        project_base_dir, 'logs', exp_pair, 'console.log'
                    ),
                )
            )
    running = []
    while len(jobs) > 0 or len(running) > 0:
        while len(jobs) > 0 and len(running) < num_parallel:
            name, center, pickup, cmd, log_dir, console = jobs.pop(0)
            running.append(
                (
                    name,
                    center,
                    pickup,
                    log_dir,
                    console,
                    subprocess.Popen(
                        cmd,
                        cwd=project_base_dir,
                        stdout=open(console, 'w'),
                        stderr=subprocess.STDOUT,
                    ),
                    time.time(),
                )
            )
        time.sleep(2.0)
        for job in list(running):
            name, center, pickup, log_dir, console, process, start = job
            if process.poll() is None:
                continue
            running.remove(job)
            report['runs'][name] = {
                'assembly': center,
                'pickup': pickup,
                **result_of(log_dir, console),
                'minutes': round((time.time() - start) / 60, 1),
            }
            save_report()
            run = report['runs'][name]
            print(
                f'[run_position_test] {name}: assembly ({center[0]:+.1f}, {center[1]:+.1f}), pickup ({pickup[0]:+.1f}, {pickup[1]:+.1f}): '
                + (
                    f'plan verified, {run["attempts"]} attempt(s)'
                    if run['verified']
                    else f'NO PLAN: {run["reason"]}'
                )
                + f' ({run["minutes"]} min)',
                flush=True,
            )
    save_report()

    runs = report['runs']
    n_verified = sum(run['verified'] for run in runs.values())
    print(
        f'\n[run_position_test] {assembly}: plan verified at {n_verified} of {len(runs)} pairs of places; report: {report_path}'
    )
    for i, place in enumerate(report['places']):
        if place.get('grasps') not in [None, 'ok']:
            print(
                f'    a{i} ({place["assembly"][0]:+6.1f}, {place["assembly"][1]:+6.1f}): not a place the real arm can build this assembly at, no pair planned: {place["grasps"]}'
            )
    print(f'    {"pair":8} {"assembly":>16} {"pickup":>16}  result')
    for name in sorted(runs, key=lambda n: [int(v[1:]) for v in n.split('_')]):
        run = runs[name]
        print(
            f'    {name:8} ({run["assembly"][0]:+6.1f}, {run["assembly"][1]:+6.1f}) ({run["pickup"][0]:+6.1f}, {run["pickup"][1]:+6.1f})  '
            + (
                f'verified, {run["attempts"]} attempt(s), {run["banned_grasps"]} grasp(s) and {run["banned_spots"]} spot(s) dropped, {run["minutes"]} min'
                + (
                    ', first part brought by the move arm'
                    if run['first_part'] == 'place'
                    else ''
                )
                + (
                    ''
                    if run['place_poses']
                    else ', no poses for place_parts.py'
                )
                if run['verified']
                else f'NO PLAN after {run["attempts"]} attempt(s), {run["minutes"]} min: {run["reason"]}'
            )
        )
    return n_verified == len(runs)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--assembly', type=str, required=True)
    parser.add_argument('--assembly-root', type=str, default='fabrica')
    parser.add_argument(
        '--precedence-from',
        type=str,
        default=None,
        help='log folder with a precedence plan at the default place to reuse (precedence.pkl, stats.json)',
    )
    parser.add_argument(
        '--n-assembly',
        type=int,
        default=N_ASSEMBLY,
        help='places of the assembly',
    )
    parser.add_argument(
        '--n-pickup',
        type=int,
        default=N_PICKUP,
        help='places of the pickup area per place of the assembly',
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-proc', type=int, default=12)
    parser.add_argument('--num-proc-grasp', type=int, default=50)
    parser.add_argument(
        '--num-parallel',
        type=int,
        default=4,
        help='plans computed at the same time',
    )
    parser.add_argument(
        '--places-only',
        action='store_true',
        help='only draw and print the places',
    )
    parser.add_argument(
        '--min-corners',
        type=int,
        default=MIN_CORNERS,
        help='corners of the footprint of the assembly that have to be reachable (4: the whole assembly)',
    )
    parser.add_argument(
        '--n-spare',
        type=int,
        default=N_SPARE,
        help='further places of the assembly, used if a part has no working grasp at one of the others',
    )
    parser.add_argument(
        '--min-separation',
        type=float,
        default=MIN_SEPARATION,
        help='[cm] between two places of the assembly, and between two pickup places of one assembly place',
    )
    args = parser.parse_args()
    sys.exit(
        0
        if run_position_test(
            args.exp_name,
            args.assembly,
            args.assembly_root,
            args.precedence_from,
            args.n_assembly,
            args.n_pickup,
            args.seed,
            args.num_proc,
            args.num_proc_grasp,
            args.num_parallel,
            args.places_only,
            args.min_corners,
            args.n_spare,
            args.min_separation,
        )
        else 1
    )
