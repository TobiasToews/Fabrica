"""Plans an assembly for the real robot without a fixture, in one go.

The parts lie flat on the table, the "move" arm picks and inserts them,
and the "hold" arm holds the assembly. Both arms are planned as robots:
each avoids the other, and no step is left to a human (since
2026-11-13). Today only the move arm is a real robot; real_robot/run.py
--single-arm lets a human do the steps of the hold arm. This file is the
driver of that
pipeline: it runs the stages below one after the other, each as a
subprocess. All stages work on one log directory,
logs/<exp-name>/<assembly>, called "the log" here. The docstring of
run_planning_flat lists the files each stage reads and writes there.

1. Precedence and grasps, only if the log has none yet, at the current
   robot height. The precedence plan says which part has to be in
   place before which. A grasp is where the gripper holds a part: a
   move grasp is one of the move arm, a hold grasp one of the hold arm
   (the arm that holds the assembly). The grasps are
   generated for the flat pipeline (run_grasp_arm_gen.py --flat):
   contact on the grid of the finger pads, more surface points for
   parts without a usable grasp, and the move grasps that are kept can
   pick their part lying flat and are the best rated ones.
2. Grasp filter (flat pickup, virtual walls, finger overlap).
3. Sequence planning.
4. Loop: sequence optimization -> pickup layout -> motion planning ->
   verify_plan.py. A stage that cannot find a clean plan says what has
   to change (planning/plan_checks.py CAUSE_*). The loop does that and
   tries again, at most MAX_SEQUENCE_ATTEMPTS times:

   * the move grasp (arm or gripper do not fit): it is removed, the
     next best sequence is tried;
   * the hold grasp (the hold arm cannot pick the first part, has no
     path, or comes too close to the move arm): it is removed, the next
     best sequence is tried;
   * the spot where a part lies: the grasp stays, the part is laid out
     at another spot (same sequence);
   * the insertion itself (the part has no way to where its insertion
     starts, however it is held): planning stops.
5. Place poses (for real_robot/place_parts.py).
6. workcell.json (the real-world values this plan was made with).

The assembly stands at --assembly-center, the parts are laid out around
--pickup-center (both on the board, in cm; stored in the log, defaults:
0 -15 and 0 5). planning/run_workspace_search.py finds a place for the
assembly.

Every path of the real arm is checked with planning/plan_checks.py
(exact geometry), and every path of either arm keeps its distance to
the other arm. The plan is only reported as done when verify_plan.py
passes.

Usage:
    python planning/run_planning_flat.py \
        --exp-name exp_human_normal_flat_v3 --assembly human_normal
"""

import os
import sys

project_base_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)
sys.path.append(project_base_dir)

import json
import pickle
import subprocess
from datetime import datetime

from planning.config import (
    ROBOT_PART_CLEARANCE,
    GRIPPER_TABLE_CLEARANCE,
    TRANSFER_PART_CLEARANCE,
    GRASP_MIN_FINGER_OVERLAP,
    ASSEMBLY_ZONE_MARGIN,
    ROBOT_SELF_CLEARANCE,
    MAX_PLAN_ATTEMPTS,
    MAX_SEQUENCE_ATTEMPTS,
    MIN_SIGMA,
    MIN_SIGMA_INSERTION,
    MIN_JOINT_LIMIT_MARGIN,
)
from planning.plan_checks import (
    FAILURE_FILE,
    CAUSE_GRASP,
    CAUSE_SPOT,
    CAUSE_INSERTION,
    CAUSE_HOLD_GRASP,
)
from planning.robot.motion_plan_arm import VIRTUAL_WALL_SAFETY
from planning.robot.workcell import (
    get_panda_base_z,
    get_panda_desk_z,
    get_assembly_center,
    get_default_assembly_center,
    use_log_workcell,
    set_log_assembly_center,
    set_log_pickup_center,
)
from planning.run_pickup_gen import (
    PART_GAP,
    GROUND_CLEARANCE,
    WRIST_CLEARANCE,
    pickup_anchor,
    ban_spot,
    clear_banned_spots,
)
from planning.config import ZONE_PART_TABLE_CLEARANCE


def run(cmd, log_file):
    """Runs one stage of the pipeline as a subprocess.

    The stage runs from the repository root, with the environment
    variable OMP_NUM_THREADS set to 1. Everything it prints (stdout and
    stderr) is appended to the log file. The lines that tell how the
    plan is going are also printed to the console.

    Args:
        cmd (list[str]): The command: the Python interpreter, the
            script and its arguments.
        log_file (str): Path of the file the output is appended to
            (planning.log in the log directory).

    Returns:
        int: The exit code of the stage; 0 means it succeeded.
    """
    print(f'[run_planning_flat] $ {" ".join(cmd)}')
    with open(log_file, 'a') as fp:
        fp.write(f'\n$ {" ".join(cmd)}\n')
        fp.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=project_base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, 'OMP_NUM_THREADS': '1'},
        )
        for line in proc.stdout:
            fp.write(line)
            # The console only shows the lines that contain one of these
            # marks: results of the checks, errors, and the lines in
            # which the stages say what they did. The full output is in
            # the log file.
            if any(
                tag in line
                for tag in [
                    'FAILED',
                    'PASSED',
                    'rejected',
                    '[run_pickup_gen] part',
                    '[filter_flat_grasps]',
                    '[run_flat_seq_opt]',
                    'Error',
                    'Traceback',
                    '[run_place_poses]',
                    '[generate_grasps] part',
                    'shown as a jump',
                    'no pickup of the hold arm',
                    'laid out around',
                ]
            ):
                print('    ' + line.rstrip())
        return proc.wait()


def workcell_values():
    """Collects the real-world values this plan is made with.

    They are printed at the start of a run and stored in workcell.json
    at its end. real_robot/run.py reads that file and refuses a plan
    made for another robot height.

    Returns:
        dict: Name and value of each quantity; the name ends with the
        unit (cm, rad) where there is one. It holds the robot height,
        the places of the assembly and of the pickup area ([x, y] in
        the board frame), and the clearances and limits that the layout
        and the checks of the plan use.
    """
    return {
        'robot_base_z_cm': get_panda_base_z(),
        'table_above_mounting_cm': get_panda_desk_z(),
        # Where the assembly stands on the board.
        'assembly_center_cm': [
            float(v) for v in get_assembly_center('panda')[:2]
        ],
        # The place around which the parts are laid out.
        'pickup_center_cm': [float(v) for v in pickup_anchor('panda')],
        # The next three: lying parts and the gripper that picks them.
        'part_gap_cm': PART_GAP,
        'pickup_gripper_clearance_cm': GROUND_CLEARANCE,
        'pickup_wrist_clearance_cm': WRIST_CLEARANCE,
        'robot_part_clearance_cm': ROBOT_PART_CLEARANCE,
        'gripper_table_clearance_cm': GRIPPER_TABLE_CLEARANCE,
        'transfer_part_clearance_cm': TRANSFER_PART_CLEARANCE,
        'zone_part_table_clearance_cm': ZONE_PART_TABLE_CLEARANCE,
        'grasp_min_finger_overlap_cm': GRASP_MIN_FINGER_OVERLAP,
        'assembly_zone_margin_cm': ASSEMBLY_ZONE_MARGIN,
        'robot_self_clearance_cm': ROBOT_SELF_CLEARANCE,
        'virtual_wall_safety_cm': VIRTUAL_WALL_SAFETY,
        'min_jacobian_sigma': MIN_SIGMA,
        'min_jacobian_sigma_insertion': MIN_SIGMA_INSERTION,
        'min_joint_limit_margin_rad': MIN_JOINT_LIMIT_MARGIN,
    }


def grasps_base_z(grasps_path):
    """Returns the robot heights the grasps of a file were made for.

    The grasp generation stores with every grasp the arm pose that
    reaches it and the position of the robot base it was computed for.
    The driver compares that height with the current one
    (get_panda_base_z): grasps made for another height need a new log.

    Args:
        grasps_path (str): Path of a grasps.pkl or
            grasps_unfiltered.pkl.

    Returns:
        set[float]: The z coordinate of the robot base in cm (relative
        to the table surface), rounded to 0.001 cm, of the first move
        grasp of every part. Parts without a move grasp and grasps
        without a stored base position are skipped. The set has a
        single value if all were made for the same height.
    """
    with open(grasps_path, 'rb') as fp:
        grasps = pickle.load(fp)
    # part['move'] is the list of move grasps of a part. Each move grasp
    # g is itself a list of grasps along the insertion path, g[0] at
    # the assembled pose. arm_pos is the position of the robot base in
    # the board frame; its z is the robot height.
    return {
        round(float(g[0].arm_pos[2]), 3)
        for part in grasps['grasps'].values()
        for g in part['move'][:1]
        if g[0].arm_pos is not None
    }


def parts_without_move_grasp(log_dir):
    """Lists the parts the move arm has to insert but has no grasp for.

    A part has to be inserted if another part is assembled before it.
    The precedence graph G_preced (precedence.pkl) has one node per
    part, and its attribute 'parts_after' lists the parts that are
    assembled before that part. A part with nothing before it can be
    the one that is held first, which needs no move grasp, whether the
    precedence plan gives it a path or not (plumbers_block and duct: it
    has one). If several parts have nothing before them, only one of
    them is held first, so at most one of them may be without a move
    grasp.

    Args:
        log_dir (str): Log directory with precedence.pkl and
            grasps.pkl.

    Returns:
        list[str]: The ids of the parts that have to be inserted and
        got no move grasp, sorted. If several parts with nothing before
        them have no move grasp, all of them are listed. Empty if no
        move grasp is missing.
    """
    with open(os.path.join(log_dir, 'precedence.pkl'), 'rb') as fp:
        G_preced = pickle.load(fp)
    with open(os.path.join(log_dir, 'grasps.pkl'), 'rb') as fp:
        grasps = pickle.load(fp)['grasps']
    # All parts without a move grasp.
    without = [
        part
        for part in G_preced.nodes
        if len(grasps.get(part, {}).get('move', [])) == 0
    ]
    # Those of them with nothing assembled before them: one of these
    # can be the part that is held first.
    first_candidates = [
        part
        for part in without
        if len(G_preced.nodes[part]['parts_after']) == 0
    ]
    # A single such part is not reported: it can be held first. If
    # there are several, all of them are reported.
    return sorted(
        [part for part in without if part not in first_candidates]
        + (first_candidates if len(first_candidates) > 1 else [])
    )


def ban_grasp(log_dir, part, grasp_id):
    """Removes a move grasp and its pairs from grasps.pkl.

    The loop of run_planning_flat does this when a stage failed because
    of the grasp, so that the sequence optimization picks another one.

    Args:
        log_dir (str): Log directory with grasps.pkl. The file is
            rewritten.
        part (str): Id of the part the grasp belongs to.
        grasp_id (int): Id of the move grasp to remove.
    """
    path = os.path.join(log_dir, 'grasps.pkl')
    with open(path, 'rb') as fp:
        grasps = pickle.load(fp)
    # A move grasp g is a list of grasps along the insertion path; its
    # id is read from g[0], the grasp at the assembled pose.
    grasps['grasps'][part]['move'] = [
        g for g in grasps['grasps'][part]['move'] if g[0].grasp_id != grasp_id
    ]
    # grasp_id_pairs: for each pair (part that is moved, part that is
    # held) the list of (move grasp id, hold grasp id) that the grasp
    # generation kept as a pair. Pairs with the removed grasp go too.
    grasps['grasp_id_pairs'] = {
        (pm, ph): [
            (m, h) for m, h in pairs if not (pm == part and m == grasp_id)
        ]
        for (pm, ph), pairs in grasps['grasp_id_pairs'].items()
    }
    with open(path, 'wb') as fp:
        pickle.dump(grasps, fp)


def next_step(failure):
    """Decides what the loop does after a stage failed.

    Args:
        failure (dict | None): The content of the planning_failure.json
            that the failed stage wrote (plan_checks.write_failure):
            'stage', 'part', 'grasp_id', 'reason' and 'cause' (one of
            the CAUSE_* values of planning/plan_checks.py). None if the
            stage named no failure.

    Returns:
        str: 'ban grasp' (drop the move grasp and take the next best
        sequence), 'ban hold grasp' (the same for a hold grasp),
        'ban spot' (lay the part out elsewhere, same sequence) or
        'stop'.
    """
    if failure is None:
        return 'stop'
    # Logs written before the causes existed blamed the grasp.
    cause = failure.get('cause', CAUSE_GRASP)
    if cause == CAUSE_INSERTION:
        return 'stop'
    # Without a grasp id there is no grasp to drop and no spot of its
    # own.
    if failure.get('grasp_id') is None:
        return 'stop'
    if cause == CAUSE_HOLD_GRASP:
        return 'ban hold grasp'
    return 'ban spot' if cause == CAUSE_SPOT else 'ban grasp'


def ban_hold_grasp(log_dir, part, grasp_id):
    """Removes a hold grasp and its pairs from grasps.pkl.

    The loop of run_planning_flat does this when the hold arm could not
    do its part with this grasp (no arm pose to pick the first part, no
    path, too close to the move arm). The sequence optimization then
    picks another hold grasp, or another sequence.

    Args:
        log_dir (str): Log directory with grasps.pkl. The file is
            rewritten.
        part (str): Id of the part the hold grasp belongs to.
        grasp_id (int): Id of the hold grasp to remove.
    """
    path = os.path.join(log_dir, 'grasps.pkl')
    with open(path, 'rb') as fp:
        grasps = pickle.load(fp)
    # Unlike a move grasp (a list of grasps along the insertion path), a
    # hold grasp is a single grasp: the part does not move while held.
    grasps['grasps'][part]['hold'] = [
        g for g in grasps['grasps'][part]['hold'] if g.grasp_id != grasp_id
    ]
    # Pairs are (move grasp id, hold grasp id) per (moved part, held
    # part); the pairs that hold this part with the removed grasp go.
    grasps['grasp_id_pairs'] = {
        (pm, ph): [
            (m, h) for m, h in pairs if not (ph == part and h == grasp_id)
        ]
        for (pm, ph), pairs in grasps['grasp_id_pairs'].items()
    }
    with open(path, 'wb') as fp:
        pickle.dump(grasps, fp)


def run_planning_flat(
    exp_name,
    assembly,
    assembly_root='fabrica',
    num_proc=12,
    num_proc_grasp=50,
    max_n_grasp=100,
    grasps_only=False,
    assembly_center=None,
    pickup_center=None,
):
    assembly_dir = os.path.join('assets', assembly_root, assembly)
    log_dir = os.path.join('logs', exp_name, assembly)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'planning.log')
    # the place of the assembly on the board: stored in the log, every stage reads it from there (planning/robot/workcell.py)
    stored = use_log_workcell(log_dir)
    if assembly_center is not None:
        planned = (
            stored
            if stored is not None
            else [float(v) for v in get_default_assembly_center('panda')[:2]]
        )
        if (
            any(
                os.path.exists(os.path.join(log_dir, name))
                for name in ['precedence.pkl', 'grasps.pkl']
            )
            and max(abs(a - b) for a, b in zip(planned, assembly_center))
            > 1e-6
        ):
            print(
                f'[run_planning_flat] {log_dir} is planned for the assembly at {planned}, not at {list(assembly_center)}: use a new --exp-name'
            )
            return False
        set_log_assembly_center(log_dir, assembly_center)
    if (
        pickup_center is not None
    ):  # only the layout and what follows depend on it: an existing log can be planned again around another place
        set_log_pickup_center(log_dir, pickup_center)
    values = workcell_values()
    print(
        '[run_planning_flat] real-world values (planning/robot/workcell.py, planning/config.py):'
    )
    for k, v in values.items():
        print(f'    {k}: {v}')
    py = sys.executable

    grasps_path = os.path.join(log_dir, 'grasps.pkl')
    if not os.path.exists(grasps_path):
        if os.path.exists(
            os.path.join(log_dir, 'precedence.pkl')
        ):  # e.g. grasps are generated again with other settings: the precedence does not depend on them
            print(
                f'[run_planning_flat] {log_dir} has a precedence plan already: only the grasps are generated'
            )
        elif (
            run(
                [
                    py,
                    'planning/run_preced_plan.py',
                    '--assembly-dir',
                    assembly_dir,
                    '--log-dir',
                    log_dir,
                    '--num-proc',
                    str(num_proc),
                    '--arm',
                    'panda',
                ],
                log_file,
            )
            != 0
        ):
            return False
        if (
            run(
                [
                    py,
                    'planning/run_grasp_arm_gen.py',
                    '--assembly-dir',
                    assembly_dir,
                    '--log-dir',
                    log_dir,
                    '--num-proc',
                    str(num_proc_grasp),
                    '--max-n-grasp',
                    str(max_n_grasp),
                    '--arm',
                    'panda',
                    '--gripper',
                    'panda',
                    '--ft-sensor',
                    'none',
                    '--flat',
                ],
                log_file,
            )
            != 0
        ):
            return False  # --flat: move grasps that can pick their part lying flat (planning/flat_grasp.py)
        if not os.path.exists(grasps_path):
            print(
                f'[run_planning_flat] grasp generation wrote no grasps.pkl, see {log_file}'
            )
            return False
    base_z = grasps_base_z(
        os.path.join(log_dir, 'grasps_unfiltered.pkl')
        if os.path.exists(os.path.join(log_dir, 'grasps_unfiltered.pkl'))
        else grasps_path
    )
    if base_z != {round(get_panda_base_z(), 3)}:
        print(
            f'[run_planning_flat] {log_dir} has grasps for robot base z {sorted(base_z)}, workcell.py has {get_panda_base_z()}: use a new --exp-name'
        )
        return False
    missing = (
        parts_without_move_grasp(log_dir)
        if not os.path.exists(os.path.join(log_dir, 'grasps_unfiltered.pkl'))
        else []
    )  # as generated (a later run may have dropped grasps)
    if missing:
        print(
            f'[run_planning_flat] no plan where the assembly stands now: no move grasp works there for part {", ".join(missing)} '
            f'(why: the lines "[generate_grasps] part ...: no usable move grasp" in {log_file})'
        )
        return False
    if grasps_only:
        print(
            f'[run_planning_flat] grasps generated: {grasps_path} (--grasps-only: run again without it for the plan)'
        )
        return True

    if (
        run(
            [
                py,
                'planning/run_pickup_gen.py',
                '--assembly-dir',
                assembly_dir,
                '--log-dir',
                log_dir,
                '--filter',
            ],
            log_file,
        )
        != 0
    ):
        return False
    if (
        run(
            [
                py,
                'planning/run_seq_plan.py',
                '--assembly-dir',
                assembly_dir,
                '--log-dir',
                log_dir,
                '--plot',
            ],
            log_file,
        )
        != 0
    ):
        return False

    banned, banned_spots = [], []
    banned_hold = []  # (part, hold grasp id) dropped by the loop
    clear_banned_spots(
        log_dir
    )  # spots banned by an earlier run belong to its sequences
    new_sequence = True
    for attempt in range(1, MAX_SEQUENCE_ATTEMPTS + 1):
        print(
            f'[run_planning_flat] sequence attempt {attempt}/{MAX_SEQUENCE_ATTEMPTS}'
            + (f', without move grasps {banned}' if banned else '')
            + (
                f', without hold grasps {banned_hold}'
                if banned_hold
                else ''
            )
            + (
                f', without the spots (part, grasp, x, y) {banned_spots}'
                if banned_spots
                else ''
            )
        )
        if (
            new_sequence
            and run(
                [
                    py,
                    'planning/run_pickup_gen.py',
                    '--assembly-dir',
                    assembly_dir,
                    '--log-dir',
                    log_dir,
                    '--seq-opt',
                    '--plot',
                ],
                log_file,
            )
            != 0
        ):
            print('[run_planning_flat] no sequence left')
            return False
        failure_path = os.path.join(log_dir, FAILURE_FILE)
        stages = [
            [
                py,
                'planning/run_pickup_gen.py',
                '--assembly-dir',
                assembly_dir,
                '--log-dir',
                log_dir,
            ],
            [
                py,
                'planning/run_motion_plan.py',
                '--assembly-dir',
                assembly_dir,
                '--log-dir',
                log_dir,
                '--optimized',
                '--no-viz',
            ],
        ]
        failed = False
        for cmd in stages:
            if run(cmd, log_file) != 0:
                failure = None
                if os.path.exists(failure_path):
                    with open(failure_path, 'r') as fp:
                        failure = json.load(fp)
                step = next_step(failure)
                if step == 'stop':
                    if failure is None:
                        print(
                            f'[run_planning_flat] {cmd[1]} failed without naming a cause, see {log_file}'
                        )
                    elif failure.get('cause') == CAUSE_INSERTION:
                        print(
                            f'[run_planning_flat] part {failure["part"]} cannot be brought to where its insertion starts, whatever the grasp and wherever it lies: {failure["reason"]}'
                        )
                    else:
                        print(
                            f'[run_planning_flat] {failure}: names no grasp '
                            f'to drop, cannot continue'
                        )
                    return False
                if step == 'ban hold grasp':
                    # The hold arm cannot do its part with this hold
                    # grasp: drop it and take the next best sequence.
                    banned_hold.append((failure['part'], failure['grasp_id']))
                    ban_hold_grasp(
                        log_dir, failure['part'], failure['grasp_id']
                    )
                    new_sequence = True
                    failed = True
                    break
                if (
                    step == 'ban spot'
                ):  # the grasp is fine: only the place where the part lies changes
                    with open(
                        os.path.join(log_dir, 'fixture', 'flat_pickup.json'),
                        'r',
                    ) as fp:
                        info = json.load(fp)[failure['part']]
                    banned_spots = ban_spot(
                        log_dir,
                        failure['part'],
                        failure['grasp_id'],
                        info['spot'],
                    )
                    new_sequence = False
                else:
                    banned.append((failure['part'], failure['grasp_id']))
                    ban_grasp(log_dir, failure['part'], failure['grasp_id'])
                    new_sequence = True
                failed = True
                break
        if failed:
            continue
        if (
            run(
                [
                    py,
                    'planning/verify_plan.py',
                    '--assembly-dir',
                    assembly_dir,
                    '--log-dir',
                    log_dir,
                ],
                log_file,
            )
            != 0
        ):
            print(
                f'[run_planning_flat] the final check failed although every stage passed its checks, see {os.path.join(log_dir, "verify_report.json")}'
            )
            return False
        break
    else:
        print('[run_planning_flat] no clean plan within the sequence attempts')
        return False

    # the poses for real_robot/place_parts.py (the robot shows where each part lies) are a help for laying the parts out, not part
    # of the plan: its joint-space moves between the spots can collide where hands lean far. Then the file is removed, so that
    # those moves cannot be executed, and the parts are laid out with the printed template
    fixture = os.path.join(log_dir, 'fixture')
    place_poses = (
        run(
            [py, 'planning/run_place_poses.py', '--log-dir', log_dir], log_file
        )
        == 0
    )
    if not place_poses and os.path.exists(
        os.path.join(fixture, 'place_poses.json')
    ):
        os.remove(os.path.join(fixture, 'place_poses.json'))

    with open(os.path.join(log_dir, 'workcell.json'), 'w') as fp:
        json.dump(
            {
                **values,
                'planned': datetime.now().isoformat(timespec='seconds'),
                'banned_move_grasps': banned,
                'banned_hold_grasps': banned_hold,
                'banned_spots': banned_spots,
                'place_poses': place_poses,
                'verified': True,
            },
            fp,
            indent=4,
        )

    templates = [
        os.path.join(fixture, f)
        for f in ['pickup_template_a4.pdf', 'assembly_template_a4.pdf']
        if os.path.exists(os.path.join(fixture, f))
    ]
    place_parts = (
        f'python real_robot/place_parts.py --log-dir {log_dir}'
        if place_poses
        else '(real_robot/place_parts.py is not available for this plan: its moves between the spots collide, see the log; lay the parts out with the template)'
    )
    print(f'''
[run_planning_flat] DONE, plan verified: {log_dir}
  print at 100 % (no "fit to page"):  {'  '.join(templates)}
  train the policy (isaacgym env):
    bash learning/preprocessing/prepare_isaac.sh {exp_name}
    cd learning/isaacgymenvs && python train.py task=FabricaFixPlugTaskAssemble task.env.assemblies=["{assembly}"] task.env.numEnvs=1024 max_iterations=200 headless=True experiment={exp_name}
  real robot (isaacgym env + ROS):
    {place_parts}
    python real_robot/run.py --single-arm --residual --fn {log_dir}/motion.pkl --checkpoint-path learning/isaacgymenvs/runs/<run>/nn/{exp_name}.pth
''')
    return True


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--assembly', type=str, required=True)
    parser.add_argument(
        '--assembly-root',
        type=str,
        default='fabrica',
        help='assets/<root>/<assembly>',
    )
    parser.add_argument('--num-proc', type=int, default=12)
    parser.add_argument('--num-proc-grasp', type=int, default=50)
    parser.add_argument('--max-n-grasp', type=int, default=100)
    parser.add_argument(
        '--grasps-only',
        action='store_true',
        help='stop after precedence and grasp generation (the stages that use all cores); run again without it for the rest',
    )
    parser.add_argument(
        '--assembly-center',
        type=float,
        nargs=2,
        default=None,
        metavar=('X', 'Y'),
        help='place of the assembly on the board (cm), for a new log; default: 0 -15. planning/run_workspace_search.py finds a place',
    )
    parser.add_argument(
        '--pickup-center',
        type=float,
        nargs=2,
        default=None,
        metavar=('X', 'Y'),
        help='place on the board (cm) around which the parts are laid out; default: 0 5. Stored in the log',
    )
    args = parser.parse_args()
    sys.exit(
        0
        if run_planning_flat(
            args.exp_name,
            args.assembly,
            args.assembly_root,
            args.num_proc,
            args.num_proc_grasp,
            args.max_n_grasp,
            args.grasps_only,
            args.assembly_center,
            args.pickup_center,
        )
        else 1
    )
