'''
Plan an assembly for the real robot without a fixture, in one go: parts lie flat on the table, the single real arm (move)
picks and inserts them, a human holds the first part (real_robot/run.py --single-arm).

  precedence + grasps (only if the log has none yet, at the current robot height)
  -> grasp filter (flat pickup, virtual walls, finger overlap) -> sequence planning
  -> loop: sequence optimization -> pickup layout -> motion planning -> verify_plan.py
     a stage that cannot find a clean plan names the move grasp that caused it; that grasp is removed and the next best
     sequence is tried (MAX_SEQUENCE_ATTEMPTS)
  -> place poses (real_robot/place_parts.py) -> workcell.json (the real-world values this plan was made with)

Every path of the real arm is checked with planning/plan_checks.py (exact geometry). The plan is only reported as done
when verify_plan.py passes.

python planning/run_planning_flat.py --exp-name exp_human_normal_flat_v3 --assembly human_normal
'''
import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import json
import pickle
import subprocess
from datetime import datetime

from planning.config import (ROBOT_PART_CLEARANCE, GRIPPER_TABLE_CLEARANCE, TRANSFER_PART_CLEARANCE, GRASP_MIN_FINGER_OVERLAP,
                             MAX_PLAN_ATTEMPTS, MAX_SEQUENCE_ATTEMPTS, MIN_SIGMA, MIN_SIGMA_INSERTION, MIN_JOINT_LIMIT_MARGIN)
from planning.plan_checks import FAILURE_FILE
from planning.robot.motion_plan_arm import VIRTUAL_WALL_SAFETY
from planning.robot.workcell import get_panda_base_z, get_panda_desk_z
from planning.run_pickup_gen import PICKUP_SHIFT_Y, MAX_TILT


def run(cmd, log_file):
    print(f'[run_planning_flat] $ {" ".join(cmd)}')
    with open(log_file, 'a') as fp:
        fp.write(f'\n$ {" ".join(cmd)}\n')
        fp.flush()
        proc = subprocess.Popen(cmd, cwd=project_base_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env={**os.environ, 'OMP_NUM_THREADS': '1'})
        for line in proc.stdout:
            fp.write(line)
            if any(tag in line for tag in ['FAILED', 'PASSED', 'rejected', '[run_pickup_gen] part', '[filter_flat_grasps]', '[run_flat_seq_opt]', 'Error', 'Traceback', '[run_place_poses]']):
                print('    ' + line.rstrip())
        return proc.wait()


def workcell_values():
    return {
        'robot_base_z_cm': get_panda_base_z(), 'table_above_mounting_cm': get_panda_desk_z(),
        'pickup_shift_y_cm': PICKUP_SHIFT_Y, 'max_pickup_tilt_deg': MAX_TILT,
        'robot_part_clearance_cm': ROBOT_PART_CLEARANCE, 'gripper_table_clearance_cm': GRIPPER_TABLE_CLEARANCE,
        'transfer_part_clearance_cm': TRANSFER_PART_CLEARANCE, 'grasp_min_finger_overlap_cm': GRASP_MIN_FINGER_OVERLAP,
        'virtual_wall_safety_cm': VIRTUAL_WALL_SAFETY,
        'min_jacobian_sigma': MIN_SIGMA, 'min_jacobian_sigma_insertion': MIN_SIGMA_INSERTION, 'min_joint_limit_margin_rad': MIN_JOINT_LIMIT_MARGIN,
    }


def grasps_base_z(grasps_path):
    with open(grasps_path, 'rb') as fp:
        grasps = pickle.load(fp)
    return {round(float(g[0].arm_pos[2]), 3) for part in grasps['grasps'].values() for g in part['move'][:1] if g[0].arm_pos is not None}


def ban_grasp(log_dir, part, grasp_id):
    '''remove a move grasp (and its pairs) from grasps.pkl, so that sequence optimization picks another one'''
    path = os.path.join(log_dir, 'grasps.pkl')
    with open(path, 'rb') as fp:
        grasps = pickle.load(fp)
    grasps['grasps'][part]['move'] = [g for g in grasps['grasps'][part]['move'] if g[0].grasp_id != grasp_id]
    grasps['grasp_id_pairs'] = {(pm, ph): [(m, h) for m, h in pairs if not (pm == part and m == grasp_id)] for (pm, ph), pairs in grasps['grasp_id_pairs'].items()}
    with open(path, 'wb') as fp:
        pickle.dump(grasps, fp)


def run_planning_flat(exp_name, assembly, assembly_root='fabrica', num_proc=12, num_proc_grasp=50, max_n_grasp=100):
    assembly_dir = os.path.join('assets', assembly_root, assembly)
    log_dir = os.path.join('logs', exp_name, assembly)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'planning.log')
    values = workcell_values()
    print('[run_planning_flat] real-world values (planning/robot/workcell.py, planning/config.py):')
    for k, v in values.items(): print(f'    {k}: {v}')
    py = sys.executable

    grasps_path = os.path.join(log_dir, 'grasps.pkl')
    if not os.path.exists(grasps_path):
        if run([py, 'planning/run_preced_plan.py', '--assembly-dir', assembly_dir, '--log-dir', log_dir, '--num-proc', str(num_proc), '--arm', 'panda'], log_file) != 0: return False
        if run([py, 'planning/run_grasp_arm_gen.py', '--assembly-dir', assembly_dir, '--log-dir', log_dir, '--num-proc', str(num_proc_grasp), '--max-n-grasp', str(max_n_grasp),
                '--arm', 'panda', '--gripper', 'panda', '--ft-sensor', 'none'], log_file) != 0: return False
    base_z = grasps_base_z(os.path.join(log_dir, 'grasps_unfiltered.pkl') if os.path.exists(os.path.join(log_dir, 'grasps_unfiltered.pkl')) else grasps_path)
    if base_z != {round(get_panda_base_z(), 3)}:
        print(f'[run_planning_flat] {log_dir} has grasps for robot base z {sorted(base_z)}, workcell.py has {get_panda_base_z()}: use a new --exp-name')
        return False

    if run([py, 'planning/run_pickup_gen.py', '--assembly-dir', assembly_dir, '--log-dir', log_dir, '--filter'], log_file) != 0: return False
    if run([py, 'planning/run_seq_plan.py', '--assembly-dir', assembly_dir, '--log-dir', log_dir, '--plot'], log_file) != 0: return False

    banned = []
    for attempt in range(1, MAX_SEQUENCE_ATTEMPTS + 1):
        print(f'[run_planning_flat] sequence attempt {attempt}/{MAX_SEQUENCE_ATTEMPTS}' + (f', without move grasps {banned}' if banned else ''))
        if run([py, 'planning/run_pickup_gen.py', '--assembly-dir', assembly_dir, '--log-dir', log_dir, '--seq-opt', '--plot'], log_file) != 0:
            print('[run_planning_flat] no sequence left'); return False
        failure_path = os.path.join(log_dir, FAILURE_FILE)
        stages = [
            [py, 'planning/run_pickup_gen.py', '--assembly-dir', assembly_dir, '--log-dir', log_dir],
            [py, 'planning/run_motion_plan.py', '--assembly-dir', assembly_dir, '--log-dir', log_dir, '--optimized', '--no-viz', '--single-arm'],
        ]
        failed = False
        for cmd in stages:
            if run(cmd, log_file) != 0:
                if not os.path.exists(failure_path):
                    print(f'[run_planning_flat] {cmd[1]} failed without naming a grasp, see {log_file}'); return False
                with open(failure_path, 'r') as fp:
                    failure = json.load(fp)
                if failure['grasp_id'] is None:
                    print(f'[run_planning_flat] {failure}: not caused by a move grasp, cannot continue'); return False
                banned.append((failure['part'], failure['grasp_id']))
                ban_grasp(log_dir, failure['part'], failure['grasp_id'])
                failed = True
                break
        if failed: continue
        if run([py, 'planning/verify_plan.py', '--assembly-dir', assembly_dir, '--log-dir', log_dir], log_file) != 0:
            print(f'[run_planning_flat] the final check failed although every stage passed its checks, see {os.path.join(log_dir, "verify_report.json")}'); return False
        break
    else:
        print('[run_planning_flat] no clean plan within the sequence attempts'); return False

    if run([py, 'planning/run_place_poses.py', '--log-dir', log_dir], log_file) != 0:
        print('[run_planning_flat] place poses for real_robot/place_parts.py collide, see the log'); return False

    with open(os.path.join(log_dir, 'workcell.json'), 'w') as fp:
        json.dump({**values, 'planned': datetime.now().isoformat(timespec='seconds'), 'banned_move_grasps': banned, 'verified': True}, fp, indent=4)

    fixture = os.path.join(log_dir, 'fixture')
    templates = [os.path.join(fixture, f) for f in ['pickup_template_a4.pdf', 'assembly_template_a4.pdf'] if os.path.exists(os.path.join(fixture, f))]
    print(f'''
[run_planning_flat] DONE, plan verified: {log_dir}
  print at 100 % (no "fit to page"):  {'  '.join(templates)}
  train the policy (isaacgym env):
    bash learning/preprocessing/prepare_isaac.sh {exp_name}
    cd learning/isaacgymenvs && python train.py task=FabricaFixPlugTaskAssemble task.env.assemblies=["{assembly}"] task.env.numEnvs=1024 max_iterations=200 headless=True experiment={exp_name}
  real robot (isaacgym env + ROS):
    python real_robot/place_parts.py --log-dir {log_dir}
    python real_robot/run.py --single-arm --residual --fn {log_dir}/motion.pkl --checkpoint-path learning/isaacgymenvs/runs/<run>/nn/{exp_name}.pth
''')
    return True


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--assembly', type=str, required=True)
    parser.add_argument('--assembly-root', type=str, default='fabrica', help='assets/<root>/<assembly>')
    parser.add_argument('--num-proc', type=int, default=12)
    parser.add_argument('--num-proc-grasp', type=int, default=50)
    parser.add_argument('--max-n-grasp', type=int, default=100)
    args = parser.parse_args()
    sys.exit(0 if run_planning_flat(args.exp_name, args.assembly, args.assembly_root, args.num_proc, args.num_proc_grasp, args.max_n_grasp) else 1)
