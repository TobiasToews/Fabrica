"""Measure how long the flat planning pipeline takes, stage by stage.

The pipeline (planning/run_planning_flat.py) is run unchanged, as a
subprocess. This script only watches it from outside:

* Wall-clock time per stage. The driver prints one line
  "[run_planning_flat] $ <command>" right before it starts a stage.
  Every console line gets a timestamp from a monotonic clock
  (time.monotonic, not affected by clock changes). A stage lasts from
  its start line to the start line of the next stage (or to the end of
  the pipeline). The short work of the driver between two stages
  (reading a result file, deciding the next step) is counted to the
  stage before it.
* CPU time and memory, sampled once per SAMPLE_PERIOD seconds from
  /proc for every process of the pipeline (the driver, the stage
  processes and their worker processes). CPU time per process is
  utime + stime + cutime + cstime: the time of the process itself and
  of its children that have already finished. Summed over all live
  processes this total never drops when a worker exits, so the CPU
  seconds of a stage are the difference of the sum at its end and at
  its start (error at most one sample period at each end). Memory is
  the proportional set size (PSS) from /proc/<pid>/smaps_rollup: pages
  shared by forked workers are split between them, so the sum is the
  memory the pipeline really occupies (plain RSS would count shared
  pages once per worker).
* The whole run is also wrapped in /usr/bin/time -v, which reports the
  total wall time, user and system CPU time of all processes and the
  largest RSS of a single process, as an independent check.
* The rest of the machine: the busy share of all CPU threads from
  /proc/stat. Busy threads minus the threads used by the pipeline is
  the load from other programs, which would distort the timing.

Usage (fabrica env, from the repository root; FABRICA_PANDA_BASE_Z and
the other settings of the pipeline are passed on to it):
    python profiling/profile_planning.py --assemblies beam gamepad
        --summary-dir logs/<folder> [-- <run_planning_flat.py args>]

Output per assembly in logs/<prefix>_<assembly>_<suffix>/profile/:
console.log (timestamped output), samples.csv (resource samples),
time_v.txt (/usr/bin/time -v), profile.json (all results). The batch
summary is written to --summary-dir (summary.csv, summary.md,
environment.json).
"""

import argparse
import csv
import datetime
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)

# Seconds between two resource samples.
SAMPLE_PERIOD = 1.0

# Clock ticks per second of the times in /proc/<pid>/stat.
CLOCK_TICKS = os.sysconf('SC_CLK_TCK')

# The line the driver prints before it starts a stage.
STAGE_LINE = re.compile(r'^\[run_planning_flat\] \$ (.*)$')


def stage_name(command):
    """Gives a short name to a stage from its command line.

    Args:
        command (str): The command the driver printed, e.g. ".../python
            planning/run_pickup_gen.py --log-dir ... --filter".

    Returns:
        str: One of precedence, grasps, grasp_filter, sequence_plan,
        sequence_opt, layout, motion_plan, verify, place_poses, or the
        script name for any other script.
    """
    script = re.search(r'planning/(\w+)\.py', command)
    script = script.group(1) if script else command.split()[0]
    words = command.split()
    if script == 'run_pickup_gen':
        if '--filter' in words:
            return 'grasp_filter'
        if '--seq-opt' in words:
            return 'sequence_opt'
        return 'layout'
    names = {
        'run_preced_plan': 'precedence',
        'run_grasp_arm_gen': 'grasps',
        'run_seq_plan': 'sequence_plan',
        'run_motion_plan': 'motion_plan',
        'verify_plan': 'verify',
        'run_place_poses': 'place_poses',
    }
    return names.get(script, script)


def descendants(root_pid):
    """Finds all processes below one process.

    Args:
        root_pid (int): Process id of the top process.

    Returns:
        list[int]: The root and every process started by it, directly
        or through other processes.
    """
    children = {}
    for entry in os.listdir('/proc'):
        if not entry.isdigit():
            continue
        try:
            with open(f'/proc/{entry}/stat') as fp:
                fields = fp.read().rsplit(')', 1)[1].split()
        except OSError:
            continue  # the process has just exited
        parent = int(fields[1])
        children.setdefault(parent, []).append(int(entry))
    found, todo = [], [root_pid]
    while todo:
        pid = todo.pop()
        found.append(pid)
        todo.extend(children.get(pid, []))
    return found


def process_usage(pid):
    """Reads the CPU time and memory of one process.

    Args:
        pid (int): Process id.

    Returns:
        tuple[float, float] | None: (CPU seconds of the process and of
        its finished children, PSS in MiB), or None if the process is
        gone.
    """
    try:
        with open(f'/proc/{pid}/stat') as fp:
            fields = fp.read().rsplit(')', 1)[1].split()
        # fields[11:15] are utime, stime, cutime, cstime (fields 14 to
        # 17 of proc(5), counted from the state field here).
        cpu = sum(int(v) for v in fields[11:15]) / CLOCK_TICKS
    except (OSError, ValueError, IndexError):
        return None
    pss_kb = 0
    try:
        # A process that has exited but was not yet collected by its
        # parent (zombie) has no memory left and no smaps_rollup.
        with open(f'/proc/{pid}/smaps_rollup') as fp:
            for line in fp:
                if line.startswith('Pss:'):
                    pss_kb = int(line.split()[1])
                    break
    except OSError:
        pass
    return cpu, pss_kb / 1024.0


def machine_busy_threads(previous):
    """Measures how many CPU threads of the machine were busy.

    Args:
        previous (tuple[int, int] | None): (busy, total) jiffies of the
            last call, None at the first call.

    Returns:
        tuple[float | None, tuple[int, int]]: The average number of busy
        hardware threads since the last call (None at the first call),
        and the new (busy, total) jiffies to pass next time.
    """
    with open('/proc/stat') as fp:
        values = [int(v) for v in fp.readline().split()[1:]]
    idle = values[3] + values[4]  # idle + iowait
    total = sum(values[:8])  # without guest times, already in user
    busy = total - idle
    if previous is None:
        return None, (busy, total)
    d_busy, d_total = busy - previous[0], total - previous[1]
    threads = os.cpu_count() or 1
    share = d_busy / d_total if d_total > 0 else 0.0
    return share * threads, (busy, total)


class Sampler(threading.Thread):
    """Samples CPU time and memory of the pipeline in the background.

    Attributes:
        rows (list[dict]): One entry per sample: t (s since start),
            stage, cpu_s (summed CPU seconds so far), pss_mb (summed
            PSS), n_proc (number of processes), machine_busy (busy
            threads of the whole machine), mem_available_mb.
    """

    def __init__(self, root_pid, start, current_stage):
        """Prepares the sampler.

        Args:
            root_pid (int): Process id of /usr/bin/time, the top
                process of the pipeline.
            start (float): time.monotonic() at the start of the run.
            current_stage (callable): Returns the name of the stage
                that runs now.
        """
        super().__init__(daemon=True)
        self.root_pid = root_pid
        self.start_time = start
        self.current_stage = current_stage
        self.rows = []
        self.stop_event = threading.Event()

    def run(self):
        """Takes one sample per SAMPLE_PERIOD until stop() is called."""
        machine = None
        while not self.stop_event.is_set():
            t = time.monotonic() - self.start_time
            cpu, pss, n = 0.0, 0.0, 0
            for pid in descendants(self.root_pid):
                usage = process_usage(pid)
                if usage is None:
                    continue
                cpu += usage[0]
                pss += usage[1]
                n += 1
            busy, machine = machine_busy_threads(machine)
            with open('/proc/meminfo') as fp:
                available = next(
                    int(line.split()[1]) / 1024.0
                    for line in fp
                    if line.startswith('MemAvailable:')
                )
            self.rows.append(
                {
                    't': round(t, 3),
                    'stage': self.current_stage(),
                    'cpu_s': round(cpu, 2),
                    'pss_mb': round(pss, 1),
                    'n_proc': n,
                    'machine_busy': None if busy is None else round(busy, 2),
                    'mem_available_mb': round(available, 1),
                }
            )
            self.stop_event.wait(SAMPLE_PERIOD)

    def stop(self):
        """Ends the sampling after the current sample."""
        self.stop_event.set()


def parse_time_v(path):
    """Reads the report of /usr/bin/time -v.

    Args:
        path (str): The file /usr/bin/time wrote (option -o).

    Returns:
        dict: wall_s, user_s, sys_s (CPU seconds of all processes that
        were waited for) and max_rss_mb (largest RSS of one process).
    """
    result = {}
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if line.startswith('User time (seconds):'):
                result['user_s'] = float(line.split(':')[1])
            elif line.startswith('System time (seconds):'):
                result['sys_s'] = float(line.split(':')[1])
            elif line.startswith('Maximum resident set size (kbytes):'):
                result['max_rss_mb'] = int(line.split(':')[1]) / 1024.0
            elif line.startswith('Elapsed (wall clock) time'):
                clock = line.split('):')[1].strip().split(':')
                seconds = 0.0
                for part in clock:
                    seconds = seconds * 60 + float(part)
                result['wall_s'] = seconds
    return result


def in_run_samples(samples, end):
    """Keeps the samples of the run and makes the CPU sum monotone.

    Samples after the end of the run see no process and are dropped.
    The CPU seconds used so far can never decrease. A sample can still
    show a lower sum than the one before: a process that has exited is
    no longer listed, and its time reaches the sum only when its parent
    collects it (or at the end, through /usr/bin/time). Such a dip is
    an undercount, so the running maximum is used instead.

    Args:
        samples (list[dict]): The rows of Sampler.
        end (float): Time of the end of the run (s since start).

    Returns:
        list[dict]: Copies of the samples with 0 < t < end and at least
        one process, with cpu_s replaced by its running maximum.
    """
    kept, highest = [], 0.0
    for s in samples:
        if not (0.0 < s['t'] < end and s['n_proc'] > 0):
            continue
        highest = max(highest, s['cpu_s'])
        kept.append({**s, 'cpu_s': highest})
    return kept


def cpu_at(t, samples, end, cpu_end):
    """Estimates the CPU seconds the pipeline had used at a given time.

    Between two samples the CPU seconds are interpolated linearly. At
    the start of the run they are 0, at its end they are the exact
    total of /usr/bin/time. Pass the samples through in_run_samples
    first.

    Args:
        t (float): Time in s since the start of the run.
        samples (list[dict]): The rows of Sampler.
        end (float): Time of the end of the run (s since start).
        cpu_end (float): CPU seconds of the whole run (user + system,
            from /usr/bin/time).

    Returns:
        float: CPU seconds used up to time t.
    """
    points = [(0.0, 0.0)]
    points += [(s['t'], s['cpu_s']) for s in samples]
    points.append((end, max(cpu_end, points[-1][1])))
    for (ta, ca), (tb, cb) in zip(points, points[1:]):
        if ta <= t <= tb:
            if tb == ta:
                return cb
            return ca + (cb - ca) * (t - ta) / (tb - ta)
    return cpu_end if t >= end else 0.0


def summarize_stages(stages, samples, end, cpu_end):
    """Combines the stage times with the resource samples.

    Args:
        stages (list[dict]): name, t_start (s since start) and command
            of every stage in the order they ran.
        samples (list[dict]): The rows of Sampler.
        end (float): Time of the end of the run (s since start).
        cpu_end (float): CPU seconds of the whole run (user + system,
            from /usr/bin/time).

    Returns:
        list[dict]: Per stage run: name, attempt (how often this stage
        ran so far, from 1), t_start, duration_s, cpu_s, cores_mean
        (cpu_s / duration_s), pss_peak_mb, other_load_mean (busy
        threads of the machine not used by the pipeline).
    """
    samples = in_run_samples(samples, end)
    threads = os.cpu_count() or 1
    result, count = [], {}
    for i, stage in enumerate(stages):
        t0 = stage['t_start']
        t1 = stages[i + 1]['t_start'] if i + 1 < len(stages) else end
        inside = [s for s in samples if t0 <= s['t'] < t1]
        # CPU seconds of the stage: used up to its end minus used up to
        # its start, both interpolated between the neighbouring samples.
        cpu0 = cpu_at(t0, samples, end, cpu_end)
        cpu1 = cpu_at(t1, samples, end, cpu_end)
        duration = t1 - t0
        busy = [s for s in inside if s['machine_busy'] is not None]
        other = None
        if len(busy) >= 2:
            # Threads the pipeline used between two samples, from the
            # increase of its CPU seconds.
            # A process that finished between two samples is counted at
            # the next one at which its time has reached its parent, so
            # single values are clipped to what the machine can do.
            values = []
            for a, b in zip(busy, busy[1:]):
                own = (b['cpu_s'] - a['cpu_s']) / (b['t'] - a['t'])
                values.append(
                    min(float(threads), max(0.0, b['machine_busy'] - own))
                )
            other = sum(values) / len(values)
        count[stage['name']] = count.get(stage['name'], 0) + 1
        result.append(
            {
                'name': stage['name'],
                'attempt': count[stage['name']],
                't_start': round(t0, 2),
                'duration_s': round(duration, 2),
                'cpu_s': round(cpu1 - cpu0, 1),
                'cores_mean': round((cpu1 - cpu0) / duration, 2)
                if duration > 0
                else None,
                'pss_peak_mb': max((s['pss_mb'] for s in inside), default=0),
                'other_load_mean': None if other is None else round(other, 2),
            }
        )
    return result


def plan_facts(console_lines, log_dir):
    """Collects the facts that explain the planning time.

    Args:
        console_lines (list[str]): The console output of the driver.
        log_dir (str): The log folder of the plan.

    Returns:
        dict: verified (bool), sequence_attempts, failures (the lines
        "FAILED for part ..."), path_rejections (re-plans of a path
        that failed the checks), grasp lines of the grasp generation,
        and n_paths / n_poses of motion.pkl if it exists.
    """
    text = '\n'.join(console_lines)
    facts = {
        'verified': 'DONE, plan verified' in text,
        'sequence_attempts': len(re.findall(r'sequence attempt \d+', text)),
        'failures': [ln.strip() for ln in console_lines if 'FAILED for' in ln],
        'path_rejections': sum('rejected:' in ln for ln in console_lines),
        'grasp_lines': [
            ln.strip()
            for ln in console_lines
            if '[generate_grasps] part' in ln
        ],
    }
    motion_path = os.path.join(log_dir, 'motion.pkl')
    if os.path.exists(motion_path):
        import pickle

        with open(motion_path, 'rb') as fp:
            motion = pickle.load(fp)
        arm_paths = [s for s in motion if s[1] == 'arm']
        facts['n_paths'] = len(arm_paths)
        facts['n_poses'] = sum(len(s[2]) for s in arm_paths)
    return facts


def assembly_size(assembly_dir):
    """Counts the parts of an assembly and the faces of their meshes.

    Args:
        assembly_dir (str): Folder of the assembly with one .obj file
            per part.

    Returns:
        dict: n_parts and n_faces (all parts together).
    """
    import trimesh

    files = sorted(f for f in os.listdir(assembly_dir) if f.endswith('.obj'))
    faces = 0
    for name in files:
        mesh = trimesh.load(os.path.join(assembly_dir, name), force='mesh')
        faces += len(mesh.faces)
    return {'n_parts': len(files), 'n_faces': faces}


def environment():
    """Records the hardware and software the measurement ran on.

    Returns:
        dict: CPU, threads, memory, kernel, OS, Python and package
        versions, git commit and whether planning/ has uncommitted
        changes, and the FABRICA_* environment variables.
    """
    import numpy
    import scipy
    import trimesh
    import networkx

    def first(path, prefix):
        with open(path) as fp:
            for line in fp:
                if line.startswith(prefix):
                    return line.split(':', 1)[1].strip()
        return None

    os_name = None
    if os.path.exists('/etc/os-release'):
        with open('/etc/os-release') as fp:
            for line in fp:
                if line.startswith('PRETTY_NAME='):
                    os_name = line.split('=', 1)[1].strip().strip('"')

    def git(*args):
        return subprocess.run(
            ['git', *args], cwd=PROJECT_DIR, capture_output=True, text=True
        ).stdout.strip()

    return {
        'date': datetime.datetime.now().isoformat(timespec='seconds'),
        'cpu': first('/proc/cpuinfo', 'model name'),
        'hardware_threads': os.cpu_count(),
        'mem_total_mb': int(first('/proc/meminfo', 'MemTotal').split()[0])
        / 1024.0,
        'kernel': platform.release(),
        'os': os_name,
        'python': platform.python_version(),
        'numpy': numpy.__version__,
        'scipy': scipy.__version__,
        'trimesh': trimesh.__version__,
        'networkx': networkx.__version__,
        'git_commit': git('rev-parse', 'HEAD'),
        'planning_uncommitted': git('status', '--porcelain', 'planning'),
        'fabrica_env': {
            k: v for k, v in os.environ.items() if k.startswith('FABRICA_')
        },
        'sample_period_s': SAMPLE_PERIOD,
    }


def other_pipelines_running():
    """Lists planning or training processes that are not ours.

    Returns:
        list[str]: Command lines of running run_planning_flat.py,
        train.py and verify_plan.py processes.
    """
    found = []
    for entry in os.listdir('/proc'):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            with open(f'/proc/{entry}/cmdline', 'rb') as fp:
                cmd = fp.read().replace(b'\0', b' ').decode(errors='replace')
        except OSError:
            continue
        if (
            any(
                key in cmd
                for key in [
                    'run_planning_flat.py',
                    'train.py',
                    'verify_plan.py',
                ]
            )
            and 'profile_planning.py' not in cmd
        ):
            found.append(cmd.strip())
    return found


def profile_one(exp_name, assembly, assembly_root, extra_args):
    """Runs the pipeline for one assembly and measures it.

    Args:
        exp_name (str): Experiment name, the log goes to
            logs/<exp_name>/<assembly>/.
        assembly (str): Name of the assembly in assets/<root>/.
        assembly_root (str): Folder in assets/, e.g. 'fabrica'.
        extra_args (list[str]): More arguments for run_planning_flat.py.

    Returns:
        dict: Everything measured, also written to profile.json.
    """
    log_dir = os.path.join(PROJECT_DIR, 'logs', exp_name, assembly)
    out_dir = os.path.join(log_dir, 'profile')
    os.makedirs(out_dir, exist_ok=True)
    time_path = os.path.join(out_dir, 'time_v.txt')
    cmd = [
        '/usr/bin/time',
        '-v',
        '-o',
        time_path,
        sys.executable,
        '-u',
        'planning/run_planning_flat.py',
        '--exp-name',
        exp_name,
        '--assembly',
        assembly,
        '--assembly-root',
        assembly_root,
        *extra_args,
    ]
    others = other_pipelines_running()
    with open('/proc/loadavg') as fp:
        load_before = fp.read().split()[:3]
    stages = []
    lines = []
    started = datetime.datetime.now().isoformat(timespec='seconds')
    start = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    sampler = Sampler(
        proc.pid,
        start,
        lambda: stages[-1]['name'] if stages else 'driver_setup',
    )
    sampler.start()
    with open(os.path.join(out_dir, 'console.log'), 'w') as fp:
        for line in proc.stdout:
            t = time.monotonic() - start
            match = STAGE_LINE.match(line.strip())
            if match:
                stages.append(
                    {
                        'name': stage_name(match.group(1)),
                        't_start': t,
                        'command': match.group(1),
                    }
                )
            lines.append(line.rstrip())
            fp.write(f'{t:10.3f} {line}')
            fp.flush()
    exit_code = proc.wait()
    end = time.monotonic() - start
    time.sleep(SAMPLE_PERIOD)
    sampler.stop()
    sampler.join()
    with open(os.path.join(out_dir, 'samples.csv'), 'w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(sampler.rows[0].keys()))
        writer.writeheader()
        writer.writerows(sampler.rows)
    # The time before the first stage belongs to the driver itself.
    all_stages = [{'name': 'driver_setup', 't_start': 0.0, 'command': ''}]
    all_stages += stages
    time_v = parse_time_v(time_path)
    cpu_end = time_v.get('user_s', 0.0) + time_v.get('sys_s', 0.0)
    result = {
        'exp_name': exp_name,
        'assembly': assembly,
        'command': ' '.join(cmd[4:]),
        'started': started,
        'exit_code': exit_code,
        'wall_s': round(end, 2),
        'load_avg_before': load_before,
        'other_pipelines_at_start': others,
        'time_v': time_v,
        'stage_starts': all_stages,
        'stages': summarize_stages(all_stages, sampler.rows, end, cpu_end),
        'pss_peak_mb': max(s['pss_mb'] for s in sampler.rows),
        'facts': plan_facts(lines, log_dir),
        'size': assembly_size(
            os.path.join(PROJECT_DIR, 'assets', assembly_root, assembly)
        ),
    }
    with open(os.path.join(out_dir, 'profile.json'), 'w') as fp:
        json.dump(result, fp, indent=2)
    return result


def resummarize(result, log_root):
    """Evaluates a measured run again from its saved raw data.

    The stage start times come from the timestamped console.log, the
    CPU and memory samples from samples.csv and the totals from
    time_v.txt, so a change of the evaluation does not need a new run.

    Args:
        result (dict): The stored result of profile_one.
        log_root (str): The logs/ folder.

    Returns:
        dict: The result with stage_starts, time_v, stages and
        pss_peak_mb computed again.
    """
    out_dir = os.path.join(
        log_root, result['exp_name'], result['assembly'], 'profile'
    )
    stages = [{'name': 'driver_setup', 't_start': 0.0, 'command': ''}]
    with open(os.path.join(out_dir, 'console.log')) as fp:
        for line in fp:
            t, _, text = line.strip().partition(' ')
            match = STAGE_LINE.match(text.strip())
            if match:
                stages.append(
                    {
                        'name': stage_name(match.group(1)),
                        't_start': float(t),
                        'command': match.group(1),
                    }
                )
    with open(os.path.join(out_dir, 'samples.csv')) as fp:
        samples = []
        for row in csv.DictReader(fp):
            busy = row['machine_busy']
            samples.append(
                {
                    't': float(row['t']),
                    'stage': row['stage'],
                    'cpu_s': float(row['cpu_s']),
                    'pss_mb': float(row['pss_mb']),
                    'n_proc': int(row['n_proc']),
                    'machine_busy': float(busy) if busy else None,
                    'mem_available_mb': float(row['mem_available_mb']),
                }
            )
    time_v = parse_time_v(os.path.join(out_dir, 'time_v.txt'))
    cpu_end = time_v.get('user_s', 0.0) + time_v.get('sys_s', 0.0)
    result = dict(result)
    result['time_v'] = time_v
    result['stage_starts'] = stages
    result['stages'] = summarize_stages(
        stages, samples, result['wall_s'], cpu_end
    )
    result['pss_peak_mb'] = max(s['pss_mb'] for s in samples)
    with open(os.path.join(out_dir, 'profile.json'), 'w') as fp:
        json.dump(result, fp, indent=2)
    return result


STAGE_ORDER = [
    'driver_setup',
    'precedence',
    'grasps',
    'grasp_filter',
    'sequence_plan',
    'sequence_opt',
    'layout',
    'motion_plan',
    'verify',
    'place_poses',
]


def write_summary(results, summary_dir):
    """Writes the table of all assemblies measured so far.

    Stages that ran more than once (the loop of the pipeline) are added
    up per assembly.

    Args:
        results (list[dict]): The results of profile_one.
        summary_dir (str): Folder for summary.csv and summary.md.
    """
    rows = []
    for r in results:
        row = {
            'assembly': r['assembly'],
            'parts': r['size']['n_parts'],
            'faces': r['size']['n_faces'],
            'verified': r['facts']['verified'],
            'attempts': r['facts']['sequence_attempts'],
            'rejections': r['facts']['path_rejections'],
            'total_min': round(r['wall_s'] / 60, 1),
            'cpu_h': round(
                (r['time_v'].get('user_s', 0) + r['time_v'].get('sys_s', 0))
                / 3600,
                2,
            ),
            'pss_peak_gib': round(r['pss_peak_mb'] / 1024, 2),
        }
        for name in STAGE_ORDER:
            seconds = sum(
                s['duration_s'] for s in r['stages'] if s['name'] == name
            )
            row[f'{name}_min'] = round(seconds / 60, 2)
        rows.append(row)
    with open(os.path.join(summary_dir, 'summary.csv'), 'w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    keys = list(rows[0].keys())
    with open(os.path.join(summary_dir, 'summary.md'), 'w') as fp:
        fp.write('| ' + ' | '.join(keys) + ' |\n')
        fp.write('|' + '---|' * len(keys) + '\n')
        for row in rows:
            fp.write('| ' + ' | '.join(str(row[k]) for k in keys) + ' |\n')
    write_report(results, summary_dir)


def md_table(header, rows):
    """Formats a Markdown table.

    Args:
        header (list[str]): Column titles.
        rows (list[list]): One list of cell values per row.

    Returns:
        str: The table, one line per row.
    """
    lines = ['| ' + ' | '.join(header) + ' |']
    lines.append('|' + '---|' * len(header))
    lines += ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows]
    return '\n'.join(lines) + '\n'


def stage_seconds(result, names, key='duration_s'):
    """Adds up a value over all runs of some stages.

    Args:
        result (dict): A result of profile_one.
        names (list[str]): Stage names.
        key (str): 'duration_s' or 'cpu_s'.

    Returns:
        float: The sum over every run of these stages.
    """
    return sum(s[key] for s in result['stages'] if s['name'] in names)


def last_driver_line(result, log_root):
    """Finds the last message of the driver, which says how it ended.

    Args:
        result (dict): A result of profile_one.
        log_root (str): The logs/ folder.

    Returns:
        str: The last console line that starts with [run_planning_flat]
        and is not a stage command, without its timestamp.
    """
    path = os.path.join(
        log_root,
        result['exp_name'],
        result['assembly'],
        'profile',
        'console.log',
    )
    last = ''
    if os.path.exists(path):
        with open(path) as fp:
            for line in fp:
                text = line.strip().partition(' ')[2].strip()
                if text.startswith('[run_planning_flat]') and (
                    not STAGE_LINE.match(text)
                ):
                    last = text
    return last


def write_report(results, summary_dir):
    """Writes report.md: setup, checks and tables of all runs.

    The report is made from the stored results only, so it can be
    written again at any time (--resummarize). Interpretation is not
    part of it; that belongs in a README.md written by a person.

    Args:
        results (list[dict]): The results of profile_one.
        summary_dir (str): Folder of the batch; environment.json is read
            from there and report.md is written there.
    """
    log_root = os.path.join(PROJECT_DIR, 'logs')
    env_path = os.path.join(summary_dir, 'environment.json')
    records = []
    if os.path.exists(env_path):
        with open(env_path) as fp:
            records = json.load(fp)
        if isinstance(records, dict):
            records = [records]
    out = [f'# Planning time profile: {os.path.basename(summary_dir)}\n']
    out.append(
        'Generated by `profiling/profile_planning.py` from `results.json` '
        '(method: see the docstring of that script). Times in minutes '
        'unless stated; one run per assembly.\n'
    )
    out.append('## Setup\n')
    for i, env in enumerate(records):
        changes = env['planning_uncommitted'] or 'none'
        out.append(
            f'Call {i + 1} ({env["date"]}): {env["cpu"]}, '
            f'{env["hardware_threads"]} hardware threads, '
            f'{env["mem_total_mb"] / 1024:.1f} GiB RAM, {env["os"]}, '
            f'kernel {env["kernel"]}, Python {env["python"]}, numpy '
            f'{env["numpy"]}, scipy {env["scipy"]}, trimesh '
            f'{env["trimesh"]}, networkx {env["networkx"]}; commit '
            f'`{env["git_commit"][:7]}`, uncommitted changes in '
            f'planning/: {changes}; environment {env["fabrica_env"]}; '
            f'sample period {env["sample_period_s"]} s.\n'
        )
    if results:
        out.append(
            'Command of the pipeline (first run; the others differ only '
            f'in the assembly): `{results[0]["command"]}`\n'
        )
    out.append('## Runs\n')
    rows = []
    for r in results:
        wall = r['wall_s']
        tv = r['time_v']
        cpu = tv.get('user_s', 0.0) + tv.get('sys_s', 0.0)
        grasps = stage_seconds(r, ['grasps'])
        loop = stage_seconds(r, ['sequence_opt', 'layout', 'motion_plan'])
        verify = stage_seconds(r, ['verify'])
        rows.append(
            [
                r['assembly'],
                r['started'],
                'yes' if r['facts']['verified'] else 'no',
                r['size']['n_parts'],
                r['size']['n_faces'],
                r['facts']['sequence_attempts'],
                r['facts']['path_rejections'],
                f'{wall / 60:.1f}',
                f'{grasps / 60:.1f} ({grasps / wall:.0%})',
                f'{loop / 60:.1f} ({loop / wall:.0%})',
                f'{verify / 60:.1f} ({verify / wall:.0%})',
                f'{cpu / 3600:.2f}',
                f'{cpu / wall:.1f}',
                f'{r["pss_peak_mb"] / 1024:.1f}',
            ]
        )
    out.append(
        md_table(
            [
                'assembly',
                'started',
                'verified',
                'parts',
                'faces',
                'attempts',
                'path re-plans',
                'total',
                'grasp generation (share)',
                'loop (share)',
                'final check (share)',
                'CPU [h]',
                'mean cores',
                'peak PSS [GiB]',
            ],
            rows,
        )
    )
    out.append(
        '\nloop = sequence optimization + layout + motion planning over '
        'all attempts; mean cores = CPU time / wall time.\n'
    )
    out.append('## Wall time per stage (minutes, all attempts added)\n')
    rows = [
        [r['assembly']]
        + [f'{stage_seconds(r, [n]) / 60:.2f}' for n in STAGE_ORDER]
        for r in results
    ]
    out.append(md_table(['assembly'] + STAGE_ORDER, rows))
    out.append('\n## Mean cores per stage (CPU time / wall time)\n')
    rows = []
    for r in results:
        row = [r['assembly']]
        for n in STAGE_ORDER:
            d = stage_seconds(r, [n])
            row.append(
                f'{stage_seconds(r, [n], "cpu_s") / d:.1f}' if d else ''
            )
        rows.append(row)
    out.append(md_table(['assembly'] + STAGE_ORDER, rows))
    out.append('\n## Checks of the measurement\n')
    rows = []
    for r in results:
        tv = r['time_v']
        cpu = tv.get('user_s', 0.0) + tv.get('sys_s', 0.0)
        loads = [
            s['other_load_mean']
            for s in r['stages']
            if s['other_load_mean'] is not None
        ]
        rows.append(
            [
                r['assembly'],
                f'{stage_seconds(r, STAGE_ORDER, "cpu_s"):.1f}',
                f'{cpu:.1f}',
                f'{stage_seconds(r, STAGE_ORDER):.1f}',
                f'{tv.get("wall_s", 0.0):.1f}',
                f'{tv.get("sys_s", 0.0) / cpu:.0%}' if cpu else '',
                f'{max(loads):.2f}' if loads else '',
                len(r['other_pipelines_at_start']),
                r['load_avg_before'][0],
            ]
        )
    out.append(
        md_table(
            [
                'assembly',
                'CPU sum of stages [s]',
                'CPU /usr/bin/time [s]',
                'sum of stage durations [s]',
                'wall /usr/bin/time [s]',
                'system share of CPU',
                'max other load [threads]',
                'other pipelines at start',
                'load avg 1 min at start',
            ],
            rows,
        )
    )
    out.append('\n## How runs without a plan ended\n')
    for r in results:
        if r['facts']['verified']:
            continue
        out.append(f'### {r["assembly"]}\n')
        out.append(f'Last message: `{last_driver_line(r, log_root)}`\n')
        counts = {}
        for line in r['facts']['failures']:
            key = re.sub(r'\(move grasp \d+\)|\(hold grasp \d+\)', '', line)
            key = re.sub(r'\(poses .*?\)', '', key).replace(' ,', ',')
            counts[key] = counts.get(key, 0) + 1
        for key, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            out.append(f'- {n}x `{key.strip()}`')
        out.append('')
    with open(os.path.join(summary_dir, 'report.md'), 'w') as fp:
        fp.write('\n'.join(out) + '\n')


def main():
    """Profiles the pipeline for each assembly, one after another."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--exp-prefix', default='v0.3_exp')
    parser.add_argument('--exp-suffix', default='flat_prof1')
    parser.add_argument('--assembly-root', default='fabrica')
    parser.add_argument('--assemblies', nargs='+', default=[])
    parser.add_argument('--summary-dir', required=True)
    parser.add_argument(
        '--resummarize',
        action='store_true',
        help='no new run: evaluate the runs in --summary-dir again from '
        'their saved raw data',
    )
    parser.add_argument(
        'extra',
        nargs=argparse.REMAINDER,
        help='after --: more arguments for run_planning_flat.py',
    )
    args = parser.parse_args()
    extra = args.extra[1:] if args.extra[:1] == ['--'] else args.extra
    os.makedirs(args.summary_dir, exist_ok=True)
    # Results of earlier calls into the same folder are kept.
    results_path = os.path.join(args.summary_dir, 'results.json')
    results = []
    if os.path.exists(results_path):
        with open(results_path) as fp:
            results = json.load(fp)
    if args.resummarize:
        log_root = os.path.join(PROJECT_DIR, 'logs')
        results = [resummarize(r, log_root) for r in results]
        with open(results_path, 'w') as fp:
            json.dump(results, fp, indent=2)
        write_summary(results, args.summary_dir)
        return
    # One environment record per call: a batch may be run in several
    # calls into the same summary folder.
    env_path = os.path.join(args.summary_dir, 'environment.json')
    records = []
    if os.path.exists(env_path):
        with open(env_path) as fp:
            records = json.load(fp)
        if isinstance(records, dict):  # written by the first version
            records = [records]
    records.append(environment())
    with open(env_path, 'w') as fp:
        json.dump(records, fp, indent=2)
    for assembly in args.assemblies:
        exp_name = f'{args.exp_prefix}_{assembly}_{args.exp_suffix}'
        print(
            f'[profile] {assembly}: start {datetime.datetime.now():%H:%M:%S}'
        )
        result = profile_one(exp_name, assembly, args.assembly_root, extra)
        results.append(result)
        with open(results_path, 'w') as fp:
            json.dump(results, fp, indent=2)
        write_summary(results, args.summary_dir)
        print(
            f'[profile] {assembly}: exit {result["exit_code"]}, '
            f'verified {result["facts"]["verified"]}, '
            f'{result["wall_s"] / 60:.1f} min',
            flush=True,
        )


if __name__ == '__main__':
    main()
