"""Sequential full-assembly evaluation.

Chains the existing single-pair FabricaTaskAssemble insertion policy across every
part of an assembly, in the order produced by the sequence optimizer during
planning (planning/run_seq_opt.py -> SequenceOptimizer.get_sequence()).

This mirrors how the real robot does it (real_robot/run.py): there is no
dedicated multi-part IsaacGym task. The same trained per-pair "specialist"
policy is simply re-invoked once per part, in order, via
`task.env.part_plug=X task.env.part_socket=Y` overrides
(see fabrica_env.py:86-89, 143-153). Everything between insertions (fetching
the next part, regrasping) is not simulated here, exactly as the RL layer of
Fabrica never simulates it either -- on the real robot that's handled by
classical motion planning outside the learned policy.

Part order is read directly off the plan_info pickle written by
learning/preprocessing/prepare_isaac_plan_info.py: its `plan_info` dict is
populated by iterating the already-ordered `sequence` list
(prepare_isaac_plan_info.py:72), and both Python dicts (3.7+) and pickle
preserve insertion order, so `list(plan_info.keys())` recovers the assembly
order without needing to touch the preprocessing script.

Usage:
    python run_sequential_eval.py --assembly duct \
        --checkpoint /path/to/duct_specialist.pth
"""

import argparse
import os
import pickle
import re
import subprocess
import sys

SUCCESS_RE = re.compile(r"Insertion Success:\s*([0-9.]+)")


def load_sequence(plan_info_path):
    """Return the ordered [(part_plug, part_socket), ...] pairs for an assembly."""
    with open(plan_info_path, "rb") as fp:
        plan_info = pickle.load(fp)
    return list(plan_info.keys())


def run_pair(train_py, assembly, part_plug, part_socket, checkpoint, num_envs, gpu):
    """Launch one train.py test=True rollout pinned to a single plug/socket pair."""
    cmd = [
        sys.executable, train_py,
        "task=FabricaTaskAssemble",
        f'task.env.assemblies=["{assembly}"]',
        f"task.env.part_plug={part_plug}",
        f"task.env.part_socket={part_socket}",
        f"task.env.numEnvs={num_envs}",
        "max_iterations=1",
        "headless=True",
        "test=True",
        f"checkpoint={checkpoint}",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    result = subprocess.run(
        cmd, cwd=os.path.dirname(train_py), capture_output=True, text=True, env=env
    )
    matches = SUCCESS_RE.findall(result.stdout)
    output = result.stdout + result.stderr
    if not matches:
        return None, output
    return float(matches[-1]), output


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--assembly", required=True)
    parser.add_argument(
        "--checkpoint", required=True,
        help="Specialist checkpoint trained for this assembly (single-pair policy)",
    )
    parser.add_argument(
        "--plan-info-path", default=None,
        help="Defaults to tasks/fabrica/data/plan_info/{assembly}.pkl",
    )
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--stop-on-failure", action="store_true",
        help="Stop chaining once a part's success rate is 0",
    )
    args = parser.parse_args()

    isaacgymenvs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_py = os.path.join(isaacgymenvs_dir, "train.py")

    plan_info_path = args.plan_info_path or os.path.join(
        isaacgymenvs_dir, "tasks", "fabrica", "data", "plan_info", f"{args.assembly}.pkl"
    )

    sequence = load_sequence(plan_info_path)
    print(f"[sequential_assembly] {args.assembly}: {len(sequence)} parts in order: {sequence}")

    results = []
    for step, (part_plug, part_socket) in enumerate(sequence):
        print(f"[sequential_assembly] step {step + 1}/{len(sequence)}: plug={part_plug} socket={part_socket}")
        success_rate, log = run_pair(
            train_py, args.assembly, part_plug, part_socket,
            args.checkpoint, args.num_envs, args.gpu,
        )
        if success_rate is None:
            print(f"[sequential_assembly] WARNING: no 'Insertion Success' line found in output for step {step + 1}")
            print(log[-2000:])
            success_rate = 0.0
        print(f"[sequential_assembly] step {step + 1} success rate: {success_rate:.3f}")
        results.append({"plug": part_plug, "socket": part_socket, "success_rate": success_rate})
        if args.stop_on_failure and success_rate == 0.0:
            print(f"[sequential_assembly] stopping early: part {step + 1} never succeeds")
            break

    print("\n=== Summary ===")
    for i, r in enumerate(results):
        print(f"  part {i + 1}: plug={r['plug']} socket={r['socket']} success_rate={r['success_rate']:.3f}")

    chained_estimate = 1.0
    for r in results:
        chained_estimate *= r["success_rate"]
    print(f"chained (independent-trials) full-assembly success estimate: {chained_estimate:.4f}")
    print(
        "NOTE: this multiplies independent per-part success *rates*; it is not a "
        "per-trial physical carry-over metric, since each part is evaluated as its "
        "own fresh IsaacGym rollout, not a single continuous assembly."
    )


if __name__ == "__main__":
    main()
