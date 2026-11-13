"""
Compare a recording of the real robot (real_robot/record_state.py) with the plan it executed (motion.pkl).
Every recorded sample is matched to the nearest planned pose of the move arm (joint space); reported per planned step:
  TCP error (real O_T_EE minus planned TCP, base frame, cm), joint tracking error, and the model check
  (planned TCP model applied to the measured joints vs the TCP the robot reports).

python real_robot/compare_recording.py --rec recordings/run.npz --fn logs/exp_human_normal_flat_v3/human_normal/motion.pkl
"""

import os
import sys

project_base_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)
sys.path.append(project_base_dir)

import argparse
import pickle
import numpy as np

PANDA_DH = [
    (0, 0, 0.333),
    (0, -np.pi / 2, 0),
    (0, np.pi / 2, 0.316),
    (0.0825, np.pi / 2, 0),
    (-0.0825, -np.pi / 2, 0.384),
    (0, np.pi / 2, 0),
    (0.088, np.pi / 2, 0),
]


def tcp(q):
    """panda flange + hand (O_T_EE convention of libfranka with the Franka hand), base frame, m"""

    def dh(a, alpha, d, theta):
        ca, sa, ct, st = (
            np.cos(alpha),
            np.sin(alpha),
            np.cos(theta),
            np.sin(theta),
        )
        return np.array(
            [
                [ct, -st, 0, a],
                [st * ca, ct * ca, -sa, -d * sa],
                [st * sa, ct * sa, ca, d * ca],
                [0, 0, 0, 1],
            ]
        )

    T = np.eye(4)
    for (a, alpha, d), theta in zip(PANDA_DH, q):
        T = T @ dh(a, alpha, d, theta)
    return T @ dh(0, 0, 0.107 + 0.1034, -np.pi / 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rec', type=str, required=True)
    parser.add_argument(
        '--fn', type=str, required=True, help='motion.pkl that was executed'
    )
    parser.add_argument(
        '--steps',
        type=int,
        nargs='*',
        default=None,
        help='only these planned steps (1-based, as run.py prints them)',
    )
    args = parser.parse_args()

    rec = np.load(args.rec)
    q_real, T_real, t = rec['q'], rec['O_T_EE'], rec['t']
    motion = pickle.load(open(args.fn, 'rb'))
    planned = [
        (i + 1, s[4], s[3], np.array(s[2]))
        for i, s in enumerate(motion)
        if s[0] == 'move' and s[1] == 'arm' and len(s[2]) > 0
    ]
    if args.steps:
        planned = [p for p in planned if p[0] in args.steps]
    all_q = np.vstack([p[3] for p in planned])
    owner = np.concatenate([[k] * len(p[3]) for k, p in enumerate(planned)])

    # model check: does the planner's kinematics reproduce what the robot reports from the measured joints?
    model_err = np.array(
        [
            np.linalg.norm(tcp(q)[:3, 3] - T[:3, 3])
            for q, T in zip(q_real[::25], T_real[::25])
        ]
    )
    print(
        f'[compare] {len(t)} samples over {t[-1] - t[0]:.0f} s; model check (planner kinematics vs robot-reported TCP): '
        f'max {model_err.max() * 1000:.1f} mm, mean {model_err.mean() * 1000:.1f} mm'
    )

    # match every sample to the nearest planned pose
    rows = {}
    for q, T in zip(q_real, T_real):
        dist = np.abs(all_q - q).max(axis=1)
        j = int(np.argmin(dist))
        if dist[j] > 0.35:
            continue  # moving between steps / not part of the plan
        k = owner[j]
        T_plan = tcp(all_q[j])
        rows.setdefault(k, []).append(
            (
                dist[j],
                (T[:3, 3] - T_plan[:3, 3]) * 100,
                T[2, 3] * 100,
                T_plan[2, 3] * 100,
            )
        )
    print(
        f'{"step":>5} {"task":10} {"part":>4} {"samples":>7} | {"joint err max [deg]":>19} | {"TCP err x/y/z mean [cm]":>26} | {"TCP z err min/max [cm]":>22} | lowest TCP z real / plan [cm]'
    )
    for k, values in sorted(rows.items()):
        step, task, part, _ = planned[k]
        jerr = np.degrees(max(v[0] for v in values))
        e = np.array([v[1] for v in values])
        zr = min(v[2] for v in values)
        zp = min(v[3] for v in values)
        print(
            f'{step:5d} {task:10} {str(part):>4} {len(values):7d} | {jerr:19.2f} | {e[:, 0].mean():8.2f} {e[:, 1].mean():8.2f} {e[:, 2].mean():8.2f} | {e[:, 2].min():10.2f} {e[:, 2].max():10.2f} | {zr:8.2f} / {zp:.2f}'
        )


if __name__ == '__main__':
    main()
