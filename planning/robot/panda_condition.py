'''
How well conditioned a panda arm pose is: distance to singularities and to the joint limits.
Used by the plan checks (planning/plan_checks.py) and by the motion planner of the real arm (planning/robot/motion_plan_arm.py).
'''
import numpy as np

PANDA_JOINT_LIMITS = (np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
                      np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]))
PANDA_DH = [(0, 0, 0.333), (0, -np.pi / 2, 0), (0, np.pi / 2, 0.316), (0.0825, np.pi / 2, 0), (-0.0825, -np.pi / 2, 0.384), (0, np.pi / 2, 0), (0.088, np.pi / 2, 0)]


def panda_jacobian(q):
    '''6x7 geometric Jacobian (m, base frame) of the panda end effector (flange + hand), q: 7 joint angles'''
    def dh(a, alpha, d, theta):
        ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
        return np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1]])
    T, frames = np.eye(4), []
    for (a, alpha, d), theta in zip(PANDA_DH, q):
        T = T @ dh(a, alpha, d, theta)
        frames.append(T.copy())
    p = (T @ dh(0, 0, 0.107 + 0.1034, 0))[:3, 3]
    return np.array([np.concatenate([np.cross(F[:3, 2], p - F[:3, 3]), F[:3, 2]]) for F in frames]).T


def joint_limit_margin(q):
    '''(smallest distance of any joint to its limits [rad], that joint 1..7), q: 7 joint angles'''
    q = np.asarray(q)[:7]
    margins = np.minimum(q - PANDA_JOINT_LIMITS[0], PANDA_JOINT_LIMITS[1] - q)
    return margins.min(), int(np.argmin(margins)) + 1


def joint_condition(q):
    '''(smallest singular value of the Jacobian, smallest distance of any joint to its limits [rad], that joint 1..7)'''
    q = np.asarray(q)[:7]
    sigma = np.linalg.svd(panda_jacobian(q), compute_uv=False).min()
    margin, joint = joint_limit_margin(q)
    return sigma, margin, joint
