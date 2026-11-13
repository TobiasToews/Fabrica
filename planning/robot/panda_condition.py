"""How well conditioned a pose of the Panda arm is.

Two things make an arm pose risky on the real robot:

* A singularity: a pose in which the arm cannot move its hand in some
  direction any more, for example with the arm fully stretched. Close to
  such a pose a small motion of the hand needs very fast joint motions.
* The joint limits: every joint can only turn within a fixed range.

This module measures how far a pose is from both. The plan checks
(planning/plan_checks.py), the motion planner of the real arm
(planning/robot/motion_plan_arm.py), the grasp filter, the layout and
the searches for a place of the assembly compare the results with the
limits in planning/config.py: MIN_SIGMA (MIN_SIGMA_INSERTION while a
part is inserted) and MIN_JOINT_LIMIT_MARGIN.
"""

import numpy as np

# Lowest and highest allowed angle of the 7 joints in rad.
PANDA_JOINT_LIMITS = (
    np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
    np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]),
)

# Geometry of the arm as Denavit-Hartenberg (DH) parameters: one tuple
# (a, alpha, d) per joint, lengths in m and angles in rad. Together with
# the joint angle they give the pose of each joint frame relative to the
# joint frame before it (see dh() in panda_jacobian).
PANDA_DH = [
    (0, 0, 0.333),
    (0, -np.pi / 2, 0),
    (0, np.pi / 2, 0.316),
    (0.0825, np.pi / 2, 0),
    (-0.0825, -np.pi / 2, 0.384),
    (0, np.pi / 2, 0),
    (0.088, np.pi / 2, 0),
]


def panda_jacobian(q):
    """Computes the Jacobian of the end effector of the Panda.

    The Jacobian says how fast the end effector moves when the joints
    turn: end effector velocity = Jacobian @ joint velocities. Column i
    belongs to joint i + 1 (the joints are numbered 1 to 7). The end
    effector is the flange with the hand on it.

    Args:
        q (array-like): Joint angles in rad. The first 7 are used.

    Returns:
        np.ndarray: The Jacobian in the frame of the robot base, shape
        (6, 7). Rows 0 to 2 give the linear velocity in m/s, rows 3 to 5
        the angular velocity in rad/s, each per rad/s of joint velocity.
    """

    def dh(a, alpha, d, theta):
        """Returns the 4x4 transform from one joint frame to the next.

        This is the DH convention in which the frame is first turned by
        alpha about x and moved by a along x, then turned by theta about
        z and moved by d along z.
        """
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

    # Chain the joint transforms: frames[i] is the pose of the frame of
    # joint i + 1 in the base frame. Its z axis is the axis the joint
    # turns about.
    T, frames = np.eye(4), []
    for (a, alpha, d), theta in zip(PANDA_DH, q):
        T = T @ dh(a, alpha, d, theta)
        frames.append(T.copy())
    # Position of the end effector: 0.107 m (flange) plus 0.1034 m
    # (hand) along the axis of the last joint.
    p = (T @ dh(0, 0, 0.107 + 0.1034, 0))[:3, 3]
    # A joint that turns about the axis z through the point o moves the
    # end effector with the velocity z x (p - o) and turns it about z.
    return np.array(
        [
            np.concatenate([np.cross(F[:3, 2], p - F[:3, 3]), F[:3, 2]])
            for F in frames
        ]
    ).T


def joint_limit_margin(q):
    """Returns how far the arm is from its nearest joint limit.

    Args:
        q (array-like): Joint angles in rad. The first 7 are used.

    Returns:
        tuple[float, int]: The smallest distance of a joint to one of
        its two limits in rad, and the number of that joint (1 to 7).
    """
    q = np.asarray(q)[:7]
    margins = np.minimum(q - PANDA_JOINT_LIMITS[0], PANDA_JOINT_LIMITS[1] - q)
    return margins.min(), int(np.argmin(margins)) + 1


def joint_condition(q):
    """Measures how far a pose is from singularities and joint limits.

    The singular values of the Jacobian say how strongly joint motion
    turns into motion of the end effector in each direction. At a
    singularity the smallest one is 0, so it serves as the distance to
    the nearest singularity.

    Args:
        q (array-like): Joint angles in rad. The first 7 are used.

    Returns:
        tuple[float, float, int]: The smallest singular value of the
        Jacobian (for lengths in m and angles in rad), the smallest
        distance of a joint to its limits in rad, and the number of that
        joint (1 to 7).
    """
    q = np.asarray(q)[:7]
    sigma = np.linalg.svd(panda_jacobian(q), compute_uv=False).min()
    margin, joint = joint_limit_margin(q)
    return sigma, margin, joint
