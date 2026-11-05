import numpy as np


def get_board_dx():
    return 2.5


def get_move_arm_pos(arm_type):
    dx = get_board_dx()
    if arm_type == 'xarm7':
        return np.array([14.5 * dx, 10 * dx, 0])
    elif arm_type == 'panda':
        return np.array([18 * dx, 8 * dx, get_panda_base_z()])
    elif arm_type == 'ur5e':
        return np.array([18 * dx, 10 * dx, 0])
    else:
        raise NotImplementedError


def get_panda_base_z():
    # height of the robot base relative to the board surface (cm); ICS lab: base sits 32 mm below the board
    return -3.7


def get_hold_arm_pos(arm_type):
    dx = get_board_dx()
    if arm_type == 'xarm7':
        return np.array([-14.5 * dx, 10 * dx, 0])
    elif arm_type == 'panda':
        return np.array([-18 * dx, 8 * dx, get_panda_base_z()])
    elif arm_type == 'ur5e':
        return np.array([-18 * dx, 10 * dx, 0])
    else:
        raise NotImplementedError


def get_dual_arm_pos(arm_type):
    return get_move_arm_pos(arm_type), get_hold_arm_pos(arm_type)


def get_single_arm_pos(arm_type):
    return get_move_arm_pos(arm_type)


def get_move_arm_euler():
    return np.array([0, 0, -np.pi / 2])


def get_hold_arm_euler():
    return np.array([0, 0, -np.pi / 2])


def get_dual_arm_euler():
    return get_move_arm_euler(), get_hold_arm_euler()


def get_single_arm_euler():
    return get_move_arm_euler()


def get_move_arm_box(arm_type):
    arm_pos = get_move_arm_pos(arm_type)
    return arm_pos - np.array([100.0, 100.0, 0.0]), arm_pos + np.array([30.0, 50.0, 80.0])


def get_hold_arm_box(arm_type):
    arm_pos = get_hold_arm_pos(arm_type)
    return arm_pos - np.array([30.0, 100.0, 0.0]), arm_pos + np.array([100.0, 50.0, 80.0])


def get_dual_arm_box(arm_type):
    return get_move_arm_box(arm_type), get_hold_arm_box(arm_type)


def get_single_arm_box(arm_type):
    return get_move_arm_box(arm_type)


def get_assembly_center(arm_type):
    dx = get_board_dx()
    if arm_type == 'xarm7':
        return np.array([0, -6 * dx, 0])
    elif arm_type == 'panda':
        return np.array([0, -6 * dx, 0])
    elif arm_type == 'ur5e':
        return np.array([0, -6 * dx, 0])
    else:
        raise NotImplementedError


def get_fixture_min_y(arm_type):
    dx = get_board_dx()
    if arm_type == 'xarm7':
        return 6 * dx
    elif arm_type == 'panda':
        return 4 * dx
    elif arm_type == 'ur5e':
        return 6 * dx
    else:
        raise NotImplementedError


def get_panda_desk_z():
    # height of the desk surface above the robot mounting (cm); the planner's ground (top at the board surface, z = 0) must stay above it
    return 3.5


def get_panda_virtual_walls():
    '''
    franka-interface's virtual walls (franka-interface/include/franka-interface/termination_handler/termination_handler.h),
    in the robot base frame (m): the robot stops when the origin of frame J2..J7 or the flange gets closer to a wall than its threshold
    '''
    planes = [(0, 0.75), (1, 0.65), (1, -0.65), (2, -0.05), (2, 1.25), (0, -0.46)] # (axis, offset)
    thresholds = [0.11, 0.11, 0.08, 0.08, 0.07, 0.07, 0.1] # J2, J3, J4, J5, J6, J7, flange
    return planes, thresholds
