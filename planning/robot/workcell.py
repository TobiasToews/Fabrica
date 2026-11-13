import os
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
    # height of the robot base relative to the board/table surface (cm); ICS lab: the robot mounting is 3.0 cm below the table (measured 2026-11-06)
    # For a test plan at another height set the environment variable FABRICA_PANDA_BASE_Z (cm, e.g. -4.0) for planning AND for
    # real_robot/run.py: every log stores its height (workcell.json), and run.py refuses a plan made for another one.
    return float(os.environ.get('FABRICA_PANDA_BASE_Z', -3.0))


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


ASSEMBLY_CENTER_ENV = 'FABRICA_ASSEMBLY_CENTER' # "x,y" in cm: the place of the assembly for this process (set by use_log_workcell)
ASSEMBLY_CENTER_FILE = 'assembly_center.json' # in a log folder: the place of the assembly this log is planned for, if not the default


def get_default_assembly_center(arm_type):
    dx = get_board_dx()
    if arm_type in ['xarm7', 'panda', 'ur5e']:
        return np.array([0, -6 * dx, 0])
    raise NotImplementedError


def get_assembly_center(arm_type):
    '''
    Where the assembly stands on the board (cm): by default 6 board holes in front of the board centre. A log can be planned for
    another place (planning/run_workspace_search.py, run_planning_flat.py --assembly-center): it stores the place in
    ASSEMBLY_CENTER_FILE, and every script that works on a log calls use_log_workcell(log_dir) first.
    '''
    override = os.environ.get(ASSEMBLY_CENTER_ENV)
    if override:
        x, y = (float(value) for value in override.split(','))
        return np.array([x, y, 0.0])
    return get_default_assembly_center(arm_type)


def use_log_workcell(log_dir):
    '''
    Makes get_assembly_center return the place of the assembly that the log in log_dir is planned for (the default place if the
    log stores none), and get_pickup_center the place its parts are laid out around (None if the log stores none). Call it
    before anything of the log is loaded or computed. Returns the stored place of the assembly [x, y] or None.
    '''
    import json
    path = os.path.join(log_dir, PICKUP_CENTER_FILE)
    if os.path.exists(path):
        with open(path, 'r') as fp:
            pickup = [float(value) for value in json.load(fp)['pickup_center_cm']]
        os.environ[PICKUP_CENTER_ENV] = f'{pickup[0]},{pickup[1]}'
    else:
        os.environ.pop(PICKUP_CENTER_ENV, None)
    path = os.path.join(log_dir, ASSEMBLY_CENTER_FILE)
    if os.path.exists(path):
        with open(path, 'r') as fp:
            center = [float(value) for value in json.load(fp)['assembly_center_cm']]
        os.environ[ASSEMBLY_CENTER_ENV] = f'{center[0]},{center[1]}'
        return center
    os.environ.pop(ASSEMBLY_CENTER_ENV, None)
    return None


PICKUP_CENTER_ENV = 'FABRICA_PICKUP_CENTER'
PICKUP_CENTER_FILE = 'pickup_center.json'


def get_pickup_center():
    '''
    The place on the board (cm, [x, y]) around which the parts of a log are laid out, if the log stores one (run_planning_flat.py
    --pickup-center; use_log_workcell reads it). None: the default pickup area of planning/run_pickup_gen.py.
    '''
    override = os.environ.get(PICKUP_CENTER_ENV)
    if override:
        return np.array([float(value) for value in override.split(',')])
    return None


def set_log_pickup_center(log_dir, center):
    '''stores the place [x, y] (cm) around which the parts of a log are laid out; None: no file (the default pickup area)'''
    import json
    path = os.path.join(log_dir, PICKUP_CENTER_FILE)
    if center is None:
        if os.path.exists(path): os.remove(path)
    else:
        os.makedirs(log_dir, exist_ok=True)
        with open(path, 'w') as fp:
            json.dump({'pickup_center_cm': [float(center[0]), float(center[1])]}, fp)
    use_log_workcell(log_dir)
    return get_pickup_center()


def set_log_assembly_center(log_dir, center):
    '''stores the place of the assembly [x, y] (cm) a new log is planned for; None or the default place: no file (the log uses the default)'''
    import json
    path = os.path.join(log_dir, ASSEMBLY_CENTER_FILE)
    if center is None or np.allclose(center[:2], get_default_assembly_center('panda')[:2]):
        if os.path.exists(path): os.remove(path)
    else:
        os.makedirs(log_dir, exist_ok=True)
        with open(path, 'w') as fp:
            json.dump({'assembly_center_cm': [float(center[0]), float(center[1])]}, fp)
    return use_log_workcell(log_dir)


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
    # height of the desk surface above the robot mounting (cm); parts lie on it, so it is the board surface z = 0: -get_panda_base_z()
    return -get_panda_base_z()


def get_panda_virtual_walls():
    '''
    franka-interface's virtual walls (franka-interface/include/franka-interface/termination_handler/termination_handler.h),
    in the robot base frame (m): the robot stops when the origin of frame J2..J7 or the flange gets closer to a wall than its threshold
    '''
    planes = [(0, 0.75), (1, 0.65), (1, -0.65), (2, -0.05), (2, 1.25), (0, -0.46)] # (axis, offset)
    thresholds = [0.11, 0.11, 0.08, 0.08, 0.07, 0.07, 0.1] # J2, J3, J4, J5, J6, J7, flange
    return planes, thresholds
