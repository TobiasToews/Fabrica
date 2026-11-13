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
    """Returns the height of the Panda base relative to the table.

    The surface of the board / table is z = 0, so a robot that is
    mounted below the table has a negative height. In the ICS lab the
    robot mounting is 3.0 cm below the table (measured 2026-11-06).

    For a test plan at another height set the environment variable
    FABRICA_PANDA_BASE_Z (cm, e.g. -4.0) for planning AND for
    real_robot/run.py: every log stores its height (workcell.json), and
    run.py refuses a plan made for another one.

    Returns:
        float: Height of the robot base above the table surface in cm
        (-3.0 unless the environment variable is set).
    """
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


# Name of the environment variable that holds the place of the assembly
# for this process, as "x,y" in cm. use_log_workcell sets it.
ASSEMBLY_CENTER_ENV = 'FABRICA_ASSEMBLY_CENTER'
# Name of the file in a log folder that stores the place of the assembly
# this log is planned for, if that is not the default place.
ASSEMBLY_CENTER_FILE = 'assembly_center.json'


def get_default_assembly_center(arm_type):
    """Returns where the assembly stands if nothing else is set.

    That is 6 board holes (6 x 2.5 cm) from the board centre in the
    direction of -y.

    Args:
        arm_type (str): Type of the arms: 'xarm7', 'panda' or 'ur5e'.
            The place is the same for all three.

    Returns:
        np.ndarray: The place [x, y, z] on the board in cm, [0, -15, 0].

    Raises:
        NotImplementedError: If arm_type is none of the three.
    """
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
    """Makes this process use the places that a log is planned for.

    A log (the folder with the results of one planning run) can be
    planned for another place of the assembly and of the parts than the
    defaults. After this call get_assembly_center returns the place of
    the assembly that the log in log_dir is planned for (the default
    place if the log stores none), and get_pickup_center the place its
    parts are laid out around (None if the log stores none). The two
    places are kept in the environment variables ASSEMBLY_CENTER_ENV
    and PICKUP_CENTER_ENV of this process.

    Call it before anything of the log is loaded or computed.

    Args:
        log_dir (str): Path of the log folder. It is searched for
            ASSEMBLY_CENTER_FILE and PICKUP_CENTER_FILE.

    Returns:
        list[float] | None: The stored place of the assembly [x, y] on
        the board in cm, or None if the log stores none.
    """
    import json

    path = os.path.join(log_dir, PICKUP_CENTER_FILE)
    if os.path.exists(path):
        with open(path, 'r') as fp:
            pickup = [
                float(value) for value in json.load(fp)['pickup_center_cm']
            ]
        os.environ[PICKUP_CENTER_ENV] = f'{pickup[0]},{pickup[1]}'
    else:
        os.environ.pop(PICKUP_CENTER_ENV, None)
    path = os.path.join(log_dir, ASSEMBLY_CENTER_FILE)
    if os.path.exists(path):
        with open(path, 'r') as fp:
            center = [
                float(value) for value in json.load(fp)['assembly_center_cm']
            ]
        os.environ[ASSEMBLY_CENTER_ENV] = f'{center[0]},{center[1]}'
        return center
    os.environ.pop(ASSEMBLY_CENTER_ENV, None)
    return None


# Name of the environment variable that holds the place around which
# the parts are laid out for this process, as "x,y" in cm.
# use_log_workcell sets it.
PICKUP_CENTER_ENV = 'FABRICA_PICKUP_CENTER'
# Name of the file in a log folder that stores that place, if the log
# has one.
PICKUP_CENTER_FILE = 'pickup_center.json'


def get_pickup_center():
    """Returns the place around which the parts of a log are laid out.

    A log stores such a place if it was planned with
    run_planning_flat.py --pickup-center; use_log_workcell reads it.

    Returns:
        np.ndarray | None: The place [x, y] on the board in cm, or None
        if none is set. None stands for the default pickup area of
        planning/run_pickup_gen.py.
    """
    override = os.environ.get(PICKUP_CENTER_ENV)
    if override:
        return np.array([float(value) for value in override.split(',')])
    return None


def set_log_pickup_center(log_dir, center):
    """Stores the place around which the parts of a log are laid out.

    The place is written to PICKUP_CENTER_FILE in the log folder and
    then made the place of this process (use_log_workcell).

    Args:
        log_dir (str): Path of the log folder. If a place is stored,
            the folder is created when it does not exist yet.
        center (array-like | None): The place [x, y] on the board in
            cm. None removes the file: the log then uses the default
            pickup area.

    Returns:
        np.ndarray | None: The place as get_pickup_center returns it
        after the change.
    """
    import json

    path = os.path.join(log_dir, PICKUP_CENTER_FILE)
    if center is None:
        if os.path.exists(path):
            os.remove(path)
    else:
        os.makedirs(log_dir, exist_ok=True)
        with open(path, 'w') as fp:
            json.dump(
                {'pickup_center_cm': [float(center[0]), float(center[1])]},
                fp,
            )
    use_log_workcell(log_dir)
    return get_pickup_center()


def set_log_assembly_center(log_dir, center):
    """Stores the place of the assembly that a new log is planned for.

    The place is written to ASSEMBLY_CENTER_FILE in the log folder and
    then made the place of this process (use_log_workcell).

    Args:
        log_dir (str): Path of the log folder. If a place is stored,
            the folder is created when it does not exist yet.
        center (array-like | None): The place [x, y] on the board in
            cm (further entries are ignored). None or the default
            place removes the file: the log then uses the default.

    Returns:
        list[float] | None: The place [x, y] now stored in the log, or
        None if the log uses the default place.
    """
    import json

    path = os.path.join(log_dir, ASSEMBLY_CENTER_FILE)
    if center is None or np.allclose(
        center[:2], get_default_assembly_center('panda')[:2]
    ):
        if os.path.exists(path):
            os.remove(path)
    else:
        os.makedirs(log_dir, exist_ok=True)
        with open(path, 'w') as fp:
            json.dump(
                {'assembly_center_cm': [float(center[0]), float(center[1])]},
                fp,
            )
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
    """Returns the height of the desk surface above the robot mounting.

    The parts lie on the desk, so the desk surface is the board surface
    z = 0. Seen from the robot base it is at -get_panda_base_z().

    Returns:
        float: Height of the desk surface above the robot base in cm.
    """
    return -get_panda_base_z()


def get_panda_virtual_walls():
    """Returns the virtual walls of franka-interface for the Panda.

    The virtual walls are safety planes of the robot controller
    (franka-interface/include/franka-interface/termination_handler/
    termination_handler.h). The robot stops when the origin of one of
    the frames J2 to J7 (the frames of joints 2 to 7) or of the flange
    gets closer to a wall than the threshold of that frame. The side
    walls (y = +0.65 m and -0.65 m) and the floor (z = -0.05 m) are
    those of the changed franka-interface that FLAT_PIPELINE.md
    describes ("Required change in franka-interface").

    Returns:
        tuple[list, list]: (planes, thresholds).

        planes holds one tuple (axis, offset) per wall, in the robot
        base frame: the wall is the plane where the coordinate axis
        (0 = x, 1 = y, 2 = z) has the value offset in m.

        thresholds holds the smallest allowed distance to a wall in m
        for the frames J2, J3, J4, J5, J6, J7 and the flange, in this
        order.
    """
    # (axis, offset)
    planes = [
        (0, 0.75),
        (1, 0.65),
        (1, -0.65),
        (2, -0.05),
        (2, 1.25),
        (0, -0.46),
    ]
    # J2, J3, J4, J5, J6, J7, flange
    thresholds = [0.11, 0.11, 0.08, 0.08, 0.07, 0.07, 0.1]
    return planes, thresholds
