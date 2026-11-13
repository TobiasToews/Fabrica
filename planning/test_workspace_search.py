"""
Tests of the search for the place of the assembly (planning/run_workspace_search.py) and of the place stored with a log
(planning/robot/workcell.py). No planning run; the arm tests use the real arm model (IK, virtual walls).

python planning/test_workspace_search.py          (fabrica env, about 1 min; exit code 1 if a test fails)
python planning/test_workspace_search.py rating   (only the tests whose name contains one of the words)
"""

import os

os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)
sys.path.append(project_base_dir)

import json
import pickle
import shutil
import tempfile
import time
import numpy as np
import networkx as nx

from planning.config import ASSEMBLY_ZONE_MARGIN
from planning.robot.geometry import (
    transform_gripper_meshes,
    get_gripper_meshes_transforms,
)
from planning.robot.motion_plan_arm import ArmMotionPlanner
from planning.robot.util_grasp import get_gripper_pos_quat
from planning.robot.workcell import (
    get_assembly_center,
    get_default_assembly_center,
    use_log_workcell,
    set_log_assembly_center,
    ASSEMBLY_CENTER_ENV,
    ASSEMBLY_CENTER_FILE,
    get_pickup_center,
    set_log_pickup_center,
    PICKUP_CENTER_ENV,
    PICKUP_CENTER_FILE,
)
from planning.run_workspace_search import (
    shift_precedence,
    candidate_places,
    grasp_passes,
    rate_place,
    rank_places,
    make_planners,
    sample_grasps,
    gripper_free,
    APPROACH_ROOM,
    N_GRASP_SAMPLES,
)

ArmMotionPlanner.VISUALIZE = False
DEFAULT = get_default_assembly_center('panda')

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def top_down_sample(xy_assembled, xy_start, height, closing, move=True):
    """
    a sampled grasp with the hand straight down: the gripper base (at the flange) `height` cm above the table over xy_assembled
    (the fingertips are about 11 cm lower), and over xy_start at the start of the insertion; the fingers close along `closing`
    """
    _, quat = get_gripper_pos_quat(
        'panda',
        np.zeros(3),
        np.array([0, 0, 1.0]),
        np.asarray(closing, dtype=float),
        0.25,
    )
    poses = [
        np.array([xy[0], xy[1], height])
        for xy in ([xy_assembled, xy_start] if move else [xy_assembled])
    ]
    return {
        'move': move,
        'pos': poses,
        'quat': np.asarray(quat),
        'open_ratio': 0.25,
    }


# ------------------------------------------------------------------------------------------- the place stored with a log


@test
def assembly_place_of_a_log(planners):
    assert (
        np.allclose(DEFAULT, [0.0, -15.0, 0.0])
        and ASSEMBLY_CENTER_ENV not in os.environ
    )
    a, b = (
        tempfile.mkdtemp(prefix='test_ws_'),
        tempfile.mkdtemp(prefix='test_ws_'),
    )
    try:
        assert use_log_workcell(a) is None and np.allclose(
            get_assembly_center('panda'), DEFAULT
        )  # a log without a stored place: the default
        assert set_log_assembly_center(a, [7.5, -25.0]) == [7.5, -25.0]
        assert os.path.exists(
            os.path.join(a, ASSEMBLY_CENTER_FILE)
        ) and np.allclose(get_assembly_center('panda'), [7.5, -25.0, 0.0])
        with open(os.path.join(a, ASSEMBLY_CENTER_FILE)) as fp:
            assert json.load(fp) == {'assembly_center_cm': [7.5, -25.0]}
        assert use_log_workcell(b) is None and np.allclose(
            get_assembly_center('panda'), DEFAULT
        )  # working on another log switches back
        assert use_log_workcell(a) == [7.5, -25.0] and np.allclose(
            get_assembly_center('panda'), [7.5, -25.0, 0.0]
        )  # ... and again
        assert set_log_assembly_center(
            a, [0.0, -15.0]
        ) is None and not os.path.exists(
            os.path.join(a, ASSEMBLY_CENTER_FILE)
        )  # the default place needs no file
        assert set_log_assembly_center(b, None) is None and np.allclose(
            get_assembly_center('panda'), DEFAULT
        )
        # the loaders of the pipeline follow the place of the log
        from planning.run_pickup_gen import load_final_part_cfg

        assembly_dir = os.path.join(
            project_base_dir, 'assets', 'fabrica', 'socket_set_small'
        )
        at_default = load_final_part_cfg(assembly_dir, 'panda')
        set_log_assembly_center(a, [5.0, -25.0])
        moved = load_final_part_cfg(assembly_dir, 'panda')
        for part in at_default['mesh']:
            assert np.allclose(
                moved['mesh'][part].bounds - at_default['mesh'][part].bounds,
                [[5.0, -10.0, 0.0]] * 2,
            )
            assert np.allclose(
                moved['pos'][part] - at_default['pos'][part], [5.0, -10.0, 0.0]
            )
    finally:
        shutil.rmtree(a)
        shutil.rmtree(b)
        os.environ.pop(ASSEMBLY_CENTER_ENV, None)


@test
def pickup_place_of_a_log(planners):
    """the place around which the parts are laid out: the default pickup area, or the one stored in the log"""
    from planning.run_pickup_gen import pickup_anchor, PICKUP_SHIFT_Y

    assert PICKUP_CENTER_ENV not in os.environ and get_pickup_center() is None
    assert (
        np.allclose(pickup_anchor('panda'), [0.0, 5.0])
        and PICKUP_SHIFT_Y == -15.0
    )  # 4 board holes (10 cm) - 15 + 10, between the arms
    assert np.allclose(
        pickup_anchor('panda', pickup_shift_y=-10.0, pickup_center_x=2.5),
        [2.5, 10.0],
    )
    a, b = (
        tempfile.mkdtemp(prefix='test_ws_'),
        tempfile.mkdtemp(prefix='test_ws_'),
    )
    try:
        assert np.allclose(
            set_log_pickup_center(a, [12.5, -2.5]), [12.5, -2.5]
        )
        with open(os.path.join(a, PICKUP_CENTER_FILE)) as fp:
            assert json.load(fp) == {'pickup_center_cm': [12.5, -2.5]}
        assert np.allclose(get_pickup_center(), [12.5, -2.5]) and np.allclose(
            pickup_anchor('panda'), [12.5, -2.5]
        )
        assert np.allclose(
            pickup_anchor('panda', pickup_shift_y=-10.0, pickup_center_x=2.5),
            [12.5, -2.5],
        )  # the stored place wins
        use_log_workcell(
            b
        )  # another log, without a stored place: the default again
        assert get_pickup_center() is None and np.allclose(
            pickup_anchor('panda'), [0.0, 5.0]
        )
        use_log_workcell(a)
        assert np.allclose(get_pickup_center(), [12.5, -2.5])
        # independent of the place of the assembly: a log can store one, both or none
        assert use_log_workcell(a) is None and np.allclose(
            get_assembly_center('panda'), DEFAULT
        )
        set_log_assembly_center(a, [15.0, -30.0])
        assert use_log_workcell(a) == [15.0, -30.0] and np.allclose(
            get_pickup_center(), [12.5, -2.5]
        )
        set_log_assembly_center(b, [5.0, -20.0])
        assert (
            use_log_workcell(b) == [5.0, -20.0] and get_pickup_center() is None
        )
        # the values the pipeline records for a plan follow the log
        from planning.run_planning_flat import workcell_values

        assert workcell_values()['pickup_center_cm'] == [
            0.0,
            5.0,
        ] and workcell_values()['assembly_center_cm'] == [5.0, -20.0]
        use_log_workcell(a)
        assert workcell_values()['pickup_center_cm'] == [
            12.5,
            -2.5,
        ] and workcell_values()['assembly_center_cm'] == [15.0, -30.0]
        # removed again
        assert (
            set_log_pickup_center(a, None) is None
            and not os.path.exists(os.path.join(a, PICKUP_CENTER_FILE))
            and np.allclose(pickup_anchor('panda'), [0.0, 5.0])
        )
        assert use_log_workcell(a) == [
            15.0,
            -30.0,
        ]  # the place of the assembly is untouched
    finally:
        shutil.rmtree(a)
        shutil.rmtree(b)
        os.environ.pop(ASSEMBLY_CENTER_ENV, None)
        os.environ.pop(PICKUP_CENTER_ENV, None)
    # a place of the assembly is never offered where the parts are laid out
    final_bounds = np.array(
        [[-4.0, -19.0], [4.0, -11.0]]
    )  # an assembly of 8 x 8 cm at the default place
    from planning.run_pickup_gen import BOARD_HALF_SIZE

    default_anchor, moved_anchor = (
        np.array([0.0, 5.0]),
        np.array([10.0, -20.0]),
    )

    def covers(place, anchor):
        return np.all(
            np.abs(np.asarray(place) - anchor) <= 4.0 + ASSEMBLY_ZONE_MARGIN
        )

    for anchor in [default_anchor, moved_anchor]:
        places = candidate_places(
            final_bounds, DEFAULT, anchor, BOARD_HALF_SIZE
        )
        assert len(places) > 20 and not any(
            covers(place, anchor)
            for place in places
            if not np.allclose(place, DEFAULT[:2])
        ), anchor
    assert any(
        covers(place, moved_anchor)
        for place in candidate_places(
            final_bounds, DEFAULT, default_anchor, BOARD_HALF_SIZE
        )
    )  # ... which depends on that place


@test
def places_for_the_position_test(planners):
    """run_position_test.sample_places: places of the assembly and of the pickup area drawn from the area both arms reach"""
    from planning.run_position_test import (
        sample_places,
        reachable,
        result_of,
        PICKUP_RADIUS,
        MIN_SEPARATION,
        PLACE_STEP,
    )
    from planning.run_pickup_gen import BOARD_HALF_SIZE

    final_bounds = np.array(
        [[-4.0, -19.0], [4.0, -11.0]]
    )  # an assembly of 8 x 8 cm at the default place

    def reach(xy):
        return (
            xy[0] >= -5.0 and abs(xy[1]) <= 30.0
        )  # a made-up reach: right of x = -5, within 30 cm of the middle in y

    asked = []
    places = sample_places(
        final_bounds,
        DEFAULT,
        lambda xy: (asked.append(tuple(xy)), reach(xy))[1],
        n_assembly=6,
        n_pickup=3,
        seed=1,
    )
    assert len(places) == 6 and all(len(p['pickup']) == 3 for p in places)
    assert len(asked) == len(set(asked))  # the reach of a point is asked once
    board = np.array(BOARD_HALF_SIZE) - 1.0

    def on_grid(xy):
        return np.allclose(
            np.asarray(xy) / PLACE_STEP, np.round(np.asarray(xy) / PLACE_STEP)
        )

    for p in places:
        centre = np.array(p['assembly'])
        lower, upper = (
            final_bounds[0] + centre - DEFAULT[:2],
            final_bounds[1] + centre - DEFAULT[:2],
        )
        zone = (lower - ASSEMBLY_ZONE_MARGIN, upper + ASSEMBLY_ZONE_MARGIN)
        assert (
            on_grid(centre)
            and np.all(zone[0] >= -board)
            and np.all(zone[1] <= board)
        )
        assert np.allclose(
            p['footprint_centre'], (lower + upper) / 2
        ) and reach(np.array(p['footprint_centre']))
        corners = [lower, upper, [lower[0], upper[1]], [upper[0], lower[1]]]
        assert p['corners_reachable'] == sum(
            reach(np.array(corner)) for corner in corners
        )
        for k, pickup in enumerate(p['pickup']):
            pickup = np.array(pickup)
            assert (
                on_grid(pickup)
                and reach(pickup)
                and np.all(np.abs(pickup) + PICKUP_RADIUS <= board)
            )
            assert (
                np.linalg.norm(pickup - np.clip(pickup, zone[0], zone[1]))
                >= PICKUP_RADIUS
            )  # the pickup area stays out of the assembly zone
            assert all(
                np.linalg.norm(pickup - np.array(other)) >= MIN_SEPARATION
                for other in p['pickup'][:k]
            )
    for i, p in enumerate(places):
        assert all(
            np.linalg.norm(np.array(p['assembly']) - np.array(q['assembly']))
            >= MIN_SEPARATION
            for q in places[:i]
        )
    assert all(
        p['corners_reachable'] == 4 for p in places
    )  # by default the whole footprint stands in the reach
    # asked for: the middle is enough; then footprints that stick out of the reach are among the places
    loose = sample_places(
        final_bounds,
        DEFAULT,
        reach,
        n_assembly=12,
        n_pickup=1,
        seed=1,
        min_corners=0,
    )
    assert (
        len(loose) == 12
        and any(p['corners_reachable'] < 4 for p in loose)
        and all(reach(np.array(p['footprint_centre'])) for p in loose)
    )
    assert all(
        p['corners_reachable'] >= 2
        for p in sample_places(
            final_bounds,
            DEFAULT,
            reach,
            n_assembly=12,
            n_pickup=1,
            seed=1,
            min_corners=2,
        )
    )
    # the same seed gives the same places, another seed others
    assert (
        sample_places(
            final_bounds, DEFAULT, reach, n_assembly=6, n_pickup=3, seed=1
        )
        == places
    )
    assert (
        sample_places(
            final_bounds, DEFAULT, reach, n_assembly=6, n_pickup=3, seed=2
        )
        != places
    )
    # spare places come after the wanted ones, which stay the same
    spare = sample_places(
        final_bounds,
        DEFAULT,
        reach,
        n_assembly=6,
        n_pickup=3,
        seed=1,
        n_spare=3,
    )
    assert len(spare) == 9 and spare[:6] == places
    assert all(
        np.linalg.norm(np.array(p['assembly']) - np.array(q['assembly']))
        >= MIN_SEPARATION
        for i, p in enumerate(spare)
        for q in spare[:i]
    )
    # places closer together when asked for
    close = sample_places(
        final_bounds,
        DEFAULT,
        reach,
        n_assembly=20,
        n_pickup=1,
        seed=1,
        min_separation=2.5,
    )
    gaps = [
        np.linalg.norm(np.array(p['assembly']) - np.array(q['assembly']))
        for i, p in enumerate(close)
        for q in close[:i]
    ]
    assert min(gaps) >= 2.5 and min(gaps) < MIN_SEPARATION
    # nothing is reachable, or no pickup place fits: no places (it gives up after max_draws points)
    assert (
        sample_places(
            final_bounds,
            DEFAULT,
            lambda xy: False,
            n_assembly=3,
            n_pickup=2,
            max_draws=300,
        )
        == []
    )

    def only_one_point(xy):
        return np.allclose(xy, [10.0, 0.0])

    assert (
        sample_places(
            final_bounds,
            [10.0, 15.0, 0.0],
            only_one_point,
            n_assembly=1,
            n_pickup=1,
            max_draws=3000,
        )
        == []
    )  # the only reachable point is the middle of the assembly itself
    # the reach of the real arms: in front of the move arm yes; left of its virtual wall, behind its base and off the board no
    np.random.seed(0)
    assert reachable((5.0, -20.0), planners) and reachable(
        (0.0, -15.0), planners
    )
    assert (
        not reachable((-20.0, -20.0), planners)
        and not reachable((25.0, 30.0), planners)
        and not reachable((0.0, -70.0), planners)
    )
    # what a planning run left behind
    tmp = tempfile.mkdtemp(prefix='test_ws_')
    try:
        console = os.path.join(tmp, 'console.log')
        with open(console, 'w') as fp:
            fp.write(
                '[run_planning_flat] sequence attempt 1/10\n    [run_motion_plan] FAILED for part 4 (move grasp 47), cause: the move grasp: transport path: x\n'
                "[run_planning_flat] sequence attempt 2/10, without move grasps [('4', 47)]\n[run_planning_flat] no sequence left\n"
            )
        nothing = {
            'verified': False,
            'attempts': 2,
            'banned_grasps': 0,
            'banned_spots': 0,
            'place_poses': None,
            'first_part': None,
        }
        assert result_of(tmp, console) == {
            **nothing,
            'reason': '[run_planning_flat] no sequence left',
        }
        with open(os.path.join(tmp, 'workcell.json'), 'w') as fp:
            json.dump(
                {
                    'verified': True,
                    'banned_move_grasps': [['4', 47]],
                    'banned_spots': [],
                    'place_poses': False,
                },
                fp,
            )
        os.makedirs(os.path.join(tmp, 'fixture'))
        with open(os.path.join(tmp, 'fixture', 'flat_pickup.json'), 'w') as fp:
            json.dump({'0': {'arm': 'place'}, '1': {'arm': 'move'}}, fp)
        assert result_of(tmp, console) == {
            'verified': True,
            'attempts': 2,
            'banned_grasps': 1,
            'banned_spots': 0,
            'place_poses': False,
            'first_part': 'place',
            'reason': None,
        }
        assert (
            result_of(tmp, os.path.join(tmp, 'missing.log'))['attempts'] == 0
        )
    finally:
        shutil.rmtree(tmp)


@test
def place_where_a_part_has_no_grasp(planners):
    """
    a part the move arm has to insert got no move grasp where the assembly stands: the pipeline says so right after the grasp
    generation (no plan at this place), and the position test names the part and what the grasp generation printed about it
    """
    from planning.robot.util_grasp import Grasp
    from planning.robot.workcell import get_panda_base_z
    from planning.run_planning_flat import (
        parts_without_move_grasp,
        run_planning_flat,
    )
    from planning.run_position_test import grasp_state

    path = np.array([[0, 0, 5.0, 0, 0, 0], [0, 0, 0.0, 0, 0, 0]])
    G = nx.DiGraph()
    G.add_node(
        '0', path=None, parts_after=[]
    )  # the base part: nothing is assembled before it, it is held, not inserted
    G.add_node('1', path=path, parts_after=['0'])
    G.add_node('2', path=path, parts_after=['0', '1'])

    def grasp(part):
        g = Grasp(part, 0, np.zeros(3), np.array([1.0, 0, 0, 0]), 0.5)
        g.arm_pos = np.array([0.0, 0.0, get_panda_base_z()])
        return g

    def write(log_dir, moves, graph=None):
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, 'precedence.pkl'), 'wb') as fp:
            pickle.dump(G if graph is None else graph, fp)
        with open(os.path.join(log_dir, 'grasps.pkl'), 'wb') as fp:
            pickle.dump(
                {
                    'grasps': {
                        part: {
                            'move': [[grasp(part)]] * n,
                            'hold': [grasp(part)],
                        }
                        for part, n in moves.items()
                    }
                },
                fp,
            )

    tmp, cwd = tempfile.mkdtemp(prefix='test_ws_'), os.getcwd()
    try:
        os.chdir(
            tmp
        )  # run_planning_flat.py works in logs/<exp>/<assembly> below the folder it is started in
        full, part_2_missing, base_only = (
            os.path.join('logs', name, 'toy')
            for name in ['full', 'missing', 'base']
        )
        write(full, {'0': 0, '1': 3, '2': 1})
        write(part_2_missing, {'0': 2, '1': 3, '2': 0})
        write(base_only, {'0': 0, '1': 0, '2': 0})
        assert (
            parts_without_move_grasp(full) == []
        )  # the base part needs no move grasp
        assert parts_without_move_grasp(part_2_missing) == [
            '2'
        ] and parts_without_move_grasp(base_only) == ['1', '2']
        # the base part is the part with nothing before it, also when the precedence plan gives it a path (plumbers_block, duct)
        with_path = G.copy()
        with_path.nodes['0']['path'] = path
        write(
            os.path.join('logs', 'base_with_path', 'toy'),
            {'0': 0, '1': 3, '2': 1},
            with_path,
        )
        assert (
            parts_without_move_grasp(
                os.path.join('logs', 'base_with_path', 'toy')
            )
            == []
        )
        # two parts with nothing before them: one of them is held first, the other has to be inserted
        two_first = G.copy()
        two_first.nodes['1']['parts_after'] = []
        two_first.nodes['2']['parts_after'] = ['0']
        for name, moves, expected in [
            ('one', {'0': 0, '1': 2, '2': 1}, []),
            ('other', {'0': 2, '1': 0, '2': 1}, []),
            ('both', {'0': 0, '1': 0, '2': 1}, ['0', '1']),
            ('all', {'0': 0, '1': 0, '2': 0}, ['0', '1', '2']),
        ]:
            write(os.path.join('logs', name, 'toy'), moves, two_first)
            assert (
                parts_without_move_grasp(os.path.join('logs', name, 'toy'))
                == expected
            ), name
        console = os.path.join(tmp, 'console.log')
        with open(console, 'w') as fp:
            fp.write(
                '    [generate_grasps] part 1: 10 candidates, 5 feasible move grasps, 3 usable lying flat -> kept 3 move + 5 hold\n'
                '    [generate_grasps] part 2: 10 candidates, 4 feasible move grasps, 0 usable lying flat -> kept 0 move + 5 hold\n'
                '    [generate_grasps] part 2: no usable move grasp; of the 4 feasible move grasps fail: walls 4, singularity / joint limits 1, finger overlap 0, flat pickup 2\n'
            )
        assert grasp_state(full, console) == 'ok'
        assert (
            grasp_state(part_2_missing, console)
            == 'no move grasp for part 2: part 2: no usable move grasp; of the 4 feasible move grasps fail: walls 4, singularity / joint limits 1, finger overlap 0, flat pickup 2'
        )
        assert (
            grasp_state(base_only, os.path.join(tmp, 'missing.log'))
            == 'no move grasp for part 1, 2'
        )
        # the pipeline stops there with the reason; with grasps for every part it goes on (here: only up to the grasps)
        import io
        import contextlib

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert (
                run_planning_flat('missing', 'toy', grasps_only=True) is False
            )
        assert (
            'no plan where the assembly stands now: no move grasp works there for part 2'
            in out.getvalue()
        ), out.getvalue()
        with contextlib.redirect_stdout(io.StringIO()):
            assert run_planning_flat('full', 'toy', grasps_only=True) is True
    finally:
        os.chdir(cwd)
        use_log_workcell(
            tmp
        )  # a folder without stored places: the defaults again
        shutil.rmtree(tmp)


@test
def precedence_moves_with_the_place(planners):
    G = nx.DiGraph()
    G.add_node(
        'base',
        path=None,
        com=np.array([0.0, -15.0, 1.0]),
        parts_after=[],
        parts_before=['A'],
        action=None,
    )
    G.add_node(
        'A',
        path=np.array([[0, -15.0, 0, 0, 0, 0.3], [2.5, -15.0, 0, 0, 0, 0.3]]),
        com=np.array([1.0, -14.0, 3.0]),
        parts_after=['base'],
        parts_before=[],
        action=np.array([1.0, 0, 0]),
    )
    G.add_edge(
        'base',
        'A',
        contact_points=np.array([[0.5, -15.0, 2.0], [-0.5, -15.5, 2.0]]),
    )
    moved = shift_precedence(G, [5.0, -10.0])
    assert np.allclose(
        moved.nodes['A']['path'],
        [[5.0, -25.0, 0, 0, 0, 0.3], [7.5, -25.0, 0, 0, 0, 0.3]],
    )  # positions move, rotations stay
    assert np.allclose(
        moved.nodes['A']['com'], [6.0, -24.0, 3.0]
    ) and np.allclose(moved.nodes['base']['com'], [5.0, -25.0, 1.0])
    assert (
        moved.nodes['base']['path'] is None
        and moved.nodes['A']['parts_after'] == ['base']
        and np.allclose(moved.nodes['A']['action'], [1.0, 0, 0])
    )
    assert np.allclose(
        moved.edges['base', 'A']['contact_points'],
        [[5.5, -25.0, 2.0], [4.5, -25.5, 2.0]],
    )
    assert np.allclose(
        G.nodes['A']['path'][0], [0, -15.0, 0, 0, 0, 0.3]
    ) and np.allclose(
        G.nodes['A']['com'], [1.0, -14.0, 3.0]
    )  # the original is untouched
    assert np.allclose(
        G.edges['base', 'A']['contact_points'][0], [0.5, -15.0, 2.0]
    )
    back = shift_precedence(moved, [-5.0, 10.0])
    assert np.allclose(
        back.nodes['A']['path'], G.nodes['A']['path']
    ) and np.allclose(back.nodes['A']['com'], G.nodes['A']['com'])


@test
def places_to_rate(planners):
    bounds = np.array(
        [[-5.0, -20.0], [5.0, -10.0]]
    )  # a 10 x 10 cm assembly at the default place
    anchor, board = np.array([0.0, 5.0]), (31.25, 46.25)
    places = candidate_places(
        bounds,
        DEFAULT,
        anchor,
        board,
        x_range=(-30.0, 30.0),
        y_range=(-45.0, 10.0),
        step=5.0,
    )
    assert [0.0, -15.0] in places and len(places) == len(
        {tuple(p) for p in places}
    )
    for (
        x,
        y,
    ) in places:  # the zone (assembly + 3 cm) stays 1 cm inside the board and does not cover the pickup anchor
        lower, upper = (
            np.array([x - 5.0, y - 5.0]) - ASSEMBLY_ZONE_MARGIN,
            np.array([x + 5.0, y + 5.0]) + ASSEMBLY_ZONE_MARGIN,
        )
        assert np.all(lower >= -np.array(board) + 1.0 - 1e-9) and np.all(
            upper <= np.array(board) - 1.0 + 1e-9
        ), (x, y)
        assert not (np.all(lower <= anchor) and np.all(upper >= anchor)), (
            x,
            y,
        )
    assert [-25.0, -15.0] not in places and [
        -20.0,
        -15.0,
    ] in places  # x - 8 >= -30.25
    assert (
        [0.0, 0.0] not in places
        and [0.0, 5.0] not in places
        and [0.0, -5.0] in places
    )  # the zone would cover the anchor at y = 5
    assert [0.0, -40.0] not in places and [
        0.0,
        -35.0,
    ] in places  # y - 8 >= -45.25
    # the default place is rated even when the grid misses it
    assert [0.0, -15.0] in candidate_places(
        bounds,
        DEFAULT,
        anchor,
        board,
        x_range=(2.0, 12.0),
        y_range=(-32.0, -22.0),
        step=5.0,
    )


# ------------------------------------------------------------------------------------------------------ rating of a place


def find_hold_case(planners, xy, deltas):
    """
    a hold grasp with the hand straight down over xy that the hold arm reaches with the assembly moved by each of `deltas`.
    The IK of the grasp generation (regularised, from the rest pose) does not reach every hand rotation, so a few are tried.
    """
    for closing in [
        [1.0, 0, 0],
        [0, -1.0, 0],
        [-1.0, 0, 0],
        [0.7071, -0.7071, 0],
        [0, 1.0, 0],
    ]:
        hold = top_down_sample(xy, None, 20.0, closing, move=False)
        if all(
            grasp_passes(hold, delta, planners['hold'], 'panda')
            for delta in deltas
        ):
            return hold
    raise AssertionError(
        f'no top-down hold grasp found over {xy} that the hold arm reaches for the shifts {deltas}'
    )


WALL_CASE = []


def find_wall_case(planners):
    """
    Two move grasps with the hand straight down, inserted along +x. `wall` is like part 4 of human_normal: the flange 5 cm left
    of the default place, 7.5 cm at the start of the insertion, i.e. 2.5 cm from where the virtual wall stops it (x = -10).
    `easy` is the same grasp 10 cm further right. Searched once over a few heights and hand rotations for a pair that the arm
    can do apart from the wall: `wall` 5 cm further right and, without the approach, at the default place; `easy` at both.
    """
    if not WALL_CASE:
        move = planners['move']
        for height in [16.0, 20.0, 24.0]:
            for closing in [
                [0, 1.0, 0],
                [1.0, 0, 0],
                [0.7071, 0.7071, 0],
                [0.7071, -0.7071, 0],
                [0, -1.0, 0],
                [-1.0, 0, 0],
            ]:
                wall = top_down_sample(
                    [-5.0, -15.0], [-7.5, -15.0], height, closing
                )
                easy = dict(
                    wall,
                    pos=[p + np.array([10.0, 0.0, 0.0]) for p in wall['pos']],
                )
                if (
                    grasp_passes(wall, [5.0, 0.0], move, 'panda')
                    and grasp_passes(
                        wall, [0.0, 0.0], move, 'panda', approach_room=0.0
                    )
                    and grasp_passes(easy, [0.0, 0.0], move, 'panda')
                    and grasp_passes(easy, [5.0, 0.0], move, 'panda')
                ):
                    WALL_CASE.extend([wall, easy])
                    break
            if WALL_CASE:
                break
        assert WALL_CASE, (
            'no top-down grasp found that the move arm can do next to the default place'
        )
    return WALL_CASE


@test
def rating_needs_room_behind_the_insertion(planners):
    """the wall left of the move arm: a grasp that reaches its insertion start but cannot back off 3 cm from it fails; 5 cm further right it passes"""
    sample, _ = find_wall_case(planners)
    move = planners['move']
    assert grasp_passes(sample, [5.0, 0.0], move, 'panda')
    assert grasp_passes(
        sample, [0.0, 0.0], move, 'panda', approach_room=0.0
    )  # the two poses of the insertion are fine at the default place
    assert not grasp_passes(
        sample, [0.0, 0.0], move, 'panda'
    )  # ... the straight approach behind them is not (APPROACH_ROOM)
    assert APPROACH_ROOM == 3.0
    assert grasp_passes(
        sample, [0.0, 0.0], move, 'panda', approach_room=0.5
    )  # 0.5 cm of room exist (flange 2.5 cm from the limit, 1 cm safety, minus 0.5)
    # further left the insertion start itself is outside the walls
    assert not grasp_passes(
        sample, [-3.0, 0.0], move, 'panda', approach_room=0.0
    )
    # out of reach: no IK
    assert not grasp_passes(
        sample, [-70.0, -60.0], move, 'panda', approach_room=0.0
    )
    # a hold grasp only needs an IK solution of the hold arm (no walls: the hold arm is a human or a simulated arm)
    hold = find_hold_case(
        planners, [-5.0, -15.0], [[0.0, 0.0], [-10.0, 0.0]]
    )  # 10 cm further left: beyond the wall of the move arm, fine for the hold arm
    assert not grasp_passes(
        dict(hold, move=True), [-10.0, 0.0], move, 'panda', approach_room=0.0
    )  # the same pose as a move grasp: outside the walls
    assert not grasp_passes(
        hold, [150.0, 0.0], planners['hold'], 'panda'
    )  # out of reach of the hold arm


@test
def rating_of_a_place(planners):
    """the share of passing grasps per part; the part with the fewest options is the rating; a part without grasps makes it 0"""
    wall, easy = find_wall_case(
        planners
    )  # easy: the same grasp 10 cm further right, with room enough
    hold = find_hold_case(planners, [0.0, -15.0], [[0.0, 0.0], [5.0, 0.0]])
    samples = {'A': [wall, easy], 'B': [easy, easy, easy], 'base': [hold]}
    at_default = rate_place(samples, [0.0, 0.0], planners, 'panda')
    assert (
        at_default['parts'] == {'A': 0.5, 'B': 1.0, 'base': 1.0}
        and at_default['rating'] == 0.5
        and abs(at_default['mean'] - 2.5 / 3) < 1e-9
    )
    right = rate_place(samples, [5.0, 0.0], planners, 'panda')
    assert (
        right['parts'] == {'A': 1.0, 'B': 1.0, 'base': 1.0}
        and right['rating'] == 1.0
    )
    assert (
        rate_place(samples, [0.0, 0.0], planners, 'panda', approach_room=0.0)[
            'rating'
        ]
        == 1.0
    )
    assert rate_place({**samples, 'C': []}, [5.0, 0.0], planners, 'panda') == {
        'rating': 0.0,
        'mean': 0.75,
        'parts': {'A': 1.0, 'B': 1.0, 'base': 1.0, 'C': 0.0},
    }
    assert rate_place({}, [0.0, 0.0], planners, 'panda') == {
        'rating': 0.0,
        'mean': 0.0,
        'parts': {},
    }
    # ranking: rating first, then the mean, then the place nearest to the default one
    ratings = [
        ([10.0, -25.0], {'rating': 0.5, 'mean': 0.9}),
        ([0.0, -15.0], {'rating': 0.5, 'mean': 0.7}),
        ([5.0, -15.0], {'rating': 0.8, 'mean': 0.8}),
        ([5.0, -20.0], {'rating': 0.5, 'mean': 0.9}),
        ([-5.0, -15.0], {'rating': 0.0, 'mean': 0.3}),
    ]
    assert [place for place, _ in rank_places(ratings, DEFAULT)] == [
        [5.0, -15.0],
        [5.0, -20.0],
        [10.0, -25.0],
        [0.0, -15.0],
        [-5.0, -15.0],
    ]


@test
def sampled_grasps_of_a_real_assembly(planners):
    """the grasps that are tested at every place, for socket_set_small (uses the precedence plan of an earlier run)"""
    log_dir = os.path.join(
        project_base_dir,
        'logs',
        'old_exp_socket_set_small_flat_run2',
        'socket_set_small',
    )
    if not os.path.exists(os.path.join(log_dir, 'precedence.pkl')):
        print(
            '      (skipped: logs/old_exp_socket_set_small_flat_run2 is not on this machine)'
        )
        return
    from planning.run_grasp_arm_gen import GraspArmGenerator
    from planning.flat_grasp import get_flat_rots, get_flat_pickup_candidates

    with open(os.path.join(log_dir, 'precedence.pkl'), 'rb') as fp:
        G = pickle.load(fp)
    np.random.seed(0)
    generator = GraspArmGenerator(
        os.path.join(project_base_dir, 'assets'),
        os.path.join(
            project_base_dir, 'assets', 'fabrica', 'socket_set_small'
        ),
        G,
        'panda',
        'panda',
        {'move': False, 'hold': False},
        0,
        200,
        10,
        0.95,
        'least_squares',
        1.0,
        0.0,
        0.1,
        flat=True,
    )
    base = [n for n in G.nodes if G.nodes[n]['path'] is None]
    assert base == ['0']
    # the flat pipeline has no fixture: the box that keeps gripper and arm out of the fixture's place in the original pipeline
    # is gone (an assembly standing there got no grasp), the table stays. A gripper in the middle of that place is free
    assert sorted(generator.ground_col_manager._objs) == ['ground']
    original = GraspArmGenerator(
        os.path.join(project_base_dir, 'assets'),
        os.path.join(
            project_base_dir, 'assets', 'fabrica', 'socket_set_small'
        ),
        G,
        'panda',
        'panda',
        {'move': False, 'hold': False},
        0,
        200,
        10,
        0.95,
        'least_squares',
        1.0,
        0.0,
        0.1,
    )
    assert sorted(original.ground_col_manager._objs) == ['fixture', 'ground']
    _, quat = get_gripper_pos_quat(
        'panda', np.zeros(3), np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), 0.5
    )
    in_the_box = get_gripper_meshes_transforms(
        'panda',
        generator.gripper_meshes,
        np.array([0.0, 30.0, 20.0]),
        np.asarray(quat),
        np.eye(4),
        0.5,
    )
    for gen, blocked in [(generator, False), (original, True)]:
        gen.apply_transforms_to_col_manager(
            gen.gripper_col_manager_buffered, in_the_box
        )
        assert (
            gen.gripper_col_manager_buffered.in_collision_other(
                gen.ground_col_manager
            )
            == blocked
        )
    samples = {
        part: sample_grasps(generator, part, n=8)
        for part in generator.part_ids
    }
    for part, part_samples in samples.items():
        assert 0 < len(part_samples) <= 8, part
        for s in part_samples:
            assert (
                s['move'] == (part not in base)
                and len(s['pos']) == (1 if part in base else 3)
                and abs(np.linalg.norm(s['quat']) - 1) < 1e-9
            )
            if s[
                'move'
            ]:  # the gripper follows the part along its insertion path: same offset as the path of the part
                path = G.nodes[part]['path']
                assert np.allclose(
                    s['pos'][-1] - s['pos'][0],
                    path[-1][:3] - path[0][:3],
                    atol=1e-6,
                )
                assert np.linalg.norm(s['pos'][-1] - s['pos'][0]) > 1.0
    # every move sample can pick its part lying flat, and its gripper is free at the assembly; a grasp inside another part is not
    part = [p for p in generator.part_ids if p not in base][0]
    mesh = (
        generator.part_meshes[part]
        .copy()
        .apply_transform(generator.part_final_transforms[part])
    )
    from planning.robot.util_grasp import Grasp
    from assets.transform import get_transform_from_path

    transforms = get_transform_from_path(G.nodes[part]['path'], n_sample=3)
    for s in samples[part]:
        grasp = Grasp(part, 0, s['pos'][0], s['quat'], s['open_ratio'])
        assert (
            len(
                get_flat_pickup_candidates(
                    grasp,
                    mesh,
                    get_flat_rots(mesh, with_fallback=True),
                    'panda',
                    generator.gripper_meshes,
                )
            )
            > 0
        )
        assert gripper_free(generator, grasp, part, transforms)
        lowest = min(
            m.bounds[0, 2]
            for m in transform_gripper_meshes(
                'panda',
                generator.gripper_meshes,
                grasp.pos,
                grasp.quat,
                np.eye(4),
                grasp.open_ratio,
            ).values()
        )
        low = Grasp(
            part,
            0,
            s['pos'][0] - np.array([0, 0, lowest + 1.0]),
            s['quat'],
            s['open_ratio'],
        )  # the same grasp with the gripper 1 cm into the table
        assert not gripper_free(generator, low, part, transforms)
    # the default place of this assembly passes for every part (it has a verified plan there)
    rating = rate_place(samples, [0.0, 0.0], planners, 'panda')
    assert rating['rating'] > 0, rating


if __name__ == '__main__':
    words = sys.argv[1:]
    planners = make_planners('panda')
    np.random.seed(0)
    failed = 0
    for fn in TESTS:
        if words and not any(w in fn.__name__ for w in words):
            continue
        start = time.time()
        try:
            fn(planners)
            print(f'PASS  {fn.__name__}  ({time.time() - start:.1f} s)')
        except Exception as e:
            failed += 1
            import traceback

            print(f'FAIL  {fn.__name__}: {type(e).__name__}: {e}')
            traceback.print_exc(limit=3)
    print(
        f'{"ALL TESTS PASSED" if failed == 0 else f"{failed} TEST(S) FAILED"}'
    )
    sys.exit(1 if failed else 0)
