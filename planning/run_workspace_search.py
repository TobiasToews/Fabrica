"""Finds a place for the assembly on the board and plans there.

Without this script the assembly is built at the fixed default place
(0, -15) cm of the board. That place is at the edge of what the real arm
can do: 9 cm to its left the flange meets a virtual wall (a safety plane
of the robot controller that the arm must not cross), and close to the
robot base the arm folds up. A part that is inserted from the far side
can then have no room for its approach (human_normal part 4). This
script searches the region of the board that both arms can work in.

The two arms: the move arm is the real Franka Panda, it picks and
inserts every part. The hold arm is the second robot of the original
Fabrica pipeline, which holds the assembly. Here a human does that, and
the hold arm is only simulated.

Steps:

1. Places: a grid over the board (WORKSPACE_X, WORKSPACE_Y,
   WORKSPACE_STEP) where the assembly zone (the footprint of the
   finished assembly plus a margin) stays on the board and away from
   the pickup area.
2. Rating of every place, without planning (rate_place). For every part
   that is inserted, a sample of grasps is taken that can pick the part
   lying flat and whose gripper is free at the assembled pose. Each
   grasp is tested with the move arm at that place: inverse kinematics
   (IK: the joint angles that put the gripper at a wanted pose), virtual
   walls, singularity (a pose in which the arm cannot move its hand in
   some direction any more) and joint limits at the assembled pose and
   at the start of the insertion, and APPROACH_ROOM behind the insertion
   start for the straight approach. For the part that is held first: IK
   of the hold arm for a sample of grasps. The rating of a place is the
   smallest share of grasps that pass over all parts (the part with the
   fewest options decides).
3. The pipeline (run_planning_flat.py --assembly-center) runs for the
   best rated places, one after the other, until one gives a verified
   plan. Every place gets its own log folder; the log stores its place
   (assembly_center.json).

The pickup positions are searched by the layout itself, over the whole
board (run_pickup_gen.py).

Usage:
    python planning/run_workspace_search.py \
        --exp-name exp_human_normal_ws --assembly human_normal

    Reuse a precedence plan:
    python planning/run_workspace_search.py --exp-name exp_x \
        --assembly duct --precedence-from logs/exp_duct_flat_run1/duct

    Only the rating:
    python planning/run_workspace_search.py --exp-name exp_x \
        --assembly human_normal --rate-only
"""

import os

os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)
sys.path.append(project_base_dir)

import copy
import json
import pickle
import shutil
import subprocess
import numpy as np

from assets.transform import get_transform_from_path, get_pos_quat_from_pose
from planning.config import (
    GRASP_MIN_FINGER_OVERLAP,
    MIN_SIGMA,
    MIN_SIGMA_INSERTION,
    MIN_JOINT_LIMIT_MARGIN,
    STRAIGHT_STEP,
    STRAIGHT_MAX_JOINT_STEP,
    ASSEMBLY_ZONE_MARGIN,
    RETRACT_OPEN_RATIO,
)
from planning.robot.geometry import get_gripper_meshes_transforms
from planning.robot.motion_plan_arm import (
    ArmMotionPlanner,
    VIRTUAL_WALL_SAFETY,
)
from planning.robot.panda_condition import joint_condition
from planning.robot.util_arm import (
    get_arm_chain,
    get_ik_target_orientation,
    get_gripper_pos_quat_from_arm_q,
    get_virtual_wall_margin,
)
from planning.robot.workcell import (
    get_default_assembly_center,
    get_dual_arm_box,
    set_log_assembly_center,
    set_log_pickup_center,
    get_pickup_center,
)
from utils.parallel import parallel_execute

# Range in cm of the places that are rated for the assembly centre, in
# board coordinates (x points towards the move arm).
WORKSPACE_X = (-15.0, 25.0)
WORKSPACE_Y = (-40.0, 0.0)
# Grid spacing of the places in cm.
WORKSPACE_STEP = 5.0
# Number of grasps per part that are tested at every place.
N_GRASP_SAMPLES = 24
# Length in cm behind the start of an insertion that must be free for
# the arm (walls, joint limits, singularity).
APPROACH_ROOM = 3.0
# Number of places the pipeline is run for.
MAX_PLACES = 3


def shift_precedence(G, delta):
    """Returns the precedence plan for the assembly moved on the board.

    The precedence plan says which part must be in place before which,
    and how each part is inserted. This script makes it once for the
    default place and moves it to every other place with this function.
    Everything in the plan that is a position on the board moves with
    the assembly: the insertion paths, the centres of mass and the
    contact points. The graph that is passed in is not changed.

    Args:
        G (networkx.DiGraph): The precedence graph. A node is a part.
            Its 'path' is the insertion path of the part, shape (n, 6):
            per row a position in cm in the board frame and a rotation
            as Euler angles in rad, or None. Its 'com' is the centre of
            mass in cm, shape (3,). An edge can carry 'contact_points'
            in cm, shape (n, 3).
        delta (array-like): How far the assembly is moved in x and y, in
            cm. The first two entries are used.

    Returns:
        networkx.DiGraph: A copy of G with all positions moved by delta.
    """
    shift = np.array([delta[0], delta[1], 0.0])
    G = copy.deepcopy(G)
    for node in G.nodes:
        if G.nodes[node].get('path') is not None:
            # Columns 0 to 2 of a path are the position. The rotation
            # in columns 3 to 5 stays as it is.
            G.nodes[node]['path'][:, :3] += shift
        if G.nodes[node].get('com') is not None:
            G.nodes[node]['com'] = (
                np.asarray(G.nodes[node]['com'], dtype=float) + shift
            )
    for edge in G.edges:
        if (
            'contact_points' in G.edges[edge]
            and len(G.edges[edge]['contact_points']) > 0
        ):
            G.edges[edge]['contact_points'] = (
                np.asarray(G.edges[edge]['contact_points'], dtype=float)
                + shift
            )
    return G


def candidate_places(
    final_bounds,
    default_center,
    anchor,
    board_half_size,
    x_range=WORKSPACE_X,
    y_range=WORKSPACE_Y,
    step=WORKSPACE_STEP,
):
    """Lists the places of the assembly centre that are to be rated.

    The places lie on a grid. A place is kept if the assembly zone (the
    assembly plus ASSEMBLY_ZONE_MARGIN) stays 1 cm inside the board and
    does not cover the pickup anchor (the parts are laid out around it).
    The default place is always part of the list.

    Args:
        final_bounds (np.ndarray): Lower and upper x, y of the assembly
            at the default place in cm, board frame, shape (2, 2).
        default_center (array-like): The default place of the assembly
            centre in cm, board frame. Its first two entries (x, y) are
            used.
        anchor (array-like): x, y of the pickup anchor in cm, board
            frame.
        board_half_size (array-like): Half the size of the board in x
            and y in cm. The origin of the board frame is the board
            centre, so the board reaches from minus to plus this value.
        x_range (tuple[float, float]): First and last x of the grid in
            cm.
        y_range (tuple[float, float]): First and last y of the grid in
            cm.
        step (float): Grid spacing in cm.

    Returns:
        list[list[float]]: The places [x, y] of the assembly centre in
        cm, board frame.
    """
    places = []
    # The added 1e-6 makes np.arange include the last value of a range.
    xs, ys = (
        np.arange(x_range[0], x_range[1] + 1e-6, step),
        np.arange(y_range[0], y_range[1] + 1e-6, step),
    )
    for x in xs:
        for y in ys:
            # The assembly zone with the assembly centre at (x, y).
            delta = np.array([x, y]) - np.asarray(default_center)[:2]
            lower, upper = (
                final_bounds[0] + delta - ASSEMBLY_ZONE_MARGIN,
                final_bounds[1] + delta + ASSEMBLY_ZONE_MARGIN,
            )
            on_board = np.all(
                lower >= -np.asarray(board_half_size) + 1.0
            ) and np.all(upper <= np.asarray(board_half_size) - 1.0)
            covers_anchor = np.all(lower <= anchor) and np.all(upper >= anchor)
            if on_board and not covers_anchor:
                places.append([float(x), float(y)])
    # The default place is rated even if the grid or the tests above
    # leave it out.
    default = [float(default_center[0]), float(default_center[1])]
    if default not in places:
        places.append(default)
    return places


def gripper_free(generator, grasp, part_id, part_transforms):
    """Tests whether the gripper of a move grasp is free at the assembly.

    These are the tests of GraspArmGenerator.check_grasp_feasible that
    do not depend on the place of the assembly: the gripper of a move
    grasp keeps clear of the table, of the part itself and of the parts
    that are assembled before it, at the assembled pose (also with the
    gripper opened) and along the insertion path.

    Args:
        generator (GraspArmGenerator): The grasp generator of the
            assembly (planning/run_grasp_arm_gen.py). It holds the
            meshes and the collision managers; this function moves the
            objects in its collision managers.
        grasp (Grasp): The grasp to test. Its pos (cm) and quat (w, x,
            y, z) are the gripper pose in the board frame for the part
            at its assembled pose.
        part_id (str): The part that is grasped.
        part_transforms (list[np.ndarray]): Poses of the part along its
            insertion path, the assembled pose first. Each is a
            homogeneous 4x4 transform (one matrix for rotation and
            translation), lengths in cm.

    Returns:
        bool: True if the gripper is free in all tested poses.
    """
    # In the precedence graph 'parts_after' of a part lists the parts
    # that are assembled before it: they stand there when it is inserted.
    parts_there = generator.G_preced.nodes[part_id]['parts_after']
    for timestep, part_transform in enumerate(part_transforms):
        # The gripper moves with the part: apply to the grasp the motion
        # that takes the part from its assembled pose to this pose.
        grasp_t = generator.transform_grasp(
            grasp, part_transform @ np.linalg.inv(part_transforms[0])
        )
        # The scene: all parts at their assembled poses, only the
        # grasped part at this pose of its path.
        transforms = generator.part_final_transforms.copy()
        transforms[part_id] = part_transform
        generator.apply_transforms_to_col_manager(
            generator.part_col_manager, transforms
        )
        # At the assembled pose (timestep 0) the gripper is tested as it
        # holds the part and opened by RETRACT_OPEN_RATIO more, on the
        # path only as it holds the part.
        for open_ratio in (
            [
                grasp_t.open_ratio,
                min(grasp_t.open_ratio + RETRACT_OPEN_RATIO, 1.0),
            ]
            if timestep == 0
            else [grasp_t.open_ratio]
        ):
            # The fingers are opened by 0.05 more for the collision
            # tests, as check_grasp_feasible does it (numerical
            # stability of the collision check).
            gripper_transforms = get_gripper_meshes_transforms(
                generator.gripper_type,
                generator.gripper_meshes,
                grasp_t.pos,
                grasp_t.quat,
                np.eye(4),
                min(open_ratio + 0.05, 1.0),
            )
            # Two copies of the gripper get this pose: the exact meshes
            # and the buffered ones (the meshes made larger by a safety
            # distance).
            generator.apply_transforms_to_col_manager(
                generator.gripper_col_manager, gripper_transforms
            )
            generator.apply_transforms_to_col_manager(
                generator.gripper_col_manager_buffered, gripper_transforms
            )
            # The buffered gripper must not touch the ground objects of
            # the generator (in the flat pipeline that is the table).
            if generator.gripper_col_manager_buffered.in_collision_other(
                generator.ground_col_manager
            ):
                return False
            # The exact gripper must not collide with the grasped part.
            # names holds the pairs of objects that collide.
            _, names = generator.gripper_col_manager.in_collision_other(
                generator.part_col_manager, return_names=True
            )
            if any(part_id in pair for pair in names):
                return False
            # The buffered gripper must not touch a part that is already
            # assembled.
            _, names = (
                generator.gripper_col_manager_buffered.in_collision_other(
                    generator.part_col_manager, return_names=True
                )
            )
            if any(other in pair for pair in names for other in parts_there):
                return False
    return True


def sample_grasps(generator, part_id, n=N_GRASP_SAMPLES):
    """Picks the grasps of one part that are tested at every place.

    The candidates are made from antipodal point pairs: two points on
    opposite sides of the part between which the two fingers can clamp
    it. Which candidates are kept depends on the part:

    * A part that is inserted: up to n grasps that can pick it lying
      flat (in a stable resting pose, else in a less stable one), hold
      it deep enough and keep the gripper free at the assembly
      (gripper_free). A resting pose is a pose in which the part lies on
      the board by itself.
    * The part that is held first (it has no insertion path in the
      precedence plan): up to n grasps with contact.

    Args:
        generator (GraspArmGenerator): The grasp generator of the
            assembly, made with flat=True (the finger contact is then
            measured on the gripping pads, which gives the contact
            depths used here).
        part_id (str): The part.
        n (int): Largest number of grasps that are returned.

    Returns:
        list[dict]: One dict per grasp, with the gripper poses for the
        assembly at the default place:

        * 'move' (bool): True for a part that is inserted, False for the
          part that is held first.
        * 'pos' (list[np.ndarray]): The gripper position in cm in the
          board frame, shape (3,), for each pose of the insertion path,
          the assembled pose first. A part that is inserted has three
          poses, the part that is held first only the assembled pose.
        * 'quat' (np.ndarray): The gripper orientation at the assembled
          pose as a quaternion (w, x, y, z).
        * 'open_ratio' (float): How far the gripper is open, as a share
          of its full opening.
    """
    from planning.flat_grasp import (
        PartFaces,
        get_flat_rots,
        get_flat_pickup_candidates,
    )
    from planning.config import FLAT_FALLBACK_POSES

    # The mesh of the part at its assembled pose. The grasps are defined
    # for the part at this pose.
    mesh = (
        generator.part_meshes[part_id]
        .copy()
        .apply_transform(generator.part_final_transforms[part_id])
    )
    # The triangles of the part, for the contact test of the pads.
    generator.part_faces[part_id] = PartFaces(mesh)
    path = generator.G_preced.nodes[part_id]['path']
    candidates = generator.antipodal_candidates(
        part_id, mesh, generator.n_surface_pt, 0
    )
    # Keep the candidates whose fingers touch the part.
    candidates = [
        g
        for g in generator.compute_contact_points_chunk(candidates, part_id)
        if len(g.contact_points) > 0
    ]
    # Held first: any grasp with contact, spread over the candidates.
    if path is None:
        # np.linspace gives up to n evenly spaced positions in the list,
        # cut to whole indices; np.unique removes any index that comes
        # up twice.
        chosen = (
            [
                candidates[i]
                for i in np.unique(
                    np.linspace(
                        0, len(candidates) - 1, min(n, len(candidates))
                    ).astype(int)
                )
            ]
            if len(candidates) > 0
            else []
        )
        return [
            {
                'move': False,
                'pos': [np.asarray(g.pos)],
                'quat': np.asarray(g.quat),
                'open_ratio': g.open_ratio,
            }
            for g in chosen
        ]
    # Three poses of the part on its insertion path, as 4x4 transforms:
    # the first, the middle and the last row of the path. The first is
    # the assembled pose, the last the start of the insertion.
    part_transforms = get_transform_from_path(path, n_sample=3)
    # Deep enough: the part reaches at least GRASP_MIN_FINGER_OVERLAP
    # into each finger (contact_depths has one value per finger).
    deep = [
        g
        for g in candidates
        if min(g.contact_depths) >= GRASP_MIN_FINGER_OVERLAP
    ]
    usable = []
    # First round: the stable resting poses of the part. Second round,
    # only if the first leaves no grasp and FLAT_FALLBACK_POSES is set:
    # the less stable resting poses as well.
    for rots in [get_flat_rots(mesh)] + (
        [get_flat_rots(mesh, with_fallback=True)]
        if FLAT_FALLBACK_POSES
        else []
    ):
        # Grasps that can pick the part from the board in at least one
        # of these resting poses, and of those the ones whose gripper is
        # free at the assembly.
        usable = [
            g
            for g in deep
            if len(
                get_flat_pickup_candidates(
                    g,
                    mesh,
                    rots,
                    generator.gripper_type,
                    generator.gripper_meshes,
                )
            )
            > 0
        ]
        usable = [
            g
            for g in usable
            if gripper_free(generator, g, part_id, part_transforms)
        ]
        if len(usable) > 0:
            break
    # Up to n of the usable grasps, spread evenly over the list (as for
    # the part that is held first).
    chosen = (
        [
            usable[i]
            for i in np.unique(
                np.linspace(0, len(usable) - 1, min(n, len(usable))).astype(
                    int
                )
            )
        ]
        if len(usable) > 0
        else []
    )
    samples = []
    for g in chosen:
        # The gripper pose (position, quaternion) at each of the three
        # poses of the part: the gripper moves with the part.
        poses = [
            get_pos_quat_from_pose(
                g.pos, g.quat, T @ np.linalg.inv(part_transforms[0])
            )
            for T in part_transforms
        ]
        samples.append(
            {
                'move': True,
                'pos': [np.asarray(p) for p, _ in poses],
                'quat': np.asarray(poses[0][1]),
                'open_ratio': g.open_ratio,
            }
        )
    return samples


def grasp_passes(
    sample, delta, planner, gripper_type, approach_room=APPROACH_ROOM
):
    """
    One sampled grasp with the assembly moved by delta (x, y), tested with the arm of `planner`.
    Move grasp: IK at the assembled pose and at the start of the insertion (as the grasp generation solves it), both inside the
    virtual walls and away from singularities and joint limits (the values of the grasp filter), and the arm can move
    approach_room further back from the insertion start in a straight line, inside the walls and the joint-limit and
    singularity rules of a transfer. Hold grasp: IK only.
    """
    chain = planner.arm_chain
    shift = np.array([delta[0], delta[1], 0.0])
    orientation = get_ik_target_orientation(
        chain, gripper_type, sample['quat']
    )
    q_prev, qs = chain.active_to_full(chain.rest_q), []
    for i, pos in enumerate(sample['pos']):
        kwargs = (
            dict(optimizer='least_squares', regularization_parameter=1.0)
            if i == 0
            else dict(optimizer='L-BFGS-B', regularization_parameter=0.0)
        )
        q, success = chain.inverse_kinematics_above_ground(
            target_position=pos + shift,
            target_orientation=orientation,
            orientation_mode='all',
            initial_position=q_prev,
            **kwargs,
        )
        if not success:
            return False
        reached_pos, reached_quat = get_gripper_pos_quat_from_arm_q(
            chain, q, gripper_type
        )
        if not (
            np.allclose(reached_pos, pos + shift, atol=1e-4)
            and np.allclose(reached_quat, sample['quat'], atol=1e-4)
        ):
            return False
        qs.append(q)
        q_prev = q
    if not sample['move']:
        return True
    for q in qs:
        sigma, margin, _ = joint_condition(chain.active_from_full(q))
        if (
            get_virtual_wall_margin(chain, q) < VIRTUAL_WALL_SAFETY
            or sigma < MIN_SIGMA_INSERTION
            or margin < MIN_JOINT_LIMIT_MARGIN
        ):
            return False
    if len(qs) < 2 or approach_room <= 0:
        return True
    # room for the straight approach: back from the insertion start along the insertion line, as ArmMotionPlanner.plan_path_straight_out
    direction = sample['pos'][-1] - sample['pos'][0]
    if np.linalg.norm(direction) < 1e-6:
        return True
    direction = direction / np.linalg.norm(direction)
    fk = chain.forward_kinematics(qs[-1])
    pos, ori, q_prev = fk[:3, 3].copy(), fk[:3, :3].copy(), qs[-1]
    for step in range(1, int(round(approach_room / STRAIGHT_STEP)) + 1):
        q = planner.inverse_kinematics(
            pos + direction * step * STRAIGHT_STEP, ori, q_init=q_prev
        )
        if (
            q is None
            or np.abs(np.asarray(q) - np.asarray(q_prev)).max()
            > STRAIGHT_MAX_JOINT_STEP
        ):
            return False
        sigma, margin, _ = joint_condition(chain.active_from_full(q))
        if (
            get_virtual_wall_margin(chain, q) < VIRTUAL_WALL_SAFETY
            or sigma < MIN_SIGMA
            or margin < MIN_JOINT_LIMIT_MARGIN
        ):
            return False
        q_prev = q
    return True


def rate_place(
    samples, delta, planners, gripper_type, approach_room=APPROACH_ROOM
):
    """
    Rating of a place of the assembly (delta from the default place). samples: {part id: list from sample_grasps}.
    Returns {'rating': smallest share of passing grasps over the parts, 'mean': mean share, 'parts': {part id: share}}; a part
    without any sampled grasp counts as 0 (it has no usable grasp at any place).
    """
    shares = {}
    for part_id, part_samples in samples.items():
        if len(part_samples) == 0:
            shares[part_id] = 0.0
            continue
        passed = sum(
            grasp_passes(
                s,
                delta,
                planners['move' if s['move'] else 'hold'],
                gripper_type,
                approach_room,
            )
            for s in part_samples
        )
        shares[part_id] = passed / len(part_samples)
    values = list(shares.values())
    return {
        'rating': float(min(values)) if values else 0.0,
        'mean': float(np.mean(values)) if values else 0.0,
        'parts': shares,
    }


def rank_places(ratings, default_center):
    """places sorted best first: by rating, then mean share, then distance from the default place (ties keep the layout known so far)"""

    def key(item):
        place, rating = item
        return (
            -round(rating['rating'], 6),
            -round(rating['mean'], 6),
            float(
                np.linalg.norm(
                    np.asarray(place) - np.asarray(default_center)[:2]
                )
            ),
        )

    return sorted(ratings, key=key)


def make_planners(gripper_type='panda'):
    """the two arms for the rating; joint limits reduced by 10 % as in the grasp generation (run_grasp_arm_gen.py --reduced-limit, default 0.1)"""
    box_move, box_hold = get_dual_arm_box('panda')
    return {
        'move': ArmMotionPlanner(
            get_arm_chain('panda', 'move', reduced_limit=0.1),
            gripper_type,
            False,
            box_move,
            virtual_walls=True,
        ),
        'hold': ArmMotionPlanner(
            get_arm_chain('panda', 'hold', reduced_limit=0.1),
            gripper_type,
            False,
            box_hold,
        ),
    }


def _rate_worker(samples, place, default_center, gripper_type, approach_room):
    ArmMotionPlanner.VISUALIZE = False
    planners = make_planners(gripper_type)
    np.random.seed(0)
    return place, rate_place(
        samples,
        np.asarray(place) - np.asarray(default_center)[:2],
        planners,
        gripper_type,
        approach_room,
    )


def run_workspace_search(
    exp_name,
    assembly,
    assembly_root='fabrica',
    precedence_from=None,
    max_places=MAX_PLACES,
    num_proc=12,
    num_proc_grasp=50,
    rate_only=False,
    rerate=False,
    pickup_center=None,
):
    from planning.run_grasp_arm_gen import GraspArmGenerator
    from planning.run_pickup_gen import (
        load_final_part_cfg,
        BOARD_HALF_SIZE,
        pickup_anchor,
    )
    from planning.robot.workcell import use_log_workcell

    assembly_dir = os.path.join(
        project_base_dir, 'assets', assembly_root, assembly
    )
    default_center = get_default_assembly_center('panda')
    py = sys.executable

    # the precedence plan at the default place (made once, moved to every place)
    reference_dir = os.path.join(project_base_dir, 'logs', exp_name, assembly)
    os.makedirs(reference_dir, exist_ok=True)
    if use_log_workcell(reference_dir) is not None:
        print(
            f'[run_workspace_search] {reference_dir} is planned for another place: use a new --exp-name'
        )
        return False
    if (
        pickup_center is not None
    ):  # around which the parts are laid out: no place of the assembly may cover it, and every plan uses it
        set_log_pickup_center(reference_dir, pickup_center)
    pickup_args = (
        []
        if get_pickup_center() is None
        else ['--pickup-center'] + [str(float(v)) for v in get_pickup_center()]
    )
    precedence_path = os.path.join(reference_dir, 'precedence.pkl')
    if not os.path.exists(precedence_path):
        if precedence_from is not None:
            if os.path.exists(
                os.path.join(precedence_from, 'assembly_center.json')
            ):
                print(
                    f'[run_workspace_search] {precedence_from} is planned for another place than the default'
                )
                return False
            shutil.copy(
                os.path.join(precedence_from, 'precedence.pkl'),
                precedence_path,
            )
            with open(os.path.join(precedence_from, 'stats.json'), 'r') as fp:
                stats = {'preced_plan': json.load(fp)['preced_plan']}
            with open(os.path.join(reference_dir, 'stats.json'), 'w') as fp:
                json.dump(stats, fp)
        elif subprocess.call(
            [
                py,
                'planning/run_preced_plan.py',
                '--assembly-dir',
                os.path.join('assets', assembly_root, assembly),
                '--log-dir',
                reference_dir,
                '--num-proc',
                str(num_proc),
                '--arm',
                'panda',
            ],
            cwd=project_base_dir,
        ) != 0 or not os.path.exists(precedence_path):
            print('[run_workspace_search] precedence planning failed')
            return False
    with open(precedence_path, 'rb') as fp:
        G_preced = pickle.load(fp)
    with open(os.path.join(reference_dir, 'stats.json'), 'r') as fp:
        stats_preced = {'preced_plan': json.load(fp)['preced_plan']}

    report_path = os.path.join(
        project_base_dir, 'logs', exp_name, f'workspace_search_{assembly}.json'
    )

    def save_report():
        with open(report_path, 'w') as fp:
            json.dump(report, fp, indent=2)

    if (
        os.path.exists(report_path) and not rerate
    ):  # the rating of an earlier call (e.g. --rate-only) is reused
        with open(report_path, 'r') as fp:
            report = json.load(fp)
        ranked = [
            (
                entry['center'],
                {key: entry[key] for key in ['rating', 'mean', 'parts']},
            )
            for entry in report['places']
        ]
        print(
            f'[run_workspace_search] using the ratings of {report_path} (--rerate to rate again)'
        )
    else:
        # grasps to test at every place
        np.random.seed(0)
        generator = GraspArmGenerator(
            os.path.join(project_base_dir, 'assets'),
            assembly_dir,
            G_preced,
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
        samples = {}
        for part_id in generator.part_ids:
            samples[part_id] = sample_grasps(generator, part_id)
            print(
                f'[run_workspace_search] part {part_id}: {len(samples[part_id])} grasps to test'
                + (
                    ''
                    if G_preced.nodes[part_id]['path'] is not None
                    else ' (held first: hold arm)'
                )
                + (
                    ' -- NO grasp picks it lying flat with a free gripper at the assembly'
                    if len(samples[part_id]) == 0
                    else ''
                ),
                flush=True,
            )

        final_meshes = load_final_part_cfg(assembly_dir, 'panda')['mesh']
        bounds = np.array([mesh.bounds for mesh in final_meshes.values()])
        final_bounds = np.array(
            [bounds[:, 0, :2].min(axis=0), bounds[:, 1, :2].max(axis=0)]
        )
        anchor = pickup_anchor(
            'panda'
        )  # around which the parts are laid out: the place stored in the reference log, else the default pickup area
        places = candidate_places(
            final_bounds, default_center, anchor, BOARD_HALF_SIZE
        )
        print(
            f'[run_workspace_search] rating {len(places)} places of the assembly (grid {WORKSPACE_STEP} cm, x {WORKSPACE_X}, y {WORKSPACE_Y})',
            flush=True,
        )
        args = [
            (samples, place, default_center, 'panda', APPROACH_ROOM)
            for place in places
        ]
        ratings = list(
            parallel_execute(
                _rate_worker, args, num_proc=num_proc, show_progress=False
            )
        )
        ranked = rank_places(ratings, default_center)
        report = {
            'assembly': assembly,
            'default_center': [float(v) for v in default_center[:2]],
            'approach_room_cm': APPROACH_ROOM,
            'n_grasp_samples': {k: len(v) for k, v in samples.items()},
            'places': [
                {'center': place, **rating} for place, rating in ranked
            ],
            'runs': [],
        }
        save_report()
    default_rating = next(
        rating
        for place, rating in ranked
        if np.allclose(place, default_center[:2])
    )
    print(
        f'[run_workspace_search] default place {report["default_center"]}: rating {default_rating["rating"]:.2f} (mean {default_rating["mean"]:.2f}), weakest part {min(default_rating["parts"], key=default_rating["parts"].get)}'
    )
    print(
        '[run_workspace_search] best places (rating = share of the tested grasps that pass, for the part with the fewest):'
    )
    for place, rating in ranked[:8]:
        print(
            f'    x {place[0]:+6.1f}, y {place[1]:+6.1f}: rating {rating["rating"]:.2f}, mean {rating["mean"]:.2f}  '
            + ' '.join(
                f'{k}:{v:.2f}' for k, v in sorted(rating['parts'].items())
            )
        )
    print(f'[run_workspace_search] all ratings: {report_path}', flush=True)
    if rate_only:
        return True
    if ranked[0][1]['rating'] <= 0:
        print(
            '[run_workspace_search] no place where every part has a grasp that passes: no plan'
        )
        return False

    for place, rating in ranked[:max_places]:
        if rating['rating'] <= 0:
            break
        is_default = bool(np.allclose(place, default_center[:2]))
        exp_place = (
            exp_name
            if is_default
            else f'{exp_name}_x{place[0]:+.1f}_y{place[1]:+.1f}'
        )
        log_dir = os.path.join(project_base_dir, 'logs', exp_place, assembly)
        os.makedirs(log_dir, exist_ok=True)
        if not is_default and not os.path.exists(
            os.path.join(log_dir, 'precedence.pkl')
        ):
            with open(os.path.join(log_dir, 'precedence.pkl'), 'wb') as fp:
                pickle.dump(
                    shift_precedence(
                        G_preced, np.asarray(place) - default_center[:2]
                    ),
                    fp,
                )
            with open(os.path.join(log_dir, 'stats.json'), 'w') as fp:
                json.dump(stats_preced, fp)
            set_log_assembly_center(log_dir, place)
        print(
            f'[run_workspace_search] planning with the assembly at x {place[0]:+.1f}, y {place[1]:+.1f} (rating {rating["rating"]:.2f}): logs/{exp_place}/{assembly}',
            flush=True,
        )
        code = subprocess.call(
            [
                py,
                '-u',
                'planning/run_planning_flat.py',
                '--exp-name',
                exp_place,
                '--assembly',
                assembly,
                '--assembly-root',
                assembly_root,
                '--num-proc',
                str(num_proc),
                '--num-proc-grasp',
                str(num_proc_grasp),
                '--assembly-center',
                str(place[0]),
                str(place[1]),
            ]
            + pickup_args,
            cwd=project_base_dir,
        )
        report['runs'].append(
            {
                'center': place,
                'log_dir': os.path.join('logs', exp_place, assembly),
                'verified': code == 0,
            }
        )
        save_report()
        if code == 0:
            print(
                f'[run_workspace_search] DONE: plan verified with the assembly at x {place[0]:+.1f}, y {place[1]:+.1f}: logs/{exp_place}/{assembly}'
            )
            return True
        print(
            f'[run_workspace_search] no plan at x {place[0]:+.1f}, y {place[1]:+.1f}',
            flush=True,
        )
    print(
        f'[run_workspace_search] no plan at the {len(report["runs"])} best places, see {report_path}'
    )
    return False


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--assembly', type=str, required=True)
    parser.add_argument('--assembly-root', type=str, default='fabrica')
    parser.add_argument(
        '--precedence-from',
        type=str,
        default=None,
        help='log folder with a precedence plan at the default place to reuse (precedence.pkl, stats.json)',
    )
    parser.add_argument(
        '--max-places',
        type=int,
        default=MAX_PLACES,
        help='best rated places the pipeline is run for, until one gives a verified plan',
    )
    parser.add_argument('--num-proc', type=int, default=12)
    parser.add_argument('--num-proc-grasp', type=int, default=50)
    parser.add_argument(
        '--rate-only',
        action='store_true',
        help='only rate the places (no planning)',
    )
    parser.add_argument(
        '--rerate',
        action='store_true',
        help='rate the places again although logs/<exp>/workspace_search_<assembly>.json exists',
    )
    parser.add_argument(
        '--pickup-center',
        type=float,
        nargs=2,
        default=None,
        metavar=('X', 'Y'),
        help='place on the board (cm) around which the parts are laid out; default: 0 5',
    )
    args = parser.parse_args()
    sys.exit(
        0
        if run_workspace_search(
            args.exp_name,
            args.assembly,
            args.assembly_root,
            args.precedence_from,
            args.max_places,
            args.num_proc,
            args.num_proc_grasp,
            args.rate_only,
            args.rerate,
            args.pickup_center,
        )
        else 1
    )
