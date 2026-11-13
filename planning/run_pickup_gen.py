"""Pickup of the parts from the board, without a printed fixture.

In the original Fabrica pipeline every part waits in a printed fixture
(run_fixture_gen.py). In the flat pipeline every part lies flat on the
board (the table) in one of its resting poses, that is, a pose in which
it lies there by itself, and the real arm picks it from there. This
file replaces run_fixture_gen.py: it writes fixture/pickup.json and no
fixture.obj.

It holds three stages of planning/run_planning_flat.py (stages 2 to 4
in FLAT_PIPELINE.md). The command line chooses the stage:

* --filter: the grasp filter (filter_flat_grasps). It keeps the move
  grasps that meet the criteria of the flat pipeline, above all that
  they can pick their part lying flat. Run it before run_seq_plan.py.
* --seq-opt: the sequence optimization (run_flat_seq_opt). It replaces
  run_seq_opt.py and adds one condition: the first hold grasp has to
  pick its part lying flat.
* neither of the two: the layout (run_pickup_gen). The parts are placed
  on the board one by one, each at a spot where its pickup passes the
  checks of the real arm. The arm poses of the pickups are found by
  inverse kinematics (IK: the joint angles that put the gripper at a
  wanted pose) and cached in the log. At the end the printable 1:1
  templates for laying out the parts are written.
* --template fit|a4|assembly: only draws a template of an existing
  layout again.

Usage, as planning/run_planning_flat.py calls it:
    python planning/run_pickup_gen.py --assembly-dir <assembly dir>
        --log-dir <log dir> [--filter | --seq-opt]

Terms used in this file:

* Board frame: lengths in cm, origin at the centre of the board, x from
  the hold arm to the move arm, z up, the table surface at z = 0.
* Move arm: the real Panda. It picks and inserts every part. Hold arm:
  the second arm of the original pipeline, which holds the assembly. On
  the real setup a human does that.
* Pose: a homogeneous 4x4 transform (a 3x3 rotation and a translation
  in one matrix), in the board frame unless said otherwise.
* Grasp: an object of the class Grasp (planning/robot/util_grasp.py).
  It stores the gripper pose (pos in cm, quat in the order w, x, y, z)
  for the part at its assembled pose, how far the gripper is open
  (open_ratio, at most 1.0) and the arm pose for it (arm_q). A move
  grasp is a list of such grasps along the insertion path, with index 0
  at the assembled pose. A hold grasp is a single one.
* Full and active arm pose: "full" has one value per link of the
  kinematic chain (0 for links that do not move), "active" only the
  angles of the joints that move, in rad.
* Footprint: the outline of a part or finger seen from above.
"""

import os

# One thread for the numeric libraries (OpenMP). This has to be set
# before numpy is imported.
os.environ['OMP_NUM_THREADS'] = '1'
import sys

# The repository root, so that "planning" and "assets" can be imported.
project_base_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)
sys.path.append(project_base_dir)

import json
import pickle
import random
import shutil
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from time import time

from assets.load import load_pos_quat_dict
from assets.transform import (
    get_transform_matrix,
    get_transform_matrix_quat,
    mat_to_pos_quat,
    get_pos_euler_from_transform_matrix,
)
from planning.config import RETRACT_OPEN_RATIO, RETRACT_DELTA_FAR
from planning.robot.geometry import (
    load_part_meshes,
    load_gripper_meshes,
    transform_gripper_meshes,
    offset_mesh,
)
from planning.robot.motion_plan_arm import (
    ArmMotionPlanner,
    VIRTUAL_WALL_SAFETY,
)
from planning.robot.util_arm import (
    get_arm_chain,
    get_virtual_wall_margin,
    get_hand_wall_margin,
)
from planning.robot.workcell import (
    get_assembly_center,
    get_fixture_min_y,
    get_dual_arm_box,
    get_pickup_center,
)
from planning.run_fixture_gen import (
    run_bin_packing,
    generate_pickup_meshes,
    check_part_gripper_collision,
    DELTA_BUFFER_SIZE,
    DX,
)
from planning.run_motion_plan import (
    get_pickup_arm_q,
    get_pickup_gripper_pose,
    get_back_retract_dir,
    make_move_planner,
    PICKUP_Q_HOLD_FILE,
)
from planning.run_seq_opt import SequenceOptimizer
from planning.config import (
    RETRACT_OPEN_RATIO as _RETRACT_OPEN_RATIO,
    GRASP_MIN_FINGER_OVERLAP,
    MIN_SIGMA,
    MIN_JOINT_LIMIT_MARGIN,
    FLAT_FALLBACK_POSES,
    OPEN_RATIO_REST,
)


# The string below is a statement without effect: only a string at the
# very top of a file is its docstring. The docstring at the top of this
# file describes the module.
"""
Fixture-free pickup: every part lies flat on the board in one of its \
stable resting poses, and is picked from (nearly) above.
Replaces run_fixture_gen.py: writes fixture/pickup.json and no \
fixture.obj.
"""

# Resting poses and the flat-pickup test live in planning/flat_grasp.py
# (shared with the grasp generation). The names stay available here.
from planning.flat_grasp import (
    MAX_TILT,
    MIN_TIP_ANGLE_FALLBACK,
    MIN_TIP_ANGLE,
    GROUND_CLEARANCE,
    WRIST_CLEARANCE,
    YAW_OFFSETS,
    get_flat_rots,
    get_flat_pickup_candidates,
    flat_move_grasps,
)

# cm. The default pickup area is moved this far from get_fixture_min_y
# towards the assembly (negative y): the real desk ends beside the robot
# base.
PICKUP_SHIFT_Y = -15.0

# cm. Largest size of the pickup area that generate_flat_pickup_pose
# tries (30 x 35 cm, DX is the spacing of the board holes): x between
# the arms, y from get_fixture_min_y (the board edge is at y = 46.25).
MAX_LAYOUT_SIZE = [12 * DX, 14 * DX]

# cm. Grid of the spots that are tried for each part: half the spacing
# of the board holes.
PLACEMENT_STEP = 1.25

# cm. Gap between the footprints of two lying parts. It gives the
# gripper room to turn between them, and it is more than
# ROBOT_PART_CLEARANCE (1.0 cm), so that a part lifted straight up
# cannot come too close to a neighbour. The value 2.0 was set by the
# user on 2026-11-12.
PART_GAP = 2.0

# Number of spots (each with one resting pose) that are checked with the
# robot per part, the spots nearest to the pickup area first.
MAX_PLACEMENT_TRIES = 150

# Number of further checks over the whole board if none of those fits.
# This wide search takes every second grid spot, and only spots where
# the fingers stay clear of the parts already placed.
MAX_PLACEMENT_TRIES_WIDE = 300

# cm above the pickup pose. The arm poses at these heights are checked
# for singularities and joint limits.
LIFT_CHECK_HEIGHTS = [2.5, 5.0, 7.5, 10.0]

# Number of random start poses of the IK per pickup, besides the rest
# pose of the arm and the arm pose of the grasp at the assembly.
N_PICKUP_IK_SAMPLES = 12

# cm. Offsets (dx, dy) to try around the default pickup area. No code in
# the repository uses this list.
PICKUP_AREA_OFFSETS = [
    (dx, dy)
    for dx in (0.0, -5.0, 5.0, -10.0, 10.0)
    for dy in (0.0, -5.0, 5.0, -10.0, 10.0)
]

# cm. Half the size of the optical board (x between the arms, y along
# them). The board frame has its origin at the centre, so the board
# reaches from -31.25 to 31.25 in x and from -46.25 to 46.25 in y.
BOARD_HALF_SIZE = (31.25, 46.25)

# Paper sizes in cm (width, height), portrait.
PAGE_SIZES = {'a4': (21.0, 29.7), 'a3': (29.7, 42.0)}

# cm. Border of a sheet that most printers cannot print on.
PAGE_MARGIN = 0.5

# cm. Overlap of neighbouring sheets of a tiled template.
TILE_OVERLAP = 1.0


def save_pickup_template(
    part_meshes_pickup, pickup_info, save_path, dx=2.5, page=None
):
    """Draws the 1:1 template for laying the parts out on the board.

    The template is a PDF in the board frame at true size. It shows the
    board holes to align it with the board and, for every part, its
    outline seen from above, the faces lying on the board (blue), the
    support area and who picks it.

    With page=None the result is one page fitted around the parts plus
    one board hole. With a paper size the parts are drawn on sheets of
    that size, portrait or landscape. If they do not fit on one sheet,
    several sheets that overlap by TILE_OVERLAP are written, all into
    one PDF.

    Args:
        part_meshes_pickup (dict): {part id: trimesh.Trimesh}, every
            part where it lies on the board (board frame, cm).
        pickup_info (dict): {part id: dict} as written to
            fixture/flat_pickup.json. Used here: 'arm' (who picks the
            part), 'fallback' (the part lies in a less stable resting
            pose) and 'tip_angle' (deg).
        save_path (str): Path of the PDF to write.
        dx (float): Spacing of the board holes in cm.
        page (str | None): None for the fitted page, else a key of
            PAGE_SIZES ('a4', 'a3').

    Returns:
        tuple[np.ndarray, np.ndarray]: The lower and the upper corner
        (x, y) of the area covered by the template, in the board frame
        in cm.
    """
    import matplotlib

    # 'Agg' draws into files and needs no display.
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union

    # mesh.bounds is [[lowest x, y, z], [highest x, y, z]]. The area of
    # all parts in x and y, with 0.5 cm around it.
    bounds = np.array([mesh.bounds for mesh in part_meshes_pickup.values()])
    parts_lower, parts_upper = (
        bounds[:, 0, :2].min(axis=0) - 0.5,
        bounds[:, 1, :2].max(axis=0) + 0.5,
    )

    # What is drawn of every part (board frame).
    outlines, drawing = [], []
    for part_id, mesh in part_meshes_pickup.items():
        # The triangles of the mesh without their z coordinate. Their
        # union is the silhouette of the part seen from above. Growing
        # it by 0.01 cm and shrinking it again closes hairline gaps
        # between the triangles.
        triangles = [
            Polygon(tri)
            for tri in mesh.triangles[:, :, :2]
            if Polygon(tri).area > 1e-8
        ]
        outline = unary_union(triangles).buffer(0.01).buffer(-0.01)
        outlines.append(outline)
        # The faces lying on the board: triangles whose three corners
        # are at most 0.001 cm above the lowest point of the part.
        contact = unary_union(
            [
                Polygon(tri[:, :2])
                for tri in mesh.triangles
                if np.all(tri[:, 2] < mesh.bounds[0, 2] + 1e-3)
                and Polygon(tri[:, :2]).area > 1e-8
            ]
        )
        # The corners that touch the board. Their convex hull (the
        # smallest convex polygon around them) is the support area of
        # the part.
        support = mesh.vertices[
            mesh.vertices[:, 2] < mesh.bounds[0, 2] + 1e-3
        ][:, :2]
        drawing.append(
            (
                part_id,
                mesh,
                outline,
                contact,
                support,
                support[ConvexHull(support).vertices],
            )
        )
    # Where the legend must not stand: the parts and 0.3 cm around them.
    occupied = unary_union(outlines).buffer(0.3)

    def draw(ax, lower, upper, sheet_note=None, overlap_edges=()):
        """Draws one sheet: the board between lower and upper.

        Args:
            ax (matplotlib.axes.Axes): The axes to draw on.
            lower (np.ndarray): Lower corner (x, y) of the sheet in the
                board frame in cm.
            upper (np.ndarray): Upper corner (x, y) of the sheet in the
                board frame in cm.
            sheet_note (str | None): Extra first line of the legend
                (which sheet of a tiled template this is).
            overlap_edges (sequence): Lines ([x, x], [y, y]) in cm that
                mark where the neighbouring sheets end on this sheet.
        """
        size = upper - lower
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_aspect('equal')
        ax.axis('off')
        # The cut line around the sheet.
        ax.add_patch(
            plt.Rectangle(lower, *size, fill=False, lw=0.3, color='lightgray')
        )
        # The board holes are at (k + 0.5) * dx for whole numbers k.
        # Each is drawn as a circle with a cross. The first argument of
        # arange is the first hole at or above the lower edge.
        for x in np.arange(
            np.ceil(lower[0] / dx - 0.5) * dx + 0.5 * dx, upper[0], dx
        ):
            for y in np.arange(
                np.ceil(lower[1] / dx - 0.5) * dx + 0.5 * dx, upper[1], dx
            ):
                ax.add_patch(
                    plt.Circle((x, y), 0.25, fill=False, lw=0.5, color='gray')
                )
                ax.plot([x - 0.4, x + 0.4], [y, y], lw=0.3, color='gray')
                ax.plot([x, x], [y - 0.4, y + 0.4], lw=0.3, color='gray')
        # Dashed: the part of this sheet covered by the neighbouring
        # sheet.
        for edge in overlap_edges:
            ax.plot(*edge, lw=0.6, ls='--', color='tab:green')
        for part_id, mesh, outline, contact, support, hull in drawing:
            # A shapely union is one polygon or several ("geoms").
            for polygon in getattr(outline, 'geoms', [outline]):
                ax.plot(*polygon.exterior.xy, lw=1.0, color='black')
            # The faces lying on the board.
            for polygon in getattr(contact, 'geoms', [contact]):
                if not polygon.is_empty:
                    ax.fill(
                        *polygon.exterior.xy,
                        color='tab:blue',
                        alpha=0.25,
                        lw=0,
                    )
            # The support polygon (also for parts resting on edges or
            # corners). The first corner is repeated to close the line.
            ax.plot(
                *np.vstack([hull, hull[:1]]).T,
                lw=0.8,
                ls='--',
                color='tab:blue',
            )
            ax.plot(support[:, 0], support[:, 1], 'o', ms=2, color='tab:blue')
            ax.text(
                *mesh.center_mass[:2],
                f'{part_id}',
                ha='center',
                va='center',
                fontsize=14,
                weight='bold',
                clip_on=True,
            )
            # Who picks the part. 'place' is the first part in the
            # fallback: the move arm brings it to the assembly spot.
            who = {'place': 'move (brings it to the assembly spot)'}.get(
                pickup_info[part_id]['arm'], pickup_info[part_id]['arm']
            )
            ax.text(
                mesh.center_mass[0],
                mesh.center_mass[1] - 0.6,
                who,
                ha='center',
                va='top',
                fontsize=6,
                clip_on=True,
            )
            if pickup_info[part_id].get('fallback'):
                ax.text(
                    mesh.center_mass[0],
                    mesh.center_mass[1] - 1.2,
                    f'tips at {pickup_info[part_id]["tip_angle"]:.0f} deg: '
                    'place carefully',
                    ha='center',
                    va='top',
                    fontsize=6,
                    color='tab:red',
                    clip_on=True,
                )
        # The scale bar and the notes go to a free spot of this sheet.
        # The spots are tried on a 0.25 cm grid, the lowest row first.
        # The "else" of the loop runs if no row has a free spot.
        legend_size = np.array([10.6, 2.3 + (0.6 if sheet_note else 0)])
        corner = lower + 0.2
        for y in np.arange(lower[1] + 0.2, upper[1] - legend_size[1], 0.25):
            free = [
                x
                for x in np.arange(
                    lower[0] + 0.2, upper[0] - legend_size[0], 0.25
                )
                if not box(
                    x, y, x + legend_size[0], y + legend_size[1]
                ).intersects(occupied)
            ]
            if free:
                corner = np.array([free[0], y])
                break
        else:
            print(
                '[save_pickup_template] no free spot for the scale bar, '
                'it overlaps a part'
            )
        # A line of 10 cm to check the scale of the print.
        ax.plot(
            [corner[0] + 0.3, corner[0] + 10.3],
            [corner[1] + 0.3] * 2,
            lw=1.5,
            color='red',
        )
        ax.text(
            corner[0] + 5.3,
            corner[1] + 0.45,
            '10 cm (print at 100%, no "fit to page")',
            ha='center',
            va='bottom',
            fontsize=6,
            color='red',
        )
        note = (
            (sheet_note + '\n' if sheet_note else '')
            + 'move robot: +x ->   assembly spot: -y\n'
            f'board x [{lower[0]:.2f}, {upper[0]:.2f}] cm, '
            f'y [{lower[1]:.2f}, {upper[1]:.2f}] cm\n'
            'blue: on the board (dashed: support area, dots: contact corners)'
        )
        ax.text(
            corner[0] + 0.3,
            corner[1] + legend_size[1] - 0.1,
            note,
            ha='left',
            va='top',
            fontsize=6,
        )

    if page is None:
        # Extend the page to one board hole beyond the parts. A figure
        # size is given in inches, 1 inch is 2.54 cm.
        lower = np.floor(parts_lower / dx - 0.5) * dx - 0.5 * dx
        upper = np.ceil(parts_upper / dx + 0.5) * dx + 0.5 * dx
        fig = plt.figure(
            figsize=((upper - lower)[0] / 2.54, (upper - lower)[1] / 2.54)
        )
        draw(fig.add_axes([0, 0, 1, 1]), lower, upper)
        fig.savefig(save_path)
        plt.close(fig)
        return lower, upper

    # Tiles: of portrait and landscape, the orientation with the fewest
    # sheets. area is the printable part of a sheet, n the number of
    # sheets in x and y. Every sheet after the first adds its area minus
    # the overlap.
    parts_size = parts_upper - parts_lower
    best = None
    for page_size in [
        np.array(PAGE_SIZES[page]),
        np.array(PAGE_SIZES[page])[::-1],
    ]:
        area = page_size - 2 * PAGE_MARGIN
        n = np.maximum(
            1, np.ceil((parts_size - TILE_OVERLAP) / (area - TILE_OVERLAP))
        ).astype(int)
        if best is None or n.prod() < best[1].prod():
            best = (page_size, n)
    page_size, n = best
    area = page_size - 2 * PAGE_MARGIN
    # span is the size of what all sheets cover together. It is centred
    # on the parts.
    span = n * area - (n - 1) * TILE_OVERLAP
    start = (parts_lower + parts_upper) / 2 - span / 2
    n_sheets = int(n.prod())
    with PdfPages(save_path) as pdf:
        # One page of the PDF per sheet, the top row first.
        for row in range(n[1] - 1, -1, -1):
            for col in range(n[0]):
                lower = start + np.array([col, row]) * (area - TILE_OVERLAP)
                upper = lower + area
                sheet = (n[1] - 1 - row) * n[0] + col + 1
                # A line on every side that has a neighbouring sheet.
                edges = []
                if col > 0:
                    edges.append(
                        ([lower[0] + TILE_OVERLAP] * 2, [lower[1], upper[1]])
                    )
                if col < n[0] - 1:
                    edges.append(
                        ([upper[0] - TILE_OVERLAP] * 2, [lower[1], upper[1]])
                    )
                if row > 0:
                    edges.append(
                        ([lower[0], upper[0]], [lower[1] + TILE_OVERLAP] * 2)
                    )
                if row < n[1] - 1:
                    edges.append(
                        ([lower[0], upper[0]], [upper[1] - TILE_OVERLAP] * 2)
                    )
                note = (
                    None
                    if n_sheets == 1
                    else f'SHEET {sheet}/{n_sheets} '
                    f'(column {col + 1}/{n[0]} from the left, '
                    f'row {n[1] - row}/{n[1]} from the top); '
                    f'green dashed: {TILE_OVERLAP:.0f} cm overlap, '
                    'lay the neighbouring sheet on it, crosses on crosses'
                )
                fig = plt.figure(
                    figsize=(page_size[0] / 2.54, page_size[1] / 2.54)
                )
                # The axes leave the unprintable border of the sheet
                # free (positions as fractions of the sheet).
                draw(
                    fig.add_axes(
                        [
                            PAGE_MARGIN / page_size[0],
                            PAGE_MARGIN / page_size[1],
                            area[0] / page_size[0],
                            area[1] / page_size[1],
                        ]
                    ),
                    lower,
                    upper,
                    note,
                    edges,
                )
                pdf.savefig(fig)
                plt.close(fig)
    if n_sheets > 1:
        print(
            f'[save_pickup_template] {n_sheets} {page.upper()} sheets '
            f'({n[0]} x {n[1]}, '
            f'{"portrait" if page_size[0] < page_size[1] else "landscape"}) '
            f'with {TILE_OVERLAP:.0f} cm overlap'
        )
    return start, start + span


def save_assembly_template(assembly_dir, log_dir, dx=2.5, page='a4'):
    """
    1:1 template of the assembly spot (board frame, cm): where the parts that touch the board in the assembled state
    (the part held by the human) have to stand, plus the outline of the finished assembly. Writes fixture/assembly_template_<page>.pdf
    """
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    with open(os.path.join(log_dir, 'grasps.pkl'), 'rb') as fp:
        arm_type = pickle.load(fp)['arm']
    meshes = load_final_part_cfg(assembly_dir, arm_type)['mesh']

    def footprint(mesh):
        return (
            unary_union(
                [
                    Polygon(tri)
                    for tri in mesh.triangles[:, :, :2]
                    if Polygon(tri).area > 1e-8
                ]
            )
            .buffer(0.01)
            .buffer(-0.01)
        )

    on_board = [
        part_id for part_id, mesh in meshes.items() if mesh.bounds[0, 2] < 0.05
    ]
    bounds = np.array([mesh.bounds for mesh in meshes.values()])
    center = (bounds[:, 0, :2].min(axis=0) + bounds[:, 1, :2].max(axis=0)) / 2
    page_size = np.array(PAGE_SIZES[page])
    lower, upper = (
        center - (page_size - 2 * PAGE_MARGIN) / 2,
        center + (page_size - 2 * PAGE_MARGIN) / 2,
    )
    size = upper - lower

    fig = plt.figure(figsize=(page_size[0] / 2.54, page_size[1] / 2.54))
    ax = fig.add_axes(
        [
            PAGE_MARGIN / page_size[0],
            PAGE_MARGIN / page_size[1],
            size[0] / page_size[0],
            size[1] / page_size[1],
        ]
    )
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_aspect('equal')
    ax.axis('off')
    ax.add_patch(
        plt.Rectangle(lower, *size, fill=False, lw=0.3, color='lightgray')
    )
    for x in np.arange(
        np.ceil(lower[0] / dx - 0.5) * dx + 0.5 * dx, upper[0], dx
    ):
        for y in np.arange(
            np.ceil(lower[1] / dx - 0.5) * dx + 0.5 * dx, upper[1], dx
        ):
            ax.add_patch(
                plt.Circle((x, y), 0.25, fill=False, lw=0.5, color='gray')
            )
            ax.plot([x - 0.4, x + 0.4], [y, y], lw=0.3, color='gray')
            ax.plot([x, x], [y - 0.4, y + 0.4], lw=0.3, color='gray')
    outline = unary_union([footprint(mesh) for mesh in meshes.values()])
    for polygon in getattr(outline, 'geoms', [outline]):
        ax.plot(
            *polygon.exterior.xy, lw=0.8, ls='--', color='gray'
        )  # finished assembly seen from above
    for part_id in on_board:
        mesh = meshes[part_id]
        for polygon in getattr(footprint(mesh), 'geoms', [footprint(mesh)]):
            ax.fill(*polygon.exterior.xy, color='tab:orange', alpha=0.3, lw=0)
            ax.plot(*polygon.exterior.xy, lw=1.2, color='black')
        ax.text(
            *mesh.bounds[:, :2].mean(axis=0),
            f'{part_id}',
            ha='center',
            va='center',
            fontsize=16,
            weight='bold',
        )
    ax.plot(
        [lower[0] + 0.3, lower[0] + 10.3],
        [lower[1] + 0.3] * 2,
        lw=1.5,
        color='red',
    )
    ax.text(
        lower[0] + 5.3,
        lower[1] + 0.45,
        '10 cm (print at 100%, no "fit to page")',
        ha='center',
        va='bottom',
        fontsize=6,
        color='red',
    )
    ax.text(
        lower[0] + 0.3,
        upper[1] - 0.3,
        f'ASSEMBLY SPOT: hold part(s) {", ".join(on_board)} exactly on the orange area (dashed: finished assembly from above)\n'
        f'move robot: +x ->   pickup area: +y\nboard x [{lower[0]:.2f}, {upper[0]:.2f}] cm, y [{lower[1]:.2f}, {upper[1]:.2f}] cm',
        ha='left',
        va='top',
        fontsize=6,
    )
    path = os.path.join(log_dir, 'fixture', f'assembly_template_{page}.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'[save_assembly_template] {path}')
    return path


def save_pickup_templates(log_dir, pages=(None, 'a4')):
    """(Re)draw the placement templates of a finished layout: fixture/pickup_template.pdf (fitted) and fixture/pickup_template_<page>.pdf"""
    import trimesh as _trimesh

    fixture_dir = os.path.join(log_dir, 'fixture')
    with open(os.path.join(fixture_dir, 'flat_pickup.json'), 'r') as fp:
        pickup_info = json.load(fp)
    scene = _trimesh.load(
        os.path.join(fixture_dir, 'pickup_layout.glb'), force='scene'
    )
    part_meshes_pickup = {}
    for node in scene.graph.nodes_geometry:
        if not node.endswith('.obj'):
            continue  # the board box of the layout
        transform, geometry_name = scene.graph[node]
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        part_meshes_pickup[node[: -len('.obj')]] = mesh
    for page in pages:
        path = os.path.join(
            fixture_dir,
            'pickup_template.pdf'
            if page is None
            else f'pickup_template_{page}.pdf',
        )
        result = save_pickup_template(
            part_meshes_pickup, pickup_info, path, page=page
        )
        if result is not None:
            print(
                f'[save_pickup_templates] {path}'
                + ('' if page is None else f' ({page.upper()})')
            )


IK_CACHE_FILE = 'pickup_ik_cache.pkl'


def load_ik_cache(log_dir):
    path = os.path.join(log_dir, IK_CACHE_FILE)
    if os.path.exists(path):
        with open(path, 'rb') as fp:
            return pickle.load(fp)
    return {}


def save_ik_cache(log_dir, cache):
    with open(os.path.join(log_dir, IK_CACHE_FILE), 'wb') as fp:
        pickle.dump(cache, fp)


def pickup_ik_solutions(
    cache, planner, grasp, part_id, pickup_mat, final_mat, has_ft_sensor
):
    """
    Arm poses of the real arm for one pickup that pass every check not depending on the other parts (IK, virtual walls,
    5 cm lift, singularity, joint limits), best conditioned first; score = min(sigma / MIN_SIGMA, joint margin / MIN_JOINT_LIMIT_MARGIN).
    Cached per robot height, part, grasp and pickup pose (in the log, reused by later runs).
    """
    from planning.plan_checks import joint_condition
    from planning.robot.workcell import get_panda_base_z

    key = (
        'lift',
        round(get_panda_base_z(), 3),
        part_id,
        int(grasp.grasp_id),
        tuple(np.round(pickup_mat[:3, :4], 3).flatten()),
    )
    if key not in cache and planner.virtual_walls:
        # the wrist frame and the flange follow the gripper rigidly. If one of them is behind a virtual wall at the pickup or
        # on the lift above it, every arm pose fails the tests below, so none is searched: that search (52 IK runs that cannot
        # succeed) took most of the time of a layout far from the arm. The limit is lower by PICKUP_POSE_SLACK, as in pickup_blocked.
        from planning.plan_checks import PICKUP_POSE_SLACK

        gripper_pose = get_pickup_gripper_pose(grasp, pickup_mat, final_mat)
        for lift in [0.0] + LIFT_CHECK_HEIGHTS:
            lifted = gripper_pose.copy()
            lifted[2, 3] += lift
            if (
                get_hand_wall_margin(
                    planner.arm_chain,
                    planner.gripper_type,
                    lifted,
                    has_ft_sensor,
                )
                < VIRTUAL_WALL_SAFETY - PICKUP_POSE_SLACK
            ):
                cache[key] = []
                break
    if key not in cache:
        rest = planner.arm_chain.active_to_full(planner.arm_chain.rest_q)
        starts = [(rest, None), (grasp.arm_q, None)] + [
            (planner.arm_chain.sample_joint_angles(), 1)
            for _ in range(N_PICKUP_IK_SAMPLES)
        ]
        solutions = []
        for q_init, n_restart in starts:
            q = get_pickup_arm_q(
                planner,
                grasp,
                pickup_mat,
                final_mat,
                arm_q_init=q_init,
                has_ft_sensor=has_ft_sensor,
                optimizer='least_squares',
                regularization=1.0,
                n_restart=n_restart,
            )
            if q is None or not planner.check_virtual_walls(q):
                continue
            # the pickup pose and the poses while lifting the part straight up (the arm folds further when lifting near the robot)
            score, q_prev = np.inf, q
            fk = planner.arm_chain.forward_kinematics(q)
            for lift in [0.0] + LIFT_CHECK_HEIGHTS:
                q_lift = (
                    q
                    if lift == 0
                    else planner.inverse_kinematics(
                        fk[:3, 3] + np.array([0, 0, lift]),
                        fk[:3, :3],
                        q_init=q_prev,
                        n_restart=1,
                    )
                )
                if (
                    q_lift is None
                    or not planner.check_virtual_walls(q_lift)
                    or np.abs(np.asarray(q_lift) - np.asarray(q_prev)).max()
                    > 0.5
                ):
                    score = 0
                    break
                sigma, margin, _ = joint_condition(
                    planner.arm_chain.active_from_full(q_lift)
                )
                score = min(
                    score, sigma / MIN_SIGMA, margin / MIN_JOINT_LIMIT_MARGIN
                )
                q_prev = q_lift
            if score < 1:
                continue
            solutions.append((score, np.asarray(q)))
        cache[key] = sorted(solutions, key=lambda x: -x[0])
    return cache[key]


def area_allowed(center_x, min_y, bin_size, final_meshes, margin=1.0):
    """the pickup area lies on the board and away from the assembly zone (assembly + ASSEMBLY_ZONE_MARGIN)"""
    from planning.config import ASSEMBLY_ZONE_MARGIN

    lower, upper = (
        np.array([center_x - bin_size[0] / 2, min_y]),
        np.array([center_x + bin_size[0] / 2, min_y + bin_size[1]]),
    )
    if np.any(lower < -np.array(BOARD_HALF_SIZE) + margin) or np.any(
        upper > np.array(BOARD_HALF_SIZE) - margin
    ):
        return False
    bounds = np.array([mesh.bounds for mesh in final_meshes.values()])
    zone_lower, zone_upper = (
        bounds[:, 0, :2].min(axis=0) - ASSEMBLY_ZONE_MARGIN,
        bounds[:, 1, :2].max(axis=0) + ASSEMBLY_ZONE_MARGIN,
    )
    return bool(np.any(upper <= zone_lower) or np.any(lower >= zone_upper))


def generate_flat_pickup_pose(
    pose_info, min_y, max_size=MAX_LAYOUT_SIZE, center_x=0.0
):
    """
    Smallest 2d layout of the resting parts on the board (generate_pickup_pose of run_fixture_gen.py without the fixture bottom and size limits)
    """
    best = None
    for bin_x in np.arange(4 * DX, max_size[0] + 0.5 * DX, DX):
        for bin_y in np.arange(4 * DX, max_size[1] + 0.5 * DX, DX):
            if best is not None and (bin_x * bin_y, bin_y) >= (
                best[1][0] * best[1][1],
                best[1][1],
            ):
                continue
            packer = run_bin_packing(pose_info, [bin_x, bin_y])
            if packer is not None:
                best = (packer, [bin_x, bin_y])
    if best is None:
        return None, None
    packer, bin_size = best

    pickup_pose = {}
    for rect in packer[0]:
        part_id = rect.rid
        part_transform = np.eye(4)
        part_transform[:3, :3] = pose_info[part_id]['rot_mat']
        part_transform[:3, 3] = np.array(
            [
                rect.x
                + rect.width / 2
                - pose_info[part_id]['center_x']
                - bin_size[0] / 2
                + center_x,
                rect.y
                + rect.height / 2
                - pose_info[part_id]['center_y']
                + min_y,
                -pose_info[part_id]['min_z'],
            ]
        )
        pickup_pose[part_id] = get_pos_euler_from_transform_matrix(
            part_transform
        ).tolist()
    return pickup_pose, bin_size


def check_lift(motion_planner, q, lift=RETRACT_DELTA_FAR):
    """
    A part lying on the board is inside the ground safety margin, so the motion planner first lifts it straight up by RETRACT_DELTA_FAR
    """
    fk = motion_planner.arm_chain.forward_kinematics(q)
    q_lift = motion_planner.inverse_kinematics(
        fk[:3, 3] + np.array([0, 0, lift]), fk[:3, :3], q_init=q
    )
    return q_lift is not None and motion_planner.check_virtual_walls(q_lift)


def load_final_part_cfg(assembly_dir, arm_type):
    part_meshes_final = load_part_meshes(assembly_dir, transform='final')
    part_meshes_final = {
        k.replace('part', ''): v for k, v in part_meshes_final.items()
    }
    for part_mesh in part_meshes_final.values():
        part_mesh.apply_translation(get_assembly_center(arm_type))
    part_pos_dict_final, part_quat_dict_final = load_pos_quat_dict(
        assembly_dir, transform='final'
    )
    part_pos_dict_final = {
        part_id: part_pos_dict_final[part_id] + get_assembly_center(arm_type)
        for part_id in part_meshes_final.keys()
    }
    return {
        'mesh': part_meshes_final,
        'pos': part_pos_dict_final,
        'quat': part_quat_dict_final,
    }


def flat_rots_of(part_mesh, part_id, grasps, min_tip_angle=MIN_TIP_ANGLE):
    """
    The resting poses of a part in this plan: the stable ones, plus the less stable ones if the grasp filter needed them to find a
    move grasp for this part (recorded in grasps['settings']['flat_pickup']['fallback_parts'])
    """
    fallback_parts = (
        grasps['settings'].get('flat_pickup', {}).get('fallback_parts', [])
    )
    return get_flat_rots(
        part_mesh, min_tip_angle, with_fallback=part_id in fallback_parts
    )


def filter_flat_grasps(
    assembly_dir,
    log_dir,
    max_tilt=MAX_TILT,
    clearance=GROUND_CLEARANCE,
    min_tip_angle=MIN_TIP_ANGLE,
    filter_hold=False,
):
    """
    Keep only the move grasps that meet the criteria of the flat pipeline (flat_grasp.move_grasp_criteria: inside the virtual walls,
    away from singularities and joint limits, deep enough in the fingers, able to pick their part lying flat). Run before sequence
    planning. If no move grasp of a part passes in its stable resting poses, the less stable ones are admitted for that part
    (get_flat_rots with_fallback); these parts are recorded for the later stages (flat_rots_of) and marked on the template.
    Hold grasps are only used on the board for the first held part, so they are kept by default.
    Returns the parts that have move grasps before, but none after the filter.
    """
    grasps_path = os.path.join(log_dir, 'grasps.pkl')
    grasps_all_path = os.path.join(log_dir, 'grasps_unfiltered.pkl')
    if not os.path.exists(grasps_all_path):
        shutil.copy(grasps_path, grasps_all_path)
    with open(grasps_all_path, 'rb') as fp:
        grasps = pickle.load(fp)

    part_cfg_final = load_final_part_cfg(assembly_dir, grasps['arm'])
    gripper_meshes = load_gripper_meshes(
        grasps['gripper'], os.path.join(project_base_dir, 'assets')
    )
    arm_chain_move = (
        get_arm_chain(grasps['arm'], 'move')
        if grasps['arm'] == 'panda'
        else None
    )  # the move arm is the real robot
    never_moved = []  # the base part is held, not inserted: it needs no move grasp
    precedence_path = os.path.join(log_dir, 'precedence.pkl')
    if os.path.exists(precedence_path):
        with open(precedence_path, 'rb') as fp:
            G_preced = pickle.load(fp)
        never_moved = [
            part_id
            for part_id in G_preced.nodes
            if G_preced.nodes[part_id]['path'] is None
        ]

    kept_ids, fallback_parts, empty_parts = {}, [], []
    for part_id, part_grasps in grasps['grasps'].items():
        part_mesh = part_cfg_final['mesh'][part_id]
        n_move, n_hold = len(part_grasps['move']), len(part_grasps['hold'])
        moves, flat_rots, fallback = flat_move_grasps(
            part_grasps['move'],
            part_mesh,
            grasps['gripper'],
            gripper_meshes,
            arm_chain_move,
            max_tilt,
            clearance,
            min_tip_angle,
            allow_fallback=FLAT_FALLBACK_POSES,
        )
        if fallback:
            fallback_parts.append(
                part_id
            )  # no grasp fits a stable resting pose: the less stable ones are used for this part
        if len(moves) == 0 and part_id not in never_moved:
            empty_parts.append(part_id)
        part_grasps['move'] = moves
        if filter_hold:
            part_grasps['hold'] = [
                g
                for g in part_grasps['hold']
                if len(
                    get_flat_pickup_candidates(
                        g,
                        part_mesh,
                        flat_rots,
                        grasps['gripper'],
                        gripper_meshes,
                        max_tilt,
                        clearance,
                    )
                )
                > 0
            ]
        kept_ids[part_id] = {
            'move': {g[0].grasp_id for g in part_grasps['move']},
            'hold': {g.grasp_id for g in part_grasps['hold']},
        }
        print(
            f'[filter_flat_grasps] part {part_id}: {len(flat_rots)} resting poses'
            + (
                ' (less stable ones included: no grasp fits a stable one)'
                if part_id in fallback_parts
                else ''
            )
            + f', move {n_move} -> {len(part_grasps["move"])}, hold {n_hold} -> {len(part_grasps["hold"])}'
        )

    grasps['grasp_id_pairs'] = {
        (part_move, part_hold): [
            (m, h)
            for m, h in pairs
            if m in kept_ids[part_move]['move']
            and h in kept_ids[part_hold]['hold']
        ]
        for (part_move, part_hold), pairs in grasps['grasp_id_pairs'].items()
    }
    grasps['settings']['flat_pickup'] = {
        'max_tilt': max_tilt,
        'clearance': clearance,
        'min_tip_angle': min_tip_angle,
        'filter_hold': filter_hold,
        'min_finger_overlap': GRASP_MIN_FINGER_OVERLAP,
        'fallback_parts': fallback_parts,
    }
    with open(grasps_path, 'wb') as fp:
        pickle.dump(grasps, fp)
    if len(empty_parts) > 0:
        print(
            f'[filter_flat_grasps] NO MOVE GRASP left for part(s) {", ".join(empty_parts)}: none of their grasps can pick the part lying flat and insert it'
        )
    return empty_parts


def first_hold_nodes(
    nodes,
    part_grasps,
    part_meshes,
    grasps,
    gripper_meshes,
    max_tilt=MAX_TILT,
    clearance=GROUND_CLEARANCE,
    min_tip_angle=MIN_TIP_ANGLE,
):
    """
    The first-hold nodes of a grasp tree, and those of them that the flat pipeline cannot use.
    nodes: the nodes of the grasp tree, (parts that remain, part held, hold grasp id), in disassembly order: in a first-hold node
    only the part that is held first remains. part_grasps: {part_id: {'hold': {grasp_id: grasp}}} (SequenceOptimizer.grasps);
    part_meshes: the parts at their assembled poses; grasps: the content of grasps.pkl (gripper, resting poses per part).
    Returns (first-hold nodes, those whose hold grasp cannot pick its part lying flat): the hold grasp of the first step is the
    only hold grasp that takes its part from the board.
    """
    first_holds = [
        node for node in nodes if len(node[0]) == 1 and node[1] in node[0]
    ]
    flat_rots, infeasible = {}, []
    for node in first_holds:
        part_id, grasp_id = node[1], node[2]
        if part_id not in flat_rots:
            flat_rots[part_id] = flat_rots_of(
                part_meshes[part_id], part_id, grasps, min_tip_angle
            )
        grasp = part_grasps[part_id]['hold'][grasp_id]
        if (
            len(
                get_flat_pickup_candidates(
                    grasp,
                    part_meshes[part_id],
                    flat_rots[part_id],
                    grasps['gripper'],
                    gripper_meshes,
                    max_tilt,
                    clearance,
                )
            )
            == 0
        ):
            infeasible.append(node)
    return first_holds, infeasible


def place_grasps_of_first_part(
    part_moves,
    part_mesh,
    flat_rots,
    gripper_type,
    gripper_meshes,
    max_tilt=MAX_TILT,
    clearance=GROUND_CLEARANCE,
    final_pose_ok=None,
):
    """
    Fallback for the part that is held first when no first hold grasp can pick it lying flat: the move arm brings the part from
    the board to the assembly spot (a planned step of the real arm), and the hold arm takes it there. The grasps the move arm
    can do that with, best first: move grasps of the part (part_moves: [[grasp, ...], ...] as in grasps.pkl) that pick it lying
    flat and whose arm pose with the part standing at the assembly spot is fine (final_pose_ok(grasp) -> bool; None: not
    tested). Best first by the rating of the grasp generation (flat_rating), without it by how upright the hand is at the pickup.
    Returns [(grasp, candidates of get_flat_pickup_candidates)].
    """
    usable = []
    for move in (
        part_moves.values() if isinstance(part_moves, dict) else part_moves
    ):  # SequenceOptimizer turns the lists of grasps.pkl into {grasp id: [grasp, ...]}
        grasp = move[0]
        candidates = get_flat_pickup_candidates(
            grasp,
            part_mesh,
            flat_rots,
            gripper_type,
            gripper_meshes,
            max_tilt,
            clearance,
        )
        if len(candidates) == 0 or grasp.arm_q is None:
            continue
        if final_pose_ok is not None and not final_pose_ok(grasp):
            continue
        usable.append((grasp, candidates))
    return sorted(
        usable,
        key=lambda item: (
            -getattr(item[0], 'flat_rating', 0.0),
            round(item[1][0]['tilt'], 1),
            item[0].grasp_id,
        ),
    )


def run_flat_seq_opt(
    assembly_dir,
    log_dir,
    max_tilt=MAX_TILT,
    clearance=GROUND_CLEARANCE,
    min_tip_angle=MIN_TIP_ANGLE,
    plot=False,
    verbose=False,
):
    """
    run_seq_opt.py, but the first hold grasp (the only one that picks its part from the board) must pick the part lying flat.
    Fallback if no first hold grasp can: the hold grasps stay as they are, and the move arm brings the first part to the
    assembly spot with one of its own move grasps (place_grasps_of_first_part; the layout chooses it). Without such a grasp
    there is no sequence.
    """
    with open(os.path.join(log_dir, 'precedence.pkl'), 'rb') as fp:
        G_preced = pickle.load(fp)
    with open(os.path.join(log_dir, 'grasps.pkl'), 'rb') as fp:
        grasps = pickle.load(fp)
    with open(os.path.join(log_dir, 'tree.pkl'), 'rb') as fp:
        tree = pickle.load(fp)

    t_start = time()
    seq_optimizer = SequenceOptimizer(G_preced, grasps)
    grasp_tree = seq_optimizer._convert_part_tree_to_grasp_tree(
        tree, verbose=verbose
    )
    if grasp_tree is None:
        print(f'[run_flat_seq_opt] No complete sequence in {log_dir}')
        return False

    part_cfg_final = load_final_part_cfg(assembly_dir, grasps['arm'])
    gripper_meshes = load_gripper_meshes(
        grasps['gripper'], os.path.join(project_base_dir, 'assets')
    )
    first_holds, infeasible = first_hold_nodes(
        grasp_tree.nodes,
        seq_optimizer.grasps,
        part_cfg_final['mesh'],
        grasps,
        gripper_meshes,
        max_tilt,
        clearance,
        min_tip_angle,
    )
    print(
        f'[run_flat_seq_opt] {len(first_holds) - len(infeasible)}/{len(first_holds)} first hold grasps can pick their part lying flat'
    )
    grasp_tree = grasp_tree.copy()
    if len(first_holds) > 0 and len(infeasible) == len(first_holds):
        # fallback: the move arm brings the first part to the assembly spot. Only for first parts that have a move grasp for it
        can_place = set()
        for part_id in sorted({node[1] for node in first_holds}):
            flat_rots = flat_rots_of(
                part_cfg_final['mesh'][part_id], part_id, grasps, min_tip_angle
            )
            if (
                len(
                    place_grasps_of_first_part(
                        grasps['grasps'][part_id]['move'],
                        part_cfg_final['mesh'][part_id],
                        flat_rots,
                        grasps['gripper'],
                        gripper_meshes,
                        max_tilt,
                        clearance,
                    )
                )
                > 0
            ):
                can_place.add(part_id)
        if len(can_place) == 0:
            print(
                '[run_flat_seq_opt] none can, and the part has no move grasp that picks it lying flat either: there is no sequence'
            )
            return False
        print(
            f'[run_flat_seq_opt] none can: fallback, the move arm brings the first part ({", ".join(sorted(can_place))}) to the assembly spot and the hold arm takes it there'
        )
        grasp_tree.remove_nodes_from(
            [node for node in first_holds if node[1] not in can_place]
        )
    else:
        grasp_tree.remove_nodes_from(infeasible)

    result = seq_optimizer._optimize_part_tree_from_grasp_tree(
        grasp_tree, verbose=verbose
    )
    if result is None:
        print(f'[run_flat_seq_opt] Failed sequence planning in {log_dir}')
        return False
    tree_opt, _ = result

    with open(os.path.join(log_dir, 'tree_opt.pkl'), 'wb') as fp:
        pickle.dump(tree_opt, fp)
    stats_path = os.path.join(log_dir, 'stats.json')
    with open(stats_path, 'r') as fp:
        stats = json.load(fp)
    stats['seq_opt'] = {'time': round(time() - t_start, 2), 'flat': True}
    with open(stats_path, 'w') as fp:
        json.dump(stats, fp)
    if plot:
        seq_optimizer.plot_tree(
            tree_opt, save_path=os.path.join(log_dir, 'tree_opt.png')
        )
    return True


def finger_footprints(grasp, pose, gripper_type, gripper_meshes):
    """
    Footprints on the board (shapely polygons, one per finger) of the gripper fingers at the pickup of a part that lies at
    `pose` (4x4, applied to the part in its final pose, where the grasp is defined), each over the opened gripper (as it arrives)
    and the gripper closed on the part.
    """
    from shapely.geometry import MultiPoint
    from planning.robot.geometry import (
        get_gripper_meshes_transforms,
        get_gripper_finger_names,
    )

    shapes = []
    for name in get_gripper_finger_names(gripper_type):
        vertices = gripper_meshes[name].convex_hull.vertices
        points = []
        for open_ratio in [
            grasp.open_ratio,
            min(grasp.open_ratio + _RETRACT_OPEN_RATIO, 1.0),
        ]:
            transform = get_gripper_meshes_transforms(
                gripper_type,
                gripper_meshes,
                grasp.pos,
                grasp.quat,
                pose,
                open_ratio,
            )[name]
            points.append(
                (vertices @ transform[:3, :3].T + transform[:3, 3])[:, :2]
            )
        shapes.append(MultiPoint(np.vstack(points)).convex_hull)
    return shapes


def fingers_clear(fingers, others, clearance):
    """
    True if the finger footprints keep `clearance` to the footprints of the other lying parts. Footprints are convex hulls seen
    from above, so this is sufficient (not necessary) for the fingers to keep that distance to the parts in space.
    """
    return all(
        finger.distance(other) >= clearance
        for finger in fingers
        for other in others
    )


class PickupMoves:
    """
    The straight moves of the real arm at a pickup spot, tested with the planner of run_motion_plan.py and its own routines. A
    spot is only taken if the arm can get into the grasp there and away with the part; else the plan would fail later, in motion
    planning, although the same grasp works a few centimetres further.
      into the grasp     as ArmMotionPlanner.plan_path_switch: from the pickup pose back along the hand axis, the gripper opened
                         for the grasp, until a pose is clear of every part (also of the one to pick); if the gripper arrives
                         less opened than that, on to where it is clear with that opening as well
      out with the part  as ArmMotionPlanner.plan_path_with_grasp: straight up with the part until it is clear of the other
                         parts and the table, then the lift to the transport height, where the free path starts
    """

    UP = np.array([0, 0, 1.0])

    def __init__(self, planner, part_meshes_raw):
        self.planner = planner  # make_move_planner of run_motion_plan.py
        self.raw = part_meshes_raw
        self.inflated = {}  # (part_id, buffer): the raw mesh inflated by buffer, computed once per part

    def mesh_at(self, part_id, mat):
        """the part lying at mat (4x4, applied to the raw mesh); its inflated versions move with it and are handed to the planner"""
        mesh = self.raw[part_id].copy().apply_transform(mat)
        for buffer in self.planner.part_buffers():
            if (part_id, buffer) not in self.inflated:
                self.inflated[(part_id, buffer)] = offset_mesh(
                    self.raw[part_id], buffer
                )
            self.planner.add_buffered(
                mesh,
                buffer,
                self.inflated[(part_id, buffer)].copy().apply_transform(mat),
            )
        return mesh

    def drop(self, mesh):
        """a mesh of mesh_at that is not used any more (the spot was not taken)"""
        self.planner.drop_buffered(mesh)

    def at_spot(self, grasp, part_mesh, gripper_pickup, others, open_before):
        """
        part_mesh: the part where it lies (mesh_at); gripper_pickup: pose of the gripper at the pickup (4x4); others: meshes of all
        other parts as they are when this part is picked (lying or assembled); open_before: open ratio of the gripper when it
        arrives. Returns test(q): None if both straight moves exist for the arm pose q (full) at the pickup, else what is missing.
        """
        planner = self.planner
        active = planner.arm_chain.active_from_full
        open_grasp = min(grasp.open_ratio + _RETRACT_OPEN_RATIO, 1.0)
        fns = {}  # the collision tests of the planner for this spot, built when the first arm pose is tested

        def blocked():
            what, moved = planner.last_block
            return str(what) + (
                f' after {moved:.2f} cm' if moved is not None else ''
            )

        def test(q):
            if len(fns) == 0:
                fns['approach'] = [
                    planner.get_fns(None, None, others + [part_mesh], ratio)[3]
                    for ratio in (
                        [open_grasp, open_before]
                        if open_before < open_grasp
                        else [open_grasp]
                    )
                ]
                fns['carry'] = planner.get_fns(
                    part_mesh, gripper_pickup, others, grasp.open_ratio
                )[3]
            back, q_clear = get_back_retract_dir(planner, q), q
            for collision_fn in fns['approach']:
                path = planner.plan_path_straight_out(
                    q_clear, collision_fn, back
                )
                if path is None:
                    return f'no straight move into the grasp ({blocked()})'
                q_clear = path[-1]
            collision_fn = fns['carry']

            def contact(q_active):
                return collision_fn(
                    q_active, buffered=False, move_ground_buffer=False
                )

            path = planner.plan_path_straight_out(q, collision_fn, self.UP)
            if path is None:
                return f'no straight move up with the part ({blocked()})'
            lift = planner.plan_path_lift(
                path[-1],
                lambda q_active: planner.check_part_too_low(
                    q_active, grasp.open_ratio, part_mesh, gripper_pickup
                ),
                contact,
            )
            if lift is None or contact(active(lift[-1])):
                return 'no straight lift of the part to the transport height'
            collision_fn.relax([active(lift[-1])])
            if collision_fn(active(lift[-1]), part_clearance=True):
                return f'the part is not clear of everything after its lift ({collision_fn.hit})'
            return None

        return test


BANNED_SPOTS_FILE = 'banned_spots.json'


def spot_key(part_id, grasp_id, spot):
    """a spot (x, y) of the layout grid for a part and the move grasp that picks it there, as stored in BANNED_SPOTS_FILE"""
    return (
        str(part_id),
        int(grasp_id),
        round(float(spot[0]), 3),
        round(float(spot[1]), 3),
    )


def load_banned_spots(log_dir):
    """
    Spots where a part must not be laid out again: {spot_key}. run_planning_flat.py adds a spot when motion planning fails
    because of the place where the part lies (planning/plan_checks.py CAUSE_SPOT): the layout then takes the next spot for the
    same grasp. A ban holds for that grasp only, in whatever resting pose the part lies at the spot.
    """
    path = os.path.join(log_dir, BANNED_SPOTS_FILE)
    if not os.path.exists(path):
        return set()
    with open(path, 'r') as fp:
        return {
            spot_key(part_id, grasp_id, (x, y))
            for part_id, grasp_id, x, y in json.load(fp)
        }


def ban_spot(log_dir, part_id, grasp_id, spot):
    """adds the spot (x, y) of a part to BANNED_SPOTS_FILE, see load_banned_spots; returns all banned spots as a sorted list"""
    banned = load_banned_spots(log_dir) | {spot_key(part_id, grasp_id, spot)}
    with open(os.path.join(log_dir, BANNED_SPOTS_FILE), 'w') as fp:
        json.dump(sorted(banned), fp)
    return sorted(banned)


def clear_banned_spots(log_dir):
    path = os.path.join(log_dir, BANNED_SPOTS_FILE)
    if os.path.exists(path):
        os.remove(path)


def pickup_anchor(
    arm_type, pickup_shift_y=PICKUP_SHIFT_Y, pickup_center_x=0.0
):
    """
    The place on the board (x, y in cm) around which the parts are laid out: the one stored in the log (run_planning_flat.py
    --pickup-center, read by use_log_workcell), else the default pickup area, centred between the arms.
    """
    stored = get_pickup_center()
    if stored is not None:
        return np.array(stored, dtype=float)
    return np.array(
        [pickup_center_x, get_fixture_min_y(arm_type) + pickup_shift_y + 10.0]
    )


def hold_pickup_q(
    planner,
    grasp,
    pickup_mat,
    final_mat,
    lying_meshes,
    has_ft_sensor,
    move_chain,
    has_ft_sensor_move,
):
    """Finds the arm pose with which the hold arm picks the first part.

    The hold arm takes the first part from the board before anything
    else moves: the move arm is at rest, and every other part lies where
    the layout put it. A spot counts only if the hold arm reaches the
    grasp there and its arm and opened gripper are then clear of the
    lying parts, the table and the move arm. The arm pose is searched
    the same way as in run_motion_plan.py, which uses the result (the
    file PICKUP_Q_HOLD_FILE). The moves into the grasp and up with the
    part are planned later; if they fail, run_planning_flat.py drops
    the hold grasp.

    Args:
        planner (ArmMotionPlanner): Planner of the hold arm.
        grasp (Grasp): The hold grasp (a gripper pose relative to the
            part at its assembled pose).
        pickup_mat (np.ndarray): 4x4 transform of the part mesh where
            it lies (world frame, cm), as in fixture/pickup.json.
        final_mat (np.ndarray): 4x4 transform of the part mesh at its
            assembled pose.
        lying_meshes (list[trimesh.Trimesh]): Every part where it lies,
            this part included (world frame, cm).
        has_ft_sensor (bool): Whether the hold arm has a force-torque
            sensor between flange and hand (it moves the hand).
        move_chain (ArmChain): Kinematic chain of the move arm.
        has_ft_sensor_move (bool): The same for the move arm.

    Returns:
        np.ndarray | None: The arm pose (9 values: 7 joints and the 2
        fingers), or None if there is none at this spot.
    """
    rest = planner.arm_chain.active_to_full(planner.arm_chain.rest_q)
    q = get_pickup_arm_q(
        planner,
        grasp,
        pickup_mat,
        final_mat,
        arm_q_init=rest,
        has_ft_sensor=has_ft_sensor,
        optimizer='least_squares',
        regularization=1.0,
    )
    if q is None:
        return None
    # The gripper arrives opened a little wider than the grasp, as in
    # run_motion_plan.py, and the move arm waits at its rest pose.
    open_ratio = min(grasp.open_ratio + _RETRACT_OPEN_RATIO, 1.0)
    _, _, _, collision_fn = planner.get_fns(
        None,
        None,
        lying_meshes,
        open_ratio,
        arm_chain_other=move_chain,
        arm_q_other=move_chain.active_to_full(move_chain.rest_q),
        open_ratio_other=OPEN_RATIO_REST,
        has_ft_sensor_other=has_ft_sensor_move,
    )
    # Exact meshes (no planner buffers): the planner of run_motion_plan.py
    # also only asks the goal pose itself to be free of contact; its
    # straight move into the grasp is planned there.
    active = planner.arm_chain.active_from_full(q)
    if collision_fn(active, buffered=False, move_ground_buffer=False):
        return None
    return q


def run_pickup_gen(
    assembly_dir,
    log_dir,
    max_tilt=MAX_TILT,
    clearance=GROUND_CLEARANCE,
    min_tip_angle=MIN_TIP_ANGLE,
    seed=0,
    pickup_shift_y=PICKUP_SHIFT_Y,
    verbose=False,
):
    """
    Lay the parts out on the table one by one, in reverse pick order, each at the free spot nearest to the pickup area
    (pickup_anchor) where its pickup passes every real-robot check and the arm can move straight into the grasp and out with the part
    """
    return _run_pickup_gen(
        assembly_dir,
        log_dir,
        max_tilt,
        clearance,
        min_tip_angle,
        seed,
        pickup_shift_y,
        0.0,
        verbose,
    )


def _run_pickup_gen(
    assembly_dir,
    log_dir,
    max_tilt,
    clearance,
    min_tip_angle,
    seed,
    pickup_shift_y,
    pickup_center_x,
    verbose=False,
    first_by_move=False,
):
    """
    Choose a flat resting pose for every part of the optimized sequence and lay the parts out on the board
    first_by_move: the move arm brings the first part to the assembly
        spot ('place'). Set when the layout found no spot where the hold
        arm can pick that part: the function then calls itself once more
        with it.
    """
    with open(os.path.join(log_dir, 'precedence.pkl'), 'rb') as fp:
        G_preced = pickle.load(fp)
    with open(os.path.join(log_dir, 'grasps.pkl'), 'rb') as fp:
        grasps = pickle.load(fp)
    with open(os.path.join(log_dir, 'tree_opt.pkl'), 'rb') as fp:
        tree = pickle.load(fp)
    gripper_type, arm_type, has_ft_sensor = (
        grasps['gripper'],
        grasps['arm'],
        grasps['ft_sensor'],
    )

    sequence, grasps_sequence = SequenceOptimizer(
        G_preced, grasps
    ).get_sequence(tree)
    if sequence is None:
        print(f'[run_pickup_gen] No feasible sequence found in {log_dir}')
        return False
    sequence_forward, grasps_sequence_forward = (
        sequence[::-1],
        grasps_sequence[::-1],
    )

    # grasp used to pick up each part: hold grasp for the first held part, move grasp for the others ('arm': who picks it)
    pickup_grasps = {}  # part_id: (grasp, arm)
    first_part = sequence_forward[0][1]
    for i, ((part_move, part_hold), (grasps_move, grasp_hold)) in enumerate(
        zip(sequence_forward, grasps_sequence_forward)
    ):
        if i == 0:
            pickup_grasps[part_hold] = (grasp_hold, 'hold')
        pickup_grasps[part_move] = (grasps_move[0], 'move')

    part_cfg_final = load_final_part_cfg(assembly_dir, arm_type)
    gripper_meshes = load_gripper_meshes(
        gripper_type, os.path.join(project_base_dir, 'assets')
    )
    from planning.plan_checks import (
        SceneChecker,
        clear_failure,
        write_failure,
        joint_condition,
        CAUSE_HOLD_GRASP,
    )
    from planning.robot.geometry import load_part_meshes

    clear_failure(log_dir)
    part_meshes_raw = {
        k[4:]: v
        for k, v in load_part_meshes(assembly_dir, transform='none').items()
    }
    checker = (
        SceneChecker(
            arm_type,
            gripper_type,
            has_ft_sensor['move'],
            part_meshes_raw,
            {k: np.eye(4) for k in part_meshes_raw},
            part_cfg_final['mesh'],
        )
        if arm_type == 'panda'
        else None
    )

    candidates = {}
    for part_id, (grasp, _) in list(pickup_grasps.items()):
        part_mesh = part_cfg_final['mesh'][part_id]
        flat_rots = flat_rots_of(part_mesh, part_id, grasps, min_tip_angle)
        candidates[part_id] = get_flat_pickup_candidates(
            grasp,
            part_mesh,
            flat_rots,
            gripper_type,
            gripper_meshes,
            max_tilt,
            clearance,
        )
        if (
            (len(candidates[part_id]) == 0 or first_by_move)
            and part_id == first_part
            and checker is not None
        ):
            # fallback: the hold grasp cannot take the first part from the
            # board (it cannot pick the part lying flat, or the hold arm
            # reaches it on no spot of the layout: first_by_move), so the
            # move arm brings it to the assembly spot ('place'): with its
            # best move grasp whose arm pose there, the part standing on
            # the table, passes the checks
            def final_pose_ok(g):
                return (
                    len(
                        checker.check_path(
                            [g.arm_q],
                            g.open_ratio,
                            {},
                            carried=part_id,
                            inserting=True,
                        )
                    )
                    == 0
                )

            places = place_grasps_of_first_part(
                grasps['grasps'][part_id]['move'],
                part_mesh,
                flat_rots,
                gripper_type,
                gripper_meshes,
                max_tilt,
                clearance,
                final_pose_ok,
            )
            why = (
                'the hold arm reaches it on no spot of the layout'
                if first_by_move
                else 'it cannot pick the part lying flat'
            )
            if len(places) == 0:
                if first_by_move:
                    # Another hold grasp may reach the part: drop this one.
                    write_failure(
                        log_dir,
                        'run_pickup_gen',
                        part_id,
                        grasp.grasp_id,
                        f'hold grasp {grasp.grasp_id}: {why}, and no move '
                        f'grasp can bring the part to the assembly spot',
                        CAUSE_HOLD_GRASP,
                    )
                    return False
                write_failure(
                    log_dir,
                    'run_pickup_gen',
                    part_id,
                    None,
                    f'hold grasp {grasp.grasp_id} cannot pick the first part lying flat, and no move grasp can bring it to the assembly spot',
                )
                return False
            print(
                f'[run_pickup_gen] part {part_id} (held first): hold grasp '
                f'{grasp.grasp_id}: {why}; fallback: the move arm brings '
                f'it to the assembly spot with move grasp '
                f'{places[0][0].grasp_id}'
            )
            pickup_grasps[part_id] = (places[0][0], 'place')
            candidates[part_id] = places[0][1]
        if len(candidates[part_id]) == 0:
            print(
                f'[run_pickup_gen] part {part_id} '
                + (
                    'has no resting pose on the board (it tips over in every pose)'
                    if len(flat_rots) == 0
                    else f'cannot be picked lying flat with grasp {grasp.grasp_id}, run with --filter first'
                )
            )
            return False

    arm_box_move, arm_box_hold = get_dual_arm_box(arm_type)
    motion_planners = {
        'move': ArmMotionPlanner(
            get_arm_chain(arm_type, 'move'),
            gripper_type,
            has_ft_sensor['move'],
            arm_box_move,
            virtual_walls=arm_type == 'panda',
        ),
        'hold': ArmMotionPlanner(
            get_arm_chain(arm_type, 'hold'),
            gripper_type,
            has_ft_sensor['hold'],
            arm_box_hold,
        ),
    }
    final_mats = {
        part_id: get_transform_matrix_quat(
            part_cfg_final['pos'][part_id], part_cfg_final['quat'][part_id]
        )
        for part_id in pickup_grasps
    }

    # clearance of the real (move) arm at each pickup: to the parts still lying and the parts already assembled at that moment
    state_at_pick, open_before = {}, {}
    assembled = {
        first_part
    }  # the first held part stands at the assembly spot before the first insertion
    picked_before = (
        first_part if pickup_grasps[first_part][1] == 'place' else None
    )  # the part the move arm had in its gripper before
    if (
        picked_before is not None
    ):  # fallback: the move arm picks the first part itself, while every other part still lies on the board
        state_at_pick[first_part] = {k: 'pickup' for k in part_meshes_raw}
        open_before[first_part] = OPEN_RATIO_REST
    for part_move, _ in sequence_forward:
        state_at_pick[part_move] = {
            k: ('final' if k in assembled else 'pickup')
            for k in part_meshes_raw
        }
        assembled.add(part_move)
        # how far the gripper is open when it arrives (run_motion_plan.py): as after the release of the part before, at rest for the first
        open_before[part_move] = (
            OPEN_RATIO_REST
            if picked_before is None
            else min(
                pickup_grasps[picked_before][0].open_ratio
                + _RETRACT_OPEN_RATIO,
                1.0,
            )
        )
        picked_before = part_move
    # the straight moves into each grasp and out with the part, tested with the planner of run_motion_plan.py
    moves = (
        PickupMoves(
            make_move_planner(
                get_arm_chain(arm_type, 'move'),
                gripper_type,
                has_ft_sensor['move'],
                arm_box_move,
                part_cfg_final['mesh'],
            ),
            part_meshes_raw,
        )
        if checker is not None
        else None
    )
    lying_meshes = {}  # part_id: its mesh where it is laid out (parts picked by the real arm)
    banned_spots = load_banned_spots(log_dir)

    ik_cache = load_ik_cache(log_dir)
    t_start = time()
    random.seed(seed)
    np.random.seed(seed)  # IK restarts are random
    from shapely.geometry import MultiPoint, box as shapely_box
    from shapely.affinity import translate as _translate

    def shapely_translate(shape, xy):
        return _translate(shape, xoff=xy[0], yoff=xy[1])

    anchor = pickup_anchor(
        arm_type, pickup_shift_y, pickup_center_x
    )  # preferred place of the parts
    print(
        f'[run_pickup_gen] the parts are laid out around ({anchor[0]:.1f}, {anchor[1]:.1f})'
        + (
            ' (stored in the log)'
            if get_pickup_center() is not None
            else ' (default pickup area)'
        )
    )
    final_bounds = np.array(
        [mesh.bounds for mesh in part_cfg_final['mesh'].values()]
    )
    from planning.config import ASSEMBLY_ZONE_MARGIN

    zone = shapely_box(
        *(final_bounds[:, 0, :2].min(axis=0) - ASSEMBLY_ZONE_MARGIN),
        *(final_bounds[:, 1, :2].max(axis=0) + ASSEMBLY_ZONE_MARGIN),
    )
    board = shapely_box(
        -BOARD_HALF_SIZE[0] + 1.0,
        -BOARD_HALF_SIZE[1] + 1.0,
        BOARD_HALF_SIZE[0] - 1.0,
        BOARD_HALF_SIZE[1] - 1.0,
    )
    grid = np.array(
        [
            [x, y]
            for x in np.arange(
                -BOARD_HALF_SIZE[0], BOARD_HALF_SIZE[0], PLACEMENT_STEP
            )
            for y in np.arange(
                -BOARD_HALF_SIZE[1], BOARD_HALF_SIZE[1], PLACEMENT_STEP
            )
        ]
    )
    grid = grid[np.argsort(np.linalg.norm(grid - anchor, axis=1))]
    coarse = np.all(
        np.round((grid + np.array(BOARD_HALF_SIZE)) / PLACEMENT_STEP).astype(
            int
        )
        % 2
        == 0,
        axis=1,
    )  # every second spot in x and y (wide search)
    from planning.config import ROBOT_PART_CLEARANCE

    # resting pose shape of every candidate: footprint (centered) and the transform that puts it on the table at (x, y)
    shapes = {}
    for part_id in pickup_grasps:
        shapes[part_id] = []
        for candidate in candidates[part_id]:
            rot4 = np.block(
                [[candidate['rot'], np.zeros((3, 1))], [np.zeros((1, 3)), 1]]
            )
            mesh = part_cfg_final['mesh'][part_id].copy().apply_transform(rot4)
            center = mesh.bounds[:, :2].mean(axis=0)
            footprint = MultiPoint(
                mesh.convex_hull.vertices[:, :2] - center
            ).convex_hull
            shapes[part_id].append(
                (footprint, rot4, center, mesh.bounds[0, 2])
            )

    def place(part_id, k, xy):
        footprint, rot4, center, min_z = shapes[part_id][k]
        transform = np.eye(4)
        transform[:3, 3] = [xy[0] - center[0], xy[1] - center[1], -min_z]
        return (
            transform @ rot4
        )  # applied to the part in its final pose (as pose_pickup of the fixture pipeline)

    order = [part_move for part_move, _ in sequence_forward][::-1] + [
        sequence_forward[0][1]
    ]  # last picked first; the human's part last
    placed, footprints, choice, pickup_q, spots = {}, {}, {}, {}, {}
    pickup_q_hold = {}  # the first part: arm pose of the hold arm there
    for part_id in order:
        grasp, kind = pickup_grasps[part_id]
        arm = (
            'hold' if kind == 'hold' else 'move'
        )  # 'place': the first part, picked by the move arm (fallback)
        planner = motion_planners[arm]
        found, tried, violations_seen = False, 0, {}
        if arm == 'hold':
            # The hold arm picks first, while every other part lies on
            # the board (they are all placed by now: it comes last).
            hold_scene = [
                part_meshes_raw[k].copy().apply_transform(m)
                for k, m in placed.items()
            ]
        checked, fingers_at_origin = (
            set(),
            {},
        )  # (spot, candidate) pairs checked with the robot; finger footprints per candidate
        if (
            arm == 'move' and moves is not None
        ):  # all other parts as they are when this one is picked: assembled, or lying where they were laid out (they are picked later)
            others = [
                part_cfg_final['mesh'][k]
                if state == 'final'
                else lying_meshes[k]
                for k, state in state_at_pick[part_id].items()
                if k != part_id
            ]
        # 1. the spots nearest to the pickup area. 2. (real arm only) if none of them fits: the whole board, every second spot,
        #    and only where the fingers stay clear of the parts already placed (a quick test before the robot check)
        for wide in (
            [False, True] if arm == 'move' and checker is not None else [False]
        ):
            if wide:
                print(
                    f'[run_pickup_gen] part {part_id}: no spot among the {tried} nearest ones, searching the whole board'
                )
            limit = MAX_PLACEMENT_TRIES + (
                MAX_PLACEMENT_TRIES_WIDE if wide else 0
            )
            for i, xy in enumerate(grid):
                if wide and not coarse[i]:
                    continue
                for k, (footprint, _, _, _) in enumerate(shapes[part_id]):
                    shape = shapely_translate(footprint, xy)
                    if (
                        not board.contains(shape)
                        or shape.intersects(zone)
                        or any(
                            shape.distance(other) < PART_GAP
                            for other in footprints.values()
                        )
                    ):
                        continue
                    pose = place(part_id, k, xy)
                    mat = pose @ final_mats[part_id]
                    if arm == 'hold':
                        # The hold arm takes the first part from here; the
                        # spot counts only if it has an arm pose for that
                        # (until 2026-11-13 any free spot: a human took it).
                        tried += 1
                        here = part_meshes_raw[part_id].copy()
                        q = hold_pickup_q(
                            planner,
                            grasp,
                            mat,
                            final_mats[part_id],
                            hold_scene + [here.apply_transform(mat)],
                            has_ft_sensor['hold'],
                            get_arm_chain(arm_type, 'move'),
                            has_ft_sensor['move'],
                        )
                        if q is None:
                            why = (
                                'hold arm: no arm pose that picks the part '
                                'there clear of the parts and the move arm'
                            )
                            violations_seen[why] = (
                                violations_seen.get(why, 0) + 1
                            )
                            continue
                        pickup_q_hold[part_id] = q
                        found = True
                    elif (
                        checker is None
                    ):  # no real-robot checks: free spot and reachable
                        q = get_pickup_arm_q(
                            planner,
                            grasp,
                            mat,
                            final_mats[part_id],
                            arm_q_init=planner.arm_chain.active_to_full(
                                planner.arm_chain.rest_q
                            ),
                            has_ft_sensor=has_ft_sensor[arm],
                            optimizer='least_squares',
                            regularization=1.0,
                        )
                        if (
                            q is None
                            or not planner.check_virtual_walls(q)
                            or not check_lift(planner, q)
                        ):
                            continue
                        found = True
                    else:
                        if (i, k) in checked:
                            continue
                        if (
                            spot_key(part_id, grasp.grasp_id, xy)
                            in banned_spots
                        ):
                            continue  # motion planning failed with the part lying there
                        if wide:
                            if k not in fingers_at_origin:
                                fingers_at_origin[k] = finger_footprints(
                                    grasp,
                                    place(part_id, k, (0.0, 0.0)),
                                    gripper_type,
                                    gripper_meshes,
                                )
                            if not fingers_clear(
                                [
                                    shapely_translate(finger, xy)
                                    for finger in fingers_at_origin[k]
                                ],
                                footprints.values(),
                                ROBOT_PART_CLEARANCE,
                            ):
                                continue
                        checked.add((i, k))
                        tried += 1
                        checker.set_pickups(
                            part_meshes_raw, {**placed, part_id: mat}
                        )
                        gripper_pickup = get_pickup_gripper_pose(
                            grasp, mat, final_mats[part_id]
                        )
                        open_ratio_pickup = min(
                            grasp.open_ratio + _RETRACT_OPEN_RATIO, 1.0
                        )
                        # gripper or wrist too close to another part: no arm pose changes that, so none is searched
                        blocked = checker.pickup_blocked(
                            gripper_pickup,
                            open_ratio_pickup,
                            part_id,
                            state_at_pick[part_id],
                            closed_ratio=grasp.open_ratio,
                        )
                        if blocked is not None:
                            violations_seen[
                                f'pickup of part {part_id}: {blocked}'
                            ] = (
                                violations_seen.get(
                                    f'pickup of part {part_id}: {blocked}', 0
                                )
                                + 1
                            )
                            continue
                        spot_mesh, straight = (
                            None,
                            None,
                        )  # the straight moves at this spot: set up when the first arm pose passes the checks of the pickup pose
                        for score, q in pickup_ik_solutions(
                            ik_cache,
                            planner,
                            grasp,
                            part_id,
                            mat,
                            final_mats[part_id],
                            has_ft_sensor[arm],
                        ):
                            violations = checker.check_pickup(
                                q,
                                open_ratio_pickup,
                                part_id,
                                state_at_pick[part_id],
                                closed_ratio=grasp.open_ratio,
                            )
                            if not violations:
                                if straight is None:
                                    spot_mesh = moves.mesh_at(part_id, mat)
                                    straight = moves.at_spot(
                                        grasp,
                                        spot_mesh,
                                        gripper_pickup,
                                        others,
                                        open_before[part_id],
                                    )
                                missing = straight(q)
                                if missing is not None:
                                    violations = [
                                        f'pickup of part {part_id}: {missing}'
                                    ]
                            if not violations:
                                pickup_q[part_id] = (
                                    q  # used by run_motion_plan.py
                                )
                                lying_meshes[part_id] = spot_mesh
                                found = True
                                break
                            for v in violations:
                                violations_seen[v.split(' (poses')[0]] = (
                                    violations_seen.get(
                                        v.split(' (poses')[0], 0
                                    )
                                    + 1
                                )
                        if not found and spot_mesh is not None:
                            moves.drop(spot_mesh)
                    if found:
                        (
                            placed[part_id],
                            footprints[part_id],
                            choice[part_id],
                            spots[part_id],
                        ) = mat, shape, k, [float(xy[0]), float(xy[1])]
                        break
                if found or tried >= limit:
                    break
            if found:
                break
        if not found and arm == 'hold' and checker is not None:
            # The hold arm can pick the first part nowhere near the pickup
            # area: the move arm brings it to the assembly spot instead.
            # The layout starts again, because the move arm then picks
            # this part first and the other parts are checked with that.
            save_ik_cache(log_dir, ik_cache)
            print(
                f'[run_pickup_gen] part {part_id} (held first): the hold '
                f'arm can pick it on none of the {tried} spots tried; '
                f'fallback: the move arm brings it to the assembly spot'
            )
            return _run_pickup_gen(
                assembly_dir,
                log_dir,
                max_tilt,
                clearance,
                min_tip_angle,
                seed,
                pickup_shift_y,
                pickup_center_x,
                verbose,
                first_by_move=True,
            )
        if not found:
            reasons = '; '.join(
                f'{v} ({n}x)'
                for v, n in sorted(
                    violations_seen.items(), key=lambda x: -x[1]
                )[:3]
            )
            write_failure(
                log_dir,
                'run_pickup_gen',
                part_id,
                grasp.grasp_id if arm == 'move' else None,
                f'no spot within {tried} tries where the pickup passes the checks: {reasons}',
            )
            save_ik_cache(log_dir, ik_cache)
            return False
        print(
            f'[run_pickup_gen] part {part_id} placed at ({footprints[part_id].centroid.x:.1f}, {footprints[part_id].centroid.y:.1f}), resting pose candidate {choice[part_id]}'
            + (
                f', {tried} spots tried'
                + (' (whole board)' if tried > MAX_PLACEMENT_TRIES else '')
                if tried
                else ''
            )
        )
        if verbose and violations_seen:
            print(
                '    rejected because: '
                + '; '.join(f'{v} ({n}x)' for v, n in violations_seen.items())
            )

    pose_pickup_global = placed
    part_meshes_pickup = {
        part_id: part_meshes_raw[part_id].copy().apply_transform(mat)
        for part_id, mat in placed.items()
    }
    layout_bounds = np.array(
        [mesh.bounds for mesh in part_meshes_pickup.values()]
    )

    save_ik_cache(log_dir, ik_cache)
    fixture_dir = os.path.join(log_dir, 'fixture')
    os.makedirs(fixture_dir, exist_ok=True)
    with open(
        os.path.join(fixture_dir, 'pickup_q.json'), 'w'
    ) as fp:  # checked arm pose of every move-arm pickup, used by run_motion_plan.py
        json.dump(
            {
                part_id: np.asarray(q).tolist()
                for part_id, q in pickup_q.items()
            },
            fp,
            indent=4,
        )
    # The arm pose of the hold arm at the first part (empty if the move
    # arm brings that part), used by run_motion_plan.py.
    with open(os.path.join(fixture_dir, PICKUP_Q_HOLD_FILE), 'w') as fp:
        json.dump(
            {
                part_id: np.asarray(q).tolist()
                for part_id, q in pickup_q_hold.items()
            },
            fp,
            indent=4,
        )
    with open(os.path.join(fixture_dir, 'pickup.json'), 'w') as fp:
        json.dump(
            {
                part_id: get_pos_euler_from_transform_matrix(mat).tolist()
                for part_id, mat in pose_pickup_global.items()
            },
            fp,
        )
    if os.path.exists(os.path.join(fixture_dir, 'fixture.obj')):
        os.remove(
            os.path.join(fixture_dir, 'fixture.obj')
        )  # no fixture: parts lie on the board
    # 'spot' (x, y of the layout grid): what run_planning_flat.py bans if motion planning fails because of the spot; 'candidate': which resting pose and yaw of the grasp
    pickup_info = {
        part_id: {
            'arm': pickup_grasps[part_id][1],
            'grasp_id': int(pickup_grasps[part_id][0].grasp_id),
            'spot': spots[part_id],
            'candidate': int(choice[part_id]),
            **{
                k: v
                for k, v in candidates[part_id][choice[part_id]].items()
                if k != 'rot'
            },
        }
        for part_id in pickup_grasps
    }
    with open(os.path.join(fixture_dir, 'flat_pickup.json'), 'w') as fp:
        json.dump(pickup_info, fp, indent=4)
    trimesh.Scene(
        [
            trimesh.creation.box(
                bounds=[
                    [*(layout_bounds[:, 0, :2].min(axis=0) - 1.0), -0.2],
                    [*(layout_bounds[:, 1, :2].max(axis=0) + 1.0), 0],
                ]
            )
        ]
        + [mesh.copy() for mesh in part_meshes_pickup.values()]
    ).export(os.path.join(fixture_dir, 'pickup_layout.glb'))
    lower, upper = save_pickup_template(
        part_meshes_pickup,
        pickup_info,
        os.path.join(fixture_dir, 'pickup_template.pdf'),
    )
    print(
        f'[run_pickup_gen] placement template: {os.path.join(fixture_dir, "pickup_template.pdf")} ({upper[0] - lower[0]:.1f} x {upper[1] - lower[1]:.1f} cm)'
    )
    save_pickup_template(
        part_meshes_pickup,
        pickup_info,
        os.path.join(fixture_dir, 'pickup_template_a4.pdf'),
        page='a4',
    )  # A4 sheets, tiled if needed
    print(
        f'[run_pickup_gen] placement template on A4: {os.path.join(fixture_dir, "pickup_template_a4.pdf")}'
    )
    save_assembly_template(assembly_dir, log_dir)
    for part_id, info in pickup_info.items():
        print(
            f'[run_pickup_gen] part {part_id} ({info["arm"]} grasp {info["grasp_id"]}): tilt {info["tilt"]:.1f} deg, tips over at {info["tip_angle"]:.0f} deg, yaw offset {info["yaw_offset"]} deg, gripper {info["gripper_min_z"]:.2f} cm above board'
        )

    stats_path = os.path.join(log_dir, 'stats.json')
    with open(stats_path, 'r') as fp:
        stats = json.load(fp)
    stats['fixture_gen'] = {'time': round(time() - t_start, 2), 'flat': True}
    with open(stats_path, 'w') as fp:
        json.dump(stats, fp)
    return True


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--assembly-dir', type=str, required=True)
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument(
        '--filter',
        action='store_true',
        default=False,
        help='step 1: filter grasps.pkl to move grasps that can pick a flat part (run before run_seq_plan.py)',
    )
    parser.add_argument(
        '--seq-opt',
        action='store_true',
        default=False,
        help='step 2: replaces run_seq_opt.py, the first hold grasp must pick a flat part',
    )
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument(
        '--filter-hold',
        action='store_true',
        default=False,
        help='also filter hold grasps (too strict: only the first hold grasp picks from the board)',
    )
    parser.add_argument(
        '--pickup-shift-y',
        type=float,
        default=PICKUP_SHIFT_Y,
        help='[cm] shift of the pickup area towards the assembly (negative y)',
    )
    parser.add_argument(
        '--template',
        type=str,
        default=None,
        choices=['fit', 'a4', 'assembly'],
        help='only (re)draw the placement template of the existing layout (assembly: the A4 template of the assembly spot)',
    )
    parser.add_argument(
        '--max-tilt',
        type=float,
        default=MAX_TILT,
        help='max angle [deg] between gripper approach and straight down',
    )
    parser.add_argument(
        '--clearance',
        type=float,
        default=GROUND_CLEARANCE,
        help='min height [cm] of the gripper above the board',
    )
    parser.add_argument(
        '--min-tip-angle',
        type=float,
        default=MIN_TIP_ANGLE,
        help='min tipping angle [deg] of a resting pose',
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    from planning.robot.workcell import use_log_workcell

    use_log_workcell(
        args.log_dir
    )  # the place of the assembly this log is planned for
    if args.template == 'assembly':
        save_assembly_template(args.assembly_dir, args.log_dir)
        sys.exit(0)
    if args.template is not None:
        save_pickup_templates(
            args.log_dir,
            pages=[None if args.template == 'fit' else args.template],
        )
        sys.exit(0)

    if args.seq_opt:
        success = run_flat_seq_opt(
            args.assembly_dir,
            args.log_dir,
            args.max_tilt,
            args.clearance,
            args.min_tip_angle,
            args.plot,
            args.verbose,
        )
    elif args.filter:
        success = (
            filter_flat_grasps(
                args.assembly_dir,
                args.log_dir,
                args.max_tilt,
                args.clearance,
                args.min_tip_angle,
                filter_hold=args.filter_hold,
            )
            is not False
        )
    else:
        success = run_pickup_gen(
            args.assembly_dir,
            args.log_dir,
            args.max_tilt,
            args.clearance,
            args.min_tip_angle,
            args.seed,
            args.pickup_shift_y,
            args.verbose,
        )
    sys.exit(0 if success else 1)
