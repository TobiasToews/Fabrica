"""
Checks of what the real (move) arm executes, shared by run_pickup_gen.py (pickup poses), run_motion_plan.py (re-plan a path
that fails) and verify_plan.py (final check). Exact geometry, no planner buffers.

Rules (clearances in planning/config.py):
  arm links 2-7 vs every part (lying or assembled, except the carried one) >= ROBOT_PART_CLEARANCE
  gripper vs lying parts >= ROBOT_PART_CLEARANCE, except the part being picked: no contact with it on the whole approach
      (the gripper is still open on that path; it touches the part only when it closes, which is a gripper step, not a path)
  gripper vs assembled parts: no contact, including the part just released (the gripper has opened before the path starts)
  gripper and carried part vs assembled parts while passing by >= ASSEMBLY_CLEARANCE: a path may start closer (leaving after a
      release) and may end closer (arriving to insert), but once it was clear it may come closer only to stay close until its end
  gripper (hand, fingers) vs the robot's own links 0-5 >= ROBOT_SELF_CLEARANCE
  carried part vs lying parts: no contact; >= ROBOT_PART_CLEARANCE from the first pose on where it is that clear (it lay among them
      and may be lifted straight up beside a taller neighbour); vs assembled parts no contact (not checked while inserting)
  carried part above the table: never below it; outside the assembly zone (assembly + ASSEMBLY_ZONE_MARGIN), once lifted,
      >= TRANSFER_PART_CLEARANCE; inside the zone >= ZONE_PART_TABLE_CLEARANCE until the last stretch of the path, where it comes
      down to its goal (not checked while inserting: an insertion may end on the table)
  gripper >= GRIPPER_TABLE_CLEARANCE, arm links >= ROBOT_PART_CLEARANCE above the table
  franka-interface virtual walls >= VIRTUAL_WALL_SAFETY
  every move grasp: finger overlap with the part >= GRASP_MIN_FINGER_OVERLAP (see grasp_finger_overlap)
  every pose: smallest singular value of the arm Jacobian >= MIN_SIGMA (MIN_SIGMA_INSERTION while inserting),
              every joint >= MIN_JOINT_LIMIT_MARGIN from its limits

Not checked (TODO):
  the wrist camera mount and its cable are not in the robot model (real_robot/_tobi_stuff/RealSenseD435_Camera_Mount_on_angle.step):
      the gripper is checked as the bare Franka hand, so the mount can hit the arm, parts or the table although every rule passes
  carried part vs the robot's own links (the parts are small; needed once parts get much larger)
"""

import os
import sys

project_base_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)
sys.path.append(project_base_dir)

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from assets.transform import get_transform_matrix_quat, mat_to_pos_quat
from planning.config import (
    ROBOT_PART_CLEARANCE,
    GRIPPER_TABLE_CLEARANCE,
    TRANSFER_PART_CLEARANCE,
    GRASP_MIN_FINGER_OVERLAP,
    MIN_SIGMA,
    MIN_SIGMA_INSERTION,
    MIN_JOINT_LIMIT_MARGIN,
    ASSEMBLY_ZONE_MARGIN,
    ROBOT_SELF_CLEARANCE,
    ASSEMBLY_CLEARANCE,
    ZONE_PART_TABLE_CLEARANCE,
)
from planning.robot.geometry import (
    load_gripper_meshes,
    transform_gripper_meshes,
    get_gripper_finger_names,
    get_gripper_meshes_transforms,
)
from planning.robot.motion_plan_arm import (
    ArmMotionPlanner,
    VIRTUAL_WALL_SAFETY,
)
from planning.robot.util_arm import (
    get_arm_chain,
    get_gripper_pos_quat_from_arm_q,
    get_virtual_wall_margin,
    get_gripper_basis_directions,
)
from planning.robot.workcell import get_panda_base_z, get_panda_desk_z

from planning.robot.panda_condition import (
    PANDA_JOINT_LIMITS,
    PANDA_DH,
    panda_jacobian,
    joint_condition,
)  # shared with the motion planner

ARM_LINKS_CHECKED = [
    'panda_link2',
    'panda_link3',
    'panda_link4',
    'panda_link5',
    'panda_link6',
    'panda_link7',
]
SELF_LINKS_CHECKED = [
    'panda_link0',
    'panda_link1',
    'panda_link2',
    'panda_link3',
    'panda_link4',
    'panda_link5',
]  # links the gripper can reach (6 and 7 carry it)


def grasp_finger_overlap(grasp, gripper_type, gripper_meshes=None):
    """
    How far (cm) the part reaches into the fingers, measured from the fingertip along the finger, from the grasp's contact
    points (final pose). With the gripper e cm too high the overlap shrinks by e, so this is the tolerance of the grasp.
    Grasps of the flat pipeline store the depth per finger, measured on the grid of the finger pads (flat_grasp.pad_contact):
    for them the weaker finger counts (the first one to lose the part). Older grasps only have a few sampled contact points:
    for them it is the deepest of those points, of either finger.
    """
    depths = getattr(grasp, 'contact_depths', None)
    if depths is not None:
        return float(min(depths)) if len(depths) > 0 else 0.0
    if grasp.contact_points is None or len(grasp.contact_points) == 0:
        return 0.0
    if gripper_meshes is None:
        gripper_meshes = load_gripper_meshes(
            gripper_type, os.path.join(project_base_dir, 'assets')
        )
    meshes = transform_gripper_meshes(
        gripper_type,
        gripper_meshes,
        grasp.pos,
        grasp.quat,
        np.eye(4),
        grasp.open_ratio,
    )
    fingers = np.vstack(
        [
            meshes[name].vertices
            for name in get_gripper_finger_names(gripper_type)
        ]
    )
    finger_dirs = np.array(
        fingers.mean(axis=0)
        - meshes[
            [
                n
                for n in meshes
                if n not in get_gripper_finger_names(gripper_type)
            ][0]
        ].vertices.mean(axis=0)
    )
    # from the hand towards the fingertips: the gripper's own axis (the line between the mesh centres is tilted by about 6 degrees
    # for the panda hand, which shifted the result by up to 0.4 cm), with the sign of that line
    axis = R.from_quat(np.asarray(grasp.quat)[[1, 2, 3, 0]]).apply(
        get_gripper_basis_directions(gripper_type)[0]
    )
    axis = axis / np.linalg.norm(axis) * np.sign(axis @ finger_dirs)
    tip = (fingers @ axis).max()
    return float(tip - (np.array(grasp.contact_points) @ axis).min())


WRIST_LINK = 'panda_link7'  # the arm link that carries the hand: where it is follows from the gripper pose alone
PICKUP_POSE_SLACK = 0.1  # cm, by how much the tests made before an arm pose is searched stay below the checked limits (SceneChecker.pickup_blocked)
DISTANCE_TOL = 1e-3  # cm: a distance that equals its limit up to rounding keeps it (parts laid out on a grid stand exactly 1.0 cm apart, the distance query then returns 0.9999 or 1.0001)


class SceneChecker:
    """Exact clearance checks of the move arm in the real workcell"""

    def __init__(
        self,
        arm_type,
        gripper_type,
        has_ft_sensor,
        part_meshes_raw,
        pickup_mats,
        final_meshes,
    ):
        assert arm_type == 'panda', (
            'the real-robot checks are written for the panda workcell'
        )
        self.chain = get_arm_chain(arm_type, 'move')
        self.planner = ArmMotionPlanner(
            self.chain, gripper_type, has_ft_sensor
        )
        self.gripper_type = gripper_type
        self.has_ft_sensor = has_ft_sensor
        self.table_z = get_panda_base_z() + get_panda_desk_z()
        self.part_ids = list(part_meshes_raw.keys())
        self.pickup_meshes = {
            k: part_meshes_raw[k].copy().apply_transform(pickup_mats[k])
            for k in self.part_ids
        }
        self.final_meshes = final_meshes
        bounds = np.array([mesh.bounds for mesh in final_meshes.values()])
        self.zone = (
            bounds[:, 0, :2].min(axis=0) - ASSEMBLY_ZONE_MARGIN,
            bounds[:, 1, :2].max(axis=0) + ASSEMBLY_ZONE_MARGIN,
        )
        self.arm_mgr = trimesh.collision.CollisionManager()
        for name in ARM_LINKS_CHECKED:
            self.arm_mgr.add_object(name, self.planner.arm_meshes[name])
        self.self_mgr = trimesh.collision.CollisionManager()
        for name in SELF_LINKS_CHECKED:
            self.self_mgr.add_object(name, self.planner.arm_meshes[name])
        self.gripper_mgr = trimesh.collision.CollisionManager()
        for name, mesh in self.planner.gripper_meshes.items():
            self.gripper_mgr.add_object(name, mesh)
        self.wrist_mgr = (
            trimesh.collision.CollisionManager()
        )  # the wrist alone, for pickup_blocked
        self.wrist_mgr.add_object(
            WRIST_LINK, self.planner.arm_meshes[WRIST_LINK]
        )
        self._wrist_offset = None
        self.arm_hull = {
            name: self.planner.arm_meshes[name].convex_hull.vertices
            for name in ARM_LINKS_CHECKED
        }
        self.gripper_hull = {
            name: mesh.convex_hull.vertices
            for name, mesh in self.planner.gripper_meshes.items()
        }

    def set_pickups(self, part_meshes_raw, pickup_mats):
        self.pickup_meshes = {
            k: part_meshes_raw[k].copy().apply_transform(pickup_mats[k])
            for k in pickup_mats
        }

    def part_mesh(self, part_id, where):
        return (
            self.final_meshes[part_id]
            if where == 'final'
            else self.pickup_meshes[part_id]
        )

    def manager(self, meshes):
        mgr = trimesh.collision.CollisionManager()
        for name, mesh in meshes.items():
            mgr.add_object(name, mesh)
        return mgr

    def gripper_transform(self, q):
        return get_transform_matrix_quat(
            *get_gripper_pos_quat_from_arm_q(
                self.chain,
                q,
                self.gripper_type,
                has_ft_sensor=self.has_ft_sensor,
            )
        )

    def check_path(
        self,
        path,
        open_ratio,
        state,
        carried=None,
        pick_part=None,
        released_part=None,
        inserting=False,
        label='',
    ):
        """
        path: list of full (9) or active (7) arm qs of the move arm
        state: {part_id: 'pickup' (lying) / 'final' (assembled)} before the path, without the carried part
        carried: part in the gripper (it lies at its pickup pose when the path starts, unless inserting)
        pick_part: part picked after this path (lying): the open gripper must not touch it on any pose of the path
        released_part: part released before this path (assembled): checked like every assembled part, only named in the message
        returns a list of violation strings (empty: ok)
        """
        path = [np.asarray(q) for q in path]
        path = [
            self.chain.active_to_full(q) if len(q) == 7 else q for q in path
        ]
        lying = {k for k, v in state.items() if v == 'pickup' and k != carried}
        assembled = {
            k for k, v in state.items() if v == 'final' and k != carried
        }
        all_parts = self.manager(
            {k: self.part_mesh(k, state[k]) for k in lying | assembled}
        )
        lying_strict = self.manager(
            {k: self.pickup_meshes[k] for k in lying if k != pick_part}
        )
        assembled_mgr = self.manager(
            {k: self.final_meshes[k] for k in assembled}
        )
        pick_mgr = (
            self.manager({pick_part: self.pickup_meshes[pick_part]})
            if pick_part is not None
            else None
        )
        if carried is not None:
            # the carried part follows the gripper: transport paths start at its pickup pose, insertion paths end at its final pose
            carried_mesh = (
                self.final_meshes[carried]
                if inserting
                else self.pickup_meshes[carried]
            )
            T_grip_ref = self.gripper_transform(
                path[-1] if inserting else path[0]
            )
            carried_mgr = self.manager({'carried': carried_mesh})
            carried_hull = carried_mesh.convex_hull.vertices

        violations = {}

        def add(key, k):
            violations.setdefault(key, []).append(k)

        lifted = False
        near_assembly = {
            'gripper': [],
            'carried part': [],
        }  # per pose: (distance to the assembled parts, nearest part)
        near_lying = []  # per pose: (distance of the carried part to the lying parts, nearest part)
        zone_low = []  # per pose: height of the carried part above the table inside the assembly zone (inf: no point of it is in the zone)
        for k, q in enumerate(path):
            arm_tf, grip_tf, _ = self.planner.get_meshes_transforms(
                q, open_ratio, None, None
            )
            for name in ARM_LINKS_CHECKED:
                self.arm_mgr.set_transform(name, arm_tf[name])
            for name in SELF_LINKS_CHECKED:
                self.self_mgr.set_transform(name, arm_tf[name])
            for name, tf in grip_tf.items():
                self.gripper_mgr.set_transform(name, tf)
            T_grip = self.gripper_transform(q)

            # table
            z_grip = min(
                (verts @ grip_tf[n][:3, :3].T + grip_tf[n][:3, 3])[:, 2].min()
                for n, verts in self.gripper_hull.items()
            )
            z_arm = min(
                (verts @ arm_tf[n][:3, :3].T + arm_tf[n][:3, 3])[:, 2].min()
                for n, verts in self.arm_hull.items()
            )
            if z_grip < self.table_z + GRIPPER_TABLE_CLEARANCE:
                add(
                    f'gripper closer than {GRIPPER_TABLE_CLEARANCE} cm to the table',
                    k,
                )
            if z_arm < self.table_z + ROBOT_PART_CLEARANCE:
                add(
                    f'arm closer than {ROBOT_PART_CLEARANCE} cm to the table',
                    k,
                )
            # walls
            if get_virtual_wall_margin(self.chain, q) < VIRTUAL_WALL_SAFETY:
                add(
                    f'closer than {VIRTUAL_WALL_SAFETY} cm to a virtual wall',
                    k,
                )
            # singularities and joint limits (all joints)
            sigma, margin, joint = joint_condition(
                self.chain.active_from_full(q)
            )
            min_sigma = MIN_SIGMA_INSERTION if inserting else MIN_SIGMA
            if sigma < min_sigma:
                add(
                    f'too close to a singularity (Jacobian sigma_min < {min_sigma})',
                    k,
                )
            if margin < MIN_JOINT_LIMIT_MARGIN:
                add(
                    f'joint {joint} closer than {MIN_JOINT_LIMIT_MARGIN} rad to its limit',
                    k,
                )
            # arm links vs parts
            if len(lying | assembled) > 0:
                d, names = self.arm_mgr.min_distance_other(
                    all_parts, return_names=True
                )
                if d < ROBOT_PART_CLEARANCE - DISTANCE_TOL:
                    add(
                        f'arm {names[0]} closer than {ROBOT_PART_CLEARANCE} cm to part {names[1]}',
                        k,
                    )
            # gripper vs lying parts
            if len(lying_strict._objs) > 0:
                d, names = self.gripper_mgr.min_distance_other(
                    lying_strict, return_names=True
                )
                if d < ROBOT_PART_CLEARANCE - DISTANCE_TOL:
                    add(
                        f'gripper {names[0]} closer than {ROBOT_PART_CLEARANCE} cm to lying part {names[1]}',
                        k,
                    )
            if pick_mgr is not None and self.gripper_mgr.in_collision_other(
                pick_mgr
            ):
                add(f'gripper touches part {pick_part} before grasping it', k)
            # gripper vs assembled parts (the part just released is one of them)
            if len(assembled_mgr._objs) > 0:
                hit, names = self.gripper_mgr.in_collision_other(
                    assembled_mgr, return_names=True
                )
                if hit:
                    part = sorted(names)[0][1]
                    add(
                        f'gripper hits part {part} (just released)'
                        if part == released_part
                        else f'gripper hits assembled part {part}',
                        k,
                    )
                d, names = self.gripper_mgr.min_distance_other(
                    assembled_mgr, return_names=True
                )
                near_assembly['gripper'].append((d, names[1]))
            # gripper vs the robot itself
            d, names = self.self_mgr.min_distance_other(
                self.gripper_mgr, return_names=True
            )
            if d < ROBOT_SELF_CLEARANCE - DISTANCE_TOL:
                add(
                    f'gripper {names[1]} closer than {ROBOT_SELF_CLEARANCE} cm to the robot ({names[0]})',
                    k,
                )
            # carried part
            if carried is not None:
                T_part = T_grip @ np.linalg.inv(T_grip_ref)
                carried_mgr.set_transform('carried', T_part)
                v = carried_hull @ T_part[:3, :3].T + T_part[:3, 3]
                if v[:, 2].min() < self.table_z - 0.05:
                    add(f'carried part {carried} below the table', k)
                outside = ~np.all(
                    (v[:, :2] >= self.zone[0]) & (v[:, :2] <= self.zone[1]),
                    axis=1,
                )
                if not inserting:
                    low = (
                        v[outside, 2].min() - self.table_z
                        if outside.any()
                        else np.inf
                    )
                    if low >= TRANSFER_PART_CLEARANCE - 1e-3:
                        lifted = True
                    elif lifted and outside.any():
                        add(
                            f'carried part {carried} lower than {TRANSFER_PART_CLEARANCE} cm above the table outside the assembly zone',
                            k,
                        )
                    zone_low.append(
                        v[~outside, 2].min() - self.table_z
                        if not outside.all()
                        else np.inf
                    )
                    if len(lying_strict._objs) > 0:
                        d, names = carried_mgr.min_distance_other(
                            lying_strict, return_names=True
                        )
                        if carried_mgr.in_collision_other(lying_strict):
                            add(
                                f'carried part {carried} hits lying part {names[1]}',
                                k,
                            )
                        near_lying.append((d, names[1]))
                    if len(assembled_mgr._objs) > 0:
                        if carried_mgr.in_collision_other(assembled_mgr):
                            add(f'carried part {carried} hits the assembly', k)
                        d, names = carried_mgr.min_distance_other(
                            assembled_mgr, return_names=True
                        )
                        near_assembly['carried part'].append((d, names[1]))
        # the carried part lay among the lying parts: it may be closer than ROBOT_PART_CLEARANCE to them until it is clear of them for
        # the first time (e.g. lifted straight up beside a taller neighbour), then it must stay clear
        clear = [
            d >= ROBOT_PART_CLEARANCE - DISTANCE_TOL for d, _ in near_lying
        ]
        if True in clear:
            for k in range(clear.index(True), len(clear)):
                if not clear[k]:
                    add(
                        f'carried part {carried} closer than {ROBOT_PART_CLEARANCE} cm to lying part {near_lying[k][1]}',
                        k,
                    )
        # inside the assembly zone the carried part keeps ZONE_PART_TABLE_CLEARANCE to the table; lower only in one stretch at the end
        # of the path (it is set down, or it reaches an insertion start pose that lies lower)
        high = [
            low >= ZONE_PART_TABLE_CLEARANCE - DISTANCE_TOL for low in zone_low
        ]
        if True in high:
            for k in range(len(high) - 1 - high[::-1].index(True)):
                if not high[k]:
                    add(
                        f'carried part {carried} lower than {ZONE_PART_TABLE_CLEARANCE} cm above the table in the assembly zone',
                        k,
                    )
        # passing by the assembly: closer than ASSEMBLY_CLEARANCE only in one stretch at the start and one at the end of the path
        for what, values in near_assembly.items():
            clear = [d >= ASSEMBLY_CLEARANCE - DISTANCE_TOL for d, _ in values]
            if True not in clear:
                continue  # the whole path stays close (e.g. a short move at the assembly)
            first, last = (
                clear.index(True),
                len(clear) - 1 - clear[::-1].index(True),
            )
            for k in range(first, last + 1):
                if not clear[k]:
                    add(
                        f'{what} passes closer than {ASSEMBLY_CLEARANCE} cm to assembled part {values[k][1]}',
                        k,
                    )
        return [
            f'{label}{key} (poses {ks[0]}..{ks[-1]} of {len(path)}, {len(ks)}x)'
            for key, ks in violations.items()
        ]

    def check_pickup(self, q, open_ratio, part_id, state, closed_ratio=None):
        """
        robot at the pickup pose of part_id (lying), state as in check_path: clearance to all other parts, with the opened gripper
        as it arrives (open_ratio) and, if closed_ratio is given, with the gripper closed on the part: that is the first pose of
        the transport, and a finger that closes can come closer to a neighbour beside it than the opened finger was
        """
        violations = self.check_path(
            [q],
            open_ratio,
            state,
            pick_part=part_id,
            label=f'pickup of part {part_id}: ',
        )
        if not violations and closed_ratio is not None:
            violations = self.check_path(
                [q],
                closed_ratio,
                state,
                carried=part_id,
                label=f'pickup of part {part_id}, gripper closed: ',
            )
        return violations

    def wrist_offset(self):
        """pose of the wrist link in the frame of the gripper (the same for every arm pose: the hand is bolted to this link)"""
        if self._wrist_offset is None:
            q = self.chain.active_to_full(self.chain.rest_q)
            arm_tf, _, _ = self.planner.get_meshes_transforms(
                q, 0.0, None, None
            )
            self._wrist_offset = (
                np.linalg.inv(self.gripper_transform(q)) @ arm_tf[WRIST_LINK]
            )
        return self._wrist_offset

    def pickup_blocked(
        self,
        T_grip,
        open_ratio,
        part_id,
        state,
        closed_ratio=None,
        slack=PICKUP_POSE_SLACK,
    ):
        """
        The clearances of check_pickup that do not depend on the arm pose, for the gripper at T_grip (4x4, the gripper pose the
        grasp asks for where the part lies): the gripper to the other lying parts, opened as it arrives and closed on the part,
        and the wrist to every part. Returns what is too close (the text of check_path), or None.
        Used before an arm pose is searched (run_pickup_gen.py): the search takes seconds, and among lying parts most spots
        fail for the gripper. It is not a check of its own, every spot that passes here is checked by check_pickup. An arm
        pose reaches T_grip only up to the tolerance of the IK, so the limit is lower by slack here: a spot that close to
        the limit goes on to the full check.
        """
        limit = ROBOT_PART_CLEARANCE - DISTANCE_TOL - slack
        others = {
            k: self.part_mesh(k, where)
            for k, where in state.items()
            if k != part_id
        }
        self.wrist_mgr.set_transform(WRIST_LINK, T_grip @ self.wrist_offset())
        d, names = self.wrist_mgr.min_distance_other(
            self.manager({**others, part_id: self.pickup_meshes[part_id]}),
            return_names=True,
        )
        if d < limit:
            return f'arm {WRIST_LINK} closer than {ROBOT_PART_CLEARANCE} cm to part {names[1]}'
        lying = self.manager(
            {k: mesh for k, mesh in others.items() if state[k] == 'pickup'}
        )
        if len(lying._objs) == 0:
            return None
        pos, quat = mat_to_pos_quat(T_grip)
        for ratio in [open_ratio] + (
            [closed_ratio] if closed_ratio is not None else []
        ):
            for name, tf in get_gripper_meshes_transforms(
                self.gripper_type,
                self.planner.gripper_meshes,
                pos,
                quat,
                np.eye(4),
                ratio,
            ).items():
                self.gripper_mgr.set_transform(name, tf)
            d, names = self.gripper_mgr.min_distance_other(
                lying, return_names=True
            )
            if d < limit:
                return f'gripper {names[0]} closer than {ROBOT_PART_CLEARANCE} cm to lying part {names[1]}'
        return None


FAILURE_FILE = 'planning_failure.json'


def clear_failure(log_dir):
    path = os.path.join(log_dir, FAILURE_FILE)
    if os.path.exists(path):
        os.remove(path)


CAUSE_GRASP = 'grasp'  # the arm or the gripper cannot do it with this move grasp: another grasp of the part can
CAUSE_SPOT = 'spot'  # the place where the part lies does not work (neighbours, approach, lift): the same grasp can at another spot
CAUSE_INSERTION = 'insertion'  # the part itself has no way to where its insertion starts, however it is grasped
CAUSE_TEXT = {
    CAUSE_GRASP: 'the move grasp',
    CAUSE_SPOT: 'the spot where the part lies',
    CAUSE_INSERTION: 'the insertion itself, whatever the grasp',
}


def write_failure(log_dir, stage, part, grasp_id, reason, cause=CAUSE_GRASP):
    """
    Tells run_planning_flat.py why planning failed and what to change (cause, see CAUSE_*): drop the move grasp and take the next
    best sequence, lay the part out at another spot with the same grasp, or stop because no grasp and no spot can help.
    """
    import json

    assert cause in CAUSE_TEXT
    with open(os.path.join(log_dir, FAILURE_FILE), 'w') as fp:
        json.dump(
            {
                'stage': stage,
                'part': part,
                'grasp_id': None if grasp_id is None else int(grasp_id),
                'reason': reason,
                'cause': cause,
            },
            fp,
            indent=4,
        )
    print(
        f'[{stage}] FAILED for part {part} (move grasp {grasp_id}), cause: {CAUSE_TEXT[cause]}: {reason}'
    )
