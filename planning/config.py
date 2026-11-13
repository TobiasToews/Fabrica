FINGER_BUFFER = 0.25  # buffer distance for robot finger collision checking (cm)
HAND_KNUCKLE_BUFFER = 1.0  # buffer distance for robot hand knuckle collision checking (cm)
ARM_BUFFER = 1.0  # buffer distance for robot arm collision checking (cm)

RETRACT_OPEN_RATIO = 0.1  # extra open ratio for retract grasp
RETRACT_DELTA_NEAR = 1.0 # incremental distance for retract grasp when near
RETRACT_DELTA_FAR = 5.0 # incremental distance for retract grasp when far away
TRANSFER_PART_CLEARANCE = 3.0 # cm, min height of a carried part above the table between pickup and the assembly approach (real robot: grasp slip, tracking error); was 5.0 until 2026-11-10
TRANSFER_LIFT_DELTA = 1.0 # cm, vertical step when lifting a carried part to TRANSFER_PART_CLEARANCE

CHECK_GRIPPERS_INTERLOCK = False # whether to check grippers interlock during dual-arm planning

OPEN_RATIO_REST = 0.5 # open ratio for resting position

# real-robot plan checks (planning/plan_checks.py), enforced by run_pickup_gen.py, run_motion_plan.py and verify_plan.py
ROBOT_PART_CLEARANCE = 1.0 # cm, arm links to every part and gripper/carried part to lying parts (except the part being picked)
GRIPPER_TABLE_CLEARANCE = 0.2 # cm, gripper to the table
GRASP_MIN_FINGER_OVERLAP = 0.3 # cm, how far the part reaches into the fingers from the fingertip (tolerance against a too high gripper); was 0.5 until 2026-11-11
ASSEMBLY_ZONE_MARGIN = 4.0 # cm, the assembly zone is the footprint of the finished assembly plus this margin: only there a carried part may come lower than TRANSFER_PART_CLEARANCE, and no part is laid out there. Was 3.0 until 2026-11-12: gamepad part 4 (6 cm long, pushed in sideways) is clear of its neighbours only where it reaches 3.3 cm beyond the footprint
ZONE_PART_TABLE_CLEARANCE = 0.2 # cm, a carried part to the table inside the assembly zone (as the gripper), until the last stretch of its path, where it comes down to its goal. The motion planner keeps it with its table slab of this thickness; until 2026-11-11 it kept its table buffer (0.78 cm) there, so a part inserted just above the table (gamepad part 4: 0.65 cm) had no approach
ROBOT_SELF_CLEARANCE = 1.0 # cm, gripper (hand, fingers) to the robot's own links 0-5
ASSEMBLY_CLEARANCE = 1.0 # cm, gripper and carried part to the assembled parts while passing by; closer only where a path starts (leaving after a release) or ends (arriving to insert), never touching
# the motion planner of the real arm (ArmMotionPlanner.strict) plans with these reserves on top of the checked values, so that
# a planned path passes the checks (poses between two planned waypoints, inflated meshes are approximate)
PLANNER_MARGIN = 0.2 # cm, added to ROBOT_PART_CLEARANCE / ASSEMBLY_CLEARANCE / VIRTUAL_WALL_SAFETY
PLANNER_TABLE_MARGIN = 0.5 # cm, added to TRANSFER_PART_CLEARANCE
PLANNER_JOINT_MARGIN = 0.03 # rad, added to MIN_JOINT_LIMIT_MARGIN
PLANNER_SIGMA_MARGIN = 0.01 # added to MIN_SIGMA
STRAIGHT_STEP = 0.25 # cm, step of the straight moves out of / into a grasp, release or insertion start (fine: next to a wall the stretch where the clearances hold can be a few mm long)
STRAIGHT_MAX = 20.0 # cm, a straight move that is not clear of everything by then fails
STRAIGHT_MAX_JOINT_STEP = 0.1 # rad, largest joint change per straight step (larger: the arm would flip)
# grasps of the flat pipeline (planning/flat_grasp.py, run_grasp_arm_gen.py --flat)
PAD_CELL = 0.025 # cm, grid on the gripping pad of each finger on which the contact with the part is evaluated
PAD_STORE_CELL = 0.1 # cm, grid of the contact points that are stored with a grasp (plus the deepest one of each finger)
PAD_CLOSING = 0.04 # cm, a pad point is in contact if the part surface is at most this far in front of it (the original contact test closes each finger by this much)
PAD_PENETRATION = 0.2 # cm, part surface up to this far behind the pad plane still belongs to the grasp: the fingers stop there first (the collision test of the grasp generation opens the fingers by 0.05 = 0.2 cm, deeper grasps are dropped there)
FLAT_MIN_GRASPS = 10 # a part with fewer usable move grasps gets more candidates (FLAT_RESAMPLE_PT further surface point pairs); was 3 in the batch of 2026-11-11: human_normal part 4 had exactly 3 grasps, all failed in motion planning
FLAT_RESAMPLE_PT = 600 # antipodal pair budget of that second round (the first round uses --n-surface-pt, 200 in the pipeline)
FLAT_LYING_PT = 200 # antipodal pair budget for the candidates made on the lying part (flat_grasp.lying_grasp_candidates), shared by its resting poses; the candidates made at the assembled pose use --n-surface-pt (200). 0: only the candidates of the assembled pose, as until 2026-11-12
FLAT_LYING_ANGLES = (0.0, 20.0, -20.0) # deg, hand directions of those candidates: straight above the pair, and leaning this much about the line between the fingers
FLAT_HOLD_ROOM = 5 # hold grasps are chosen at random; for every part that can be inserted while this part is held, at least this many hold grasps whose gripper leaves room for that insertion are kept as well, if they exist (car: 3 of the 441 hold grasps of the chassis leave room for the body, the random 100 can miss them all). 0: only the random choice
FLAT_FALLBACK_POSES = True # a part without any move grasp for its stable resting poses may lie in a less stable one (tips over between MIN_TIP_ANGLE_FALLBACK and MIN_TIP_ANGLE, marked on the template); False: such a part has no grasp
FLAT_RATING_WEIGHTS = {'grip': 0.4, 'balance': 0.25, 'upright': 0.2, 'insertion': 0.15} # rate_flat_grasp, sum 1
FLAT_RATING_DEPTH = 1.0 # cm, grip: a contact that reaches this far into both fingers counts as full
FLAT_RATING_AREA = 2.0 # cm^2, grip: this much contact area on both pads together counts as full (one pad is about 3 cm^2)
FLAT_RATING_LEVER = 3.0 # cm, balance: the centre of mass this far beside the line between the fingers counts as worst
MAX_PLAN_ATTEMPTS = 8 # re-plans of a path that fails the checks
MAX_SEQUENCE_ATTEMPTS = 10 # sequences tried by run_planning_flat.py before giving up
MIN_SIGMA = 0.10 # smallest singular value of the 6x7 arm Jacobian (m, rad), every pose of the real arm: distance to any singularity
MIN_SIGMA_INSERTION = 0.15 # same while inserting (cartesian impedance control reacts strongly near singularities)
MIN_JOINT_LIMIT_MARGIN = 0.5 # rad, every joint stays this far from both of its limits (e.g. elbow not folded up)
