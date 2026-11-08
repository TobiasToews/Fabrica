FINGER_BUFFER = 0.25  # buffer distance for robot finger collision checking (cm)
HAND_KNUCKLE_BUFFER = 1.0  # buffer distance for robot hand knuckle collision checking (cm)
ARM_BUFFER = 1.0  # buffer distance for robot arm collision checking (cm)

RETRACT_OPEN_RATIO = 0.1  # extra open ratio for retract grasp
RETRACT_DELTA_NEAR = 1.0 # incremental distance for retract grasp when near
RETRACT_DELTA_FAR = 5.0 # incremental distance for retract grasp when far away
TRANSFER_PART_CLEARANCE = 5.0 # cm, min height of a carried part above the table between pickup and the assembly approach (real robot: grasp slip, tracking error)
TRANSFER_LIFT_DELTA = 1.0 # cm, vertical step when lifting a carried part to TRANSFER_PART_CLEARANCE

CHECK_GRIPPERS_INTERLOCK = False # whether to check grippers interlock during dual-arm planning

OPEN_RATIO_REST = 0.5 # open ratio for resting position

# real-robot plan checks (planning/plan_checks.py), enforced by run_pickup_gen.py, run_motion_plan.py and verify_plan.py
ROBOT_PART_CLEARANCE = 1.0 # cm, arm links to every part and gripper/carried part to lying parts (except the part being picked)
GRIPPER_TABLE_CLEARANCE = 0.2 # cm, gripper to the table
GRASP_MIN_FINGER_OVERLAP = 0.5 # cm, how far the part reaches into the fingers from the fingertip (tolerance against a too high gripper)
CONTACT_RADIUS = 5.0 # cm, around the grasp / release pose the gripper may touch the part it picks or releases
MAX_PLAN_ATTEMPTS = 8 # re-plans of a path that fails the checks
MAX_SEQUENCE_ATTEMPTS = 10 # sequences tried by run_planning_flat.py before giving up
MIN_SIGMA = 0.10 # smallest singular value of the 6x7 arm Jacobian (m, rad), every pose of the real arm: distance to any singularity
MIN_SIGMA_INSERTION = 0.15 # same while inserting (cartesian impedance control reacts strongly near singularities)
MIN_JOINT_LIMIT_MARGIN = 0.5 # rad, every joint stays this far from both of its limits (e.g. elbow not folded up)
