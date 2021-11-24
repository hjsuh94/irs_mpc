from irs_lqr.irs_lqr import IrsLqr
import os
import numpy as np

from irs_lqr.irs_lqr_quasistatic import IrsLqrQuasistaticParameters
from quasistatic_simulator.examples.model_paths import models_dir

object_sdf_path = os.path.join(models_dir, "sphere_yz_rotation_r_0.25m.sdf")
model_directive_path = os.path.join(models_dir, "planar_hand.yml")

# robots.
Kp = np.array([50, 25], dtype=float)
robot_l_name = "arm_left"
robot_r_name = "arm_right"
robot_stiffness_dict = {robot_l_name: Kp, robot_r_name: Kp}

# object
object_name = "sphere"
object_sdf_dict = {object_name: object_sdf_path}

# environment
h = 0.1
gravity = np.array([0, 0, -10.])
contact_detection_tolerance = 10.0
gradient_lstsq_tolerance = 1e-3

# gradient exac
gradient_mode = "first_order"
decouple_AB = True

# num_workers
use_workers = True
num_workers = 10
task_stride = 1
num_iters = 10
num_samples = 50
