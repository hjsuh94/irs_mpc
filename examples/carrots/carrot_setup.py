import os
import numpy as np

from quasistatic_simulator.examples.model_paths import models_dir

object_sdf_path = os.path.join(models_dir, "sphere_yz_small.sdf")
model_directive_path = os.path.join(models_dir, "carrot.yml")

# robots.
Kp = np.array([50, 50, 50, 200, 200], dtype=float)
robot_stiffness_dict = {"gripper": Kp}

# object
num_pieces = 20
object_sdf_dict = {}
for i in range(num_pieces):
    object_name = "carrot_{:02d}".format(i)
    object_sdf_dict[object_name] = object_sdf_path

# environment
h = 1.0
gravity = np.array([0, 0, 0])
contact_detection_tolerance = 3.0
gradient_lstsq_tolerance = 1e-3

# gradient mode
gradient_mode = "zero_order_B"
decouple_AB = True

# num_workers
use_workers = True
num_workers = 30
task_stride = 1
num_iters = 30
num_samples = 100