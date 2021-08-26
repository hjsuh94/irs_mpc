import os
import numpy as np

from quasistatic_simulator.examples.model_paths import models_dir

object_sdf_path = os.path.join(models_dir, "box_1m_rotation.sdf")
model_directive_path = os.path.join(models_dir, "box_pivoting.yml")

# robots.
Kp = np.array([50000, 50000], dtype=float)
robot_stiffness_dict = {"hand": Kp}

# object
object_name = "box"
object_sdf_dict = {object_name: object_sdf_path}

# environment
h = 0.1
gravity = np.array([0, 0, -9.81])
contact_detection_tolerance = 100.0
gradient_lstsq_tolerance = 1e-3

# gradient mode
gradient_mode = "zero_order_B"
decouple_AB = True

# num_workers
use_workers = True
num_workers = 30
task_stride = 1
num_samples = 100
num_iters = 30