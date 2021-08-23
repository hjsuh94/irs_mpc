import os
import numpy as np

from quasistatic_simulator.examples.model_paths import models_dir

object_sdf_path = os.path.join(models_dir, "plate.sdf")
model_directive_path = os.path.join(models_dir, "plate.yml")

# robots.
Kp = np.array([50, 50, 50, 200, 200], dtype=float)
robot_stiffness_dict = {"gripper": Kp}

# object
object_name = "plate"
object_sdf_dict = {object_name: object_sdf_path}

# environment
h = 0.1
gravity = np.array([0, 0, -9.81])
contact_detection_tolerance = 1.0
gradient_lstsq_tolerance = 1e-3
