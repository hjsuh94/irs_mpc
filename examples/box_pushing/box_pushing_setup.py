import os
import numpy as np

from quasistatic_simulator.examples.model_paths import models_dir

#object_sdf_path = os.path.join(models_dir, "box_1m_rotation.sdf")
object_sdf_path = os.path.join(models_dir, "sphere_yz_rotation_r_0.5m.sdf")
model_directive_path = os.path.join(models_dir, "box_pushing.yml")

# robots.
Kp = np.array([500, 500], dtype=float)
robot_stiffness_dict = {"hand": Kp}

# object
object_name = "box"
object_sdf_dict = {object_name: object_sdf_path}

# environment
h = 0.2
gravity = np.array([0, 0, 0])
contact_detection_tolerance = 0.1
gradient_lstsq_tolerance = 1e-3
