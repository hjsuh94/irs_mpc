import numpy as np
from tqdm import tqdm
import meshcat
import matplotlib.pyplot as plt

from qsim.simulator import (QuasistaticSimulator, QuasistaticSimParameters)
from qsim.system import (cpp_params_from_py_params)
from quasistatic_simulator.examples.setup_simulation_diagram import (
    create_dict_keyed_by_model_instance_index)
from quasistatic_simulator_py import (QuasistaticSimulatorCpp)

from irs_lqr.quasistatic_dynamics import QuasistaticDynamics

from box_pushing_setup import (object_sdf_path, model_directive_path, Kp,
                               robot_stiffness_dict, object_sdf_dict,
                               gravity, contact_detection_tolerance,
                               gradient_lstsq_tolerance)

viz = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
#%%
h = 0.1

sim_params = QuasistaticSimParameters(
    gravity=gravity,
    nd_per_contact=2,
    contact_detection_tolerance=contact_detection_tolerance,
    is_quasi_dynamic=True)

# robot
robot_name = "hand"
nq_a = 2
q_a0 = np.array([0., -0.1])

# box
object_name = "box"
q_u0 = np.array([0.0, 0.5, 0.0])

q0_dict_str = {object_name: q_u0,
               robot_name: q_a0[0]}

# Python sim.
q_sim_py = QuasistaticSimulator(
    model_directive_path=model_directive_path,
    robot_stiffness_dict=robot_stiffness_dict,
    object_sdf_paths=object_sdf_dict,
    sim_params=sim_params,
    internal_vis=True)

# C++ sim.
sim_params_cpp = cpp_params_from_py_params(sim_params)
sim_params_cpp.gradient_lstsq_tolerance = gradient_lstsq_tolerance
q_sim_cpp = QuasistaticSimulatorCpp(
    model_directive_path=model_directive_path,
    robot_stiffness_str=robot_stiffness_dict,
    object_sdf_paths=object_sdf_dict,
    sim_params=sim_params_cpp)

q_dynamics = QuasistaticDynamics(h=h, q_sim_py=q_sim_py, q_sim=q_sim_cpp)
n_a = q_dynamics.dim_u
n_u = q_dynamics.dim_x - n_a
idx_a = q_sim_py.plant.GetModelInstanceByName(robot_name)
idx_u = q_sim_py.plant.GetModelInstanceByName(object_name)
q0_dict = create_dict_keyed_by_model_instance_index(
    q_sim_py.plant, q_dict_str=q0_dict_str)

#%% generate samples
n_samples = 1000
radius = 0.2
qu_samples = np.zeros((n_samples, n_u))
qa_samples = np.zeros((n_samples, n_a))
du = np.random.rand(n_samples, n_a) * radius - radius / 2

x0 = q_dynamics.get_x_from_q_dict(q0_dict)
for i in tqdm(range(n_samples)):
    u = q_dynamics.get_u_from_q_cmd_dict({idx_a: q_a0 + du[i]})
    x = q_dynamics.dynamics_more_steps(x0, u, n_steps=10)
    q_dict = q_dynamics.get_q_dict_from_x(x)

    qu_samples[i] = q_dict[idx_u]
    qa_samples[i] = q_dict[idx_a]


#%% visualize

fig = plt.figure()
plt.title('qa_samples')
plt.scatter(qa_samples[:, 0], qa_samples[:, 1])
plt.axis('equal')
plt.grid(True)
plt.show()

viz['/qu_samples'].set_object(
    meshcat.geometry.PointCloud(
        position=qu_samples.T, color=np.ones_like(qu_samples).T * 0.5))


