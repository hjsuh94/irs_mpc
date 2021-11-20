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

from planar_hand_setup import (object_sdf_path, model_directive_path, Kp,
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
robot_l_name = "arm_left"
robot_r_name = "arm_right"
nq_a = 4
q_al0 = np.array([-0.77459643, -0.78539816])
q_ar0 = np.array([0.77459643, 0.78539816])

# box
object_name = "sphere"
q_u0 = np.array([0.0, 0.317, 0.0])

q0_dict_str = {object_name: q_u0,
               robot_l_name: q_al0,
               robot_r_name: q_ar0}

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
model_a_l = q_sim_py.plant.GetModelInstanceByName(robot_l_name)
model_a_r = q_sim_py.plant.GetModelInstanceByName(robot_r_name)
model_u = q_sim_py.plant.GetModelInstanceByName(object_name)
q0_dict = create_dict_keyed_by_model_instance_index(
    q_sim_py.plant, q_dict_str=q0_dict_str)


#%% generate samples
n_samples = 10000
radius = 0.1
qu_samples = np.zeros((n_samples, n_u))
qa_l_samples = np.zeros((n_samples, 2))
qa_r_samples = np.zeros((n_samples, 2))
du = np.random.rand(n_samples, n_a) * radius - radius / 2

x0 = q_dynamics.get_x_from_q_dict(q0_dict)
u0 = q_dynamics.get_u_from_q_cmd_dict(q0_dict)

for i in tqdm(range(n_samples)):
    u = u0 + du[i]
    # x = q_dynamics.dynamics(x0, u, requires_grad=False)
    x = q_dynamics.dynamics_more_steps(x0, u, n_steps=10)
    q_dict = q_dynamics.get_q_dict_from_x(x)

    qu_samples[i] = q_dict[model_u]
    qa_l_samples[i] = q_dict[model_a_l]
    qa_r_samples[i] = q_dict[model_a_r]


#%% visualize
__, axes = plt.subplots(1, 2)
plt.title('qa_samples')
axes[0].scatter(qa_l_samples[:, 0], qa_l_samples[:, 1])
axes[1].scatter(qa_r_samples[:, 0], qa_r_samples[:, 1])
for ax in axes:
    ax.axis('equal')
    ax.grid(True)
plt.show()

viz['qu_samples_2'].set_object(
    meshcat.geometry.PointCloud(
        position=(qu_samples - qu_samples.mean(axis=0)).T * 10,
        color=np.ones_like(qu_samples).T))


