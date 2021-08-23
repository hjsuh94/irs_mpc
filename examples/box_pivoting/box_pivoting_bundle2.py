import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from pydrake.all import PiecewisePolynomial

from quasistatic_simulator.core.quasistatic_simulator import (
    QuasistaticSimulator, QuasistaticSimParameters)
from quasistatic_simulator.core.quasistatic_system import (
    cpp_params_from_py_params)
from quasistatic_simulator.examples.setup_simulation_diagram import (
    create_dict_keyed_by_model_instance_index)
from quasistatic_simulator_py import (QuasistaticSimulatorCpp)

from irs_lqr.quasistatic_dynamics import QuasistaticDynamics
from irs_lqr.irs_lqr_quasistatic import IrsLqrQuasistatic

from box_pivoting_setup import *

#%% sim setup
T = int(round(0.1 / h))  # num of time steps to simulate forward.
duration = T * h
sim_params = QuasistaticSimParameters(
    gravity=gravity,
    nd_per_contact=2,
    contact_detection_tolerance=contact_detection_tolerance,
    is_quasi_dynamic=True)

# trajectory and initial conditions.
nq_a = 2
z_now = 1.11

robot_name = "hand"
object_name = "box"

q_sim_py = QuasistaticSimulator(
    model_directive_path=model_directive_path,
    robot_stiffness_dict=robot_stiffness_dict,
    object_sdf_paths=object_sdf_dict,
    sim_params=sim_params,
    internal_vis=True)

# construct C++ backend.
sim_params_cpp = cpp_params_from_py_params(sim_params)
sim_params_cpp.gradient_lstsq_tolerance = gradient_lstsq_tolerance
q_sim_cpp = QuasistaticSimulatorCpp(
    model_directive_path=model_directive_path,
    robot_stiffness_str=robot_stiffness_dict,
    object_sdf_paths=object_sdf_dict,
    sim_params=sim_params_cpp)

plant = q_sim_cpp.get_plant()
q_sim_py.get_robot_name_to_model_instance_dict()
idx_a = plant.GetModelInstanceByName(robot_name)
idx_u = plant.GetModelInstanceByName(object_name)


#%%



z = np.linspace(1.0, 1.2, 21)
q_dynamics = QuasistaticDynamics(h=h, q_sim_py=q_sim_py, q_sim=q_sim_cpp)
data_lst = []
for i in range(len(z)):
    z_now = z[i]
    qa_knots = np.zeros((2, nq_a))
    qa_knots[0] = [0.0, z_now]
    qa_knots[1] = [0.05, z_now]
    q_robot_traj = PiecewisePolynomial.FirstOrderHold(
        [0, T * h], qa_knots.T)

    q_u0 = np.array([0.0, 0.505, 0.0])

    q0_dict_str = {object_name: q_u0,
                robot_name: qa_knots[0]}    

    q0_dict = create_dict_keyed_by_model_instance_index(
    q_sim_py.plant, q_dict_str=q0_dict_str)

    q_a_traj_dict_str = {robot_name: q_robot_traj}

    dim_x = q_dynamics.dim_x
    dim_u = q_dynamics.dim_u

    #%% try running the dynamics.

    q_dict_traj = [q0_dict]

    # print('--------------------------------')
    x0 = q_dynamics.get_x_from_q_dict(q0_dict)
    u_traj_0 = np.zeros((T, dim_u))

    x = np.copy(x0)    

    qa_knots = np.zeros((2, nq_a))
    qa_knots[0] = [0.0, z_now]
    qa_knots[1] = [0.02, z_now]
    q_robot_traj = PiecewisePolynomial.FirstOrderHold(
        [0, T * h], qa_knots.T)

    t = h * i
    q_cmd_dict = {idx_a: q_robot_traj.value(t + h).ravel()}

    u = q_dynamics.get_u_from_q_cmd_dict(q_cmd_dict)

    ABhat_first_order = \
        q_dynamics.calc_AB_first_order(x, u, 10000, np.array([0.001, 0.05]))
    ABhat_zero_order = \
        q_dynamics.calc_B_zero_order(x, u, 10000, np.array([0.001, 0.05]))

    data_lst.append([z_now, ABhat_first_order[1,6], ABhat_zero_order[1,6]])
 
    u_traj_0[0] = u

plt.figure()
data_lst = np.array(data_lst)
plt.plot(data_lst[:,1], data_lst[:,0], color="springgreen", label='first order')
plt.plot(data_lst[:,2], data_lst[:,0], 'r', label='zero order')
plt.ylim([1.1-0.1, 1.1+0.1])
plt.legend()
plt.xlabel('dx_x_box / du_y_ball')
plt.ylabel('nominal uy')
plt.show()