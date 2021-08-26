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



w = np.random.normal(0, (0.001, 0.05), size=(1000, 2))
x_lst = np.zeros((w.shape[0], 5))
q_dynamics = QuasistaticDynamics(h=h, q_sim_py=q_sim_py, q_sim=q_sim_cpp)
data_lst = []
for i in range(w.shape[0]):
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
    xp = q_dynamics.dynamics(x, u + w[i], requires_grad=True)

    ABhat = q_dynamics.jacobian_xu(x, u + w[i])

    data_lst.append([(u + w[i])[1], ABhat[1,6]])
 
    u_traj_0[0] = u
    q_dict_traj = [q0_dict, q_dynamics.get_q_dict_from_x(xp)]
    q_sim_py.animate_system_trajectory(h, q_dict_traj)

    x_lst[i,:] = xp

plt.figure()
data_lst = np.array(data_lst)
plt.scatter(data_lst[:,1], data_lst[:,0], marker='o', color='springgreen')
plt.ylim([1.1-0.02, 1.1+0.1])
plt.xlabel('dx_x_box / du_y_ball')
plt.ylabel('nominal uy')

print(np.mean(x_lst, axis=0)[1])
print(np.std(x_lst, axis=0)[1])

x = np.copy(x0)
u = q_dynamics.get_u_from_q_cmd_dict(q_cmd_dict)    
print(x)
print(u)   
ABhat_first_order = \
    q_dynamics.calc_AB_first_order(x, u, 1000, np.array([0.1, 0.1]))
x = np.copy(x0)
u = q_dynamics.get_u_from_q_cmd_dict(q_cmd_dict)     
print(x)
print(u)   
ABhat_zero_order = \
    q_dynamics.calc_B_zero_order(x, u, 1000, np.array([0.1, 0.1]))

plt.figure()
ax = plt.subplot(2,1,1)
plt.imshow(ABhat_zero_order)
for i in range(ABhat_zero_order.shape[0]):
    for j in range(ABhat_zero_order.shape[1]):
        text = ax.text(j, i, "{:02f}".format(ABhat_zero_order[i,j]),
                       ha="center", va="center", color="w")
plt.colorbar()
ax = plt.subplot(2,1,2)
plt.imshow(ABhat_first_order)
for i in range(ABhat_zero_order.shape[0]):
    for j in range(ABhat_zero_order.shape[1]):
        text = ax.text(j, i, "{:02f}".format(ABhat_first_order[i,j]),
                       ha="center", va="center", color="w")
plt.colorbar()
plt.show()
#plt.scatter(w[:,0], x_lst[:,3])
#plt.show()

