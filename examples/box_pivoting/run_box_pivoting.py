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
T = int(round(8 / h))  # num of time steps to simulate forward.
duration = T * h
sim_params = QuasistaticSimParameters(
    gravity=gravity,
    nd_per_contact=2,
    contact_detection_tolerance=contact_detection_tolerance,
    is_quasi_dynamic=True)

# trajectory and initial conditions.
nq_a = 2

qa_knots = np.zeros((2, nq_a))
qa_knots[0] = [-0.7, 0.5]
qa_knots[1] = [0.5, 0.5]
q_robot_traj = PiecewisePolynomial.FirstOrderHold(
    [0, T * h], qa_knots.T)

robot_name = "hand"
object_name = "box"
q_a_traj_dict_str = {robot_name: q_robot_traj}

q_u0 = np.array([0.0, 0.5, 0.0])

q0_dict_str = {object_name: q_u0,
               robot_name: qa_knots[0]}

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

q0_dict = create_dict_keyed_by_model_instance_index(
    q_sim_py.plant, q_dict_str=q0_dict_str)

#%%
q_dynamics = QuasistaticDynamics(h=h, q_sim_py=q_sim_py, q_sim=q_sim_cpp)
dim_x = q_dynamics.dim_x
dim_u = q_dynamics.dim_u

#%% try running the dynamics.
x0 = q_dynamics.get_x_from_q_dict(q0_dict)
u_traj_0 = np.zeros((T, dim_u))

x = np.copy(x0)
print(x)

q_dict_traj = [q0_dict]
for i in tqdm(range(T)):
    # print('--------------------------------')
    t = h * i
    q_cmd_dict = {idx_a: q_robot_traj.value(t + h).ravel()}

    u = q_dynamics.get_u_from_q_cmd_dict(q_cmd_dict)
    x = q_dynamics.dynamics(x, u, requires_grad=True)
    Dq_nextDq_cpp = q_dynamics.q_sim.get_Dq_nextDq()
    Dq_nextDqa_cmd_cpp = q_dynamics.q_sim.get_Dq_nextDqa_cmd()

    u_traj_0[i] = u

    q_dict_traj.append(q_dynamics.get_q_dict_from_x(x))

    print(x)

q_sim_py.animate_system_trajectory(h, q_dict_traj)

#%%
# gripper_x plate_x gripper_y plate_y gripper_theta plate_theta gd1 gd2
"""
dx_bounds = np.array([
        np.array([-1, -1, -1, -1, -5, -5, -1, -1]),
        np.array([1, 1, 1, 1, 5, 5, 1, 1])])

du_bounds = np.array([
        np.array([-0.05, -0.05, -0.1, -0.05, -0.05]),
        np.array([0.05, 0.05, 0.1, 0.05, 0.05])])
"""

dx_bounds = np.array([-np.ones(dim_x) * 10.0, np.ones(dim_x) * 10.0])
du_bounds = np.array([-np.ones(dim_u) * 0.5 * h, np.ones(dim_u) * 0.5 * h])
xd_dict = {idx_u: q_u0 + np.array([1.0, 0.5, -np.pi/2]),
           idx_a: qa_knots[0]}
xd = q_dynamics.get_x_from_q_dict(xd_dict)
x_trj_d = np.tile(xd, (T + 1, 1))

Q_dict = {idx_u: np.array([100, 100, 500]),
          idx_a: np.array([0.001, 0.001])}

Qd_dict = {model: Q_i * 1 for model, Q_i in Q_dict.items()}

R_dict = {idx_a: 1e5 * np.array([1, 1])}

irs_lqr_q = IrsLqrQuasistatic(
    q_dynamics=q_dynamics,
    std_u_initial= 5.0 * np.array([0.1, 0.1]),
    T=T,
    Q_dict=Q_dict,
    Qd_dict=Qd_dict,
    R_dict=R_dict,
    x_trj_d=x_trj_d,
    dx_bounds=dx_bounds,
    du_bounds=du_bounds,
    x0=x0,
    u_trj_0=u_traj_0)

#%% compare zero-order and first-order gradient estimation.
"""
std_dict = {idx_u: np.ones(3) * 1e-3,
            idx_a_r: np.ones(2) * 0.1,
            idx_a_l: np.ones(2) * 0.1}
std_x = q_dynamics.get_x_from_q_dict(std_dict)
std_u = q_dynamics.get_u_from_q_cmd_dict(std_dict)
ABhat1 = q_dynamics.calc_AB_first_order(x, u, 100, std_u)
ABhat0 = q_dynamics.calc_B_zero_order(x, u, 100, std_u=std_u)
"""

#%% test multi vs single threaded execution
# x_trj = sqp_ls_q.x_trj
# u_trj = sqp_ls_q.u_trj
#
# t1 = time.time()
# At2, Bt2, ct2 = sqp_ls_q.get_TV_matrices_batch(x_trj, u_trj)
# t2 = time.time()
# print('parallel time', t2 - t1)
# time.time()
#
# t1 = time.time()
# At, Bt, ct = sqp_ls_q.get_TV_matrices(x_trj, u_trj)
# # At1, Bt1, ct1 = sqp_ls_q.get_TV_matrices(x_trj, u_trj)
# t2 = time.time()
# print('single-thread time', (t2 - t1))

#%%
try:
    t0 = time.time()
    irs_lqr_q.iterate(2)
except Exception as e:
    print(e)
    pass

t1 = time.time()

print(f"iterate took {t1 - t0} seconds.")

#%% profile iterate
# cProfile.runctx('irs_lqr_q.iterate(10)',
#                 globals=globals(), locals=locals(),
#                 filename='contact_first_order_stats_multiprocessing')


#%%
x_traj_to_publish = irs_lqr_q.x_trj_best
q_dynamics.publish_trajectory(x_traj_to_publish)
print('x_goal:', xd)
print('x_final:', x_traj_to_publish[-1])


#%% plot different components of the cost for all iterations.
plt.figure()
plt.plot(irs_lqr_q.cost_all_list, label='all')
plt.plot(irs_lqr_q.cost_Qa_list, label='Qa')
plt.plot(irs_lqr_q.cost_Qu_list, label='Qu')
plt.plot(irs_lqr_q.cost_Qa_final_list, label='Qa_f')
plt.plot(irs_lqr_q.cost_Qu_final_list, label='Qu_f')
plt.plot(irs_lqr_q.cost_R_list, label='R')

plt.title('Trajectory cost')
plt.xlabel('Iterations')
# plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()
