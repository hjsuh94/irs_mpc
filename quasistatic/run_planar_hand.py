import os
import cProfile
import time

import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import PiecewisePolynomial, ModelInstanceIndex

from quasistatic_simulator.core.quasistatic_simulator import (
    QuasistaticSimulator, QuasistaticSimParameters)
from quasistatic_simulator.core.quasistatic_system import (
    cpp_params_from_py_params)
from quasistatic_simulator.examples.setup_simulation_diagram import (
    create_dict_keyed_by_model_instance_index)
from quasistatic_simulator_py import (QuasistaticSimulatorCpp,
    QuasistaticSimParametersCpp)

from quasistatic.quasistatic_dynamics import QuasistaticDynamics
from sqp_ls_quasistatic import SqpLsQuasistatic

from planar_hand_setup import *

#%% sim setup
T = int(round(3 / h))  # num of time steps to simulate forward.
duration = T * h
sim_params = QuasistaticSimParameters(
    gravity=gravity,
    nd_per_contact=2,
    contact_detection_tolerance=contact_detection_tolerance,
    is_quasi_dynamic=True)

# trajectory and initial conditions.
nq_a = 2
qa_l_knots = np.zeros((2, nq_a))
qa_l_knots[0] = [-np.pi / 4, -np.pi / 4]
q_robot_l_traj = PiecewisePolynomial.ZeroOrderHold(
    [0, T * h], qa_l_knots.T)

qa_r_knots = np.zeros((2, nq_a))
qa_r_knots[0] = [np.pi / 4, np.pi / 4]
q_robot_r_traj = PiecewisePolynomial.ZeroOrderHold(
    [0, T * h], qa_r_knots.T)

q_a_traj_dict_str = {robot_l_name: q_robot_l_traj,
                     robot_r_name: q_robot_r_traj}

q_u0 = np.array([0, 0.35, 0])

q0_dict_str = {object_name: q_u0,
               robot_l_name: qa_l_knots[0],
               robot_r_name: qa_r_knots[0]}


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
idx_a_l = plant.GetModelInstanceByName(robot_l_name)
idx_a_r = plant.GetModelInstanceByName(robot_r_name)
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

q_dict_traj = [q0_dict]
for i in range(0):
    # print('--------------------------------')
    t = h * i
    q_cmd_dict = {idx_a_l: q_robot_l_traj.value(t + h).ravel(),
                  idx_a_r: q_robot_r_traj.value(t + h).ravel()}
    u = q_dynamics.get_u_from_q_cmd_dict(q_cmd_dict)
    x = q_dynamics.dynamics_py(x, u, mode='qp_mp', requires_grad=True)
    Dq_nextDq = q_dynamics.q_sim_py.get_Dq_nextDq()
    Dq_nextDqa_cmd = q_dynamics.q_sim_py.get_Dq_nextDqa_cmd()

    q_dynamics.dynamics_py(x, u, mode='qp_cvx', requires_grad=True)
    Dq_nextDq_cvx = q_dynamics.q_sim_py.get_Dq_nextDq()
    Dq_nextDqa_cmd_cvx = q_dynamics.q_sim_py.get_Dq_nextDqa_cmd()

    print('t={},'.format(t), 'x:', x, 'u:', u)
    print('Dq_nextDq\n', Dq_nextDq)
    print('cvx\n', Dq_nextDq_cvx)
    print('Dq_nextDqa_cmd\n', Dq_nextDqa_cmd)
    print('cvx\n', Dq_nextDqa_cmd_cvx)
    u_traj_0[i] = u

    q_dict_traj.append(q_dynamics.get_q_dict_from_x(x))


q_sim_py.animate_system_trajectory(h, q_dict_traj)


#%%
dx_bounds = np.array([-np.ones(dim_x) * 1, np.ones(dim_x) * 1])
du_bounds = np.array([-np.ones(dim_u) * 0.5 * h, np.ones(dim_u) * 0.5 * h])

xd_dict = {idx_u: q_u0 + np.array([0.30, 0, 0]),
           idx_a_l: qa_l_knots[0],
           idx_a_r: qa_r_knots[0]}
xd = q_dynamics.get_x_from_q_dict(xd_dict)
x_trj_d = np.tile(xd, (T + 1, 1))

Q_dict = {idx_u: np.array([10, 0.001, 0.001]),
          idx_a_l: np.array([0.001, 0.001]),
          idx_a_r: np.array([0.001, 0.001])}

Qd_dict = {model: Q_i * 10 for model, Q_i in Q_dict.items()}

R_dict = {idx_a_l: np.array([1, 1]),
          idx_a_r: np.array([1, 1])}

sqp_ls_q = SqpLsQuasistatic(
    q_dynamics=q_dynamics,
    std_u_initial=np.ones(dim_u) * 0.3,
    T=T,
    Q_dict=Q_dict,
    Qd_dict=Qd_dict,
    R_dict=R_dict,
    x_trj_d=x_trj_d,
    dx_bounds=dx_bounds,
    du_bounds=du_bounds,
    x0=x0,
    u_trj_0=u_traj_0)

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
# At1, Bt1, ct1 = sqp_ls_q.get_TV_matrices(x_trj, u_trj)
# t2 = time.time()
# print('single-thread time', t2 - t1)

#%%
sqp_ls_q.iterate(1e-6, 20)
# cProfile.runctx('sqp_ls_q.iterate(1e-6, 10)',
#                 globals=globals(), locals=locals(),
#                 filename='contact_first_order_stats')


#%%
x_traj_to_publish = sqp_ls_q.x_trj_list[-1]
q_dynamics.publish_trajectory(x_traj_to_publish)
print('x_goal:', xd)
print('x_final:', x_traj_to_publish[-1])


#%% plot different components of the cost for all iterations.
plt.figure()
plt.plot(sqp_ls_q.cost_all_list, label='all')
plt.plot(sqp_ls_q.cost_Qa_list, label='Qa')
plt.plot(sqp_ls_q.cost_Qu_list, label='Qu')
plt.plot(sqp_ls_q.cost_Qa_final_list, label='Qa_f')
plt.plot(sqp_ls_q.cost_Qu_final_list, label='Qu_f')
plt.plot(sqp_ls_q.cost_R_list, label='R')

plt.title('Trajectory cost')
plt.xlabel('Iterations')
# plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()

#%%

