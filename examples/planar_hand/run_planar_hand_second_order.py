import time
import matplotlib.pyplot as plt
import numpy as np

from pydrake.all import PiecewisePolynomial

from quasistatic_simulator.core.quasistatic_simulator import (
    QuasistaticSimulator, QuasistaticSimParameters)
from quasistatic_simulator.core.quasistatic_system import (
    cpp_params_from_py_params)
from quasistatic_simulator.examples.setup_simulation_diagram import (
    create_dict_keyed_by_model_instance_index)
from quasistatic_simulator_py import (QuasistaticSimulatorCpp)

from irs_lqr.quasistatic_dynamics import QuasistaticDynamics
from irs_lqr.mbp_dynamics import MbpDynamics
from irs_lqr.irs_lqr_quasistatic import (
    IrsLqrQuasistatic, IrsLqrQuasistaticParameters)
from irs_lqr.irs_lqr_mbp import IrsLqrMbp

from planar_hand_setup import *

#%% sim setup
T = int(round(6 / h))  # num of time steps to simulate forward.
duration = T * h
sim_params = QuasistaticSimParameters(
    gravity=gravity,
    nd_per_contact=2,
    contact_detection_tolerance=contact_detection_tolerance,
    is_quasi_dynamic=True)

# trajectory and initial conditions.
nq_a = 4
qa_l_knots = np.zeros((2, nq_a))
qa_l_knots[0] = [-np.pi / 4, -np.pi / 4, 0, 0]
q_robot_l_traj = PiecewisePolynomial.ZeroOrderHold(
    [0, T * h], qa_l_knots.T)

qa_r_knots = np.zeros((2, nq_a))
qa_r_knots[0] = [np.pi / 4, np.pi / 4, 0, 0]
q_robot_r_traj = PiecewisePolynomial.ZeroOrderHold(
    [0, T * h], qa_r_knots.T)

q_a_traj_dict_str = {robot_l_name: q_robot_l_traj,
                     robot_r_name: q_robot_r_traj}

q_u0 = np.array([0, 0.35, 0, 0, 0, 0])

q0_dict_str = {object_name: q_u0,
               robot_l_name: qa_l_knots[0],
               robot_r_name: qa_r_knots[0]}

mbp_dynamics = MbpDynamics(h=0.1, model_directive_path=model_directive_path,
    robot_stiffness_dict=robot_stiffness_dict, object_sdf_paths=object_sdf_dict,
    sim_params=sim_params)

idx_a_l = mbp_dynamics.plant.GetModelInstanceByName(robot_l_name)
idx_a_r = mbp_dynamics.plant.GetModelInstanceByName(robot_r_name)
idx_u = mbp_dynamics.plant.GetModelInstanceByName(object_name)

q0_dict = create_dict_keyed_by_model_instance_index(
    mbp_dynamics.plant, q_dict_str=q0_dict_str)

#%%
dim_x = mbp_dynamics.dim_x
dim_u = mbp_dynamics.dim_u

#%% try running the dynamics.
x0 = mbp_dynamics.get_x_from_qv_dict(q0_dict)
u_traj_0 = np.zeros((T, dim_u))

x = np.copy(x0)

q_dict_traj = [q0_dict]
for i in range(T):
    # print('--------------------------------')
    t = h * i
    q_cmd_dict = {idx_a_l: q_robot_l_traj.value(t + h).ravel(),
                  idx_a_r: q_robot_r_traj.value(t + h).ravel()}
    u = -1.0 * np.array([1, 1, -1, -1])
    x = mbp_dynamics.dynamics(x, u, requires_grad=True)

    print('t={},'.format(t), 'x:', x, 'u:', u)
    u_traj_0[i] = u

    q_dict_traj.append(mbp_dynamics.get_qv_dict_from_x(x))

mbp_dynamics.animate_system_trajectory(h, q_dict_traj)


#%%

params = IrsLqrQuasistaticParameters()
params.Q_dict = {
    idx_u: np.array([10, 10, 10, 0.0, 0.0, 0.0]),
    idx_a_l: np.array([0.0, 0.0, 0.0, 0.0]),
    idx_a_r: np.array([0.0, 0.0, 0.0, 0.0])}
params.Qd_dict = {model: Q_i * 100 for model, Q_i in params.Q_dict.items()}
params.R_dict = {
    idx_a_l: 1e2 * np.array([1, 1]),
    idx_a_r: 1e2 * np.array([1, 1])}


xd_dict = {idx_u: q_u0 + np.array([0.0, 0.0, -np.pi/4, 0, 0, 0]),
           idx_a_l: qa_l_knots[0],
           idx_a_r: qa_r_knots[0]}
xd = mbp_dynamics.get_x_from_qv_dict(xd_dict)
x_trj_d = np.tile(xd, (T + 1, 1))

params.x0 = x0
params.x_trj_d = x_trj_d
params.u_trj_0 = u_traj_0
params.T = T

params.u_bounds_abs = np.array([
    -np.ones(dim_u) * 0.5, np.ones(dim_u) * 0.5])

def sampling(u_initial, iter):
    return u_initial ** (0.5 * iter)

params.sampling = sampling
params.std_u_initial = np.ones(dim_u) * 0.4

params.decouple_AB = decouple_AB
params.use_workers = False
params.gradient_mode = gradient_mode
params.task_stride = task_stride

irs_lqr_q = IrsLqrMbp(mbp_dynamics, params)

t0 = time.time()
irs_lqr_q.iterate(5)
t1 = time.time()

print(f"iterate took {t1 - t0} seconds.")

#%% profile iterate
# cProfile.runctx('irs_lqr_q.iterate(10)',
#                 globals=globals(), locals=locals(),
#                 filename='contact_first_order_stats_multiprocessing')


#%%
x_traj_to_publish = irs_lqr_q.x_trj_best
mbp_dynamics.publish_trajectory(x_traj_to_publish)
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


