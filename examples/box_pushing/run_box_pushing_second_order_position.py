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

from irs_lqr.quasistatic_dynamics import QuasistaticDynamics
from irs_lqr.mbp_dynamics_position import MbpDynamicsPosition
from irs_lqr.irs_lqr_quasistatic import (
    IrsLqrQuasistatic, IrsLqrQuasistaticParameters)
from irs_lqr.irs_lqr_mbp_position import IrsLqrMbpPosition

from box_pushing_setup import *

#%% sim setup
T = int(round(6 / h))  # num of time steps to simulate forward.
duration = T * h
sim_params = QuasistaticSimParameters(
    gravity=gravity,
    nd_per_contact=2,
    contact_detection_tolerance=contact_detection_tolerance,
    is_quasi_dynamic=True)

# trajectory and initial conditions.
# trajectory and initial conditions.
nq_a = 2
qa_knots = np.zeros((2, nq_a))
qa_knots[0] = [0.0, -0.2]
qa_knots[1] = [0.0, 0.2]
q_robot_traj = PiecewisePolynomial.FirstOrderHold(
    [0, T * h], qa_knots.T)

nqv_a = 4
qva_knots = np.zeros((2, nqv_a))
qva_knots[0] = [0.0, -0.2, 0.0, 0.0]
qva_knots[1] = [0.0, 0.2, 0.0, 0.0]

robot_name = "hand"
object_name = "box"
q_a_traj_dict_str = {robot_name: q_robot_traj}

q_u0 = np.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0])

q0_dict_str = {object_name: q_u0,
               robot_name: qva_knots[0]}

mbp_dynamics = MbpDynamicsPosition(h=h, 
    model_directive_path=model_directive_path,
    robot_stiffness_dict=robot_stiffness_dict,
    object_sdf_paths=object_sdf_dict,
    sim_params=sim_params,
    internal_vis=True)

idx_a = mbp_dynamics.plant.GetModelInstanceByName(robot_name)
idx_u = mbp_dynamics.plant.GetModelInstanceByName(object_name)

q0_dict = create_dict_keyed_by_model_instance_index(
    mbp_dynamics.plant, q_dict_str=q0_dict_str)

print(q0_dict)

#%%
dim_x = mbp_dynamics.dim_x
dim_u = mbp_dynamics.dim_u

#%% try running the dynamics.
print(q0_dict)
x0 = mbp_dynamics.get_x_from_qv_dict(q0_dict)
u_traj_0 = np.zeros((T, dim_u))

x = np.copy(x0)

qv_dict_traj = [q0_dict]
for i in range(T):
    # print('--------------------------------')
    t = h * i
    q_cmd_dict = {idx_a: q_robot_traj.value(t + h).ravel()}

    u = mbp_dynamics.get_u_from_q_cmd_dict(q_cmd_dict)
    x = mbp_dynamics.dynamics(x, u, requires_grad=True)
    u_traj_0[i] = u

    qv_dict_traj.append(mbp_dynamics.get_qv_dict_from_x(x))

mbp_dynamics.animate_system_trajectory(h, qv_dict_traj)



#%%

params = IrsLqrQuasistaticParameters()
params.Q_dict = {
    idx_u: np.array([300, 300, 120, 0.0, 0.0, 0.0]),
    idx_a: np.array([0.0, 0.0, 0.0, 0.0])}
params.Qd_dict = {model: Q_i * 1 for model, Q_i in params.Q_dict.items()}
params.R_dict = {
    idx_a: 1e2 * np.array([1, 1])}

xd_dict = {idx_u: q_u0 + np.array([0.0, 1.0, 0.0, 0, 0, 0]),
           idx_a: qva_knots[0]}
xd = mbp_dynamics.get_x_from_qv_dict(xd_dict)
x_trj_d = np.tile(xd, (T + 1, 1))

params.x0 = x0
params.x_trj_d = x_trj_d
params.u_trj_0 = u_traj_0
params.T = T

params.u_bounds_rel = np.array([
    -np.ones(dim_u) * 0.03, np.ones(dim_u) * 0.03])

def sampling(u_initial, iter):
    return u_initial ** (0.5 * iter)

params.sampling = sampling
params.std_u_initial = np.ones(dim_u) * 0.04

params.decouple_AB = decouple_AB
params.use_workers = use_workers
params.gradient_mode = gradient_mode
params.task_stride = task_stride
params.num_samples = num_samples

irs_lqr_q = IrsLqrMbpPosition(mbp_dynamics, params)

t0 = time.time()
irs_lqr_q.iterate(num_iters)
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


