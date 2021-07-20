import os
import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import PiecewisePolynomial, ModelInstanceIndex

from quasistatic_simulator.core.quasistatic_simulator import (
    QuasistaticSimulator, QuasistaticSimParameters)
from quasistatic_simulator.examples.model_paths import models_dir

from two_spheres_quasistatic.quasistatic_dynamics import QuasistaticDynamics
from sqp_ls_quasistatic import SqpLsQuasistatic

#%% sim setup
object_sdf_path = os.path.join(models_dir, "sphere_y.sdf")
model_directive_path = os.path.join(models_dir, "sphere_y_actuated.yml")

h = 0.1
T = int(round(2 / h))  # num of time steps to simulate forward.
duration = T * h
quasistatic_sim_params = QuasistaticSimParameters(
    gravity=np.array([0, 0, 0.]),
    nd_per_contact=2,
    contact_detection_tolerance=np.inf,
    is_quasi_dynamic=True)

# robot
Kp = np.array([500], dtype=float)
robot_name = "sphere_y_actuated"
robot_stiffness_dict = {robot_name: Kp}

# object
object_name = "sphere_y"
object_sdf_dict = {object_name: object_sdf_path}

# trajectory and initial conditions.
nq_a = 1
qa_knots = np.zeros((3, nq_a))
qa_knots[0] = [0]
qa_knots[1] = [0.5]
qa_knots[2] = qa_knots[1]
qa_traj = PiecewisePolynomial.FirstOrderHold([0, duration * 0.7, duration],
                                             qa_knots.T)
qu0 = np.array([0.5])

q_sim = QuasistaticSimulator(
    model_directive_path=model_directive_path,
    robot_stiffness_dict=robot_stiffness_dict,
    object_sdf_paths=object_sdf_dict,
    sim_params=quasistatic_sim_params,
    internal_vis=False)

name_to_model_instance_dict = q_sim.get_robot_name_to_model_instance_dict()
idx_a = name_to_model_instance_dict[robot_name]
idx_u = name_to_model_instance_dict[object_name]
q0_dict = {idx_u: qu0, idx_a: qa_knots[0]}

#%%
q_dynamics = QuasistaticDynamics(h=h, q_sim=q_sim)
dim_x = q_dynamics.dim_x
dim_u = q_dynamics.dim_u

#%% try running the dynamics.
x0 = q_dynamics.get_x_from_q_dict(q0_dict)
u_traj_0 = np.zeros((T, nq_a))

x = np.copy(x0)
# for i in range(T):
#     print('--------------------------------')
#     t = h * i
#     q_cmd_dict = {idx_a: qa_traj.value(t + h).ravel()}
#     u = q_dynamics.get_u_from_q_cmd_dict(q_cmd_dict)
#     x = q_dynamics.dynamics(x, u, mode='qp_mp', requires_grad=True)
#     _, _, Dq_nextDq, Dq_nextDqa_cmd = \
#         q_dynamics.q_sim.get_dynamics_derivatives()
#
#     q_dynamics.dynamics(x, u, mode='qp_cvx', requires_grad=True)
#     _, _, Dq_nextDq_cvx, Dq_nextDqa_cmd_cvx = \
#         q_dynamics.q_sim.get_dynamics_derivatives()
#
#     print('t={},'.format(t), 'x:', x, 'u:', u)
#     print('Dq_nextDq\n', Dq_nextDq)
#     print('cvx\n', Dq_nextDq_cvx)
#     print('Dq_nextDqa_cmd\n', Dq_nextDqa_cmd)
#     print('cvx\n', Dq_nextDqa_cmd_cvx)
#     u_traj_0[i] = u


#%%
x_bounds = np.array([-np.ones(dim_x) * 10, np.ones(dim_x) * 10])
u_bounds = np.array([-np.ones(dim_u) * 10, np.ones(dim_u) * 10])
xd = np.array([0.2, 2.0])
x_trj_d = np.tile(xd, (T + 1, 1))

sqp_ls_q = SqpLsQuasistatic(
    q_dynamics=q_dynamics,
    std_u_initial=np.ones(dim_u) * 0.1,
    T=T,
    Q=np.diag([5, 5]),
    R=np.diag([1]),
    x_trj_d=x_trj_d,
    x_bounds=x_bounds,
    u_bounds=u_bounds,
    x0=x0,
    u_trj_0=u_traj_0)


#%%
sqp_ls_q.iterate(1e-6, 10)


#%% first order estimate of B.
x_nominal = np.array([0, 0.5])
u_nominal = np.array([0.2])

Ahat1, Bhat1 = sqp_ls_q.calc_AB_first_order(
    x_nominal=x_nominal,
    u_nominal=u_nominal,
    n_samples=100,
    std=0.05)
print('Bhat1\n', Bhat1)


#%% zero-th order estimate of B.
Bhat0, du = sqp_ls_q.calc_B_zero_order(
    x_nominal=x_nominal,
    u_nominal=u_nominal,
    n_samples=10000,
    std=0.05)

print('Bhat0\n', Bhat0)

plt.figure()
plt.scatter(x_nominal, [0, 0])
plt.scatter(u_nominal + du.squeeze(), np.zeros(len(du)))
plt.axvline(x_nominal[1] - 0.2)  # contact boundary
plt.grid(True)
plt.show()
