import os
import timeit
import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import PiecewisePolynomial, ModelInstanceIndex

from quasistatic_simulator.core.quasistatic_simulator import (
    QuasistaticSimulator, QuasistaticSimParameters)
from quasistatic_simulator.examples.model_paths import models_dir
from quasistatic_simulator.examples.setup_simulation_diagram import (
    create_dict_keyed_by_model_instance_index)

from two_spheres_quasistatic.quasistatic_dynamics import QuasistaticDynamics
from sqp_ls_quasistatic import SqpLsQuasistatic

#%% sim setup
object_sdf_path = os.path.join(models_dir, "sphere_yz_rotation_r_0.25m.sdf")
model_directive_path = os.path.join(models_dir, "planar_hand.yml")

h = 0.1
T = int(round(2 / h))  # num of time steps to simulate forward.
duration = T * h
quasistatic_sim_params = QuasistaticSimParameters(
    gravity=np.array([0, 0, -10.]),
    nd_per_contact=2,
    contact_detection_tolerance=0.5,
    is_quasi_dynamic=True)

# robots.
Kp = np.array([50, 25], dtype=float)
robot_l_name = "arm_left"
robot_r_name = "arm_right"
robot_stiffness_dict = {robot_l_name: Kp, robot_r_name: Kp}

# object
object_name = "sphere"
object_sdf_dict = {object_name: object_sdf_path}

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


q_sim = QuasistaticSimulator(
    model_directive_path=model_directive_path,
    robot_stiffness_dict=robot_stiffness_dict,
    object_sdf_paths=object_sdf_dict,
    sim_params=quasistatic_sim_params,
    internal_vis=True)

name_to_model_instance_dict = q_sim.get_robot_name_to_model_instance_dict()
idx_a_l = name_to_model_instance_dict[robot_l_name]
idx_a_r = name_to_model_instance_dict[robot_r_name]
idx_u = name_to_model_instance_dict[object_name]

q0_dict = create_dict_keyed_by_model_instance_index(
    q_sim.plant, q_dict_str=q0_dict_str)

#%%
q_dynamics = QuasistaticDynamics(h=h, q_sim=q_sim)
dim_x = q_dynamics.dim_x
dim_u = q_dynamics.dim_u

#%% try running the dynamics.
x0 = q_dynamics.get_x_from_q_dict(q0_dict)
u_traj_0 = np.zeros((T, dim_u))

x = np.copy(x0)

q_dict_traj = [q0_dict]
for i in range(T):
    # print('--------------------------------')
    t = h * i
    # q_cmd_dict = {idx_a_l: q_a_traj_dict_str[robot_l_name].value(t + h).ravel(),
    #               idx_a_r: q_a_traj_dict_str[robot_r_name].value(t + h).ravel()}
    q_cmd_dict = {idx_a_l: q_robot_l_traj.value(t + h).ravel(),
                  idx_a_r: q_robot_r_traj.value(t + h).ravel()}
    u = q_dynamics.get_u_from_q_cmd_dict(q_cmd_dict)
    x = q_dynamics.dynamics(x, u, mode='qp_mp', requires_grad=True)
    _, _, Dq_nextDq, Dq_nextDqa_cmd = \
        q_dynamics.q_sim.get_dynamics_derivatives()

    q_dynamics.dynamics(x, u, mode='qp_cvx', requires_grad=True)
    _, _, Dq_nextDq_cvx, Dq_nextDqa_cmd_cvx = \
        q_dynamics.q_sim.get_dynamics_derivatives()

    print('t={},'.format(t), 'x:', x, 'u:', u)
    print('Dq_nextDq\n', Dq_nextDq)
    print('cvx\n', Dq_nextDq_cvx)
    print('Dq_nextDqa_cmd\n', Dq_nextDqa_cmd)
    print('cvx\n', Dq_nextDqa_cmd_cvx)
    u_traj_0[i] = u

    q_dict_traj.append(q_dynamics.get_q_dict_from_x(x))


q_sim.animate_system_trajectory(h, q_dict_traj)


#%%
dx_bounds = np.array([-np.ones(dim_x) * 1, np.ones(dim_x) * 1])
du_bounds = np.array([-np.ones(dim_u) * 0.5 * h, np.ones(dim_u) * 0.5 * h])

xd_dict = {idx_u: q_u0 + np.array([-0.15, 0, 0]),
           idx_a_l: qa_l_knots[0],
           idx_a_r: qa_r_knots[0]}
xd = q_dynamics.get_x_from_q_dict(xd_dict)
x_trj_d = np.tile(xd, (T + 1, 1))

Q_dict = {idx_u: np.array([10, 10., 10]),
          idx_a_l: np.array([1, 1.]),
          idx_a_r: np.array([1, 1.])}
Q = q_dynamics.get_Q_from_Q_dict(Q_dict)

sqp_ls_q = SqpLsQuasistatic(
    q_dynamics=q_dynamics,
    std_u_initial=np.ones(dim_u) * 0.1,
    T=T,
    Q=Q,
    R=np.eye(dim_u) * 5,
    x_trj_d=x_trj_d,
    dx_bounds=dx_bounds,
    du_bounds=du_bounds,
    x0=x0,
    u_trj_0=u_traj_0)

#%%
sqp_ls_q.iterate(1e-6, 10)

#%%
q_dynamics.publish_trajectory(sqp_ls_q.x_trj_best)
print('x_goal:', xd)
print('x_final:', sqp_ls_q.x_trj_best[-1])
assert False
#%%

q_dict_traj = [q0_dict]
x = x0
for u in sqp_ls_q.u_trj_best:
    x = q_dynamics.dynamics(x, u, mode='qp_mp', requires_grad=False)
    q_dict_traj.append(q_dynamics.get_q_dict_from_x(x))

q_sim.animate_system_trajectory(h, q_dict_traj)
