import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from pydrake.all import PiecewisePolynomial

from qsim.simulator import (QuasistaticSimulator, QuasistaticSimParameters)
from qsim.system import (cpp_params_from_py_params)
from quasistatic_simulator.examples.setup_simulation_diagram import (
    create_dict_keyed_by_model_instance_index)
from quasistatic_simulator_py import (QuasistaticSimulatorCpp)

from irs_lqr.quasistatic_dynamics import QuasistaticDynamics
from irs_lqr.irs_lqr_quasistatic import (
    IrsLqrQuasistatic, IrsLqrQuasistaticParameters)

from box_pushing_setup import *

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
qa_knots[0] = [0.0, -0.2]
qa_knots[1] = [0.0, 0.2]
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

q_sim_py.animate_system_trajectory(h, q_dict_traj)

#%%
# gripper_x plate_x gripper_y plate_y gripper_theta plate_theta gd1 gd2
params = IrsLqrQuasistaticParameters()
params.Q_dict = {
    idx_u: np.array([300, 300, 120]),
    idx_a: np.array([0.0, 0.0])}
params.Qd_dict = {model: Q_i * 0 for model, Q_i in params.Q_dict.items()}
params.R_dict = {idx_a: 1e1 * np.array([1, 1])}

xd_dict = {idx_u: q_u0 + np.array([0.0, 1.0, 0.0]),
           idx_a: qa_knots[0]}
xd = q_dynamics.get_x_from_q_dict(xd_dict)
x_trj_d = np.tile(xd, (T + 1, 1))

params.x0 = x0
params.x_trj_d = x_trj_d
params.u_trj_0 = u_traj_0
params.T = T

params.u_bounds_rel = np.array([
    -np.ones(dim_u) * 0.2 * h, np.ones(dim_u) * 0.2 * h])

def sampling(u_initial, iter):
    return u_initial ** (0.5 * iter)

params.sampling = sampling
params.std_u_initial = np.ones(dim_u) * 0.1

params.decouple_AB = decouple_AB
params.use_workers = use_workers
params.gradient_mode = gradient_mode
params.task_stride = task_stride
params.num_samples = num_samples

irs_lqr_q = IrsLqrQuasistatic(q_dynamics=q_dynamics, params=params)

try:
    t0 = time.time()
    irs_lqr_q.iterate(num_iters)
except Exception as e:
    print(e)
    pass

t1 = time.time()

print(f"iterate took {t1 - t0} seconds.")
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
