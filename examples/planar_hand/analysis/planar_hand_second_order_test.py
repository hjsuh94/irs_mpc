import time
import matplotlib.pyplot as plt
import numpy as np

from pydrake.all import (
    PiecewisePolynomial, DiagramBuilder, ConnectMeshcatVisualizer,
    MeshcatContactVisualizer, Simulator_, AutoDiffXd,
    initializeAutoDiff, autoDiffToGradientMatrix)

from quasistatic_simulator.core.quasistatic_simulator import (
    QuasistaticSimulator, QuasistaticSimParameters)
from quasistatic_simulator.core.quasistatic_system import (
    cpp_params_from_py_params)
from quasistatic_simulator.examples.setup_simulation_diagram import (
    create_dict_keyed_by_model_instance_index)
from quasistatic_simulator_py import (QuasistaticSimulatorCpp)
from quasistatic_simulator.core.utils import create_plant_with_robots_and_objects

from irs_lqr.quasistatic_dynamics import QuasistaticDynamics
from irs_lqr.mbp_dynamics import MbpDynamics 
from irs_lqr.irs_lqr_quasistatic import (
    IrsLqrQuasistatic, IrsLqrQuasistaticParameters)
from irs_lqr.irs_lqr_mbp import IrsLqrMbp

from planar_hand_setup import *

# Test raw building.
builder = DiagramBuilder()
plant, scene_graph, robot_models, object_models = \
    create_plant_with_robots_and_objects(
        builder=builder,
        model_directive_path=model_directive_path,
        robot_names=[name for name in robot_stiffness_dict.keys()],
        object_sdf_paths=object_sdf_dict,
        time_step=1e-3,
        gravity=gravity
    )

diagram_f = builder.Build()
diagram_ad = diagram_f.ToAutoDiffXd()
#context = diagram_f.CreateDefaultContext()
context = diagram_ad.CreateDefaultContext()
plant_ad = diagram_ad.GetSubsystemByName(plant.get_name())


xu_ad = initializeAutoDiff([
    -np.pi/4, -np.pi/4, 0, 0,
    np.pi/4, np.pi/4, 0, 0,
    0, 0.35, 0.0, 0, 0, 0,
    0, 0, 0, 0])

xu_ad = initializeAutoDiff([
    0, -np.pi/4, np.pi/4, 
    0.35, -np.pi/4, np.pi/4,
    0,
    0, 0, 0,
    0, 0, 0,
    0,
    0, 0, 0, 0])


plant_context = plant_ad.GetMyMutableContextFromRoot(context)

# Set 
plant_ad.SetPositionsAndVelocities(plant_context, 
    plant_ad.GetModelInstanceByName("arm_left"), xu_ad[[1,4,8,11]])

plant_ad.SetPositionsAndVelocities(plant_context,
    plant_ad.GetModelInstanceByName("arm_right"), xu_ad[[2,5,9,12]])

plant_ad.SetPositionsAndVelocities(plant_context, 
    plant_ad.GetModelInstanceByName("sphere"), xu_ad[[0,3,6,7,10,13]])

plant_ad.get_actuation_input_port(
    plant_ad.GetModelInstanceByName("arm_left")).FixValue(
        plant_context, xu_ad[14:16])
plant_ad.get_actuation_input_port(
    plant_ad.GetModelInstanceByName("arm_right")).FixValue(
        plant_context, xu_ad[16:18])

simulator_ad = Simulator_[AutoDiffXd](diagram_ad, context) 
simulator_ad.AdvanceTo(0.1)

plant_context = plant_ad.GetMyMutableContextFromRoot(context)

x_next = plant_ad.GetPositionsAndVelocities(plant_context)
AB = autoDiffToGradientMatrix(x_next)

## Test out the MbpDynamics.
sim_params = QuasistaticSimParameters(
    gravity=gravity,
    nd_per_contact=2,
    contact_detection_tolerance=contact_detection_tolerance,
    is_quasi_dynamic=True)

mbp_dynamics = MbpDynamics(h=0.1, model_directive_path=model_directive_path,
    robot_stiffness_dict=robot_stiffness_dict, object_sdf_paths=object_sdf_dict,
    sim_params=sim_params)

## trajectory and initial conditions.
nq_a = 4
qa_l_knots = np.zeros((2, nq_a))
qa_l_knots[0] = [-np.pi / 4, -np.pi / 4, 0, 0]

qa_r_knots = np.zeros((2, nq_a))
qa_r_knots[0] = [np.pi / 4, np.pi / 4, 0, 0]

q_u0 = np.array([0, 0.35, 0, 0, 0, 0])

q0_dict_str = {object_name: q_u0,
               robot_l_name: qa_l_knots[0],
               robot_r_name: qa_r_knots[0]}

## Load in x and u.
q0_dict = create_dict_keyed_by_model_instance_index(
    mbp_dynamics.plant, q_dict_str=q0_dict_str
)
x0 = mbp_dynamics.get_x_from_qv_dict(q0_dict)
xnext = mbp_dynamics.dynamics(x0, np.array([0.0, 0.0, 0.0, 0.0]))
xnext = mbp_dynamics.dynamics_py(x0, np.array([0.0, 0.0, 0.0, 0.0]))

AB1 = mbp_dynamics.jacobian_xu(x0, np.array([0.0, 0.0, 0.0, 0.0]))
print(np.array_equal(AB, AB1))
print(AB1.shape)
print(AB - AB1)

AB2 = mbp_dynamics.jacobian_xu(x0, np.array([0.0, 0.0, 0.0, 0.0]))
AB1hat = mbp_dynamics.calc_AB_first_order(x0, np.array([0.1, 0.0, 0.0, 0.0]),
    n_samples=100, std_u= 0.01 * np.ones(4))
AB2hat = mbp_dynamics.calc_B_zero_order(x0, np.array([0.1, 0.0, 0.0, 0.0]),
    n_samples=100, std_u= 0.01 * np.ones(4))    
AB3hat = mbp_dynamics.calc_AB_zero_order(x0, np.array([0.1, 0.0, 0.0, 0.0]),
    n_samples=100, std_u= 0.01 * np.ones(4))        
print(AB1hat)
print(AB2hat)
print(AB3hat)

plt.figure()
plt.subplot(4,1,1)
plt.title("Exact AB")
plt.imshow(AB1)
plt.colorbar()

plt.subplot(4,1,2)
plt.title("First order Smoothing AB")
plt.imshow(AB1hat)
plt.colorbar()

plt.subplot(4,1,3)
plt.title("Zero order Smoothing B")
plt.imshow(AB2hat)
plt.colorbar()

plt.subplot(4,1,4)
plt.title("Zero order Smoothing AB")
plt.imshow(AB3hat)
plt.colorbar()
plt.show()

## Test out trajopt related stuff. 
idx_a_l = mbp_dynamics.plant.GetModelInstanceByName(robot_l_name)
idx_a_r = mbp_dynamics.plant.GetModelInstanceByName(robot_r_name)
idx_u = mbp_dynamics.plant.GetModelInstanceByName(object_name)

Q_dict = {
    idx_u: np.array([10, 10, 0.2, 0.1, 0.1, 0.1]),
    idx_a_l: np.array([0.3, 0.4, 0.5, 0.6]),
    idx_a_r: np.array([0.7, 0.8, 0.9, 1.0])}
R_dict = {
    idx_a_l: np.array([0.3, 0.4]),
    idx_a_r: np.array([0.7, 0.8])}

params = IrsLqrQuasistaticParameters()
params.Q_dict = Q_dict
params.Qd_dict = {model: Q_i * 1 for model, Q_i in params.Q_dict.items()}
params.R_dict = R_dict

xd_dict = {idx_u: q_u0 + np.array([0.4, -0.1, 0, 0, 0, 0]),
           idx_a_l: qa_l_knots[0],
           idx_a_r: qa_r_knots[0]}
xd = mbp_dynamics.get_x_from_q_dict(xd_dict)
x_trj_d = np.tile(xd, (T + 1, 1))

params.x0 = x0
params.x_trj_d = x_trj_d
params.u_trj_0 = u_traj_0
params.T = T

params.u_bounds_rel = np.array([
    -np.ones(dim_u) * 0.5 * h, np.ones(dim_u) * 0.5 * h])

def sampling(u_initial, iter):
    return u_initial ** (0.5 * iter)

params.sampling = sampling
params.std_u_initial = np.ones(dim_u) * 0.3

params.decouple_AB = decouple_AB
params.use_workers = use_workers
params.gradient_mode = gradient_mode
params.task_stride = task_stride

