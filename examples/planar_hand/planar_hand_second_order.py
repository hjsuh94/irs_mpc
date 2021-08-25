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

from planar_hand_setup import *

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
    -np.pi/4, -np.pi/4,
    np.pi/4, np.pi/4,
    0, 0.35, 0.0,
    0, 0, 0, 0])

plant_context = plant_ad.GetMyMutableContextFromRoot(context)

# Set 
plant_ad.SetPositions(plant_context, 
    plant_ad.GetModelInstanceByName("arm_left"), xu_ad[0:2])

plant_ad.SetPositions(plant_context,
    plant_ad.GetModelInstanceByName("arm_right"), xu_ad[2:4])

plant_ad.SetPositions(plant_context, 
    plant_ad.GetModelInstanceByName("sphere"), xu_ad[4:7])

plant_ad.get_actuation_input_port(
    plant_ad.GetModelInstanceByName("arm_left")).FixValue(
        plant_context, xu_ad[7:9])
plant_ad.get_actuation_input_port(
    plant_ad.GetModelInstanceByName("arm_right")).FixValue(
        plant_context, xu_ad[9:11])

simulator_ad = Simulator_[AutoDiffXd](diagram_ad, context) 
simulator_ad.AdvanceTo(0.1)

plant_context = plant_ad.GetMyMutableContextFromRoot(context)

x_next = plant_ad.GetPositions(plant_context, plant_ad.GetModelInstanceByName("sphere"))
AB = autoDiffToGradientMatrix(x_next)

## Test out the MbpDynamics.
sim_params = QuasistaticSimParameters(
    gravity=gravity,
    nd_per_contact=2,
    contact_detection_tolerance=contact_detection_tolerance,
    is_quasi_dynamic=True)
q_sim_py = QuasistaticSimulator(
    model_directive_path=model_directive_path,
    robot_stiffness_dict=robot_stiffness_dict,
    object_sdf_paths=object_sdf_dict,
    sim_params=sim_params,
    internal_vis=False)
sim_params_cpp = cpp_params_from_py_params(sim_params)
sim_params_cpp.gradient_lstsq_tolerance = gradient_lstsq_tolerance
q_sim_cpp = QuasistaticSimulatorCpp(
    model_directive_path=model_directive_path,
    robot_stiffness_str=robot_stiffness_dict,
    object_sdf_paths=object_sdf_dict,
    sim_params=sim_params_cpp)

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
mbp_dynamics = MbpDynamics(h=0.1, q_sim_py=q_sim_py, q_sim=q_sim_cpp)
q0_dict = create_dict_keyed_by_model_instance_index(
    mbp_dynamics.plant, q_dict_str=q0_dict_str
)
x0 = mbp_dynamics.get_x_from_q_dict(q0_dict)
xnext = mbp_dynamics.dynamics_py(x0, np.array([0.0, 0.0, 0.0, 0.0]))
AB1 = mbp_dynamics.jacobian_xu(x0, np.array([0.0, 0.0, 0.0, 0.0]))
AB2 = mbp_dynamics.jacobian_xu(x0, np.array([0.0, 0.0, 0.0, 0.0]))
AB1hat = mbp_dynamics.calc_AB_first_order(x0, np.array([0.0, 0.0, 0.0, 0.0]),
    n_samples=100, std_u= 0.01 * np.ones(4))
AB2hat = mbp_dynamics.calc_B_zero_order(x0, np.array([0.0, 0.0, 0.0, 0.0]),
    n_samples=100, std_u= 0.01 * np.ones(4))    
AB3hat = mbp_dynamics.calc_AB_zero_order(x0, np.array([0.0, 0.0, 0.0, 0.0]),
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
