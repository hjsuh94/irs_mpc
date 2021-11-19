from typing import Dict, Set, Union, List

import numpy as np
from pydrake.all import (
    ModelInstanceIndex, MultibodyPlant, Simulator, Simulator_,
    AutoDiffXd, initializeAutoDiff, autoDiffToGradientMatrix,
    DiagramBuilder, ConnectMeshcatVisualizer, PidController)
from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer
from qsim.simulator import (QuasistaticSimulator, QuasistaticSimParameters)
from qsim.system import (cpp_params_from_py_params)
from qsim.utils import create_plant_with_robots_and_objects
from irs_lqr.quasistatic_dynamics import QuasistaticDynamics

from irs_lqr.mbp_dynamics import MbpDynamics

class MbpDynamicsPosition(MbpDynamics):
    def __init__(self, h: float, model_directive_path: str,
        robot_stiffness_dict: Dict[str, np.ndarray],
        object_sdf_paths: Dict[str, str],
        sim_params: QuasistaticSimParameters,
        internal_vis: bool = False):
        super().__init__(h, model_directive_path, robot_stiffness_dict,
            object_sdf_paths, sim_params, internal_vis)
        """
        Position controlled MbpDynamics. Same implementation with MbpDynamics
        in all but the following:
        1. Plant has been wired with a PidController.
        2. Now the input 
        """

    def get_u_indices_into_x(self):
        u_indices = np.zeros(self.dim_u, dtype=int)
        i_start = 0
        for model in self.models_actuated:
            indices = self.velocity_indices[model]
            n_a_i = len(indices)
            u_indices[i_start: i_start + n_a_i] = indices
            i_start += n_a_i
        return u_indices

    def create_diagram(self, internal_vis: bool = False):
        builder = DiagramBuilder()
        plant, scene_graph, robot_models, object_models = \
            create_plant_with_robots_and_objects(
                builder=builder,
                model_directive_path=self.model_directive_path,
                robot_names=[
                    name for name in self.robot_stiffness_dict.keys()],
                object_sdf_paths=self.object_sdf_paths,
                time_step=1e-3,  # Only useful for MBP simulations.
                gravity=self.sim_params.gravity)

        # Add the pid controllers for each arm.
        controller_dict = {}
        for robot_name in self.robot_stiffness_dict.keys():
            stiffness = self.robot_stiffness_dict[robot_name]
            controller = builder.AddSystem(PidController(
                kp=stiffness, ki=np.zeros(2), kd=0.2 * stiffness))
            controller_dict[robot_name] = controller

        for model in self.models_actuated:
            robot_name = plant.GetModelInstanceName(model)
            controller = controller_dict[robot_name]
            builder.Connect(
                controller.get_output_port_control(),
                plant.get_actuation_input_port(model))
            builder.Connect(
                plant.get_state_output_port(model),
                controller.get_input_port_estimated_state())
            builder.ExportInput(controller.get_input_port_desired_state(),
                robot_name + "_desired_position")

        if internal_vis:
            self.viz = ConnectMeshcatVisualizer(builder, scene_graph)

        diagram = builder.Build()
        return diagram, plant, scene_graph, robot_models, object_models

    def update_mbp_inputs(self, plant, diagram, context, q_a_dict):
        for model in self.models_actuated:
            robot_name = plant.GetModelInstanceName(model)
            # The following here is to resolve differences between 
            # the double q_a_dict[model], and the 
            # AutodiffXd q_a_dict[model].

            if len(np.array(q_a_dict[model]).shape) == 1:
                diagram.GetInputPort(robot_name + "_desired_position").FixValue(
                    context, np.hstack((
                        np.array(q_a_dict[model]), np.zeros(2))))
            else:
                diagram.GetInputPort(robot_name + "_desired_position").FixValue(
                    context, np.vstack((
                        np.array(q_a_dict[model]), np.zeros((2,1)))))

    def dynamics_py(self, x: np.ndarray, u: np.ndarray,
                    mode: str = 'qp_mp', requires_grad: bool = False):
        """
        :param x: the position vector of self.q_sim.plant.
        :param u: commanded positions of models in
            self.q_sim.models_actuated, concatenated into one vector.
        """
        qv_dict = self.get_qv_dict_from_x(x)
        q_a_cmd_dict = self.get_q_a_cmd_dict_from_u(u)

        self.update_mbp_positions_and_velocities_from_dict(
            self.plant, self.context_plant,
            self.scene_graph, self.context_sg, qv_dict)
        self.update_mbp_inputs(self.plant, self.diagram, self.context,
            q_a_cmd_dict)

        self.simulator_time += self.h
        self.simulator.AdvanceTo(self.simulator_time)

        x_next = self.plant.GetPositionsAndVelocities(self.context_plant)
        return x_next

    def dynamics(self, x: np.ndarray, u: np.ndarray,
                 requires_grad: bool = False):
        """
        :param x: the position vector of self.q_sim.plant.
        :param u: commanded positions of models in
            self.q_sim.models_actuated, concatenated into one vector.
        """
        q_dict = self.get_qv_dict_from_x(x)
        q_a_cmd_dict = self.get_q_a_cmd_dict_from_u(u)

        self.update_mbp_positions_and_velocities_from_dict(
            self.plant, self.context_plant,
            self.scene_graph, self.context_sg, q_dict)
        self.update_mbp_inputs(self.plant, self.diagram, self.context,
            q_a_cmd_dict)

        self.simulator_time += self.h
        self.simulator.AdvanceTo(self.simulator_time)

        x_next = self.plant.GetPositionsAndVelocities(self.context_plant)
        return x_next

    def jacobian_xu(self, x, u):
        xu_ad = initializeAutoDiff(np.hstack((x,u)))

        x_ad = xu_ad[:self.dim_x]
        u_ad = xu_ad[self.dim_x:]

        q_a_cmd_dict = self.get_q_a_cmd_dict_from_u(u_ad)

        self.update_mbp_positions_and_velocities_from_vec(
            self.plant_ad, self.context_plant_ad,
            self.scene_graph_ad, self.context_sg_ad, x_ad)
        self.update_mbp_inputs(self.plant_ad, self.diagram_ad,
            self.context_ad, q_a_cmd_dict)

        self.simulator_time_ad += self.h
        self.simulator_ad.AdvanceTo(self.simulator_time_ad)
        x_next = self.plant_ad.GetPositionsAndVelocities(self.context_plant_ad)
        return autoDiffToGradientMatrix(x_next)
