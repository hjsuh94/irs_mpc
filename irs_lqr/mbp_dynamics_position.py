from typing import Dict, Set, Union, List

import numpy as np
from pydrake.all import (
    ModelInstanceIndex, MultibodyPlant, Simulator, Simulator_,
    AutoDiffXd, initializeAutoDiff, autoDiffToGradientMatrix,
    DiagramBuilder, ConnectMeshcatVisualizer, PidController)
from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer
from quasistatic_simulator.core.quasistatic_simulator import (
    QuasistaticSimulator, QuasistaticSimParameters)
from quasistatic_simulator.core.utils import create_plant_with_robots_and_objects
from irs_lqr.quasistatic_dynamics import QuasistaticDynamics


from irs_lqr.mbp_dynamics import MbpDynamics

class MbpDynamicsPosition(MbpDynamics):
    def __init__(self, h: float, model_directive_path: str,
        robot_stiffness_dict: Dict[str, np.ndarray],
        object_sdf_paths: Dict[str, str],
        sim_params: QuasistaticSimParameters):
        super().__init__(h, model_directive_path, robot_stiffness_dict,
            object_sdf_paths, sim_params)
        """
        Position controlled MbpDynamics. Same implementation with MbpDynamics
        in all but the following:
        1. Plant has been wired with a PidController.
        2. Now the input 
        """

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
            diagram.GetInputPort(robot_name + "_desired_position").FixValue(
                context, np.hstack((q_a_dict[model], np.zeros(2)))
        )
