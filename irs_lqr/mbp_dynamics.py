from typing import Dict, Set, Union, List

import numpy as np
from pydrake.all import (
    ModelInstanceIndex, MultibodyPlant, Simulator, Simulator_,
    AutoDiffXd, initializeAutoDiff, autoDiffToGradientMatrix,
    DiagramBuilder, ConnectMeshcatVisualizer)
from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer
from quasistatic_simulator.core.quasistatic_simulator import (
    QuasistaticSimulator, QuasistaticSimParameters)
from quasistatic_simulator.core.utils import create_plant_with_robots_and_objects
from irs_lqr.quasistatic_dynamics import QuasistaticDynamics


from irs_lqr.dynamical_system import DynamicalSystem

class MbpDynamics(DynamicalSystem):
    def __init__(self, h: float, model_directive_path: str,
        robot_stiffness_dict: Dict[str, np.ndarray],
        object_sdf_paths: Dict[str, str],
        sim_params: QuasistaticSimParameters):
        super().__init__()
        """
        Prerequisites for using MBP dynamics.
        1. q_sim_py has internal_vis to True.
        2. q_sim_py_ad has internal_vis to False, since Meshcat cannot be 
           autodiffed as a diagram.
        """
        self.h = h 
        self.model_directive_path = model_directive_path
        self.robot_stiffness_dict = robot_stiffness_dict
        self.object_sdf_paths = object_sdf_paths
        self.sim_params = sim_params

        # The quasistatic sim is used for using convenient methods like
        # getting indices of actuated / unactuated objects.
        self.q_sim = QuasistaticSimulator(
            model_directive_path=self.model_directive_path,
            robot_stiffness_dict=self.robot_stiffness_dict,
            object_sdf_paths=self.object_sdf_paths,
            sim_params=self.sim_params)

        # 1. Set up plants. 
        # This plant is for applications that do not require autodiff, and
        # has a Meshcat connected. 
        self.diagram, self.plant, self.scene_graph, self.robot_models, \
            self.object_models = self.create_diagram(
            internal_vis=True)

        # Set up stuff related to plant.
        self.dim_x = self.plant.num_positions() + self.plant.num_velocities()
        self.dim_u = self.q_sim.num_actuated_dofs()
        self.models_all = self.q_sim.models_all
        self.models_actuated = self.q_sim.models_actuated
        self.models_unactuated = self.q_sim.models_unactuated
        self.position_indices = self.q_sim.get_velocity_indices()
        self.velocity_indices = self.position_indices      

        # Currently only support two dimensional systems.
        assert(self.plant.num_positions() == self.plant.num_velocities())

        # This diagram is for autodiff
        self.diagram_ad, _, _, _, _ = self.create_diagram(
            internal_vis=False)
        self.diagram_ad = self.diagram_ad.ToAutoDiffXd()

        # Get plant and scene graph as well as their autodiff components.
        self.plant_ad = self.diagram_ad.GetSubsystemByName(
            self.plant.get_name())
        self.scene_graph_ad = self.diagram_ad.GetSubsystemByName(
            self.scene_graph.get_name()
        )

        # Get contexts.
        self.context = self.diagram.CreateDefaultContext()
        self.context_plant = self.diagram.GetMutableSubsystemContext(
            self.plant, self.context)
        self.context_sg = self.diagram.GetMutableSubsystemContext(
            self.scene_graph, self.context)

        self.context_ad = self.diagram_ad.CreateDefaultContext()
        self.context_plant_ad = self.diagram_ad.GetMutableSubsystemContext(
            self.plant_ad, self.context_ad)
        self.context_sg_ad = self.diagram_ad.GetMutableSubsystemContext(
            self.scene_graph_ad, self.context_ad)

        self.context_meshcat = self.diagram.GetMutableSubsystemContext(
            self.viz, self.context)

        # Set up simulators.
        self.simulator = Simulator(self.diagram, self.context)
        self.simulator_ad = Simulator_[AutoDiffXd](self.diagram_ad,
            self.context_ad)

        # Set up initial time for the simulators.
        self.simulator_time = 0.0
        self.simulator_time_ad = 0.0

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
        
        if internal_vis:
            self.viz = ConnectMeshcatVisualizer(builder, scene_graph)

        diagram = builder.Build()
        return diagram, plant, scene_graph, robot_models, object_models

    def get_x_from_qv_dict(self, q_dict: Dict[ModelInstanceIndex, np.ndarray]):
        """
        Current assumes len(positions) == len(velocities)
        """
        x = np.zeros(self.plant.num_positions())
        v = np.zeros(self.plant.num_velocities())
        for model, n_q_indices in self.position_indices.items():
            x[n_q_indices] = q_dict[model][:len(n_q_indices)]
        for model, n_q_indices in self.velocity_indices.items():
            v[n_q_indices] = q_dict[model][len(n_q_indices):]
        return np.hstack((x,v))

    def get_qv_dict_from_x(self, x: np.ndarray):
        """
        Current assumes len(positions) == len(velocities)
        """        
        offset = self.plant.num_velocities()
        qv_dict = {
            model: np.hstack((x[n_q_indices], x[np.array(n_q_indices) + offset]))
            for model, n_q_indices in self.position_indices.items()}
        return qv_dict

    def get_q_dict_from_x(self, x: np.ndarray):
        q_dict = {
            model: x[n_q_indices]
            for model, n_q_indices in self.position_indices.items()}

        return q_dict

    def get_q_a_cmd_dict_from_u(self, u: np.ndarray):
        q_a_cmd_dict = dict()
        i_start = 0
        for model in self.models_actuated:
            n_v_i = self.plant.num_velocities(model)
            q_a_cmd_dict[model] = u[i_start: i_start + n_v_i]
            i_start += n_v_i

        return q_a_cmd_dict

    def publish_trajectory(self, x_traj):
        q_dict_traj = [self.get_q_dict_from_x(x) for x in x_traj]
        self.animate_system_trajectory(h=self.h, q_dict_traj=q_dict_traj)

    def animate_system_trajectory(self, h: float,
                                  q_dict_traj: List[
                                      Dict[ModelInstanceIndex, np.ndarray]]):
        self.viz.draw_period = h
        self.viz.reset_recording()
        self.viz.start_recording()
        for q_dict in q_dict_traj:
            self.update_mbp_positions(q_dict, self.plant, self.context_plant,
                self.scene_graph, self.context_sg)
            self.viz.DoPublish(self.context_meshcat)

        self.viz.stop_recording()
        self.viz.publish_recording()

    def update_mbp_positions(self, plant, plant_context, scene_graph,
        scene_graph_context, q_dict):
        for model_instance_idx, q in q_dict.items():
            plant.SetPositions(
                plant_context, model_instance_idx, q)

        # Update query object.
        self.query_object = scene_graph.get_query_output_port().Eval(
                scene_graph_context)

    def update_mbp_positions_and_velocities_from_dict(
        self, plant, plant_context, 
        scene_graph, scene_graph_context, qv_dict):

        for model_instance_idx, qv in qv_dict.items():
            plant.SetPositionsAndVelocities(
                plant_context,
                plant.GetModelInstanceByName(
                    self.plant.GetModelInstanceName(model_instance_idx)), qv)

        # Update query object.
        self.query_object = scene_graph.get_query_output_port().Eval(
                scene_graph_context)

    def update_mbp_positions_and_velocities_from_vec(self, plant, plant_context,
        scene_graph, scene_graph_context, qv_vec):
        plant.SetPositionsAndVelocities(plant_context, qv_vec)
        # Update query object.
        self.query_object = scene_graph.get_query_output_port().Eval(
                scene_graph_context)

    def update_mbp_inputs(self, plant, plant_context, q_a_dict):
        for model in self.models_actuated:
            plant.get_actuation_input_port(model).FixValue(
                    plant_context, q_a_dict[model])

    def get_Q_from_Q_dict(self,
                          Q_dict: Dict[ModelInstanceIndex, np.ndarray]):
        Q = np.eye(self.dim_x)
        offset = self.plant.num_velocities()
        for model, idx in self.position_indices.items():
            Q[idx, idx] = Q_dict[model][:len(idx)]
        for model, idx in self.velocity_indices.items():
            Q[np.array(idx) + offset, np.array(idx) + offset] = Q_dict[
                model][len(idx):]
        return Q

    def get_R_from_R_dict(self,
                          R_dict: Dict[ModelInstanceIndex, np.ndarray]):
        R = np.eye(self.dim_u)
        i_start = 0
        for model in self.models_actuated:
            n_v_i = self.plant.num_velocities(model)
            R[i_start: i_start + n_v_i, i_start: i_start + n_v_i] = \
                R_dict[model]
        return R

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
        self.update_mbp_inputs(self.plant, self.context_plant,
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
        self.update_mbp_inputs(self.plant, self.context_plant,
            q_a_cmd_dict)

        self.simulator_time += self.h
        self.simulator.AdvanceTo(self.simulator_time)

        x_next = self.plant.GetPositionsAndVelocities(self.context_plant)
        return x_next

    def dynamics_batch(self, x, u):
        """
        Batch dynamics. Uses pytorch for
        -args:
            x (np.array, dim: B x n): batched state
            u (np.array, dim: B x m): batched input
        -returns:
            x_next (np.array, dim: B x n): batched next state
        """
        n_batch = x.shape[0]
        x_next = np.zeros((n_batch, self.dim_x))

        for i in range(n_batch):
            x_next[i] = self.dynamics(x[i], u[i])
        return x_next

    def jacobian_xu(self, x, u):
        xu_ad = initializeAutoDiff(np.hstack((x,u)))

        x_ad = xu_ad[:self.dim_x]
        u_ad = xu_ad[self.dim_x:]

        q_a_cmd_dict = self.get_q_a_cmd_dict_from_u(u_ad)

        self.update_mbp_positions_and_velocities_from_vec(
            self.plant_ad, self.context_plant_ad,
            self.scene_graph_ad, self.context_sg_ad, x_ad)
        self.update_mbp_inputs(self.plant_ad, self.context_plant_ad,
            q_a_cmd_dict)

        self.simulator_time_ad += self.h
        self.simulator_ad.AdvanceTo(self.simulator_time_ad)
        x_next = self.plant_ad.GetPositionsAndVelocities(self.context_plant_ad)
        return autoDiffToGradientMatrix(x_next)

    def calc_AB_exact(self, x_nominal: np.ndarray, u_nominal: np.ndarray):
        return self.jacobian_xu(x_nominal, u_nominal)

    def calc_AB_first_order(self, x_nominal: np.ndarray, u_nominal: np.ndarray,
                            n_samples: int, std_u: Union[np.ndarray, float]):
        """
        x_nominal: (n_x,) array, 1 state.
        u_nominal: (n_u,) array, 1 input.
        """
        # np.random.seed(2021)
        du = np.random.normal(0, std_u, size=[n_samples, self.dim_u])
        ABhat = np.zeros((self.dim_x, self.dim_x + self.dim_u))
        for i in range(n_samples):
            ABhat += self.jacobian_xu(x_nominal, u_nominal + du[i])

        ABhat /= n_samples
        return ABhat

    def calc_AB_batch(
            self, x_nominals: np.ndarray, u_nominals: np.ndarray,
            n_samples: int, std_u: Union[np.ndarray, float], mode: str):
        """
        x_nominals: (n, n_x) array, n states.
        u_nominals: (n, n_u) array, n inputs.
        mode: "first_order", "zero_order_B", "zero_order_AB", or "exact."
        """
        n = x_nominals.shape[0]
        ABhat_list = np.zeros((n, self.dim_x, self.dim_x + self.dim_u))

        if mode == "first_order":
            for i in range(n):
                ABhat_list[i] = self.calc_AB_first_order(
                    x_nominals[i], u_nominals[i], n_samples, std_u)
        elif mode == "zero_order_B":
            for i in range(n):
                ABhat_list[i] = self.calc_B_zero_order(
                    x_nominals[i], u_nominals[i], n_samples, std_u)
        elif mode == "zero_order_AB":
            for i in range(n):
                ABhat_list[i] = self.calc_AB_zero_order(
                    x_nominals[i], u_nominals[i], n_samples, std_u)                    
        elif mode == "exact":
            for i in range(n):
                ABhat_list[i] = self.calc_AB_exact(
                    x_nominals[i], u_nominals[i])

        return ABhat_list

    def calc_B_zero_order(self, x_nominal: np.ndarray, u_nominal: np.ndarray,
                          n_samples: int, std_u: Union[np.ndarray, float]):
        """
        Computes B:=df/du using least-square fit, and A:=df/dx using the
            exact gradient at x_nominal and u_nominal.
        :param std_u: standard deviation of the normal distribution when
            sampling u.
        """

        n_x = self.dim_x
        n_u = self.dim_u
        x_next_nominal = self.dynamics(
            x_nominal, u_nominal, requires_grad=True)
        ABhat = np.zeros((n_x, n_x + n_u))
        AB_first_order = self.calc_AB_first_order(x_nominal, u_nominal,
            n_samples, std_u)
        ABhat[:, :n_x] = AB_first_order[:, :n_x]

        du = np.random.normal(0, std_u, size=[n_samples, self.dim_u])
        x_next = np.zeros((n_samples, self.dim_x))

        for i in range(n_samples):
            x_next[i] = self.dynamics(x_nominal, u_nominal + du[i])

        dx_next = x_next - x_next_nominal
        ABhat[:, n_x:] = np.linalg.lstsq(du, dx_next, rcond=None)[0].transpose()

        return ABhat

    def calc_AB_zero_order(self, x_nominal: np.ndarray, u_nominal: np.ndarray,
                           n_samples: int, std_u: Union[np.ndarray, float],
                           std_x: Union[np.ndarray, float] = 1e-3,
                           damp: float = 1e-10):
        """
        Computes both A:=df/dx and B:=df/du using least-square fit.
        :param std_x (n_x,): standard deviation of the normal distribution
            when sampling x.
        :param damp, weight of norm-regularization when solving for A and B.
        """
        n_x = self.dim_x
        n_u = self.dim_u
        dx = np.random.normal(0, std_x, size=[n_samples, n_x])
        du = np.random.normal(0, std_u, size=[n_samples, n_u])

        x_next_nominal = self.dynamics(x_nominal, u_nominal)
        x_next = np.zeros((n_samples, n_x))

        for i in range(n_samples):
            x_next[i] = self.dynamics(x_nominal + dx[i], u_nominal + du[i])

        dx_next = x_next - x_next_nominal
        # A, B as in AX = B, not the linearized dynamics.
        A = np.zeros((n_samples + n_x + n_u, n_x + n_u))
        A[:n_samples, :n_x] = dx
        A[:n_samples, n_x:] = du
        A[n_samples:] = np.eye(n_x + n_u, n_x + n_u) * damp
        B = np.zeros((n_samples + n_x + n_u, n_x))
        B[:n_samples] = dx_next

        ABhat = np.linalg.lstsq(A, B, rcond=None)[0].transpose()

        return ABhat
