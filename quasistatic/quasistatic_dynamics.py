from typing import Dict, Set

import numpy as np
from pydrake.all import ModelInstanceIndex, MultibodyPlant
from quasistatic_simulator.core.quasistatic_simulator import (
    QuasistaticSimulator)
from quasistatic_simulator_py import (QuasistaticSimulatorCpp)


class QuasistaticDynamics:
    def __init__(self, h: float, q_sim_py: QuasistaticSimulator,
                 q_sim: QuasistaticSimulatorCpp):
        self.h = h
        self.q_sim_py = q_sim_py
        self.q_sim = q_sim
        self.plant = q_sim.get_plant()
        self.dim_x = self.plant.num_positions()
        self.dim_u = q_sim.num_actuated_dofs()

        self.models_all = self.q_sim.get_all_models()
        self.models_actuated = self.q_sim.get_actuated_models()
        self.models_unactuated = self.q_sim.get_unactuated_models()
        # TODO: distinguish between position indices and velocity indices for
        #  3D systems.
        self.position_indices = self.q_sim.get_velocity_indices()
        self.velocity_indices = self.position_indices

        # make sure that q_sim_py and q_sim have the same underlying plant.
        self.check_plants(
            plant_a=q_sim.get_plant(),
            plant_b=q_sim_py.get_plant(),
            models_all_a=q_sim.get_all_models(),
            models_all_b=q_sim_py.get_all_models(),
            velocity_indices_a=q_sim.get_velocity_indices(),
            velocity_indices_b=q_sim.get_velocity_indices())

    @staticmethod
    def check_plants(plant_a: MultibodyPlant, plant_b: MultibodyPlant,
                     models_all_a: Set[ModelInstanceIndex],
                     models_all_b: Set[ModelInstanceIndex],
                     velocity_indices_a: Dict[ModelInstanceIndex, np.ndarray],
                     velocity_indices_b: Dict[ModelInstanceIndex, np.ndarray]):
        """
        Make sure that plant_a and plant_b are identical.
        """
        assert models_all_a == models_all_b
        for model in models_all_a:
            name_a = plant_a.GetModelInstanceName(model)
            name_b = plant_b.GetModelInstanceName(model)
            assert name_a == name_b

            idx_a = velocity_indices_a[model]
            idx_b = velocity_indices_b[model]
            assert idx_a == idx_b

    def get_u_indices_into_x(self):
        u_indices = np.zeros(self.dim_u, dtype=int)
        i_start = 0
        for model in self.models_actuated:
            indices = self.velocity_indices[model]
            n_a_i = len(indices)
            u_indices[i_start: i_start + n_a_i] = indices
            i_start += n_a_i
        return u_indices

    def get_q_a_cmd_dict_from_u(self, u: np.ndarray):
        q_a_cmd_dict = dict()
        i_start = 0
        for model in self.models_actuated:
            n_v_i = self.plant.num_velocities(model)
            q_a_cmd_dict[model] = u[i_start: i_start + n_v_i]
            i_start += n_v_i

        return q_a_cmd_dict

    def get_q_dict_from_x(self, x: np.ndarray):
        q_dict = {
            model: x[n_q_indices]
            for model, n_q_indices in self.position_indices.items()}

        return q_dict

    def get_x_from_q_dict(self, q_dict: Dict[ModelInstanceIndex, np.ndarray]):
        x = np.zeros(self.dim_x)
        for model, n_q_indices in self.position_indices.items():
            x[n_q_indices] = q_dict[model]

        return x

    def get_u_from_q_cmd_dict(self,
                              q_cmd_dict: Dict[ModelInstanceIndex, np.ndarray]):
        u = np.zeros(self.dim_u)
        i_start = 0
        for model in self.models_actuated:
            n_v_i = self.plant.num_velocities(model)
            u[i_start: i_start + n_v_i] = q_cmd_dict[model]
            i_start += n_v_i

        return u

    def get_Q_from_Q_dict(self,
                          Q_dict: Dict[ModelInstanceIndex, np.ndarray]):
        Q = np.eye(self.dim_x)
        for model, idx in self.velocity_indices.items():
            Q[idx, idx] = Q_dict[model]
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
        q_dict = self.get_q_dict_from_x(x)
        q_a_cmd_dict = self.get_q_a_cmd_dict_from_u(u)
        tau_ext_dict = self.q_sim_py.calc_tau_ext([])

        self.q_sim_py.update_mbp_positions(q_dict)
        q_next_dict = self.q_sim_py.step(
            q_a_cmd_dict, tau_ext_dict, self.h,
            mode=mode, requires_grad=requires_grad)

        return self.get_x_from_q_dict(q_next_dict)

    def dynamics(self, x: np.ndarray, u: np.ndarray,
                 requires_grad: bool = False):
        """
        :param x: the position vector of self.q_sim.plant.
        :param u: commanded positions of models in
            self.q_sim.models_actuated, concatenated into one vector.
        """
        q_dict = self.get_q_dict_from_x(x)
        q_a_cmd_dict = self.get_q_a_cmd_dict_from_u(u)
        tau_ext_dict = self.q_sim.calc_tau_ext([])

        self.q_sim.update_mbp_positions(q_dict)
        self.q_sim.step(
            q_a_cmd_dict, tau_ext_dict, self.h,
            self.q_sim_py.sim_params.contact_detection_tolerance,
            requires_grad=requires_grad)
        q_next_dict = self.q_sim.get_mbp_positions()
        return self.get_x_from_q_dict(q_next_dict)

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

    def calc_AB_first_order(self, x_nominal: np.ndarray, u_nominal: np.ndarray,
                            n_samples: int, std: float):
        """
        x_nominal: (n_x,) array, 1 state.
        u_nominal: (n_u,) array, 1 input.
        """
        np.random.seed(2021)
        du = np.random.normal(0, std, size=[n_samples, self.dim_u])
        Ahat = np.zeros((self.dim_x, self.dim_x))
        Bhat = np.zeros((self.dim_x, self.dim_u))
        for i in range(n_samples):
            self.dynamics(x_nominal, u_nominal + du[i], requires_grad=True)
            Ahat += self.q_sim.get_Dq_nextDq()
            Bhat += self.q_sim.get_Dq_nextDqa_cmd()

        Ahat /= n_samples
        Bhat /= n_samples

        return Ahat, Bhat

    def calc_AB_first_order_batch(
            self, x_nominals: np.ndarray, u_nominals: np.ndarray,
            n_samples: int, std: float):
        """
        x_nominals: (n, n_x) array, n states.
        u_nominals: (n, n_u) array, n inputs.
        """
        n = x_nominals.shape[0]
        Ahat_list = np.zeros((n, self.dim_x, self.dim_x))
        Bhat_list = np.zeros((n, self.dim_x, self.dim_u))

        for i in range(n):
            Ahat_list[i], Bhat_list[i] = self.calc_AB_first_order(
                x_nominals[i], u_nominals[i], n_samples, std)

        return Ahat_list, Bhat_list

    def publish_trajectory(self, x_traj):
        q_dict_traj = [self.get_q_dict_from_x(x) for x in x_traj]
        self.q_sim_py.animate_system_trajectory(h=self.h,
                                                q_dict_traj=q_dict_traj)
