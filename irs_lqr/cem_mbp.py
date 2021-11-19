import time
from typing import TypeVar, Dict, List

import numpy as np
from pydrake.all import ModelInstanceIndex

from irs_lqr.mbp_dynamics import MbpDynamics
from irs_lqr.tv_lqr import solve_tvlqr, get_solver
from irs_lqr.cem_quasistatic import CemQuasistaticParameters

class CrossEntropyMethodMbp:
    def __init__(self, mbp_dynamics: MbpDynamics,
        params: CemQuasistaticParameters):
        """
        Base class for CrossEntropyMethod.

        system (DynamicalSystem class): dynamics class.
        parms (IrsLqrParameters class): parameters class.
        """

        self.mbp_dynamics = mbp_dynamics
        self.dim_x = mbp_dynamics.dim_x
        self.dim_u = mbp_dynamics.dim_u

        self.params = params

        # These store iterations for plotting.
        self.T = params.T
        self.x0 = params.x0
        self.Q_dict = params.Q_dict
        self.Q = self.mbp_dynamics.get_Q_from_Q_dict(self.Q_dict)
        self.Qd_dict = params.Qd_dict
        self.Qd = self.mbp_dynamics.get_Q_from_Q_dict(self.Qd_dict)
        self.R_dict = params.R_dict
        self.R = self.mbp_dynamics.get_R_from_R_dict(self.R_dict)
        self.x_trj_d = params.x_trj_d
        self.u_trj_0 = params.u_trj_0

        self.publish_every_iteration = params.publish_every_iteration

        self.x_trj = self.rollout(self.x0, self.u_trj_0)
        self.u_trj = self.u_trj_0  # T x m

        (cost_Qu, cost_Qu_final, cost_Qa, cost_Qa_final,
         cost_R) = self.eval_cost(self.x_trj, self.u_trj)
        self.cost = cost_Qu + cost_Qu_final + cost_Qa + cost_Qa_final + cost_R

        self.n_elite = params.n_elite
        self.batch_size = params.batch_size
        self.initial_std = params.initial_std
        self.std_trj = np.tile(self.initial_std, (self.T,1))

        self.x_trj_best = None
        self.u_trj_best = None
        self.cost_best = np.inf

        # logging
        self.x_trj_list = [self.x_trj]
        self.u_trj_list = [self.u_trj]

        self.cost_all_list = [self.cost]
        self.cost_Qu_list = [cost_Qu]
        self.cost_Qu_final_list = [cost_Qu_final]
        self.cost_Qa_list = [cost_Qa]
        self.cost_Qa_final_list = [cost_Qa_final]
        self.cost_R_list = [cost_R]

        self.current_iter = 1
        self.start_time = time.time()


    def rollout(self, x0: np.ndarray, u_trj: np.ndarray):
        T = u_trj.shape[0]
        assert T == self.T
        x_trj = np.zeros((T + 1, self.dim_x))
        x_trj[0, :] = x0
        for t in range(T):
            x_trj[t + 1, :] = self.mbp_dynamics.dynamics(x_trj[t, :], u_trj[t, :])
        return x_trj

    @staticmethod
    def calc_Q_cost(models_list: List[ModelInstanceIndex],
                    x_dict: Dict[ModelInstanceIndex, np.ndarray],
                    xd_dict: Dict[ModelInstanceIndex, np.ndarray],
                    Q_dict: Dict[ModelInstanceIndex, np.ndarray]):
        cost = 0.
        for model in models_list:
            x_i = x_dict[model]
            xd_i = xd_dict[model]
            Q_i = Q_dict[model]
            dx_i = x_i - xd_i
            cost += (dx_i * Q_i * dx_i).sum()

        return cost

    def eval_cost(self, x_trj, u_trj):
        T = u_trj.shape[0]
        assert T == self.T and x_trj.shape[0] == T + 1

        # Final cost Qd.
        x_dict = self.mbp_dynamics.get_qv_dict_from_x(x_trj[-1])
        xd_dict = self.mbp_dynamics.get_qv_dict_from_x(self.x_trj_d[-1])
        cost_Qu_final = self.calc_Q_cost(
            models_list=self.mbp_dynamics.models_unactuated,
            x_dict=x_dict, xd_dict=xd_dict, Q_dict=self.Qd_dict)
        cost_Qa_final = self.calc_Q_cost(
            models_list=self.mbp_dynamics.models_actuated,
            x_dict=x_dict, xd_dict=xd_dict, Q_dict=self.Qd_dict)

        # Q and R costs.
        cost_Qu = 0.
        cost_Qa = 0.
        cost_R = 0.
        for t in range(T):
            x_dict = self.mbp_dynamics.get_qv_dict_from_x(x_trj[t])
            xd_dict = self.mbp_dynamics.get_qv_dict_from_x(self.x_trj_d[t])
            # Q cost.
            cost_Qu += self.calc_Q_cost(
                models_list=self.mbp_dynamics.models_unactuated,
                x_dict=x_dict, xd_dict=xd_dict, Q_dict=self.Q_dict)
            cost_Qa += self.calc_Q_cost(
                models_list=self.mbp_dynamics.models_actuated,
                x_dict=x_dict, xd_dict=xd_dict, Q_dict=self.Q_dict)

            # R cost.
            cost_R += u_trj[t] @ self.R @ u_trj[t]

        return cost_Qu, cost_Qu_final, cost_Qa, cost_Qa_final, cost_R

    def local_descent(self, x_trj, u_trj):
        """
        Forward pass using a TV-LQR controller on the linearized dynamics.
        - args:
            x_trj (np.array, shape (T + 1) x n): nominal state trajectory.
            u_trj (np.array, shape T x m) : nominal input trajectory
        """

        # 1. Produce candidate trajectories according to u_std.
        u_trj_mean = u_trj
        u_trj_candidates = np.random.normal(u_trj_mean, self.std_trj,
            (self.batch_size, self.T, self.dim_u))
        cost_array = np.zeros(self.batch_size)

        # 2. Roll out the trajectories.
        for k in range(self.batch_size):
            u_trj_cand = u_trj_candidates[k,:,:]
            (cost_Qu, cost_Qu_final, cost_Qa, cost_Qa_final,
             cost_R) = self.eval_cost(
                 self.rollout(self.x0, u_trj_cand), u_trj_cand)
            cost_array[k] = cost_Qu + cost_Qu_final + cost_Qa + cost_Qa_final + cost_R
                

        # 3. Pick the best K trajectories.
        # NOTE(terry-suh): be careful what "best" means. 
        # In the reward setting, this is the highest. In cost, it's lowest.
        best_idx = np.argpartition(cost_array, self.n_elite)[:self.n_elite]

        best_trjs = u_trj_candidates[best_idx,:,:]

        # 4. Set mean as the new trajectory, and update std.
        u_trj_new = np.mean(best_trjs, axis=0)
        u_trj_std_new = np.std(best_trjs, axis=0)
        self.std_trj = u_trj_std_new
        x_trj_new = self.rollout(self.x0, u_trj_new)

        return x_trj_new, u_trj_new

    def iterate(self, max_iterations):
        while True:
            print('Iter {:02d},'.format(self.current_iter),
                  'cost: {:0.4f}.'.format(self.cost),
                  'time: {:0.2f}.'.format(time.time() - self.start_time))

            x_trj_new, u_trj_new = self.local_descent(self.x_trj, self.u_trj)
            (cost_Qu, cost_Qu_final, cost_Qa, cost_Qa_final,
             cost_R) = self.eval_cost(x_trj_new, u_trj_new)
            cost = cost_Qu + cost_Qu_final + cost_Qa + cost_Qa_final + cost_R
            self.x_trj_list.append(x_trj_new)
            self.u_trj_list.append(u_trj_new)
            self.cost_Qu_list.append(cost_Qu)
            self.cost_Qu_final_list.append(cost_Qu_final)
            self.cost_Qa_list.append(cost_Qa)
            self.cost_Qa_final_list.append(cost_Qa_final)
            self.cost_R_list.append(cost_R)
            self.cost_all_list.append(cost)

            if self.publish_every_iteration:
                self.q_dynamics.publish_trajectory(x_trj_new)

            if self.cost_best > cost:
                self.x_trj_best = x_trj_new
                self.u_trj_best = u_trj_new
                self.cost_best = cost

            if self.current_iter > max_iterations:
                break

            # Go over to next iteration.
            self.cost = cost
            self.x_trj = x_trj_new
            self.u_trj = u_trj_new
            self.current_iter += 1

        return self.x_trj, self.u_trj, self.cost
