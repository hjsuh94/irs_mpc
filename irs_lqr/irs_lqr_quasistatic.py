from typing import Dict

from pydrake.all import ModelInstanceIndex

from irs_lqr.quasistatic_dynamics import QuasistaticDynamics
from irs_lqr.tv_lqr import solve_tvlqr_quasistatic, get_solver

from zmq_parallel_cmp.array_io import *


class IrsLqrQuasistatic:
    def __init__(self, q_dynamics: QuasistaticDynamics,
                 std_u_initial: np.ndarray, T: int,
                 Q_dict: Dict[ModelInstanceIndex, np.ndarray],
                 R_dict: Dict[ModelInstanceIndex, np.ndarray],
                 Qd_dict: Dict[ModelInstanceIndex, np.ndarray],
                 x_trj_d: np.ndarray, dx_bounds: np.ndarray,
                 du_bounds: np.ndarray,
                 x0: np.ndarray, u_trj_0: np.ndarray):
        """

        Arguments are similar to those of SqpLsImplicit.
        Only samples u to estimate B.
        A uses the first derivative of the dynamics at x.
        """
        self.q_dynamics = q_dynamics
        self.dim_x = q_dynamics.dim_x
        self.dim_u = q_dynamics.dim_u

        self.T = T
        self.x0 = x0
        self.Q_dict = Q_dict
        self.Q = self.q_dynamics.get_Q_from_Q_dict(Q_dict)
        self.Qd_dict = Qd_dict
        self.Qd = self.q_dynamics.get_Q_from_Q_dict(Qd_dict)
        self.R_dict = R_dict
        self.R = self.q_dynamics.get_R_from_R_dict(R_dict)
        self.x_trj_d = x_trj_d
        self.dx_bounds = dx_bounds
        self.du_bounds = du_bounds
        self.indices_u_into_x = q_dynamics.get_u_indices_into_x()

        self.x_trj = self.rollout(x0, u_trj_0)
        self.u_trj = u_trj_0  # T x m

        (cost_Qu, cost_Qu_final, cost_Qa, cost_Qa_final,
         cost_R) = self.eval_cost(self.x_trj, self.u_trj)
        self.cost = cost_Qu + cost_Qu_final + cost_Qa + cost_Qa_final + cost_R

        self.x_trj_best = None
        self.u_trj_best = None
        self.cost_best = np.inf

        # sampling standard deviation.
        self.std_u_initial = std_u_initial

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

        # solver
        self.solver = get_solver('gurobi')

        # parallelization.
        context = zmq.Context()

        # Socket to send messages on
        self.sender = context.socket(zmq.PUSH)
        self.sender.bind("tcp://*:5557")

        # Socket to receive messages on
        self.receiver = context.socket(zmq.PULL)
        self.receiver.bind("tcp://*:5558")

        print("Press Enter when the workers are ready: ")
        input()
        print("Sending tasks to workers...")

    def rollout(self, x0: np.ndarray, u_trj: np.ndarray):
        T = u_trj.shape[0]
        assert T == self.T
        x_trj = np.zeros((T + 1, self.dim_x))
        x_trj[0, :] = x0
        for t in range(T):
            x_trj[t + 1, :] = self.q_dynamics.dynamics(x_trj[t, :], u_trj[t, :])
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
        idx_u_into_x = self.q_dynamics.get_u_indices_into_x()

        # Final cost Qd.
        x_dict = self.q_dynamics.get_q_dict_from_x(x_trj[-1])
        xd_dict = self.q_dynamics.get_q_dict_from_x(self.x_trj_d[-1])
        cost_Qu_final = self.calc_Q_cost(
            models_list=self.q_dynamics.models_unactuated,
            x_dict=x_dict, xd_dict=xd_dict, Q_dict=self.Qd_dict)
        cost_Qa_final = self.calc_Q_cost(
            models_list=self.q_dynamics.models_actuated,
            x_dict=x_dict, xd_dict=xd_dict, Q_dict=self.Qd_dict)

        # Q and R costs.
        cost_Qu = 0.
        cost_Qa = 0.
        cost_R = 0.
        for t in range(T):
            x_dict = self.q_dynamics.get_q_dict_from_x(x_trj[t])
            xd_dict = self.q_dynamics.get_q_dict_from_x(self.x_trj_d[t])
            # Q cost.
            cost_Qu += self.calc_Q_cost(
                models_list=self.q_dynamics.models_unactuated,
                x_dict=x_dict, xd_dict=xd_dict, Q_dict=self.Q_dict)
            cost_Qa += self.calc_Q_cost(
                models_list=self.q_dynamics.models_actuated,
                x_dict=x_dict, xd_dict=xd_dict, Q_dict=self.Q_dict)

            # R cost.
            if t == 0:
                du = u_trj[t] - x_trj[t, idx_u_into_x]
            else:
                du = u_trj[t] - u_trj[t - 1]
            cost_R += du @ self.R @ du

        return cost_Qu, cost_Qu_final, cost_Qa, cost_Qa_final, cost_R

    def calc_current_std(self):
        a = self.current_iter ** 0.8
        return self.std_u_initial / a

    def get_TV_matrices(self, x_trj, u_trj):
        """
        Get time varying linearized dynamics given a nominal trajectory.
        - args:
            x_trj (np.array, shape (T + 1) x n)
            u_trj (np.array, shape T x m)
        """
        T = u_trj.shape[0]
        assert self.T == T
        At = np.zeros((T, self.dim_x, self.dim_x))
        Bt = np.zeros((T, self.dim_x, self.dim_u))
        ct = np.zeros((T, self.dim_x))
        std_u = self.calc_current_std()

        for t in range(T):
            ABhat = self.q_dynamics.calc_AB_first_order(
                x_nominal=x_trj[t],
                u_nominal=u_trj[t],
                n_samples=100,
                std=std_u)

            At[t] = ABhat[:, :self.dim_x]
            Bt[t] = ABhat[:, self.dim_x:]
            x_next_nominal = self.q_dynamics.dynamics(x_trj[t], u_trj[t])
            ct[t] = x_next_nominal - At[t].dot(x_trj[t]) - Bt[t].dot(u_trj[t])

        return At, Bt, ct

    def get_TV_matrices_batch(self, x_trj, u_trj):
        """
        Get time varying linearized dynamics given a nominal trajectory,
        using worker processes launched separately.
        - args:
            x_trj (np.array, shape (T + 1) x n)
            u_trj (np.array, shape T x m)
        """
        T = u_trj.shape[0]
        assert self.T == T
        At = np.zeros((T, self.dim_x, self.dim_x))
        Bt = np.zeros((T, self.dim_x, self.dim_u))
        ct = np.zeros((T, self.dim_x))
        std_u = self.calc_current_std()

        # send tasks.
        stride = 2
        n_tasks_sent = 0
        for t in range(0, T, stride):
            t1 = min(t + stride, T)
            x_u = np.zeros((t1 - t, self.dim_x + self.dim_u))
            x_u[:, :self.dim_x] = x_trj[t: t1]
            x_u[:, self.dim_x:] = u_trj[t: t1]
            send_array(
                self.sender, x_u,
                t=np.arange(t, t1).tolist(),
                n_samples=100, std=std_u.tolist())
            n_tasks_sent += 1

        # receive tasks.
        for _ in range(n_tasks_sent):
            ABhat, t_list, _, _ = recv_array(self.receiver)
            At[t_list] = ABhat[:, :, :self.dim_x]
            Bt[t_list] = ABhat[:, :, self.dim_x:]

        # compute ct
        for t in range(T):
            x_next_nominal = self.q_dynamics.dynamics(x_trj[t], u_trj[t])
            ct[t] = x_next_nominal - At[t].dot(x_trj[t]) - Bt[t].dot(u_trj[t])

        return At, Bt, ct

    def local_descent(self, x_trj, u_trj):
        """
        Forward pass using a TV-LQR controller on the linearized dynamics.
        - args:
            x_trj (np.array, shape (T + 1) x n): nominal state trajectory.
            u_trj (np.array, shape T x m) : nominal input trajectory
        """
        At, Bt, ct = self.get_TV_matrices_batch(x_trj, u_trj)
        x_trj_new = np.zeros(x_trj.shape)
        x_trj_new[0, :] = x_trj[0, :]
        u_trj_new = np.zeros(u_trj.shape)

        '''
        x_bounds: (2, T + 1, dim_x). 
            - x_bounds[0]: lower bounds.
            - x_bounds[1]: upper bounds.
        u_bounds: (2, T, dim_u)
            - u_bounds[0]: lower bounds.
            - u_bounds[1]: upper bounds. 
        '''
        x_bounds = np.zeros((2, self.T + 1, self.dim_x))
        u_bounds = np.zeros((2, self.T, self.dim_u))
        x_bounds[0] = x_trj + self.dx_bounds[0]
        x_bounds[1] = x_trj + self.dx_bounds[1]
        u_bounds[0] = x_trj[:-1, self.indices_u_into_x] + self.du_bounds[0]
        u_bounds[1] = x_trj[:-1, self.indices_u_into_x] + self.du_bounds[1]

        for t in range(self.T):
            x_star, u_star = solve_tvlqr_quasistatic(
                At[t:self.T],
                Bt[t:self.T],
                ct[t:self.T], self.Q, self.Qd,
                self.R, x_trj_new[t, :],
                self.x_trj_d[t:],
                x_bounds[:, t:, :], u_bounds[:, t:, :],
                indices_u_into_x=self.indices_u_into_x,
                solver=self.solver,
                xinit=None,
                uinit=None)
            u_trj_new[t, :] = u_star[0]
            x_trj_new[t + 1, :] = self.q_dynamics.dynamics(
                x_trj_new[t], u_trj_new[t])

        return x_trj_new, u_trj_new

    def iterate(self, max_iterations):
        while True:
            print('Iter {},'.format(self.current_iter),
                  'cost: {}.'.format(self.cost))

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