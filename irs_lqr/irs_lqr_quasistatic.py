from typing import Dict
import time 

from pydrake.all import ModelInstanceIndex

from irs_lqr.quasistatic_dynamics import QuasistaticDynamics
from irs_lqr.tv_lqr import solve_tvlqr, get_solver

from zmq_parallel_cmp.array_io import *

class IrsLqrQuasistaticParameters:
    def __init__(self):
        # Necessary arguments defining optimal control problem.
        self.Q_dict = None
        self.Qd_dict = None
        self.R_dict = None
        self.x0 = None
        self.x_trj_d = None
        self.u_trj_0 = None
        self.T = None

        # Optional arguments defining bounds.
        self.x_bounds_abs = None
        self.u_bounds_abs = None
        self.x_bounds_rel = None
        self.u_bounds_rel = None

        # Necessary arguments related to sampling.
        self.sampling = None
        self.std_u_initial = None        
        self.num_samples = 100

        # Arguments related to various options.
        self.decouple_AB = True
        self.use_workers = True
        self.gradient_mode = "zero_order_B"
        self.solver_name = "gurobi"
        self.task_stride = 1
        self.publish_every_iteration = True

class IrsLqrQuasistatic:
    def __init__(self, q_dynamics: QuasistaticDynamics, 
        params: IrsLqrQuasistaticParameters):
        """

        Arguments are similar to those of SqpLsImplicit.
        Only samples u to estimate B.
        A uses the first derivative of the dynamics at x.

        sampling receives sampling(initial_std, iter) and returns the 
        current std.
        """
        self.q_dynamics = q_dynamics
        self.dim_x = q_dynamics.dim_x
        self.dim_u = q_dynamics.dim_u

        self.params = params

        self.T = params.T
        self.x0 = params.x0
        self.Q_dict = params.Q_dict
        self.Q = self.q_dynamics.get_Q_from_Q_dict(self.Q_dict)
        self.Qd_dict = params.Qd_dict
        self.Qd = self.q_dynamics.get_Q_from_Q_dict(self.Qd_dict)
        self.R_dict = params.R_dict
        self.R = self.q_dynamics.get_R_from_R_dict(self.R_dict)
        self.x_trj_d = params.x_trj_d
        self.u_trj_0 = params.u_trj_0
        self.x_bounds_abs = params.x_bounds_abs
        self.u_bounds_abs = params.u_bounds_abs
        self.x_bounds_rel = params.x_bounds_rel
        self.u_bounds_rel = params.u_bounds_rel
        self.indices_u_into_x = q_dynamics.get_u_indices_into_x()

        self.decouple_AB = params.decouple_AB 
        self.use_workers = params.use_workers
        self.gradient_mode = params.gradient_mode
        self.task_stride = params.task_stride
        self.publish_every_iteration = params.publish_every_iteration

        self.x_trj = self.rollout(self.x0, self.u_trj_0)
        self.u_trj = self.u_trj_0  # T x m

        (cost_Qu, cost_Qu_final, cost_Qa, cost_Qa_final,
         cost_R) = self.eval_cost(self.x_trj, self.u_trj)
        self.cost = cost_Qu + cost_Qu_final + cost_Qa + cost_Qa_final + cost_R

        self.x_trj_best = None
        self.u_trj_best = None
        self.cost_best = np.inf

        # sampling standard deviation.
        self.std_u_initial = params.std_u_initial
        self.sampling = params.sampling
        self.num_samples = params.num_samples

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

        # solver
        self.solver = get_solver(params.solver_name)

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
        std_u = self.sampling(self.std_u_initial, self.current_iter)

        # Compute ABhat.
        ABhat_list = self.q_dynamics.calc_AB_batch(
            x_trj, u_trj, n_samples=self.num_samples, std_u=std_u,
            mode=self.gradient_mode
        )

        for t in range(T):
            At[t] = ABhat_list[t, :, :self.dim_x]
            Bt[t] = ABhat_list[t, :, self.dim_x:]

            if self.decouple_AB:
                At[t, self.indices_u_into_x, :] = 0.0
                At[t, :, self.indices_u_into_x] = 0.0
                Bt[t, self.indices_u_into_x, :] = 0.0

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
        std_u = self.sampling(self.std_u_initial, self.current_iter)
        print(std_u)

        # send tasks.
        # TODO: make the stride a parameter of the class.
        stride = self.task_stride
        n_tasks_sent = 0
        for t in range(0, T, stride):
            t1 = min(t + stride, T)
            x_u = np.zeros((t1 - t, self.dim_x + self.dim_u))
            x_u[:, :self.dim_x] = x_trj[t: t1]
            x_u[:, self.dim_x:] = u_trj[t: t1]
            # TODO: support 1-order and first-order computation of the gradient.
            send_array(
                self.sender, x_u,
                t=np.arange(t, t1).tolist(),
                n_samples=self.num_samples, std=std_u.tolist())
            n_tasks_sent += 1

        # receive tasks.
        for _ in range(n_tasks_sent):
            ABhat, t_list, _, _ = recv_array(self.receiver)
            At[t_list] = ABhat[:, :, :self.dim_x]
            Bt[t_list] = ABhat[:, :, self.dim_x:]

            if self.decouple_AB:
                At[t_list, self.indices_u_into_x, :] = 0.0
                At[t_list, :, self.indices_u_into_x] = 0.0
                Bt[t_list, self.indices_u_into_x, :] = 0.0

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
        if self.use_workers:
            At, Bt, ct = self.get_TV_matrices_batch(x_trj, u_trj)
        else:
            At, Bt, ct = self.get_TV_matrices(x_trj, u_trj) 

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
        if self.x_bounds_abs is not None:
            # x_bounds_abs are used to establish a trust region around a current
            # trajectory.
            x_bounds_abs = np.zeros((2, self.T + 1, self.dim_x))
            x_bounds_abs[0] = x_trj + self.x_bounds_abs[0]
            x_bounds_abs[1] = x_trj + self.x_bounds_abs[1]
        if self.u_bounds_abs is not None:
            # u_bounds_abs are used to establish a trust region around a current
            # trajectory.
            u_bounds_abs = np.zeros((2, self.T, self.dim_u))
            u_bounds_abs[0] = x_trj[:-1, self.indices_u_into_x] + self.u_bounds_abs[0]
            u_bounds_abs[1] = x_trj[:-1, self.indices_u_into_x] + self.u_bounds_abs[1]
        if self.x_bounds_rel is not None:
            # this should be rarely used.
            x_bounds_rel = np.zeros((2, self.T, self.dim_x))
            x_bounds_rel[0] = self.x_bounds_rel[0]
            x_bounds_rel[1] = self.x_bounds_rel[1]
        if self.u_bounds_rel is not None:
            # u_bounds_rel are used to impose input constraints.
            u_bounds_rel = np.zeros((2, self.T, self.dim_u))
            u_bounds_rel[0] = self.u_bounds_rel[0]
            u_bounds_rel[1] = self.u_bounds_rel[1]


        for t in range(self.T):
            x_star, u_star = solve_tvlqr(
                At[t:self.T],
                Bt[t:self.T],
                ct[t:self.T], self.Q, self.Qd,
                self.R, x_trj_new[t, :],
                self.x_trj_d[t:],
                solver = self.solver,
                indices_u_into_x=self.indices_u_into_x,
                x_bound_abs = x_bounds_abs[:, t:, :] if (
                    self.x_bounds_abs is not None) else None,
                u_bound_abs = u_bounds_abs[:, t:, :] if (
                    self.u_bounds_abs is not None) else None,                
                x_bound_rel = x_bounds_rel[:, t:, :] if (
                    self.x_bounds_rel is not None) else None,
                u_bound_rel = u_bounds_rel[:, t:, :] if (
                    self.u_bounds_rel is not None) else None,
                xinit=None,
                uinit=None)
            u_trj_new[t, :] = u_star[0]
            x_trj_new[t + 1, :] = self.q_dynamics.dynamics(
                x_trj_new[t], u_trj_new[t])

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
