import numpy as np
from two_spheres_quasistatic.quasistatic_dynamics import QuasistaticDynamics
from tv_lqr import solve_tvlqr, get_solver


class SqpLsQuasistatic:
    def __init__(self, q_dynamics: QuasistaticDynamics,
                 std_u_initial: np.ndarray, T: int,
                 Q: np.ndarray, R: np.ndarray,
                 x_trj_d: np.ndarray, x_bounds: np.ndarray,
                 u_bounds: np.ndarray,
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
        self.Q = Q
        self.R = R
        self.x_trj_d = x_trj_d
        self.x_bounds = x_bounds
        self.u_bounds = u_bounds

        self.x_trj = self.rollout(x0, u_trj_0)
        self.u_trj = u_trj_0  # T x m
        self.cost = self.eval_cost(self.x_trj, self.u_trj)

        # sampling standard deviation.
        self.std_u_initial = std_u_initial

        # logging
        self.x_trj_list = [self.x_trj]
        self.u_trj_list = [self.u_trj]
        self.cost_list = [self.cost]
        self.current_iter = 1

        # solver
        self.solver = get_solver('gurobi')

    def rollout(self, x0: np.ndarray, u_trj: np.ndarray):
        T = u_trj.shape[0]
        assert T == self.T
        x_trj = np.zeros((T + 1, self.dim_x))
        x_trj[0, :] = x0
        for t in range(T):
            x_trj[t + 1, :] = self.q_dynamics.dynamics(x_trj[t, :], u_trj[t, :])
        return x_trj

    def eval_cost(self, x_trj, u_trj):
        cost = 0.0
        T = u_trj.shape[0]
        assert T == self.T
        for t in range(T):
            et = x_trj[t, :] - self.x_trj_d[t, :]
            cost += et.dot(self.Q).dot(et)
            cost += (u_trj[t, :]).dot(self.R).dot(u_trj[t, :])
        et = x_trj[self.T, :] - self.x_trj_d[self.T, :]
        cost += et.dot(self.Q).dot(et)
        return cost

    def calc_B_zero_order(self, x_nominal: np.ndarray, u_nominal: np.ndarray,
                          n_samples: int, std: float):
        """
        :param std: standard deviation of the normal distribution.
        """
        du = np.random.normal(0, std, size=[n_samples, self.dim_u])
        x_next_nominal = self.q_dynamics.dynamics(x_nominal, u_nominal)
        x_next = np.zeros((n_samples, self.dim_x))

        for i in range(n_samples):
            x_next[i] = self.q_dynamics.dynamics(x_nominal, u_nominal + du[i])

        dx_next = x_next - x_next_nominal
        Bhat = np.linalg.lstsq(du, dx_next, rcond=None)[0].transpose()

        return Bhat, du

    def calc_current_std(self):
        a = self.current_iter ** 0.5
        return self.std_u_initial / a

    def calc_AB_first_order(self, x_nominal: np.ndarray, u_nominal: np.ndarray,
                           n_samples: int, std: float):
        du = np.random.normal(0, std, size=[n_samples, self.dim_u])
        Ahat_list = np.zeros((n_samples, self.dim_x, self.dim_x))
        Bhat_list = np.zeros((n_samples, self.dim_x, self.dim_u))

        for i in range(n_samples):
            self.q_dynamics.dynamics(x_nominal, u_nominal + du[i],
                                     mode='qp_cvx',
                                     requires_grad=True)
            _, _, Dq_nextDq, Dq_nextDqa_cmd = \
                self.q_dynamics.q_sim.get_dynamics_derivatives()
            Ahat_list[i] = Dq_nextDq
            Bhat_list[i] = Dq_nextDqa_cmd

        return np.mean(Ahat_list, axis=0), np.mean(Bhat_list, axis=0)

    def calc_AB_exact(self, x_nominal: np.ndarray, u_nominal: np.ndarray):
        self.q_dynamics.dynamics(x_nominal, u_nominal,
                                 mode='qp_cvx',
                                 requires_grad=True)
        _, _, Ahat, Bhat = self.q_dynamics.q_sim.get_dynamics_derivatives()
        return Ahat, Bhat

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
            Ahat, Bhat = self.calc_AB_first_order(
                x_nominal=x_trj[t],
                u_nominal=u_trj[t],
                n_samples=100,
                std=std_u)

            # Ahat, Bhat = self.calc_AB_exact(
            #     x_nominal=x_trj[t],
            #     u_nominal=u_trj[t])

            At[t] = Ahat
            Bt[t] = Bhat
            x_next_nominal = self.q_dynamics.dynamics(x_trj[t], u_trj[t])
            ct[t] = x_next_nominal - Ahat.dot(x_trj[t]) - Bhat.dot(u_trj[t])

        return At, Bt, ct

    def local_descent(self, x_trj, u_trj):
        """
        Forward pass using a TV-LQR controller on the linearized dynamics.
        - args:
            x_trj (np.array, shape (T + 1) x n): nominal state trajectory.
            u_trj (np.array, shape T x m) : nominal input trajectory
        """
        At, Bt, ct = self.get_TV_matrices(x_trj, u_trj)
        x_trj_new = np.zeros(x_trj.shape)
        x_trj_new[0, :] = x_trj[0, :]
        u_trj_new = np.zeros(u_trj.shape)

        x_star = None
        u_star = None
        for t in range(self.T):
            x_star, u_star = solve_tvlqr(
                At[t:self.T],
                Bt[t:self.T],
                ct[t:self.T], self.Q, self.Q,
                self.R, x_trj_new[t, :],
                self.x_trj_d[t:],
                self.x_bounds, self.u_bounds,
                solver=self.solver,
                xinit=None,
                uinit=None)
            u_trj_new[t, :] = u_star[0]
            x_trj_new[t + 1, :] = self.q_dynamics.dynamics(
                x_trj_new[t], u_trj_new[t])

        return x_trj_new, u_trj_new

    def iterate(self, convergence_gap, max_iterations):
        """
        Iterate local descent until convergence.
        """
        while True:
            print('Iter {},'.format(self.current_iter),
                  'cost: {}.'.format(self.cost))

            x_trj_new, u_trj_new = self.local_descent(self.x_trj, self.u_trj)
            cost_new = self.eval_cost(x_trj_new, u_trj_new)

            self.x_trj_list.append(x_trj_new)
            self.u_trj_list.append(u_trj_new)
            self.cost_list.append(cost_new)

            if self.current_iter > max_iterations:
                break

            """
            if ((self.cost - cost_new) < convergence_gap) or (iter > max_iterations):
                break
            """

            # Go over to next iteration.
            self.cost = cost_new
            self.x_trj = x_trj_new
            self.u_trj = u_trj_new
            self.current_iter += 1

        return self.x_trj, self.u_trj, self.cost