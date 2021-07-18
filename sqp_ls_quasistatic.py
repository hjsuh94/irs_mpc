import numpy as np

from tv_lqr import solve_tvlqr, get_solver


class SqpLsQuasistatic:
    def __init__(self, dynamics, dim_x: int, dim_u: int):
        """

        Arguments are similar to those of SqpLsImplicit.
        Only samples u to estimate B.
        A uses the first derivative of the dynamics at x.
        """
        self.dynamics = dynamics
        self.dim_x = dim_x
        self.dim_u = dim_u

    def rollout(self, x0: np.ndarray, u_trj: np.ndarray):
        T = u_trj.shape[0]
        x_trj = np.zeros((T + 1, self.dim_x))
        x_trj[0, :] = x0
        for t in range(T):
            x_trj[t + 1, :] = self.dynamics(x_trj[t, :], u_trj[t, :])
        return x_trj

    def eval_cost(self, x_trj, u_trj):
        cost = 0.0
        T = u_trj.shape[0]
        for t in range(T):
            et = x_trj[t, :] - self.xdt[t, :]
            cost += et.dot(self.Q).dot(et)
            cost += (u_trj[t, :]).dot(self.R).dot(u_trj[t, :])
        et = x_trj[self.timesteps, :] - self.xdt[self.timesteps, :]
        cost += et.dot(self.Q).dot(et)
        return cost

    def calc_B_zero_order(self, x_nominal: np.ndarray, u_nominal: np.ndarray,
                          n_samples: int, std: float):
        """
        :param std: standard deviation of the normal distribution.
        """
        du = np.random.normal(0, std, size=[n_samples, self.dim_u])
        x_next_nominal = self.dynamics(x_nominal, u_nominal)
        x_next = np.zeros((n_samples, self.dim_x))

        for i in range(n_samples):
            x_next[i] = self.dynamics(x_nominal, u_nominal + du[i])

        dx_next = x_next - x_next_nominal
        Bhat = np.linalg.lstsq(du, dx_next, rcond=None)[0].transpose()

        return Bhat, du



