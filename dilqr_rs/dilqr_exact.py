import numpy as np
import time

from dilqr_rs.dilqr import DiLQR

class DiLQR_Exact(DiLQR):
    def __init__(self, dynamics, jacobian_xu, 
        Q, Qd, R, x0, xdt, u_trj, xbound, ubound, solver="osqp"):
        super(DiLQR_Exact, self).__init__(dynamics, Q, Qd, R, x0, xdt, u_trj,
            xbound, ubound, solver)
        """
        Direct Iterative LQR using exact gradients.

        jacobian_x: jacobian of the dynamics. 
                  should have the signature dfdx_t = jacobian_x(x_t, u_t)
        jacobian_u: jacobian of the dynamics. 
                  should have the signature dfdu_t = jacobian_u(x_t, u_t)
        Refer to DiLQR for other arguments.
        """

        self.jacobian_xu = jacobian_xu

    def get_TV_matrices(self, x_trj, u_trj):
        """
        Get time varying linearized dynamics given a nominal trajectory.
        - args:
            x_trj (np.array, shape (T + 1) x n)
            u_trj (np.array, shape T x m)
        """
        At = np.zeros((self.timesteps, self.dim_x, self.dim_x))
        Bt = np.zeros((self.timesteps, self.dim_x, self.dim_u))
        ct = np.zeros((self.timesteps, self.dim_x))
        for t in range(self.timesteps):
            AB = self.jacobian_xu(x_trj[t], u_trj[t])
            At[t] = AB[:,0:self.dim_x]
            Bt[t] = AB[:,self.dim_x:self.dim_x+self.dim_u]
            ct[t] = self.dynamics(x_trj[t], u_trj[t]) - At[t].dot(
                x_trj[t]) - Bt[t].dot(u_trj[t])
        return At, Bt, ct
