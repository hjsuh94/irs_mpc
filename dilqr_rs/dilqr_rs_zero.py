import numpy as np

from dilqr_rs.dilqr import DiLQR

class DiLQR_RS_Zero(DiLQR):
    def __init__(self, system, sampling,
        Q, Qd, R, x0, xdt, u_trj, xbound, ubound, solver_name="osqp"):
        super(DiLQR_RS_Zero, self).__init__(system, Q, Qd, R, x0, xdt, u_trj,
            xbound, ubound, solver_name)
        """
        Direct Iterative LQR using Randomized Smoothing.
        This is a zero-order variant that samples dynamics directly.

        sampling: sampling function to use for the least squares estimate.
                  We accept this as a function so that users may input their choice
                  of sampling distribution and variance stepping schemes.
                  Should have the signature: 
                  dx, du = sample(x, u, iter)
                  where x (dim n) and u (dim m) are nominal points, and 
                  dx (dim B x n) and du (dim B x u) are batches of zero-mean samples
                  around the nominal point. (Note that these are variations dx, not
                  the actual displacement x + dx). The iter parameters controls how 
                  variance is tuned down as iteration progresses.
        Refer to DiLQR for other arguments.
        """

        self.sampling = sampling

    def compute_least_squares(self, dxdu, deltaf):
        """
        Compute least squares matrices and returns AB matrices.
        dxdu (np.array N x (dimx + dimu)) 
        deltaf (np.array, N x (dim x))
        """
        ABhat = np.linalg.lstsq(dxdu, deltaf)[0].transpose()
        Ahat = ABhat[:,0:self.dim_x]
        Bhat = ABhat[:,self.dim_x:self.dim_x + self.dim_u]
        return Ahat, Bhat
    
    def get_TV_matrices(self, x_trj, u_trj):
        """
        Get time varying dynamics given a nominal trajectory.
        - args:
            x_trj (np.array, shape (T + 1) x n)
            u_trj (np.array, shape T x m)
        """
        At = np.zeros((self.T, self.dim_x, self.dim_x))
        Bt = np.zeros((self.T, self.dim_x, self.dim_u))
        ct = np.zeros((self.T, self.dim_x))
        
        for t in range(self.T):
            dx, du = self.sampling(x_trj[t], u_trj[t], self.iter)
            fdt = self.system.dynamics_batch(x_trj[t] + dx, u_trj[t] + du)
            ft = self.system.dynamics(x_trj[t], u_trj[t])

            deltaf = fdt - ft
            dxdu = np.hstack((dx, du))

            Ahat, Bhat = self.compute_least_squares(dxdu, deltaf)

            At[t] = Ahat
            Bt[t] = Bhat
            ct[t] = self.system.dynamics(x_trj[t], u_trj[t]) - At[t].dot(
                x_trj[t]) - Bt[t].dot(u_trj[t])
        return At, Bt, ct
