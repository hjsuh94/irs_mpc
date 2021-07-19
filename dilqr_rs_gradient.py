import numpy as np

from dilqr import DiLQR

class DiLQR_RS_Gradient(DiLQR):
    def __init__(self, dynamics, jacobian_xu_batch, sampling,
        Q, Qd, R, x0, xdt, u_trj, xbound, ubound, solver="osqp"):
        super(DiLQR_RS_Gradient, self).__init__(dynamics, Q, Qd, R, x0, xdt, u_trj,
            xbound, ubound, solver)
        """
        Direct iterative LQR using Randomized Smoothing.
        This variant samples gradients directly.

        jacobian_xu_batch: batch function for jacobian_xu for fast sampling. If you
                  don't have a parallelized implementation, then you can do a 
                  simple for loop over the jacobian. Signature is 
                  dxdu_batch = jacobian_xu_batch(x_batch, u_batch)
                  where x_batch is np.array of B x n, 
                        u_batch is np.array of B x m,
                        dxdu_batch is np.array of B x n x (n + m).
        sampling: sampling function to use for gradient averaging.
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

        self.jacobian_xu_batch = jacobian_xu_batch
        self.sampling = sampling

    def get_TV_matrices(self, x_trj, u_trj):
        """
        Get time varying dynamics given a nominal trajectory.
        - args:
            x_trj (np.array, shape (T + 1) x n)
            u_trj (np.array, shape T x m)
        """
        At = np.zeros((self.timesteps, self.dim_x, self.dim_x))
        Bt = np.zeros((self.timesteps, self.dim_x, self.dim_u))
        ct = np.zeros((self.timesteps, self.dim_x))
        
        for t in range(self.timesteps):
            """
            Average gradients here.
            """
            # Sampling process outputs dx of B x n, du of B x m
            dx, du = self.sampling(x_trj[t], u_trj[t], self.iter)
            # AB_batch has shape B x n x (n + m).
            AB_batch = self.jacobian_xu_batch(x_trj[t] + dx, u_trj[t] + du)
            # Average across batch dim to get estimates.
            ABhat = np.mean(AB_batch, axis=0)

            At[t] = ABhat[:,0:self.dim_x]
            Bt[t] = ABhat[:,self.dim_x:self.dim_x+self.dim_u]
            ct[t] = self.dynamics(x_trj[t], u_trj[t]) - At[t].dot(
                x_trj[t]) - Bt[t].dot(u_trj[t])
        return At, Bt, ct
