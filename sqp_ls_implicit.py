import numpy as np

from tv_lqr import TV_LQR

class SQP_LS_Implicit():
    def __init__(self, dynamics, dynamics_batch, projection,
        x_initial_var, u_initial_var, num_samples, 
        Q, Qd, R, x0, xdt, u_trj, xbound, ubound):
        """
        dynamics: function representing state-space model of the system.
                  should have the signature dynamics(x, u)
        dynamics_batch: batch function for dynamics for fast sampling.
        projection: projection function to use for the sampling process. Has the
                  following signature: projection(x, dx, u, du) where
                    - Nominal x (np.array, shape n)
                    - A batch of dx (np.array, shape B x n)
                    - Nominal u (np.array, shape m)
                    - A batch of du (np.array, shape B x m) 
                  and returns a batch of x + dx, u + du that is consistent
                  with the penetration constraints of the system.
        x_initial_var (np.array, shape n): initial variance of the dynamics.
        u_initial_var (np.array, shape m): initial variance of the input.
        num_samples (int): number of samples for each collocation point.
        Q (np.array, shape n x n): cost matrix for state.
        Qd (np.array, shape n x n): cost matrix for final state.        
        R (np.array, shape m x m): cost matrix for input.
        x0 (np.array, shape n): initial point in state-space.
        xdt (np.array, shape (T+1) x n): desired trajectory.
        u_trj (np.array, shape T x m): initial guess of the input trajectory.
        """

        self.dynamics = dynamics
        self.dynamics_batch = dynamics_batch
        self.projection = projection
        self.x_initial_var = x_initial_var
        self.u_initial_var = u_initial_var
        self.xu_initial_var = np.concatenate(
            (self.x_initial_var, self.u_initial_var)
        )
        self.num_samples = num_samples

        self.x0 = x0
        self.u_trj = u_trj # T x m
        self.Q = Q
        self.Qd = Qd
        self.R = R
        self.xdt = xdt        
        self.xbound = xbound
        self.ubound = ubound

        self.timesteps = self.u_trj.shape[0] # Recover T.
        self.dim_x = self.x0.shape[0]
        self.dim_u = self.u_trj.shape[1]
        self.x_trj = self.rollout(self.x0, u_trj)
        self.cost = self.evaluate_cost(self.x_trj, self.u_trj)

        # These store iterations for plotting.
        self.x_trj_lst = [self.x_trj]
        self.u_trj_lst = [self.u_trj]
        self.cost_lst = [self.cost]

        self.iter = 1

    def rollout(self, x0, u_trj):
        """
        Given the initial state and an input trajectory, get an open-loop
        state trajectory of the system that is consistent with the dynamics
        of the system.
        - args:
            x0 (np.arraay)
        """
        x_trj = np.zeros((self.timesteps + 1, self.dim_x))
        x_trj[0,:] = x0
        for t in range(self.timesteps):
            x_trj[t+1,:] = self.dynamics(x_trj[t,:], u_trj[t,:])
        return x_trj

    def evaluate_cost(self, x_trj, u_trj):
        """
        Evaluate cost given an state-input trajectory.
        - args:
            x_trj (np.array, shape (T + 1) x n): state trajectory to evaluate cost with.
            u_trj (np.array, shape T x m): state trajectory to evaluate cost with.
        """
        cost = 0.0
        for t in range(self.timesteps):
            et = x_trj[t,:] - self.xdt[t,:]
            cost += et.dot(self.Q).dot(et)
            cost += (u_trj[t,:]).dot(self.R).dot(u_trj[t,:])
        et = x_trj[self.timesteps,:] - self.xdt[self.timesteps,:]
        cost += et.dot(self.Q).dot(et)
        return cost        

    def generate_random_samples(self):
        """
        Genearte random samples according to iteration count.
        """
        dxdu = np.random.normal(0.0, (self.xu_initial_var / (self.iter ** 0.9)),
            size=(self.num_samples, self.dim_x + self.dim_u))
        return dxdu

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
        Get time varying linearized dynamics given a nominal trajectory.
        - args:
            x_trj (np.array, shape (T + 1) x n)
            u_trj (np.array, shape T x m)
        """
        At = np.zeros((self.timesteps, self.dim_x, self.dim_x))
        Bt = np.zeros((self.timesteps, self.dim_x, self.dim_u))
        ct = np.zeros((self.timesteps, self.dim_x))

        dxdu = self.generate_random_samples()
        dx = dxdu[:,0:self.dim_x]
        du = dxdu[:,self.dim_x:self.dim_x+self.dim_u]
        
        for t in range(self.timesteps):
            """
            Get least squares estimates here.
            """
            x_proj, u_proj = self.projection(x_trj[t], dx, u_trj[t], du)
            fdt = self.dynamics_batch(x_proj, u_proj)
            ft = self.dynamics(x_trj[t], u_trj[t])

            deltaf = fdt - ft

            Ahat, Bhat = self.compute_least_squares(dxdu, deltaf)

            At[t] = Ahat
            Bt[t] = Bhat
            ct[t] = self.dynamics(x_trj[t], u_trj[t]) - At[t].dot(
                x_trj[t]) - Bt[t].dot(u_trj[t])
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
        x_trj_new[0,:] = x_trj[0,:]
        u_trj_new = np.zeros(u_trj.shape)

        for t in range(self.timesteps):
            x_star, u_star = TV_LQR(
                At[t:self.timesteps],
                Bt[t:self.timesteps],
                ct[t:self.timesteps],
                self.Q, self.Qd, self.R,
                x_trj_new[t,:],
                self.xdt[t:self.timesteps+1],
                self.xbound, self.ubound)
            u_trj_new[t,:] = u_star[0]
            x_trj_new[t+1,:] = self.dynamics(x_trj_new[t,:], u_trj_new[t,:])

        return x_trj_new, u_trj_new

    def iterate(self, convergence_gap, max_iterations):
        """
        Iterate local descent until convergence.
        """
        while(True):
            x_trj_new, u_trj_new = self.local_descent(self.x_trj, self.u_trj)
            cost_new = self.evaluate_cost(x_trj_new, u_trj_new)

            self.x_trj_lst.append(x_trj_new)
            self.u_trj_lst.append(u_trj_new)
            self.cost_lst.append(cost_new)

            if (self.iter > max_iterations):
                break

            """
            if ((self.cost - cost_new) < convergence_gap) or (iter > max_iterations):
                break
            """

            # Go over to next iteration.
            self.cost = cost_new            
            self.x_trj = x_trj_new
            self.u_trj = u_trj_new
            self.iter += 1

        return self.x_trj, self.u_trj, self.cost
