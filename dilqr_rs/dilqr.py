import numpy as np
import time

from dilqr_rs.tv_dlqr import TV_DLQR, get_solver

class DiLQR():
    def __init__(self, dynamics, Q, Qd, R, x0, xdt, u_trj, 
        xbound, ubound, solver="osqp"):
        """
        Base class for Direct iterative LQR.

        dynamics: function representing state-space model of the system.
                  should have the signature x_{t+1} = dynamics(x_t, u_t)
                  where x_t: np.array of dim n, u_t: np.array of dim m.
        Q (np.array, shape n x n): cost matrix for state.
        Qd (np.array, shape n x n): cost matrix for final state.
        R (np.array, shape m x m): cost matrix for input.
        x0 (np.array, shape n): initial point in state-space.
        xdt (np.array, shape (T+1) x n): desired trajectory.
        u_trj (np.array, shape T x m): initial guess of the input trajectory.
        xbound (np.array, shape 2 x n): (lb, ub) bounds on state.
        xbound (np.array, shape 2 x m): (lb, ub) bounds on input.
        solver (str): solver name to use for direct LQR.
        """

        self.dynamics = dynamics

        self.x0 = x0
        self.u_trj = u_trj # T x m
        self.Q = Q
        self.Qd = Qd
        self.R = R
        self.xdt = xdt        
        self.xbound = xbound
        self.ubound = ubound
        self.solver = get_solver(solver)

        self.timesteps = self.u_trj.shape[0] # Recover T.
        self.dim_x = self.x0.shape[0]
        self.dim_u = self.u_trj.shape[1]
        self.x_trj = self.rollout(self.x0, u_trj)
        self.cost = self.evaluate_cost(self.x_trj, self.u_trj)

        # These store iterations for plotting.
        self.x_trj_lst = [self.x_trj]
        self.u_trj_lst = [self.u_trj]
        self.cost_lst = [self.cost]

        self.start_time = time.time()

        self.iter = 1

    def rollout(self, x0, u_trj):
        """
        Given the initial state and an input trajectory, get an open-loop
        state trajectory of the system that is consistent with the dynamics
        of the system.
        - args:
            x0 (np.array, shape n): initial state.
            u_traj (np.array, shape T x m): initial input guess.
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
        NOTE(terry-suh): this function can be jitted, but we don't do it here to minimize
        dependency.
        """
        cost = 0.0
        for t in range(self.timesteps):
            et = x_trj[t,:] - self.xdt[t,:]
            cost += et.dot(self.Q).dot(et)
            cost += (u_trj[t,:]).dot(self.R).dot(u_trj[t,:])
        et = x_trj[self.timesteps,:] - self.xdt[self.timesteps,:]
        cost += et.dot(self.Q).dot(et)
        return cost        

    def get_TV_matrices(self, x_trj, u_trj):
        """
        Get time varying linearized dynamics given a nominal trajectory.
        - args:
            x_trj (np.array, shape (T + 1) x n)
            u_trj (np.array, shape T x m)
        """
        raise NotImplementedError("This class is virtual.")

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
            x_star, u_star = TV_DLQR(
                At[t:self.timesteps],
                Bt[t:self.timesteps],
                ct[t:self.timesteps],
                self.Q, self.Qd, self.R,
                x_trj_new[t,:],
                self.xdt[t:self.timesteps+1],
                self.xbound, self.ubound,
                solver=self.solver)
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

            print("Iteration: {:02d} ".format(self.iter) + " || " + 
                  "Current Cost: {0:05f} ".format(cost_new) + " || " +
                  "Elapsed time: {0:05f} ".format(time.time() - self.start_time))

            self.x_trj_lst.append(x_trj_new)
            self.u_trj_lst.append(u_trj_new)
            self.cost_lst.append(cost_new)

            if (self.iter > max_iterations):
                break

            #TODO(terry-suh): this should be here, but sometimes it makes it
            # terminate too early.
            """
            if ((self.cost - cost_new) < convergence_gap):
                break
            """

            # Go over to next iteration.
            self.cost = cost_new            
            self.x_trj = x_trj_new
            self.u_trj = u_trj_new
            self.iter += 1

        return self.x_trj, self.u_trj, self.cost
