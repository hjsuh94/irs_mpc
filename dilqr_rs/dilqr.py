import numpy as np
import time

from dilqr_rs.tv_dlqr import TV_DLQR, get_solver

class DiLQR():
    def __init__(self, system, Q, Qd, R, x0, xd_trj, u_trj, 
        xbound, ubound, solver_name="osqp"):
        """
        Base class for Direct iterative LQR.

        system (DynamicalSystem class): dynamics class.
        Q (np.array, shape n x n): cost matrix for state.
        Qd (np.array, shape n x n): cost matrix for final state.
        R (np.array, shape m x m): cost matrix for input.
        x0 (np.array, shape n): initial point in state-space.
        xd_trj (np.array, shape (T+1) x n): desired trajectory.
        u_trj (np.array, shape T x m): initial guess of the input trajectory.
        xbound (np.array, shape 2 x n): (lb, ub) bounds on state.
        xbound (np.array, shape 2 x m): (lb, ub) bounds on input.
        solver (str): solver name to use for direct LQR.
        """

        self.system = system
        self.check_valid_system(system)

        self.x0 = x0
        self.u_trj = u_trj # T x m
        self.Q = Q
        self.Qd = Qd
        self.R = R
        self.xd_trj = xd_trj
        self.xbound = xbound
        self.ubound = ubound
        self.solver = get_solver(solver_name)

        self.T = self.u_trj.shape[0] # horizon of the problem
        self.dim_x = self.system.dim_x
        self.dim_u = self.system.dim_u
        self.x_trj = self.rollout(self.x0, u_trj)
        self.cost = self.evaluate_cost(self.x_trj, self.u_trj)

        # These store iterations for plotting.
        self.x_trj_lst = [self.x_trj]
        self.u_trj_lst = [self.u_trj]
        self.cost_lst = [self.cost]

        self.start_time = time.time()

        self.iter = 1

    def check_valid_system(self, system):
        """
        Check if the system is valid. Otherwise, throw exception.
        """
        if (system.dim_x == 0):
            raise RuntimeError(
                "System has zero states. Did you forget to set dim_x?")
        elif (system.dim_u == 0):
            raise RuntimeError(
                "System has zero inputs. Did you forget to set dim_u?")
        try:
            system.dynamics(np.zeros(system.dim_x), np.zeros(system.dim_u))
        except:
            raise RuntimeError(
                "Could not evaluate dynamics. Have you implemented it?")

    def rollout(self, x0, u_trj):
        """
        Given the initial state and an input trajectory, get an open-loop
        state trajectory of the system that is consistent with the dynamics
        of the system.
        - args:
            x0 (np.array, shape n): initial state.
            u_traj (np.array, shape T x m): initial input guess.
        """
        x_trj = np.zeros((self.T + 1, self.dim_x))
        x_trj[0,:] = x0
        for t in range(self.T):
            x_trj[t+1,:] = self.system.dynamics(x_trj[t,:], u_trj[t,:])
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
        for t in range(self.T):
            et = x_trj[t,:] - self.xd_trj[t,:]
            cost += et.dot(self.Q).dot(et)
            cost += (u_trj[t,:]).dot(self.R).dot(u_trj[t,:])
        et = x_trj[self.T,:] - self.xd_trj[self.T,:]
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

        for t in range(self.T):
            x_star, u_star = TV_DLQR(
                At[t:self.T],
                Bt[t:self.T],
                ct[t:self.T],
                self.Q, self.Qd, self.R,
                x_trj_new[t,:],
                self.xd_trj[t:self.T+1],
                self.xbound, self.ubound,
                solver=self.solver)
            u_trj_new[t,:] = u_star[0]
            x_trj_new[t+1,:] = self.system.dynamics(
                x_trj_new[t,:], u_trj_new[t,:])

        return x_trj_new, u_trj_new

    def iterate(self, max_iterations):
        """
        Iterate local descent until convergence.
        NOTE(terry-suh): originally, there is a convergence criteria.
        However, given the "non-local" nature of some randomized smoothing
        algorithms, setting such a criteria might cause it to terminate early.
        Thus we only provide a max iterations input.
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

            # Go over to next iteration.
            self.cost = cost_new            
            self.x_trj = x_trj_new
            self.u_trj = u_trj_new
            self.iter += 1

        return self.x_trj, self.u_trj, self.cost

