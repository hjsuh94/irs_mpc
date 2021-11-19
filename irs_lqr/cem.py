import time

import numpy as np
from irs_lqr.tv_lqr import solve_tvlqr, get_solver


class CemParameters:
    """
    Parmeters class for IrsLqr.

    Q (np.array, shape n x n): cost matrix for state.
    Qd (np.array, shape n x n): cost matrix for final state.
    R (np.array, shape m x m): cost matrix for input.
    x0 (np.array, shape n): initial point in state-space.
    xd_trj (np.array, shape (T+1) x n): desired trajectory.
    u_trj_initial (np.array, shape T x m): initial guess of the input.
    xbound (np.array, shape 2 x n): (lb, ub) bounds on state.
    xbound (np.array, shape 2 x m): (lb, ub) bounds on input.
    solver (str): solver name to use for direct LQR.
    """

    def __init__(self):
        self.Q = None
        self.Qd = None
        self.R = None
        self.x0 = None
        self.xd_trj = None
        self.u_trj_initial = None
        self.n_elite = None
        self.batch_size = None 
        self.elite_frac = None
        self.initial_std = None # dim u array of initial stds.

class CrossEntropyMethod:
    def __init__(self, system, params):
        """
        Base class for CrossEntropyMethod.

        system (DynamicalSystem class): dynamics class.
        parms (IrsLqrParameters class): parameters class.
        """

        self.system = system
        self.params = params
        self.check_valid_system(self.system)
        self.check_valid_params(self.params, self.system)

        self.Q = params.Q
        self.Qd = params.Qd
        self.R = params.R
        self.x0 = params.x0
        self.xd_trj = params.xd_trj
        self.u_trj = params.u_trj_initial
        self.n_elite = params.n_elite
        self.batch_size = params.batch_size
        self.elite_frac = params.elite_frac
        self.initial_std = params.initial_std

        self.T = self.u_trj.shape[0]  # horizon of the problem
        self.dim_x = self.system.dim_x
        self.dim_u = self.system.dim_u
        self.x_trj = self.rollout(self.x0, self.u_trj)
        self.cost = self.evaluate_cost(self.x_trj, self.u_trj)

        self.std_trj = np.tile(self.initial_std, (self.T,1))

        # These store iterations for plotting.
        self.x_trj_lst = [self.x_trj]
        self.u_trj_lst = [self.u_trj]
        self.cost_lst = [self.cost]

        self.start_time = time.time()

        self.iter = 1

    def check_valid_system(self, system):
        """
        Check if the system is valid. Otherwise, throw exception.
        TODO(terry-suh): we can add more error checking later.        
        """
        if system.dim_x == 0:
            raise RuntimeError(
                "System has zero states. Did you forget to set dim_x?")
        elif system.dim_u == 0:
            raise RuntimeError(
                "System has zero inputs. Did you forget to set dim_u?")
        try:
            system.dynamics(np.zeros(system.dim_x), np.zeros(system.dim_u))
        except:
            raise RuntimeError(
                "Could not evaluate dynamics. Have you implemented it?")

    def check_valid_params(self, params, system):
        """
        Check if the parameter is valid. Otherwise, throw exception.
        TODO(terry-suh): we can add more error checking later.
        """
        if params.Q.shape != (system.dim_x, system.dim_x):
            raise RuntimeError(
                "Q matrix must be diagonal with dim_x x dim_x.")
        if params.Qd.shape != (system.dim_x, system.dim_x):
            raise RuntimeError(
                "Qd matrix must be diagonal with dim_x x dim_x.")
        if params.R.shape != (system.dim_u, system.dim_u):
            raise RuntimeError(
                "R matrix must be diagonal with dim_u x dim_u.")

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
        x_trj[0, :] = x0
        for t in range(self.T):
            x_trj[t + 1, :] = self.system.dynamics(x_trj[t, :], u_trj[t, :])

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
            et = x_trj[t, :] - self.xd_trj[t, :]
            cost += et.dot(self.Q).dot(et)
            cost += (u_trj[t, :]).dot(self.R).dot(u_trj[t, :])
        et = x_trj[self.T, :] - self.xd_trj[self.T, :]
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

        # 1. Produce candidate trajectories according to u_std.
        u_trj_mean = u_trj
        u_trj_candidates = np.random.normal(u_trj_mean, self.std_trj,
            (self.batch_size, self.T, self.dim_u))
        cost_array = np.zeros(self.batch_size)

        # 2. Roll out the trajectories.
        for k in range(self.batch_size):
            u_trj_cand = u_trj_candidates[k,:,:]
            cost_array[k] = self.evaluate_cost(
                self.rollout(self.x0, u_trj_cand), u_trj_cand)

        # 3. Pick the best K trajectories.
        # NOTE(terry-suh): be careful what "best" means. 
        # In the reward setting, this is the highest. In cost, it's lowest.
        best_idx = np.argpartition(cost_array, self.n_elite)[:self.n_elite]

        best_trjs = u_trj_candidates[best_idx,:,:]

        # 4. Set mean as the new trajectory, and update std.
        u_trj_new = np.mean(best_trjs, axis=0)
        u_trj_std_new = np.std(best_trjs, axis=0)
        self.std_trj = u_trj_std_new
        x_trj_new = self.rollout(self.x0, u_trj_new)

        return x_trj_new, u_trj_new

    def iterate(self, max_iterations):
        """
        Iterate local descent until convergence.
        NOTE(terry-suh): originally, there is a convergence criteria.
        However, given the "non-local" nature of some randomized smoothing
        algorithms, setting such a criteria might cause it to terminate early.
        Thus we only provide a max iterations input.
        """
        while True:
            x_trj_new, u_trj_new = self.local_descent(self.x_trj, self.u_trj)
            cost_new = self.evaluate_cost(x_trj_new, u_trj_new)

            print("Iteration: {:02d} ".format(self.iter) + " || " +
                  "Current Cost: {0:05f} ".format(cost_new) + " || " +
                  "Elapsed time: {0:05f} ".format(
                      time.time() - self.start_time))

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
