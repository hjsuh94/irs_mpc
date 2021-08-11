import numpy as np
import pydrake.symbolic as ps
import torch
import time

from algorithm.dynamical_system import DynamicalSystem

class BicycleDynamics(DynamicalSystem):
    def __init__(self, h):
        super().__init__()
        """
        x = [x pos, y pos, heading, speed, steering_angle]
        u = [acceleration, steering_velocity]
        """
        self.h = h
        self.dim_x = 5
        self.dim_u = 2

        """Jacobian computations"""
        self.x_sym = np.array([ps.Variable("x_{}".format(i)) for i in range(self.dim_x)])
        self.u_sym = np.array([ps.Variable("u_{}".format(i)) for i in range(self.dim_u)])
        self.f_sym = self.dynamics_sym(self.x_sym, self.u_sym)

        self.jacobian_xu_sym = ps.Jacobian(self.f_sym, np.hstack((self.x_sym, self.u_sym)))
        
    def dynamics_sym(self, x, u):
        """
        Symbolic expression for dynamics. Used to compute
        linearizations of the system.
        x (np.array, dim: n): state
        u (np.array, dim: m): action        
        """
        heading = x[2]
        v = x[3]
        steer = x[4]
        dxdt = np.array([
            v * ps.cos(heading),
            v * ps.sin(heading),
            v * ps.tan(steer),
            u[0],
            u[1]
        ])
        x_new = x + self.h * dxdt
        return x_new


    def dynamics(self, x, u):
        """
        Numeric expression for dynamics.
        x (np.array, dim: n): state
        u (np.array, dim: m): action
        """
        heading = x[2]
        v = x[3]
        steer = x[4] 
        dxdt = np.array([
            v * np.cos(heading),
            v * np.sin(heading),
            v * np.tan(steer),
            u[0],
            u[1]
        ])
        x_new = x + self.h * dxdt
        return x_new

    def dynamics_batch(self, x, u):
        """
        Batch dynamics. Uses pytorch for 
        -args:
            x (np.array, dim: B x n): batched state
            u (np.array, dim: B x m): batched input
        -returns:
            xnext (np.array, dim: B x n): batched next state
        """
        heading = x[:,2]
        v = x[:,3]
        steer = x[:,4]
        dxdt = np.vstack((
            v * np.cos(heading),
            v * np.sin(heading),
            v * np.tan(steer),
            u[:,0],
            u[:,1]
        )).transpose()
        x_new = x + self.h * dxdt
        return x_new


    def dynamics_batch_torch(self, x, u):
        """
        Batch dynamics. Uses pytorch for 
        -args:
            x (np.array, dim: B x n): batched state
            u (np.array, dim: B x m): batched input
        -returns:
            xnext (np.array, dim: B x n): batched next state
        """
        x = torch.Tensor(x).cuda()
        u = torch.Tensor(u).cuda()

        heading = x[:,2]
        v = x[:,3]
        steer = x[:,4]

        dxdt = torch.vstack((
            v * torch.cos(heading),
            v * torch.sin(heading),
            v * torch.tan(steer),
            u[:,0],
            u[:,1]
        )).T
        x_new = x + self.h * dxdt
        return x_new

    def jacobian_xu(self, x, u):
        """
        Recoever linearized dynamics dfdx as a function of x, u
        """
        env = {self.x_sym[i]: x[i] for i in range(self.dim_x)}
        env.update({self.u_sym[i]: u[i] for i in range(self.dim_u)})
        f_x = ps.Evaluate(self.jacobian_xu_sym, env)
        return f_x 

    def jacobian_xu_batch(self, x, u):
        """
        Recoever linearized dynamics dfd(xu) as a function of x, u
        """ 
        dxdu_batch = np.zeros((
            x.shape[0], x.shape[1], x.shape[1] + u.shape[1]))
        for i in range(x.shape[0]):
            dxdu_batch[i] = self.jacobian_xu(x[i], u[i])
        return dxdu_batch        
