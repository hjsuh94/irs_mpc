import numpy as np
import pydrake.symbolic as ps
import torch
import time

class CarDynamicsExplicit():
    def __init__(self, dt):
        """
        x = [x pos, y pos, heading, speed, steering_angle]
        u = [acceleration, steering_velocity]
        """
        self.dt = dt
        self.dim_x = 5
        self.dim_u = 2

        """Jacobian computations"""
        self.x_sym = np.array([ps.Variable("x_{}".format(i)) for i in range(self.dim_x)])
        self.u_sym = np.array([ps.Variable("u_{}".format(i)) for i in range(self.dim_u)])
        self.f_sym = self.dynamics_sym(self.x_sym, self.u_sym)

        self.jacobian_x_sym = ps.Jacobian(self.f_sym, self.x_sym)
        self.jacobian_u_sym = ps.Jacobian(self.f_sym, self.u_sym)
        
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
        x_new = x + self.dt * dxdt
        return x_new


    def dynamics_np(self, x, u):
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
        x_new = x + self.dt * dxdt
        return x_new

    def dynamics_batch_np(self, x, u):
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
        x_new = x + self.dt * dxdt
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
        x_new = x + self.dt * dxdt
        return x_new


        raise NotImplementedError        

    def jacobian_x(self, x, u):
        """
        Recoever linearized dynamics dfdx as a function of x, u
        """
        env = {self.x_sym[i]: x[i] for i in range(self.dim_x)}
        env.update({self.u_sym[i]: u[i] for i in range(self.dim_u)})
        f_x = ps.Evaluate(self.jacobian_x_sym, env)
        return f_x 

    def jacobian_u(self, x, u):
        """
        Recoever linearized dynamics dfdu as a function of x, u
        """
        env = {self.x_sym[i]: x[i] for i in range(self.dim_x)}
        env.update({self.u_sym[i]: u[i] for i in range(self.dim_u)})
        f_u = ps.Evaluate(self.jacobian_u_sym, env)
        return f_u 


"""
dynamics = CarDynamicsExplicit(0.1)

sample_size = 10000000

time_now = time.time()
x_batch = np.zeros((sample_size, 5))
u_batch = np.zeros((sample_size, 2))
x_noise = np.random.normal(0.0, 1.0, size=(sample_size,5))
x_noise += x_batch
u_noise = np.random.normal(0.0, 1.0, size=(sample_size,2))
u_noise += u_batch

print(dynamics.dynamics_batch_np(x_noise, u_noise).shape)
print("numpy: " + str(time.time() - time_now))

time_now = time.time()
x_batch = torch.zeros((sample_size, 5))
u_batch = torch.zeros((sample_size, 2))
x_noise = torch.normal(0.0, 1.0, size=(sample_size,5))
x_noise += x_batch
u_noise = torch.normal(0.0, 1.0, size=(sample_size,2))
u_noise += u_batch

print(dynamics.dynamics_batch_torch(x_noise, u_noise).shape)
print("torch: " + str(time.time() - time_now))
"""

