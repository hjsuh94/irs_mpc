from algorithm.irs_lqr import IrsLqr
from pendulum_dynamics import PendulumDynamics
import numpy as np
import pydrake.symbolic as ps
import torch.nn as nn
import torch.optim as optim
import torch
import time

import matplotlib.pyplot as plt
from matplotlib import cm

from algorithm.dynamical_system import DynamicalSystem
from algorithm.irs_lqr import IrsLqrParameters
from algorithm.irs_lqr_exact import IrsLqrExact
from algorithm.irs_lqr_zero_order import IrsLqrZeroOrder

"""1. Define some random ReLU NLP."""
class DynamicsNLP(nn.Module):
    def __init__(self):
        super(DynamicsNLP, self).__init__()

        self.dynamics_mlp = nn.Sequential(
            nn.Linear(3, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )

    def forward(self, x):
        return self.dynamics_mlp(x)

"""2. Collect artificial data."""
pendulum = PendulumDynamics(0.05)
dynamics = pendulum.dynamics_np
dynamics_batch = pendulum.dynamics_batch_np

num_data = 20000
xu = np.random.rand(num_data, 3)
xu[:,0] = 6 * np.pi * (xu[:,0] - 0.5)
xu[:,1] = 30.0 * (xu[:,1] - 0.5)
xu[:,2] = 30.0 * (xu[:,2] - 0.5)

xtarget = dynamics_batch(xu[:,0:2], xu[:,2,None])

"""3. Train the network."""
dynamics_net = DynamicsNLP()
dynamics_net.train()
optimizer = optim.Adam(dynamics_net.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500)
criterion = nn.MSELoss()

num_iter = 600
for iter in range(num_iter):
    optimizer.zero_grad()
    output = dynamics_net(torch.Tensor(xu))
    loss = criterion(output, torch.Tensor(xtarget))
    loss.backward()
    optimizer.step()
    scheduler.step()

"""4. Wrap up functions to pass to IrsLqr."""
dynamics_net.eval()

class PendulumNN(DynamicalSystem):
    def __init__(self):
        super().__init__()

        self.dim_x = 2 
        self.dim_u = 1

    def dynamics(x, u):
        xu = torch.Tensor(np.concatenate((x,u))).unsqueeze(0)
        xnext = dynamics_net(xu)
        return xnext.detach().numpy()

    def dynamics_batch(x, u):
        xu = torch.Tensor(np.hstack((x,u)))
        xnext = dynamics_net(xu)
        return xnext.detach().numpy()    

    def jacobian_xu(x, u):
        xu = torch.Tensor(np.concatenate((x,u))).unsqueeze(0)
        xu.requires_grad = True
        xnext = dynamics_net(xu)
        dJdxu0 = torch.autograd.grad(xnext[0,0], xu, retain_graph=True)[0].numpy()
        dJdxu1 = torch.autograd.grad(xnext[0,1], xu)[0].numpy()
        dJdxu = np.vstack((dJdxu0, dJdxu1))
        return dJdxu[0:2]

pendulum_nn = PendulumNN()

"""5. Test IrsLqr."""
timesteps = 200

params = IrsLqrParameters()

params.Q = np.diag([1, 1])
params.Qd = np.diag([20., 20.])
params.R = np.diag([1])
params.x0 = np.array([0, 0])
params.xd_trj = np.tile(np.array([np.pi, 0]), (timesteps+1,1))
params.xbound = [
    -np.array([1e4, 1e4]),
     np.array([1e4, 1e4])
]
params.ubound = np.array([
    -np.array([1e4]),
     np.array([1e4])
])
params.u_trj_initial = np.tile(np.array([0.1]), (timesteps,1))

# 3. Try exact.
try:
    solver = IrsLqrExact(pendulum_nn, params)

    time_now = time.time()
    solver.iterate(1e-6, 30)
    print("Final cost: " + str(solver.cost))
    print("Elapsed time: " + str(time.time() - time_now))

    plt.figure()
    plt.axis('equal')
    colormap = cm.get_cmap("jet")
    num_iters = len(solver.x_trj_lst)
    for i in range(num_iters):
        x_trj = solver.x_trj_lst[i]
        jm = colormap(i/ num_iters)
        plt.plot(x_trj[:,0], x_trj[:,1], color=(jm[0], jm[1], jm[2], i / num_iters))        

    plt.xlabel("theta")
    plt.ylabel("theta_dot")
    plt.show()

except ValueError:
    print("IrsLqrExact Failed to find a solution.")

try:

    num_samples = 10000
    x_initial_var = np.array([1.0, 1.0])
    u_initial_var = np.array([1.0])

    def sampling(xbar, ubar, iter):
        dx = np.random.normal(0.0, (x_initial_var / (iter ** 0.5)),
            size = (num_samples, pendulum.dim_x))
        du = np.random.normal(0.0, (u_initial_var / (iter ** 0.5)),
            size = (num_samples, pendulum.dim_u))        
        return dx, du

    solver = IrsLqrZeroOrder(pendulum_nn, params, sampling)

    time_now = time.time()
    solver.iterate(1e-6, 30)
    print("Final cost: " + str(solver.cost))
    print("Elapsed time: " + str(time.time() - time_now))

    plt.figure()
    plt.axis('equal')
    colormap = cm.get_cmap("jet")
    num_iters = len(solver.x_trj_lst)
    for i in range(num_iters):
        x_trj = solver.x_trj_lst[i]
        jm = colormap(i/ num_iters)
        plt.plot(x_trj[:,0], x_trj[:,1], color=(jm[0], jm[1], jm[2], i / num_iters))

    plt.xlabel("theta")
    plt.ylabel("theta_dot")
    plt.show()

except ValueError:
    print("IrsLqrZeroOrder Failed to find a solution.")
