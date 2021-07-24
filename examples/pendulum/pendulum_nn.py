from pendulum_dynamics import PendulumDynamics
import numpy as np
import pydrake.symbolic as ps
import torch.nn as nn
import torch.optim as optim
import torch
import time

import matplotlib.pyplot as plt
from matplotlib import cm

from irs_lqr.irs_lqr_exact import IrsLqrExact
from irs_lqr.irs_lqr_zero_order import IrsLqrZeroOrder

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

def dynamics_nn(x, u):
    xu = torch.Tensor(np.concatenate((x,u))).unsqueeze(0)
    xnext = dynamics_net(xu)
    return xnext.detach().numpy()

def dynamics_batch_nn(x, u):
    xu = torch.Tensor(np.hstack((x,u)))
    xnext = dynamics_net(xu)
    return xnext.detach().numpy()    

def jacobian_xu_nn(x, u):
    xu = torch.Tensor(np.concatenate((x,u))).unsqueeze(0)
    xu.requires_grad = True
    xnext = dynamics_net(xu)
    dJdxu0 = torch.autograd.grad(xnext[0,0], xu, retain_graph=True)[0].numpy()
    dJdxu1 = torch.autograd.grad(xnext[0,1], xu)[0].numpy()
    dJdxu = np.vstack((dJdxu0, dJdxu1))
    return dJdxu[0:2]

"""5. Test IrsLqr."""
timesteps = 200
Q = np.diag([1, 1])
Qd = np.diag([20., 20.])
R = np.diag([1])
x0 = np.array([0, 0])
xd = np.array([np.pi, 0])
xd_trj = np.tile(xd, (timesteps+1,1))
xbound = [
    -np.array([1e4, 1e4]),
     np.array([1e4, 1e4])
]
ubound = np.array([
    -np.array([1e4]),
     np.array([1e4])
])

# 3. Set up initial guess.
u_trj_initial = np.tile(np.array([0.1]), (timesteps,1))
x_initial_var = np.array([1.0, 1.0])
u_initial_var = np.array([1.0])
num_samples = 10000

# 4. Solve.
try:
    solver = IrsLqrExact(
        dynamics_nn,
        jacobian_xu_nn,
        Q, Qd, R, x0, xd_trj, u_trj_initial,
        xbound, ubound)

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
    print("DiLQR_Exact Failed to find a solution.")

try:

    def sampling(xbar, ubar, iter):
        dx = np.random.normal(0.0, (x_initial_var / (iter ** 0.5)),
            size = (num_samples, pendulum.dim_x))
        du = np.random.normal(0.0, (u_initial_var / (iter ** 0.5)),
            size = (num_samples, pendulum.dim_u))        
        return dx, du

    solver = IrsLqrZeroOrder(
        dynamics_nn,
        dynamics_batch_nn,
        sampling,
        Q, Qd, R, x0, xd_trj, u_trj_initial,
        xbound, ubound)

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
    print("DiLQR_RS Failed to find a solution.")
