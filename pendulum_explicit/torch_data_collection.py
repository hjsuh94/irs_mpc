from pendulum_explicit.pendulum_dynamics import PendulumDynamicsExplicit
import numpy as np
import pydrake.symbolic as ps
import torch.nn as nn
import torch.optim as optim
import torch
import time

import matplotlib.pyplot as plt
from matplotlib import cm

from sqp_exact_explicit import SqpExactExplicit
from sqp_ls_explicit import SqpLsExplicit

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
pendulum = PendulumDynamicsExplicit(0.05)
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

num_iter = 1500
for iter in range(num_iter):
    optimizer.zero_grad()
    output = dynamics_net(torch.Tensor(xu))
    loss = criterion(output, torch.Tensor(xtarget))
    loss.backward()
    print(loss)
    optimizer.step()
    scheduler.step()

"""4. Wrap up functions to pass to SQP"""
dynamics_net.eval()

def dynamics_nn(x, u):
    xu = torch.Tensor(np.concatenate((x,u))).unsqueeze(0)
    xnext = dynamics_net(xu)
    return xnext.detach().numpy()

def dynamics_batch_nn(x, u):
    xu = torch.Tensor(np.hstack((x,u)))
    xnext = dynamics_net(xu)
    return xnext.detach().numpy()    

def jacobian_x_nn(x, u):
    xu = torch.Tensor(np.concatenate((x,u))).unsqueeze(0)
    xu.requires_grad = True
    xnext = dynamics_net(xu)
    dJdxu0 = torch.autograd.grad(xnext[0,0], xu, retain_graph=True)[0].numpy()
    dJdxu1 = torch.autograd.grad(xnext[0,1], xu)[0].numpy()
    dJdxu = np.vstack((dJdxu0, dJdxu1))
    return dJdxu[0:2, 0:2]

def jacobian_u_nn(x, u):
    xu = torch.Tensor(np.concatenate((x,u))).unsqueeze(0)
    xu.requires_grad = True
    xnext = dynamics_net(xu)
    dJdxu0 = torch.autograd.grad(xnext[0,0], xu, retain_graph=True)[0].numpy()
    dJdxu1 = torch.autograd.grad(xnext[0,1], xu)[0].numpy()
    dJdxu = np.vstack((dJdxu0, dJdxu1))
    return dJdxu[0:2, 2, None]

"""5. Test SQP."""
timesteps = 200
Q = np.diag([5, 5])
R = 0.1 * np.diag([1])
x0 = np.array([0, 0])
xd = np.array([np.pi, 0])
xdt = np.tile(xd, (timesteps+1,1))
xbound = [
    -np.array([4.0 * np.pi, 20.0]),
     np.array([4.0 * np.pi, 20.0])
]
ubound = np.array([
    -np.array([20.0]),
     np.array([20.0])
])

# 3. Set up initial guess.
u_trj = np.tile(np.array([0.1]), (timesteps,1))
x_initial_var = np.array([10.0, 40.0])
u_initial_var = np.array([40.0])
num_samples = 10000

# 4. Solve.
try:
    sqp_exact = SqpExactExplicit(
        dynamics_nn,
        jacobian_x_nn,
        jacobian_u_nn,
        Q, R, x0, xdt, u_trj,
        xbound, ubound)

    time_now = time.time()
    sqp_exact.iterate(1e-6, 30)
    print("Final cost: " + str(sqp_exact.cost))
    print("Elapsed time: " + str(time.time() - time_now))

    plt.figure()
    plt.axis('equal')
    colormap = cm.get_cmap("jet")
    num_iters = len(sqp_exact.x_trj_lst)
    print(num_iters)
    for i in range(num_iters):
        x_trj = sqp_exact.x_trj_lst[i]
        jm = colormap(i/ num_iters)
        plt.plot(x_trj[:,0], x_trj[:,1], color=(jm[0], jm[1], jm[2], i / num_iters))        

    np.save("pendulum_explicit/results/x_trj.npy", np.array(sqp_exact.x_trj_lst))
    np.save("pendulum_explicit/results/u_trj.npy", np.array(sqp_exact.u_trj_lst))

    plt.xlabel("theta")
    plt.ylabel("theta_dot")
    plt.show()

except ValueError:
    print("SQP_EXACT Failed to find a solution.")


"""
try:
    sqp_exact = SQP_LS_Explicit(
        dynamics_nn,
        dynamics_batch_nn,
        x_initial_var,
        u_initial_var,
        num_samples,
        Q, R, x0, xdt, u_trj,
        xbound, ubound)

    time_now = time.time()
    sqp_exact.iterate(1e-6, 30)
    print("Final cost: " + str(sqp_exact.cost))
    print("Elapsed time: " + str(time.time() - time_now))

    plt.figure()
    plt.axis('equal')
    colormap = cm.get_cmap("jet")
    num_iters = len(sqp_exact.x_trj_lst)
    print(num_iters)
    for i in range(num_iters):
        x_trj = sqp_exact.x_trj_lst[i]
        jm = colormap(i/ num_iters)
        plt.plot(x_trj[:,0], x_trj[:,1], color=(jm[0], jm[1], jm[2], i / num_iters))

    np.save("pendulum_explicit/results/x_trj_01.npy", np.array(sqp_exact.x_trj_lst))
    np.save("pendulum_explicit/results/u_trj_01.npy", np.array(sqp_exact.u_trj_lst))

    plt.xlabel("theta")
    plt.ylabel("theta_dot")
    plt.show()

except ValueError:
    print("SQP_LS Failed to find a solution.")
"""