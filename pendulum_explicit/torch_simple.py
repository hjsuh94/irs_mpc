from pendulum_explicit.pendulum_dynamics import PendulumDynamicsExplicit
import numpy as np
import pydrake.symbolic as ps
import torch.nn as nn
import torch.optim as optim
import torch
import time

import matplotlib.pyplot as plt

"""1. Define some random ReLU NLP."""
class DynamicsNLP(nn.Module):
    def __init__(self):
        super(DynamicsNLP, self).__init__()

        self.dynamics_mlp = nn.Sequential(
            nn.Linear(1, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        return self.dynamics_mlp(x)

"""2. Collect artificial data."""
num_data = 50
x = 10.0 * (np.random.rand(num_data, 1) - 0.5)
xtarget = x ** 2.0 + 3.0 * np.sin(3.0 * x) # this is our target function.

"""3. Train the network."""
dynamics_net = DynamicsNLP()
dynamics_net.train()
optimizer = optim.Adam(dynamics_net.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000)
criterion = nn.MSELoss(reduction='sum')

num_iter = 1500
for iter in range(num_iter):
    optimizer.zero_grad()
    output = dynamics_net(torch.Tensor(x))
    loss = criterion(output, torch.Tensor(xtarget))
    loss.backward()
    print(loss)
    optimizer.step()
    scheduler.step()

"""4. Do some queries."""
dynamics_net.eval()

x_eval = np.linspace(-5, 5, 10000)

plt.figure()
plt.plot(x_eval, 
    dynamics_net(torch.unsqueeze(torch.Tensor(x_eval), 1)).detach().numpy(), 'k-')
plt.plot(x, xtarget, 'go')
plt.show()
