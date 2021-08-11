import numpy as np
import time

from three_cart_dynamics import ThreeCartDynamics
from algorithm.all import IrsLqrParameters, IrsLqrZeroOrder

import matplotlib.pyplot as plt 
from matplotlib import cm

# 1. Load dynamics.
carts = ThreeCartDynamics(0.05)

# 2. Set up desried trajectory and cost parameters.
timesteps = 100

params = IrsLqrParameters()
params.Q = 0.01 * np.diag([50, 50, 50, 20, 100, 20])
params.Qd = np.diag([50, 50, 50, 20, 100, 20])
params.R = 0.01 * np.diag([1, 1])
params.x0 = np.array([0, 1, 2, 0, 0, 0])
xd = np.array([2, 3, 4, 0, 0, 0])
params.xd_trj = np.tile(xd, (timesteps+1,1))
params.xbound = [
    -np.array([1e4, 1e4, 1e4, 1e4, 1e4, 1e4]),
     np.array([1e4, 1e4, 1e4, 1e4, 1e4, 1e4])
]
params.ubound = np.array([
    -1000 * np.array([1, 1]),
     1000 * np.array([1, 1])
])

# 3. Set up initial guess.
params.u_trj_initial = np.tile(np.array([0.1, -0.1]), (timesteps,1))
x_initial_var = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
u_initial_var = np.array([0.5, 0.5])
num_samples = 1000

def sampling(xbar, ubar, iter):
    dx = np.random.normal(0.0, (x_initial_var / (iter ** 0.2)),
        size = (num_samples, carts.dim_x))
    du = np.random.normal(0.0, (u_initial_var / (iter ** 0.2)),
        size = (num_samples, carts.dim_u))   
    return carts.projection(xbar, dx, ubar, du)

# 4. Solve.
solver = IrsLqrZeroOrder(carts, params, sampling)

time_now = time.time()
solver.iterate(20)
print("Final cost: " + str(solver.cost))
print("Elapsed time: " + str(time.time() - time_now))

plt.figure()
colormap = cm.get_cmap("jet")

purples = cm.get_cmap("Purples")
greens = cm.get_cmap("Greens")
oranges = cm.get_cmap("Oranges")

num_iters = len(solver.x_trj_lst)
for i in range(num_iters):
    x_trj = solver.x_trj_lst[i]
    jm = colormap(i/ num_iters)
    plt.plot(range(timesteps+1), x_trj[:,0], color=purples(i / num_iters))    
    plt.plot(range(timesteps+1), x_trj[:,1], color=oranges(i / num_iters))    
    plt.plot(range(timesteps+1), x_trj[:,2], color=greens(i / num_iters))
plt.show()

plt.figure()    

for i in range(num_iters):
    u_trj = solver.x_trj_lst[i]    
    plt.plot(range(timesteps+1), u_trj[:,0], color=purples(i/num_iters))
    plt.plot(range(timesteps+1), u_trj[:,2], color=greens(i/num_iters))    

plt.show()
