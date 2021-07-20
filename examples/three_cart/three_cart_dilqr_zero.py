import numpy as np
import time

from three_cart_dynamics import ThreeCartDynamics
from dilqr_rs.dilqr_rs_zero import DiLQR_RS_Zero

import matplotlib.pyplot as plt 
from matplotlib import cm

# 1. Load dynamics.
carts = ThreeCartDynamics(0.05)
dynamics = carts.dynamics_np
dynamics_batch = carts.dynamics_batch_np
projection = carts.projection

# 2. Set up desried trajectory and cost parameters.
timesteps = 100
Q = 0.01 * np.diag([50, 50, 50, 20, 100, 20])
Qd = np.diag([50, 50, 50, 20, 100, 20])
R = 0.01 * np.diag([1, 1])
x0 = np.array([0, 1, 2, 0, 0, 0])
xd = np.array([2, 3, 4, 0, 0, 0])
xdt = np.tile(xd, (timesteps+1,1))
xbound = [
    -np.array([1e4, 1e4, 1e4, 1e4, 1e4, 1e4]),
     np.array([1e4, 1e4, 1e4, 1e4, 1e4, 1e4])
]
ubound = np.array([
    -1000 * np.array([1, 1]),
     1000 * np.array([1, 1])
])

# 3. Set up initial guess.
u_trj = np.tile(np.array([0.1, -0.1]), (timesteps,1))
x_initial_var = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
u_initial_var = np.array([0.5, 0.5])
num_samples = 1000

def sampling(xbar, ubar, iter):
    dx = np.random.normal(0.0, (x_initial_var / (iter ** 0.2)),
        size = (num_samples, carts.dim_x))
    du = np.random.normal(0.0, (u_initial_var / (iter ** 0.2)),
        size = (num_samples, carts.dim_u))   
    return projection(xbar, dx, ubar, du)

# 4. Solve.
sqp_exact = DiLQR_RS_Zero(
    dynamics,
    dynamics_batch,
    sampling,
    Q, Qd, R, x0, xdt, u_trj,
    xbound, ubound)

time_now = time.time()
sqp_exact.iterate(1e-6, 20)
print("Final cost: " + str(sqp_exact.cost))
print("Elapsed time: " + str(time.time() - time_now))

plt.figure()
colormap = cm.get_cmap("jet")

purples = cm.get_cmap("Purples")
greens = cm.get_cmap("Greens")
oranges = cm.get_cmap("Oranges")

num_iters = len(sqp_exact.x_trj_lst)
for i in range(num_iters):
    x_trj = sqp_exact.x_trj_lst[i]
    jm = colormap(i/ num_iters)
    plt.plot(range(timesteps+1), x_trj[:,0], color=purples(i / num_iters))    
    plt.plot(range(timesteps+1), x_trj[:,1], color=oranges(i / num_iters))    
    plt.plot(range(timesteps+1), x_trj[:,2], color=greens(i / num_iters))
plt.show()

plt.figure()    

for i in range(num_iters):
    u_trj = sqp_exact.x_trj_lst[i]    
    plt.plot(range(timesteps+1), u_trj[:,0], color=purples(i/num_iters))
    plt.plot(range(timesteps+1), u_trj[:,2], color=greens(i/num_iters))    

plt.show()

np.save("three_cart_implicit/results/x_trj.npy", np.array(sqp_exact.x_trj_lst))
