import numpy as np
import time

from three_cart_dynamics import ThreeCartDynamicsImplicit
from sqp_ls_implicit import SqpLsImplicit

import matplotlib.pyplot as plt 
from matplotlib import cm

# 1. Load dynamics.
carts = ThreeCartDynamicsImplicit(0.05)
dynamics = carts.dynamics_np
dynamics_batch = carts.dynamics_batch_np
projection = carts.projection

# 2. Set up desried trajectory and cost parameters.
timesteps = 100
Q = 0.001 * np.diag([50, 50, 50, 20, 100, 20])
Qd = np.diag([50, 50, 50, 20, 100, 20])
R = 0.001 * np.diag([1, 1])
x0 = np.array([0, 1, 2, 0, 0, 0])
xd = np.array([2, 3, 4, 0, 0, 0])
xdt = np.tile(xd, (timesteps+1,1))
xbound = [
    -np.array([1e4, 1e4, 1e4, 1e4, 1e4, 1e4]),
     np.array([1e4, 1e4, 1e4, 1e4, 1e4, 1e4])
]
ubound = np.array([
    -100 * np.array([1, 1]),
     100 * np.array([1, 1])
])

# 3. Set up initial guess.
u_trj = np.tile(np.array([0.1, -0.1]), (timesteps,1))
x_initial_var = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
u_initial_var = np.array([4.0, 4.0])
num_samples = 10000

# 4. Solve.
sqp_exact = SqpLsImplicit(
    dynamics,
    dynamics_batch,
    projection,
    x_initial_var,
    u_initial_var,
    num_samples,
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
