import numpy as np
import time

from pendulum_dynamics import PendulumDynamicsExplicit
from sqp_ls_explicit import SQP_LS_Explicit

import matplotlib.pyplot as plt 
from matplotlib import cm

# 1. Load dynamics.
pendulum = PendulumDynamicsExplicit(0.05)
dynamics = pendulum.dynamics_np
dynamics_batch = pendulum.dynamics_batch_np

# 2. Set up desried trajectory and cost parameters.
timesteps = 200
Q = np.diag([5, 5])
R = np.diag([1])
x0 = np.array([0, 0])
xd = np.array([np.pi, 0])
xdt = np.tile(xd, (timesteps+1,1))
xbound = [
    -np.array([1e4, 1e4]),
     np.array([1e4, 1e4])
]
ubound = np.array([
    -np.array([1e4]),
     np.array([1e4])
])

# 3. Set up initial guess.
u_trj = np.tile(np.array([0.1]), (timesteps,1))
x_initial_var = np.array([1.0, 1.0])
u_initial_var = np.array([1.0])
num_samples = 1000


# 4. Solve.
sqp_exact = SQP_LS_Explicit(
    dynamics,
    dynamics_batch,
    x_initial_var,
    u_initial_var,
    num_samples,
    Q, R, x0, xdt, u_trj,
    xbound, ubound)

time_now = time.time()
sqp_exact.iterate(1e-6, 10)
print("Final cost: " + str(sqp_exact.cost))
print("Elapsed time: " + str(time.time() - time_now))

plt.figure()
plt.axis('equal')
colormap = cm.get_cmap("jet")
num_iters = len(sqp_exact.x_trj_lst)
for i in range(num_iters):
    x_trj = sqp_exact.x_trj_lst[i]
    jm = colormap(i/ num_iters)
    plt.plot(x_trj[:,0], x_trj[:,1], color=(jm[0], jm[1], jm[2], i / num_iters))    

plt.show()
