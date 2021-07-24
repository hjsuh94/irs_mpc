import numpy as np
import time

import matplotlib.pyplot as plt 
from matplotlib import cm

from pendulum_dynamics import PendulumDynamics
from irs_lqr.all import IrsLqrParameters, IrsLqrExact

# 1. Load dynamics.
pendulum = PendulumDynamics(0.05)

# 2. Set up desried trajectory and cost parameters.
timesteps = 200

params = IrsLqrParameters()
params.Q = np.diag([1., 1.])
params.Qd = np.diag([20., 20.])
params.R = np.diag([1.])
params.x0 = np.array([0., 0.])
params.xd_trj = np.tile(
    np.array([np.pi, 0.]), (timesteps+1,1))
params.xbound = [
    -np.array([1e4, 1e4]),
     np.array([1e4, 1e4])
]
params.ubound = np.array([
    -np.array([1e4]),
     np.array([1e4])
])
params.u_trj_initial = np.tile(np.array([0.1]), (timesteps,1))

# 3. Solve.
solver = IrsLqrExact(pendulum, params)

time_now = time.time()
solver.iterate(10)
print("Final cost: " + str(solver.cost))
print("Elapsed time: " + str(time.time() - time_now))

plt.figure()
plt.axis('equal')
colormap = cm.get_cmap("jet")
num_iters = len(solver.x_trj_lst)

for i in range(num_iters):
    x_trj = solver.x_trj_lst[i]
    jm = colormap(i/ num_iters)
    plt.plot(x_trj[:,0], x_trj[:,1], color=(
        jm[0], jm[1], jm[2], (i + 1) / num_iters))

plt.xlabel("theta")
plt.ylabel("theta_dot")
plt.show()
