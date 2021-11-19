import numpy as np
import time

import matplotlib.pyplot as plt 
from matplotlib import cm

from pendulum_dynamics import PendulumDynamics
from irs_lqr.all import CemParameters, CrossEntropyMethod

# 1. Load dynamics.
pendulum = PendulumDynamics(0.05)

# 2. Set up desried trajectory and cost parameters.
timesteps = 200

params = CemParameters()
params.Q = np.diag([1., 1.])
params.Qd = np.diag([20., 20.])
params.R = np.diag([1.])
params.x0 = np.array([0., 0.])
params.xd_trj = np.tile(
    np.array([np.pi, 0.]), (timesteps+1,1))
params.u_trj_initial = np.tile(np.array([0.1]), (timesteps,1))
params.initial_std = np.array([1.0])
params.batch_size = 1000
params.n_elite = 10

# 3. Solve.
solver = CrossEntropyMethod(pendulum, params)

solver.local_descent(solver.x_trj, solver.u_trj)


time_now = time.time()
solver.iterate(7)
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

np.save("examples/pendulum/analysis/pendulum_cem.npy", solver.cost_lst)
