from irs_lqr.irs_lqr import IrsLqr
import numpy as np
import time

from bicycle_dynamics import BicycleDynamics
from irs_lqr.all import CemParameters, CrossEntropyMethod

import matplotlib.pyplot as plt 
from matplotlib import cm

# 1. Load dynamics.
bicycle = BicycleDynamics(0.1)

# 2. Set up desried trajectory and cost parameters.
timesteps = 100

params = CemParameters()
params.Q = np.diag([5, 5, 3, 0.1, 0.1])
params.Qd = np.diag([50, 50, 30, 1, 1])
params.R = np.diag([1, 0.1])
params.x0 = np.array([0, 0, 0, 0, 0])
xd = np.array([3.0, 1.0, np.pi/2, 0, 0])
params.xd_trj = np.tile(xd, (timesteps+1,1))
params.u_trj_initial = np.tile(np.array([0.1, 0.0]), (timesteps,1))
params.initial_std = np.array([1.0, 1.0])
params.batch_size = 100
params.n_elite = 10

# 3. Solve.
solver = CrossEntropyMethod(bicycle, params)

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
        jm[0], jm[1], jm[2], (i+1)/ num_iters))    

plt.show()

np.savetxt("examples/bicycle/analysis/bicycle_easy_cem.csv", solver.cost_lst,
    delimiter=",")
