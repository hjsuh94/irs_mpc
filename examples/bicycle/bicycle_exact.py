from algorithm.irs_lqr import IrsLqr
import numpy as np
import time

from bicycle_dynamics import BicycleDynamics
from algorithm.all import IrsLqrParameters, IrsLqrExact

import matplotlib.pyplot as plt 
from matplotlib import cm

# 1. Load dynamics.
bicycle = BicycleDynamics(0.1)

# 2. Set up desried trajectory and cost parameters.
timesteps = 100

params = IrsLqrParameters()
params.Q = np.diag([5, 5, 3, 0.1, 0.1])
params.Qd = np.diag([50, 50, 30, 1, 1])
params.R = np.diag([1, 0.1])
params.x0 = np.array([0, 0, 0, 0, 0])
xd = np.array([3.0, 1.0, np.pi/2, 0, 0])
params.xd_trj = np.tile(xd, (timesteps+1,1))
params.xbound = [
    -np.array([1e4, 1e4, 1e4, 1e4, np.pi/4]),
     np.array([1e4, 1e4, 1e4, 1e4, np.pi/4])
]
params.ubound = np.array([
    -np.array([1e4, 1e4]),
     np.array([1e4, 1e4])
])
params.u_trj_initial = np.tile(np.array([0.1, 0.0]), (timesteps,1))

# 3. Solve.
solver = IrsLqrExact(bicycle, params)

time_now = time.time()
solver.iterate(20)
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
