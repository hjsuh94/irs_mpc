import numpy as np
import time

from quadrotor_dynamics import QuadrotorDynamics
from algorithm.all import IrsLqrParameters, IrsLqrZeroOrder

import matplotlib.pyplot as plt 
from matplotlib import cm
from mpl_toolkits import mplot3d

# 1. Load dynamics.
quadrotor = QuadrotorDynamics(0.05)
# 2. Set up desried trajectory and cost parameters.
timesteps = 200

params = IrsLqrParameters()
params.Q = 1.0 * np.diag([10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0])
params.Qd = 10.0 * np.diag([10, 10, 10, 10, 10, 10, 1, 1, 1, 1, 1, 1])
params.R = 1.0 * np.diag([1, 1, 1, 1])
params.x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
params.xd_trj = np.zeros((timesteps+1,12))

for i in range(timesteps + 1):
    params.xd_trj[i,:] = np.array(
        [1.5 * np.cos(0.05 * float(i)), 
         1.5 * np.sin(0.05 * float(i)), 
         0.02 * float(i), 0, 0, 0, 0, 0, 0, 0, 0, 0])

params.xbound = [
    -np.array([1e5, 1e5, 1e5, 2.0 * np.pi, np.pi/2, 2.0 * np.pi, 
        1e5, 1e5, 1e5, 1e5 ,1e5, 1e5]),
     np.array([1e5, 1e5, 1e5, 2.0 * np.pi, np.pi/2, 2.0 * np.pi,
        1e5, 1e5, 1e5, 1e5 ,1e5, 1e5])
]
params.ubound = np.array([
    -np.array([1e5, 1e5, 1e5, 1e5]),
    np.array([1e5 ,1e5 ,1e5, 1e5])
])
params.u_trj_initial = np.tile(np.array([2.0, 2.0, 2.0, 2.0]), (timesteps,1))

# 3. Set up initial guess.
x_initial_var = 0.1 * np.ones(12)
u_initial_var = 0.1 * np.ones(4)
num_samples = 1000

# Sampling function for variance stepping.
def sampling(xbar, ubar, iter):
    dx = np.random.normal(0.0, (x_initial_var / (iter ** 0.5)),
        size = (num_samples, quadrotor.dim_x))
    du = np.random.normal(0.0, (u_initial_var / (iter ** 0.5)),
        size = (num_samples, quadrotor.dim_u))
    return dx, du

# 4. Solve.
solver = IrsLqrZeroOrder(quadrotor, params, sampling)

time_now = time.time()
solver.iterate(3)
print("Final cost: " + str(solver.cost))
print("Elapsed time: " + str(time.time() - time_now))

plt.figure()
ax = plt.axes(projection='3d')
ax.set_aspect('auto')
colormap = cm.get_cmap("jet")

num_iters = len(solver.x_trj_lst)
for i in range(num_iters):
    x_trj = solver.x_trj_lst[i]
    jm = colormap(i/ num_iters)
    ax.plot3D(x_trj[:,0], x_trj[:,1], x_trj[:,2], color=(jm[0], jm[1], jm[2],
        (i + 1) / num_iters))    

ax.plot3D(params.xd_trj[:,0], params.xd_trj[:,1], params.xd_trj[:,2], color='blue')

ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-0.5, 4.0])
plt.show()
