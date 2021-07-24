import numpy as np
import time

import matplotlib.pyplot as plt 
from matplotlib import cm

from pendulum_dynamics import PendulumDynamics
from irs_lqr.irs_lqr_exact import IrsLqrExact

# 1. Load dynamics.
pendulum = PendulumDynamics(0.05)

# 2. Set up desried trajectory and cost parameters.
timesteps = 200
Q = np.diag([1., 1.])
Qd = np.diag([20., 20.])
R = np.diag([1.])
x0 = np.array([0., 0.])
xd = np.array([np.pi, 0.])
xd_trj = np.tile(xd, (timesteps+1,1))
xbound = [
    -np.array([1e4, 1e4]),
     np.array([1e4, 1e4])
]
ubound = np.array([
    -np.array([1e4]),
     np.array([1e4])
])

# 3. Set up initial guess.
u_trj_initial = np.tile(np.array([0.1]), (timesteps,1))

# 4. Solve.
solver = IrsLqrExact(
    pendulum,
    Q, Qd, R, x0, xd_trj, u_trj_initial,
    xbound, ubound, solver_name="osqp")

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
