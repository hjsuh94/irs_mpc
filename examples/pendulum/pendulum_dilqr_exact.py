import numpy as np
import time

from pendulum_dynamics import PendulumDynamics
from dilqr_rs.dilqr_exact import DiLQR_Exact

import matplotlib.pyplot as plt 
from matplotlib import cm

# 1. Load dynamics.
pendulum = PendulumDynamics(0.05)
dynamics = pendulum.dynamics_np
jacobian_xu = pendulum.jacobian_xu

# 2. Set up desried trajectory and cost parameters.
timesteps = 200
Q = np.diag([1., 1.])
Qd = np.diag([20., 20.])
R = np.diag([1.])
x0 = np.array([0., 0.])
xd = np.array([np.pi, 0.])
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

# 4. Solve.
sqp_exact = DiLQR_Exact(
    dynamics,
    jacobian_xu,
    Q, Qd, R, x0, xdt, u_trj,
    xbound, ubound, solver="osqp")

time_now = time.time()
sqp_exact.iterate(1e-6, 10)
print("Final cost: " + str(sqp_exact.cost))
print("Elapsed time: " + str(time.time() - time_now))

plt.figure()
plt.axis('equal')
colormap = cm.get_cmap("jet")
num_iters = len(sqp_exact.x_trj_lst)
print(num_iters)
for i in range(num_iters):
    x_trj = sqp_exact.x_trj_lst[i]
    jm = colormap(i/ num_iters)
    plt.plot(x_trj[:,0], x_trj[:,1], color=(jm[0], jm[1], jm[2], i / num_iters))
# np.save("pendulum_explicit/results/x_trj.npy", np.array(sqp_exact.x_trj_lst))
# np.save("pendulum_explicit/results/u_trj.npy", np.array(sqp_exact.u_trj_lst))

plt.xlabel("theta")
plt.ylabel("theta_dot")
plt.show()

"""
plt.figure()
for i in range(num_iters):
    u_trj = sqp_exact.u_trj_lst[i]
    plt.plot(u_trj, color=colormap(i / num_iters))
plt.show()
"""
