import numpy as np
import time

from pendulum_dynamics import PendulumDynamics
from dilqr_rs_gradient import DiLQR_RS_Gradient

import matplotlib.pyplot as plt 
from matplotlib import cm

# 1. Load dynamics.
pendulum = PendulumDynamics(0.05)
dynamics = pendulum.dynamics_np
jacobian_xu = pendulum.jacobian_xu

# Batch function for Jacobian handling.
def jacobian_xu_batch(x_batch, u_batch):
    dxdu_batch = np.zeros((
        x_batch.shape[0], x_batch.shape[1],
        x_batch.shape[1] + u_batch.shape[1]))
    for i in range(x_batch.shape[0]):
        dxdu_batch[i] = jacobian_xu(x_batch[i], u_batch[i])
    return dxdu_batch

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
num_samples = 1000
x_initial_var = np.array([1.0, 1.0])
u_initial_var = np.array([1.0])

# Sampling function for variance stepping.
def sampling(xbar, ubar, iter):
    dx = np.random.normal(0.0, (x_initial_var / (iter ** 0.5)),
        size = (num_samples, pendulum.dim_x))
    du = np.random.normal(0.0, (u_initial_var / (iter ** 0.5)),
        size = (num_samples, pendulum.dim_u))        
    return dx, du

# 4. Solve.
sqp_exact = DiLQR_RS_Gradient(
    dynamics,
    jacobian_xu_batch, sampling,
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
