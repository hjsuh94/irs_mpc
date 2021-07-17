import numpy as np
import time

from car_dynamics import CarDynamicsExplicit
from sqp_exact_explicit import SqpExactExplicit

import matplotlib.pyplot as plt 
from matplotlib import cm

# 1. Load dynamics.
car = CarDynamicsExplicit(0.1)
dynamics = car.dynamics_np
jacobian_x = car.jacobian_x
jacobian_u = car.jacobian_u

# 2. Set up desried trajectory and cost parameters.
timesteps = 100
Q = np.diag([5, 5, 3, 0.1, 0.1])
R = np.diag([1, 0.1])
x0 = np.array([0, 0, 0, 0, 0])
xd = np.array([-3.0, -1.0, -np.pi/2, 0, 0])
xdt = np.tile(xd, (timesteps+1,1))
xbound = [
    -np.array([1e4, 1e4, 1e4, 1e4, np.pi/4]),
     np.array([1e4, 1e4, 1e4, 1e4, np.pi/4])
]
ubound = np.array([
    -np.array([1e4, 1e4]),
     np.array([1e4, 1e4])
])

# 3. Set up initial guess.
u_trj = np.tile(np.array([0.1, 0.0]), (timesteps,1))

# 4. Solve.
sqp_exact = SqpExactExplicit(
    dynamics,
    jacobian_x,
    jacobian_u,
    Q, R, x0, xdt, u_trj,
    xbound, ubound)

time_now = time.time()
sqp_exact.iterate(1e-6, 5)
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
