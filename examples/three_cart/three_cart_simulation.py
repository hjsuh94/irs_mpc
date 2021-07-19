from three_cart_implicit.three_cart_dynamics import ThreeCartDynamicsImplicit
import numpy as np
import time, os
import matplotlib.pyplot as plt

carts = ThreeCartDynamicsImplicit(0.05)
dynamics = carts.dynamics_np
dynamics_batch = carts.dynamics_batch_np
projection = carts.projection

timesteps = 100
x0 = np.array([0, 2, 3, 0, 0, 0])
u_trj = np.tile(np.array([0.8, -0.3]), (timesteps,1))
x_trj = np.zeros((timesteps+1, 6))

x_trj[0,:] = x0

"""Test correction of dynamics."""
for i in range(timesteps):
    x_trj[i+1,:] = dynamics(x_trj[i,:], u_trj[i,:])


plt.figure()
plt.plot(range(timesteps+1), x_trj[:,0])
plt.plot(range(timesteps+1), x_trj[:,1])
plt.plot(range(timesteps+1), x_trj[:,2])
plt.show()

"""Test correctness of projection."""
x = np.array([0, 2, 4, 0, 0, 0])
dx = np.random.normal(0, 1.0, size=(1000, 6))

u = np.array([0, 0])
du = np.random.normal(0, 1.0, size=(1000, 2))

before_proj = x + dx
after_proj = projection(x, dx, u, du)[0]

fig = plt.figure()
plt.axis('equal')
ax = plt.axes(projection='3d')
ax.scatter3D(before_proj[:,0], before_proj[:,1], before_proj[:,2])
ax.scatter3D(after_proj[:,0], after_proj[:,1], after_proj[:,2])
ax.set_xlabel('q1')
ax.set_ylabel('q2')
ax.set_zlabel('q3')
plt.show()

"""Test correction of batch dynamics."""

batch_size = 1000
x0 = np.array([0, 2, 3, 0, 0, 0]) + np.random.rand(batch_size, 6)
u_trj = np.tile(np.array([0.8, -0.3]), (batch_size, timesteps,1))
print(u_trj.shape)
x_trj = np.zeros((batch_size, timesteps+1, 6))

x_trj[:,0,:] = x0

"""Test correction of dynamics."""
for i in range(timesteps):
    x_trj[:,i+1,:] = dynamics_batch(x_trj[:,i,:], u_trj[:,i,:])

test_dir = "three_cart_implicit/test" 
#os.mkdir(test_dir)
for b in range(batch_size):
    plt.figure()
    plt.plot(range(timesteps+1), x_trj[b,:,0])
    plt.plot(range(timesteps+1), x_trj[b,:,1])    
    plt.plot(range(timesteps+1), x_trj[b,:,2])
    plt.savefig(os.path.join(test_dir, "{:04d}.png".format(b)))
    plt.close()
