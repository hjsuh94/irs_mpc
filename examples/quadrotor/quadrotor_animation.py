import numpy as np
import pydrake.symbolic as ps
import torch
import time, os

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from pydrake.all import (
    RollPitchYaw, RotationMatrix,
)

x_trj = np.load("examples/quadrotor/results/x_trj.npy")
xd_trj = np.load("examples/quadrotor/results/xd_trj.npy")

image_dir = "examples/quadrotor/results/video/"

for i in range(x_trj.shape[0]):
    state = x_trj[i,:]
    
    p_WQ = state[0:3]
    R_WQ = RotationMatrix(RollPitchYaw(state[3:6]))

    R_WQ = R_WQ.matrix()

    p_Qx_W = p_WQ + 0.5 * R_WQ.dot(np.array([1, 0, 0]))
    p_Qy_W = p_WQ + 0.5 * R_WQ.dot(np.array([0, 1, 0]))
    p_Qz_W = p_WQ + 0.5 * R_WQ.dot(np.array([0, 0, 1]))

    print(p_Qx_W)

    plt.figure(figsize = (8,8))
    ax = plt.axes(projection='3d')
    ax.set_aspect('auto')
    ax.view_init(elev=40., azim=23)

    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])
    ax.set_zlim([-0.5, 4.5])

    ax.plot3D(x_trj[:,0], x_trj[:,1], x_trj[:,2], color=[1, 0, 0, 0.5])
    ax.plot3D(xd_trj[:,0], xd_trj[:,1], xd_trj[:,2], color=[0, 0, 1, 0.5])

    plt.plot(p_WQ[0], p_WQ[1], p_WQ[2], 'ko', markersize=3)

    plt.plot([p_WQ[0], p_Qx_W[0]], [p_WQ[1], p_Qx_W[1]], [p_WQ[2], p_Qx_W[2]], 'r-')
    plt.plot([p_WQ[0], p_Qy_W[0]], [p_WQ[1], p_Qy_W[1]], [p_WQ[2], p_Qy_W[2]], 'g-')
    plt.plot([p_WQ[0], p_Qz_W[0]], [p_WQ[1], p_Qz_W[1]], [p_WQ[2], p_Qz_W[2]], 'b-')

    plt.savefig(os.path.join(image_dir, "{:03d}.png".format(i)))

    plt.close()