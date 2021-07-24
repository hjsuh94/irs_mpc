import numpy as np
import matplotlib.pyplot as plt
import os

x_trj = np.load("pendulum/results/x_trj_06.npy")
image_save_dir = "pendulum/results/animation"

for iter in [6]:
    image_dir = os.path.join(image_save_dir, "{:02d}".format(iter))
    os.mkdir(image_dir)
    for t in range(x_trj.shape[1]):
        angle = x_trj[iter, t, 0]
        plt.figure(figsize=(8, 8))
        plt.plot([0, np.sin(angle)], [0, -np.cos(angle)], 'k-', linewidth=2)
        circle = plt.Circle((np.sin(angle), -np.cos(angle)), 0.1, color='springgreen')
        ax = plt.gca()
        ax.add_patch(circle)
        plt.axis('equal')
        plt.xlim([-1.6, 1.6])
        plt.ylim([-1.6, 1.6])
        plt.savefig(os.path.join(image_dir, "{:04d}".format(t)))
        plt.close()
