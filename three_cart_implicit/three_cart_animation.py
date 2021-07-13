import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

x_trj = np.load("three_cart_implicit/results/x_trj.npy")
image_save_dir = "three_cart_implicit/results/animation"

for iter in [20]:
    image_dir = os.path.join(image_save_dir, "{:02d}".format(iter))
    os.mkdir(image_dir)

    for t in range(x_trj.shape[1]):
        cart1 = x_trj[iter, t, 0]
        cart2 = x_trj[iter, t, 1]
        cart3 = x_trj[iter, t, 2]

        plt.figure(figsize=(8,2))
        ax = plt.gca()
        rect1 = patches.Rectangle((cart1, 0.1), 0.2, 0.2, edgecolor='k', facecolor='Purple')
        rect2 = patches.Rectangle((cart2, 0.1), 0.2, 0.2, edgecolor='k', facecolor='Orange')
        rect3 = patches.Rectangle((cart3, 0.1), 0.2, 0.2, edgecolor='k', facecolor='Green')

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)        

        target1 = patches.Rectangle((2.0, 0.1), 0.2, 0.2, edgecolor='k', facecolor='Purple', alpha=0.2)
        target2 = patches.Rectangle((3.0, 0.1), 0.2, 0.2, edgecolor='k', facecolor='Orange', alpha=0.2)
        target3 = patches.Rectangle((4.0, 0.1), 0.2, 0.2, edgecolor='k', facecolor='Green', alpha=0.2)

        ax.add_patch(target1)
        ax.add_patch(target2)
        ax.add_patch(target3)        


        plt.axis('equal')
        plt.xlim([0, 6])
        plt.ylim([-2, 2])        

        plt.savefig(os.path.join(image_dir, "{:04d}.png".format(t)))
        plt.close()
        