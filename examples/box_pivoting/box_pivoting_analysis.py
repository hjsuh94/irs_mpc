import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm

ABC_storage = np.load("examples/box_pivoting/ABC_storage.npy")
x_trj_storage = np.load("examples/box_pivoting/x_trj_lst.npy")
T = x_trj_storage.shape[1]

image_folder = "examples/box_pivoting/analysis"

"""
for i in range(21):
    for j in range(40):
        imagename = "iter_{:02d}_timestep_{:03d}".format(i,j)
        plt.figure()
        plt.imshow(ABC_storage[i,j], cmap='jet')
        plt.colorbar()
        plt.savefig(os.path.join(image_folder, imagename))
        plt.close()
"""

np.set_printoptions(precision=1)
bias_matrix = np.array([
    [0,0,0,0,0,1,0,0],
    [0,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,0],
    [0,0,0,1,0,0,0,0],
    [0,0,0,0,1,0,0,0]
])

bias_matrix = np.zeros((5,8))

polygon = 0.5 * np.array([
    [1, 1, -1, -1],
    [1, -1, -1, 1],
    [1, 1, 1 ,1]
])

plt.figure()
colormap = cm.get_cmap("jet")
num_iters = x_trj_storage.shape[0]
for i in range(num_iters):
    jm = colormap(i / num_iters)

    color1_r = (1, 0, 0, 0.2 * (i+1)/num_iters)
    color2_r = (1, 0, 0, (i+1)/num_iters)        
    
    color1_g = (0, 1, 0, 0.2 * (i+1)/num_iters)
    color2_g = (0, 1, 0, (i+1)/num_iters)                

    # Plot circle.
    plt.plot(x_trj_storage[i,:,0], x_trj_storage[i,:,2], color=color2_r)
    
    for j in range(len(x_trj_storage[i,:,0])):
        if j % 2 == 0:
            circle = plt.Circle(
                (x_trj_storage[i,j,0], x_trj_storage[i,j,2]), 0.1, 
                edgecolor=color1_r, fill=False) 
            plt.gca().add_patch(circle)

            x_box = x_trj_storage[i,j,1]
            y_box = x_trj_storage[i,j,3]
            theta = x_trj_storage[i,j,4]

            transform = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])

            box_poly = transform.dot(polygon)[0:2,:] + np.array([[x_box, y_box]]).transpose()

            plt_polygon = plt.Polygon(np.transpose(box_poly), edgecolor=color1_g,
             fill=False)
            plt.gca().add_patch(plt_polygon)



    plt.plot(x_trj_storage[i,:,1], x_trj_storage[i,:,3], color=color2_g)

# Plot goal

x_trj_d = []
for t in range(T + 1):
    x_box = 1.0
    y_box = 0.5
    theta_box = -np.pi/2

    x_trj_d.append([x_box, y_box])

    transform = np.array([
        [np.cos(theta_box), -np.sin(theta_box), 0],
        [np.sin(theta_box), np.cos(theta_box), 0],
        [0, 0, 1]
    ])

    box_poly = transform.dot(polygon)[0:2,:] + np.array([[x_box, y_box]]).transpose()

    plt_polygon = plt.Polygon(np.transpose(box_poly), edgecolor=[0,0,0,0.1],fill=False)
    plt.gca().add_patch(plt_polygon)
x_trj_d = np.array(x_trj_d)

plt.plot(x_trj_d[:,0], x_trj_d[:,1], 'k-')



plt.axis('equal')
plt.show()

plt.figure()
for i in range(10):
    plt.subplot(10,3,3*i+1)
    plt.imshow(ABC_storage[i,10] - bias_matrix, cmap='plasma')
    plt.colorbar()
    plt.clim(-1, 1)
    plt.subplot(10,3,3*i+2)
    plt.imshow(ABC_storage[i,20] - bias_matrix, cmap='plasma')
    plt.colorbar() 
    plt.clim(-1, 1)       
    plt.subplot(10,3,3*i+3)
    plt.imshow(ABC_storage[i,30] - bias_matrix, cmap='plasma')
    plt.colorbar() 
    plt.clim(-1, 1)           
plt.show()

plt.figure()
plt.imshow(ABC_storage[5,2], cmap='plasma')
plt.show()
