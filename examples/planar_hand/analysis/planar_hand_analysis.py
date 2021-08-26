import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm

ABC_storage = np.load("examples/quasistatic/ABC_storage.npy")
x_trj_storage = np.load("examples/quasistatic/x_trj_lst.npy")
T = x_trj_storage.shape[1]

image_folder = "examples/box_pushing/analysis"

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

def make_R(theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return R

polygon1 = np.array([
    0.3 * np.array([1, 1, 0, 0]),
    0.1 * np.array([0.5, -0.5, -0.5, 0.5]),
    np.array([1, 1, 1 ,1])
])

polygon2 = np.array([
    0.2 * np.array([1, 1, 0, 0]),
    0.1 * np.array([0.5, -0.5, -0.5, 0.5]),
    np.array([1, 1, 1 ,1])
])

plt.figure()
colormap = cm.get_cmap("jet")
T = x_trj_storage.shape[1]
num_iter = x_trj_storage.shape[0] - 1
num_iter = 11
print(T)
for i in range(T):
    jm = colormap(i / T)

    color1_r = (1, 0, 0, 0.5 * (i+1)/T)
    color2_r = (1, 0, 0, 0.8 * (i+1)/T)
    
    color1_g = (0, 1, 0, 0.5 * (i+1)/T)
    color2_g = (0, 1, 0, 0.8 * (i+1)/T)

    # Plot circle.
    x_box = x_trj_storage[num_iter,i,0]
    y_box = x_trj_storage[num_iter,i,3]
    theta_box = x_trj_storage[num_iter,i,6] 
    q1_l = x_trj_storage[num_iter,i,1] + np.pi
    q1_r = x_trj_storage[num_iter,i,2]
    q2_l = x_trj_storage[num_iter,i,4] 
    q2_r = x_trj_storage[num_iter,i,5] 

    circle_l0 = np.array([-0.1, 0.0])
    circle_l1 = circle_l0 + make_R(q1_l)[:2,:2].dot(np.array([0.3, 0.0]))
    circle_l2 = circle_l1 + make_R(q1_l + q2_l)[:2,:2].dot(np.array([0.2, 0.0]))

    circle_r0 = np.array([0.1, 0.0])
    circle_r1 = circle_r0 + make_R(q1_r)[:2,:2].dot(np.array([0.3, 0.0]))
    circle_r2 = circle_r1 + make_R(q1_r + q2_r)[:2,:2].dot(np.array([0.2, 0.0]))    

    circle_main = np.array([x_box, y_box])

    plt.gca().add_patch(plt.Circle(circle_l0, 0.05, edgecolor=color2_r, fill=False))
    plt.gca().add_patch(plt.Circle(circle_l1, 0.05, edgecolor=color2_r, fill=False))
    plt.gca().add_patch(plt.Circle(circle_l2, 0.05, edgecolor=color2_r, fill=False))

    body_1 = make_R(q1_l).dot(polygon1)[0:2] + np.array([circle_l0]).transpose()
    plt.gca().add_patch(plt.Polygon(
        np.transpose(body_1), edgecolor=color2_r, fill=False))
    body_2 = make_R(q1_l + q2_l).dot(polygon2)[0:2] + np.array([circle_l1]).transpose()
    plt.gca().add_patch(plt.Polygon(
        np.transpose(body_2), edgecolor=color2_r, fill=False))        

    plt.gca().add_patch(plt.Circle(circle_r0, 0.05, edgecolor=color2_r, fill=False))
    plt.gca().add_patch(plt.Circle(circle_r1, 0.05, edgecolor=color2_r, fill=False))
    plt.gca().add_patch(plt.Circle(circle_r2, 0.05, edgecolor=color2_r, fill=False))

    body_3 = make_R(q1_r).dot(polygon1)[0:2] + np.array([circle_r0]).transpose()
    plt.gca().add_patch(plt.Polygon(
        np.transpose(body_3), edgecolor=color2_r, fill=False))
    body_4 = make_R(q1_r + q2_r).dot(polygon2)[0:2] + np.array([circle_r1]).transpose()
    plt.gca().add_patch(plt.Polygon(
        np.transpose(body_4), edgecolor=color2_r, fill=False))            

    plt.gca().add_patch(plt.Circle(circle_main, 0.25, edgecolor=color2_g, fill=False))
    circle_up = circle_main + make_R(theta_box)[:2,:2].dot(np.array([0.0, 0.25]))

    plt.plot([circle_main[0], circle_up[0]], [circle_main[1], circle_up[1]], color=color1_g)
    


# Plot goal
x_box = 0.0
y_box = 0.5
theta_box = -np.pi/4

circle_main = np.array([x_box, y_box])
plt.gca().add_patch(plt.Circle((x_box, y_box), 0.25, edgecolor='k', fill=False))
circle_up = circle_main + make_R(theta_box)[:2,:2].dot(np.array([0.0, 0.25]))
plt.plot([circle_main[0], circle_up[0]], [circle_main[1], circle_up[1]], color='k')
    


plt.axis('equal')
plt.show()




"""
print(ABC_storage.shape)

plt.figure()
for i in range(10):
    plt.subplot(10,3,3*i+1)
    plt.imshow(ABC_storage[i,10] , cmap='plasma')
    plt.colorbar()
    plt.clim(-1, 1)
    plt.subplot(10,3,3*i+2)
    plt.imshow(ABC_storage[i,20] , cmap='plasma')
    plt.colorbar() 
    plt.clim(-1, 1)       
    plt.subplot(10,3,3*i+3)
    plt.imshow(ABC_storage[i,30] , cmap='plasma')
    plt.colorbar() 
    plt.clim(-1, 1)           
plt.show()

plt.figure()
plt.imshow(ABC_storage[5,2], cmap='plasma')
plt.show()
"""