import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
#from matplotlib import rc
#rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
#rc('text', usetex=True)

polygon = 0.1 * np.array([
    [2, 2, -2, -2],
    [1, -1, -1, 1],
    [1, 1, 1 ,1]
])

def draw_square(x, y, theta, color):
    transform = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    box_poly = transform.dot(polygon)[0:2,:] + np.array([[x, y]]).transpose()
    plt_polygon = plt.Polygon(np.transpose(box_poly), edgecolor=color, fill=False)
    plt.gca().add_patch(plt_polygon)

#%% Easy
folder_name = "bicycle"

result_exact = np.load("examples/bicycle/analysis/exact_trj.npy")
result_zero = np.load("examples/bicycle/analysis/zero_trj.npy")

plt.figure()

plt.subplot(2,1,1)
for i in range(result_exact.shape[0]):
    draw_square(result_exact[i,0], result_exact[i,1], result_exact[i,2], [0,1,0,0.5])
plt.plot(result_exact[:,0], result_exact[:,1], color=[1,0,0,0.2])
draw_square(-2.9, 1, np.pi/2, [0,0,0,1])
plt.axis('equal')

plt.subplot(2,1,2)
for i in range(result_zero.shape[0]):
    draw_square(result_zero[i,0], result_zero[i,1], result_zero[i,2], [0,1,0,0.5])
plt.plot(result_zero[:,0], result_zero[:,1], color=[1,0,0,0.2])
draw_square(-2.9, 1, np.pi/2, [0,0,0,1])
plt.axis('equal')
plt.show()

