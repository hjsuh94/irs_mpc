import numpy as np
import matplotlib.pyplot as plt

exact = np.loadtxt(
    "examples/box_pushing/analysis/box_pushing_exact.csv",
    delimiter=",")
first_order = np.loadtxt(
    "examples/box_pushing/analysis/box_pushing_first_order.csv",
    delimiter=",")
zero_order = np.loadtxt(
    "examples/box_pushing/analysis/box_pushing_zero_order_AB.csv",
    delimiter=",")

plt.figure()
plt.plot(exact[0:-2], marker='x', color='red', label='exact')
plt.plot(first_order[0:-2], marker='v', color='springgreen', label='first order')
plt.plot(zero_order[0:-2], marker='^', color='blue', label='zero order')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('Cost')
plt.title("Planar Pushing")
plt.grid()
plt.show()
