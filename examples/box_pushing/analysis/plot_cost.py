import numpy as np
import matplotlib.pyplot as plt

exact = np.loadtxt(
    "examples/box_pushing/analysis/box_pushing_exact_new.csv",
    delimiter=",")
first_order = np.loadtxt(
    "examples/box_pushing/analysis/box_pushing_first_order_new.csv",
    delimiter=",")
zero_order = np.loadtxt(
    "examples/box_pushing/analysis/box_pushing_zero_order_new.csv",
    delimiter=",")
cem = np.loadtxt(
    "examples/box_pushing/analysis/box_pushing_cem_new.csv",
    delimiter=",")    

plt.figure()
plt.plot(exact, marker='x', color='red', label='exact')
plt.plot(first_order, marker='v', color='springgreen', label='first order')
plt.plot(zero_order, marker='^', color='blue', label='zero order')
plt.plot(cem, marker='*', color='magenta', label='cem')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('Cost')
plt.title("Planar Pushing")
plt.grid()
plt.show()
