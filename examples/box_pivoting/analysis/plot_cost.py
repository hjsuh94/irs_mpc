import numpy as np
import matplotlib.pyplot as plt

exact = np.loadtxt(
    "examples/box_pivoting/analysis/box_pivoting_exact2.csv",
    delimiter=",")
first_order = np.loadtxt(
    "examples/box_pivoting/analysis/box_pivoting_first.csv",
    delimiter=",")
zero_order = np.loadtxt(
    "examples/box_pivoting/analysis/box_pivoting_zero.csv",
    delimiter=",")

plt.figure()
plt.plot(exact[0:6], marker='x', color='red', label='exact')
plt.plot(first_order[0:6], marker='v', color='springgreen', label='first order')
plt.plot(zero_order[0:6], marker='^', color='blue', label='zero order')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('Cost')
plt.title("Box Pivoting")
plt.grid()
plt.show()
