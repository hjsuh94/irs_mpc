import numpy as np
import matplotlib.pyplot as plt

## Easy
folder_name = "quadrotor"

result_exact = np.loadtxt(
    "examples/" + folder_name + "/analysis/" + folder_name + "_exact.csv",
    delimiter=",")
result_first = np.loadtxt(
    "examples/" + folder_name + "/analysis/" + folder_name + "_first.csv",
    delimiter=",")
result_zero = np.loadtxt(
    "examples/" + folder_name + "/analysis/" + folder_name + "_zero.csv",
    delimiter=",")

plt.figure()

plt.plot(result_exact, marker='x', color='red', label="exact")
plt.plot(result_first, marker='v', color='springgreen', label="first order")
plt.plot(result_zero, marker='^', color='blue', label="zero order")
plt.yscale('log')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('Cost (log scale)')
plt.title("Quadrotor")
plt.grid()
plt.show()
