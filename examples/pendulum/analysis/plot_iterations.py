import numpy as np
import matplotlib.pyplot as plt

result_exact = np.load("examples/pendulum/analysis/exact_cost.npy")
result_first = np.load("examples/pendulum/analysis/first_order_cost.npy")
result_zero = np.load("examples/pendulum/analysis/zero_order_cost.npy")

plt.figure()

plt.plot(result_exact, marker='x',color='red', label="exact")
plt.plot(result_first, marker='v', color='springgreen', label="first order")
plt.plot(result_zero, marker='^', color='blue', label="zero order")
plt.legend()
plt.xlabel('iterations')
plt.ylabel('Cost')
plt.title('Pendulum')
plt.grid()
plt.show()

