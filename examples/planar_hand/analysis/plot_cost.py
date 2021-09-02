import numpy as np
import matplotlib.pyplot as plt

exact = np.loadtxt(
    "examples/planar_hand/analysis/planar_hand_exact.csv",
    delimiter=",")
first_order = np.loadtxt(
    "examples/planar_hand/analysis/planar_hand_first_order.csv",
    delimiter=",")
zero_order_B = np.loadtxt(
    "examples/planar_hand/analysis/planar_hand_zero_order_B.csv",
    delimiter=",")
zero_order_AB = np.loadtxt(
    "examples/planar_hand/analysis/planar_hand_zero_order_AB.csv",
    delimiter=",")    

plt.figure()
plt.plot(exact, marker='x', color='red', label='exact')
plt.plot(first_order, marker='v', color='springgreen', label='first order')
plt.plot(zero_order_AB, marker='^', color='blue', label='zero order')
#plt.plot(zero_order_AB, marker='+', color='magenta', label='zero order_AB')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('Cost')
plt.title("Planar Hand (Move Right)")
plt.grid()
plt.show()

exact = np.loadtxt(
    "examples/planar_hand/analysis/planar_hand_spin_exact.csv",
    delimiter=",")
first_order = np.loadtxt(
    "examples/planar_hand/analysis/planar_hand_spin_first_order.csv",
    delimiter=",")
zero_order_B = np.loadtxt(
    "examples/planar_hand/analysis/planar_hand_spin_zero_order_B.csv",
    delimiter=",")
zero_order_AB = np.loadtxt(
    "examples/planar_hand/analysis/planar_hand_spin_zero_order_AB.csv",
    delimiter=",")

plt.figure()
plt.plot(exact, marker='x', color='red', label='exact')
plt.plot(first_order, marker='v', color='springgreen', label='first order')
plt.plot(zero_order_AB, marker='^', color='blue', label='zero order')
#plt.plot(zero_order_AB, marker='^', color='magenta', label='zero order_AB')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('Cost')
plt.title("Planar Hand (Spin In-Place)")
plt.grid()
plt.show()

exact = np.loadtxt(
    "examples/planar_hand/analysis/planar_hand_spin_second_exact.csv",
    delimiter=",")
first_order = np.loadtxt(
    "examples/planar_hand/analysis/planar_hand_spin_second_first.csv",
    delimiter=",")
zero_order = np.loadtxt(
    "examples/planar_hand/analysis/planar_hand_spin_second_zero.csv",
    delimiter=",")

plt.figure()
plt.plot(exact, marker='x', color='red', label='exact')
plt.plot(first_order, marker='v', color='springgreen', label='first order')
plt.plot(zero_order, marker='^', color='blue', label='zero order')
plt.legend()
plt.yscale('log')
plt.xlabel('iterations')
plt.ylabel('Cost (log scale)')
plt.title("Planar Hand (Spin In-hand, Second-Order Sim)")
plt.grid()
plt.show()