import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

plt.figure(figsize=(8, 4), dpi=200)

#%% Pendulum

result_exact = np.load("examples/pendulum/analysis/exact_cost.npy")
result_first = np.load("examples/pendulum/analysis/first_order_cost.npy")
result_zero = np.load("examples/pendulum/analysis/zero_order_cost.npy")

plt.subplot(4,2,1)
plt.plot(result_exact, marker='x',color='red', label="exact")
plt.plot(result_first, marker='v', color='springgreen', label="first order")
plt.plot(result_zero, marker='^', color='blue', label="zero order")
plt.legend()
plt.xlabel('iterations')
plt.ylabel('Cost')
plt.title('Pendulum')
plt.grid()
plt.show()


#%% Easy
folder_name = "bicycle"

result_exact = np.loadtxt(
    "examples/" + folder_name + "/analysis/" + folder_name + "_easy_exact.csv",
    delimiter=",")
result_first = np.loadtxt(
    "examples/" + folder_name + "/analysis/" + folder_name + "_easy_first.csv",
    delimiter=",")
result_zero = np.loadtxt(
    "examples/" + folder_name + "/analysis/" + folder_name + "_easy_zero.csv",
    delimiter=",")

plt.subplot(4,2,2)
plt.plot(result_exact, marker='x', color='red', label="exact")
plt.plot(result_first, marker='v', color='springgreen', label="first order")
plt.plot(result_zero, marker='^', color='blue', label="zero order")
plt.yscale('log')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('Cost (log scale)')
plt.title("Dubin's Car (Easy)")
plt.grid()

#%% Hard
result_exact = np.loadtxt(
    "examples/" + folder_name + "/analysis/" + folder_name + "_hard_exact.csv",
    delimiter=",")
result_first = np.loadtxt(
    "examples/" + folder_name + "/analysis/" + folder_name + "_hard_first.csv",
    delimiter=",")
result_zero = np.loadtxt(
    "examples/" + folder_name + "/analysis/" + folder_name + "_hard_zero.csv",
    delimiter=",")

plt.subplot(4,2,3)
plt.plot(result_exact, marker='x', color='red', label="exact")
plt.plot(result_first, marker='v', color='springgreen', label="first order")
plt.plot(result_zero, marker='^', color='blue', label="zero order")
plt.yscale('log')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('Cost (logscale)')
plt.title("Dubin's Car (Hard)")
plt.grid()

#%% Quadrotor

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

plt.subplot(4,2,4)
plt.plot(result_exact, marker='x', color='red', label="exact")
plt.plot(result_first, marker='v', color='springgreen', label="first order")
plt.plot(result_zero, marker='^', color='blue', label="zero order")
plt.yscale('log')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('Cost (log scale)')
plt.title("Quadrotor")
plt.grid()

#%% Planar Hand 1


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

plt.subplot(4,2,5)
plt.plot(exact, marker='x', color='red', label='exact')
plt.plot(first_order, marker='v', color='springgreen', label='first order')
plt.plot(zero_order_AB, marker='^', color='blue', label='zero order')
#plt.plot(zero_order_AB, marker='+', color='magenta', label='zero order_AB')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('Cost')
plt.title("Planar Hand (Move Right)")
plt.grid()

#%% Planar Hand 2

exact = np.loadtxt(
    "examples/planar_hand/analysis/planar_hand_spin_second_exact.csv",
    delimiter=",")
first_order = np.loadtxt(
    "examples/planar_hand/analysis/planar_hand_spin_second_first.csv",
    delimiter=",")
zero_order = np.loadtxt(
    "examples/planar_hand/analysis/planar_hand_spin_second_zero.csv",
    delimiter=",")

plt.subplot(4,2,6)
plt.plot(exact, marker='x', color='red', label='exact')
plt.plot(first_order, marker='v', color='springgreen', label='first order')
plt.plot(zero_order, marker='^', color='blue', label='zero order')
plt.legend()
plt.yscale('log')
plt.xlabel('iterations')
plt.ylabel('Cost (log scale)')
plt.title("Planar Hand (Spin In-hand, Second-Order Sim)")
plt.grid()

#%% Box Pushing

exact = np.loadtxt(
    "examples/box_pushing/analysis/box_pushing_exact.csv",
    delimiter=",")
first_order = np.loadtxt(
    "examples/box_pushing/analysis/box_pushing_first_order.csv",
    delimiter=",")
zero_order = np.loadtxt(
    "examples/box_pushing/analysis/box_pushing_zero_order_AB.csv",
    delimiter=",")

plt.subplot(4,2,7)
plt.plot(exact[0:-2], marker='x', color='red', label='exact')
plt.plot(first_order[0:-2], marker='v', color='springgreen', label='first order')
plt.plot(zero_order[0:-2], marker='^', color='blue', label='zero order')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('Cost')
plt.title("Planar Pushing")
plt.grid()

#%% Box Pivoting

exact = np.loadtxt(
    "examples/box_pivoting/analysis/box_pivoting_exact2.csv",
    delimiter=",")
first_order = np.loadtxt(
    "examples/box_pivoting/analysis/box_pivoting_first.csv",
    delimiter=",")
zero_order = np.loadtxt(
    "examples/box_pivoting/analysis/box_pivoting_zero.csv",
    delimiter=",")

plt.subplot(4,2,8)
plt.plot(exact[0:6], marker='x', color='red', label='exact')
plt.plot(first_order[0:6], marker='v', color='springgreen', label='first order')
plt.plot(zero_order[0:6], marker='^', color='blue', label='zero order')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('Cost')
plt.title("Box Pivoting")
plt.grid()

#%%
plt.savefig("results.pdf", bbox_inches='tight', pad_inches=0.01)