import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 10})
from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=False)

fig, axes = plt.subplots(2, 4, figsize=(16, 5.1), dpi=300)
title_font_size = 10

# %% Pendulum
result_exact = np.load("examples/pendulum/analysis/exact_cost.npy")
result_first = np.load("examples/pendulum/analysis/first_order_cost.npy")
result_zero = np.load("examples/pendulum/analysis/zero_order_cost.npy")
result_cem = np.load("examples/pendulum/analysis/pendulum_cem.npy")

ax = axes[0, 0]
ax.plot(result_exact, marker='x', color='red', label="exact")
ax.plot(result_first, marker='v', color='springgreen', label="first order")
ax.plot(result_zero, marker='^', color='blue', label="zero order")
ax.plot(result_cem, marker='*', color='magenta', label='cem')
ax.legend()
# ax.set_xlabel('iterations')
ax.set_ylabel('Cost')
ax.set_title('Pendulum', fontsize=title_font_size)
ax.grid()

# %% Easy
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
result_cem = np.loadtxt(
    "examples/" + folder_name + "/analysis/" + folder_name + "_easy_cem.csv",
    delimiter=",")


ax = axes[0, 1]
ax.plot(result_exact, marker='x', color='red', label="exact")
ax.plot(result_first, marker='v', color='springgreen', label="first order")
ax.plot(result_zero, marker='^', color='blue', label="zero order")
ax.plot(result_cem, marker='*', color='magenta', label="cem")
ax.set_yscale('log')
ax.legend()
# ax.set_xlabel('iterations')
# ax.set_ylabel('Cost (log scale)')
ax.set_title("Dubin's Car (Easy)", fontsize=title_font_size)
ax.grid()

# %% Hard
result_exact = np.loadtxt(
    "examples/" + folder_name + "/analysis/" + folder_name + "_hard_exact.csv",
    delimiter=",")
result_first = np.loadtxt(
    "examples/" + folder_name + "/analysis/" + folder_name + "_hard_first.csv",
    delimiter=",")
result_zero = np.loadtxt(
    "examples/" + folder_name + "/analysis/" + folder_name + "_hard_zero.csv",
    delimiter=",")
result_cem = np.loadtxt(
    "examples/" + folder_name + "/analysis/" + folder_name + "_hard_cem.csv",
    delimiter=",")    

ax = axes[0, 2]
ax.plot(result_exact[:15], marker='x', color='red', label="exact")
ax.plot(result_first[:15], marker='v', color='springgreen', label="first order")
ax.plot(result_zero[:15], marker='^', color='blue', label="zero order")
ax.plot(result_cem[:15], marker='*', color='magenta', label="cem")
ax.set_yscale('log')
ax.legend()
# ax.set_xlabel('iterations')
# ax.set_ylabel('Cost (logscale)')
ax.set_title("Dubin's Car (Hard)", fontsize=title_font_size)
ax.grid()

# %% Quadrotor
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
result_cem = np.loadtxt(
    "examples/" + folder_name + "/analysis/" + folder_name + "_cem.csv",
    delimiter=",")    

ax = axes[0, 3]
ax.plot(result_exact, marker='x', color='red', label="exact")
ax.plot(result_first, marker='v', color='springgreen', label="first order")
ax.plot(result_zero, marker='^', color='blue', label="zero order")
ax.plot(result_cem, marker='*', color='magenta', label="cem")
ax.set_yscale('log')
ax.legend()
# ax.set_xlabel('iterations')
# ax.set_ylabel('Cost (log scale)')
ax.set_title("Quadrotor", fontsize=title_font_size)
ax.grid()

# %% Planar Hand 1
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
cem = np.loadtxt(
    "examples/planar_hand/analysis/planar_hand_cem.csv",
    delimiter=",")    


ax = axes[1, 0]
ax.plot(exact, marker='x', color='red', label='exact')
ax.plot(first_order, marker='v', color='springgreen', label='first order')
ax.plot(zero_order_AB, marker='^', color='blue', label='zero order')
# ax.plot(zero_order_AB, marker='+', color='magenta', label='zero order_AB')
ax.plot(cem, marker='*', color='magenta', label='cem')
ax.legend()
ax.set_xlabel('iterations')
ax.set_ylabel('Cost')
ax.set_title("Planar Hand (Quasistatic Sim)", fontsize=title_font_size)
ax.grid()

# %% Planar Hand 2

exact = np.loadtxt(
    "examples/planar_hand/analysis/planar_hand_spin_second_exact.csv",
    delimiter=",")
first_order = np.loadtxt(
    "examples/planar_hand/analysis/planar_hand_spin_second_first.csv",
    delimiter=",")
zero_order = np.loadtxt(
    "examples/planar_hand/analysis/planar_hand_spin_second_zero.csv",
    delimiter=",")
cem = np.loadtxt(
    "examples/planar_hand/analysis/planar_hand_spin_second_cem.csv",
    delimiter=",")    

ax = axes[1, 1]
ax.plot(exact, marker='x', color='red', label='exact')
ax.plot(first_order, marker='v', color='springgreen', label='first order')
ax.plot(zero_order, marker='^', color='blue', label='zero order')
ax.plot(cem[:12], marker='*', color='magenta', label='cem')
ax.legend()
ax.set_yscale('log')
ax.set_xlabel('iterations')
# ax.set_ylabel('Cost (log scale)')
ax.set_title("Planar Hand (Second-Order Sim)", fontsize=title_font_size)
ax.grid()

# %% Box Pushing
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

ax = axes[1, 2]
ax.plot(exact, marker='x', color='red', label='exact')
ax.plot(first_order, marker='v', color='springgreen', label='first order')
ax.plot(zero_order, marker='^', color='blue', label='zero order')
ax.plot(cem, marker='*', color='magenta', label='cem')
ax.legend()
ax.set_xlabel('iterations')
# ax.set_ylabel('Cost')
ax.set_title("Planar Pushing", fontsize=title_font_size)
ax.grid()

# %% Box Pivoting

exact = np.loadtxt(
    "examples/box_pivoting/analysis/box_pivoting_exact2.csv",
    delimiter=",")
first_order = np.loadtxt(
    "examples/box_pivoting/analysis/box_pivoting_first.csv",
    delimiter=",")
zero_order = np.loadtxt(
    "examples/box_pivoting/analysis/box_pivoting_zero.csv",
    delimiter=",")
cem = np.loadtxt(
    "examples/box_pivoting/analysis/box_pivoting_cem.csv",
    delimiter=",")    

ax = axes[1, 3]
ax.plot(1e-2 * exact[0:6], marker='x', color='red', label='exact')
ax.plot(1e-2 * first_order[0:6], marker='v', color='springgreen', label='first order')
ax.plot(1e-2 * zero_order[0:6], marker='^', color='blue', label='zero order')
ax.plot(1e-2 * cem[0:6], marker='*', color='magenta', label='cem')
ax.legend()
ax.set_xlabel('iterations')
# ax.set_ylabel('Cost')
ax.set_title("Box Pivoting", fontsize=title_font_size)
ax.grid()

# %%
plt.tight_layout()
plt.savefig("results.pdf", bbox_inches='tight', pad_inches=0.01)
plt.show()
