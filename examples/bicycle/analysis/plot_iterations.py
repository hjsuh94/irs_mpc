import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

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

plt.figure(figsize=(4, 3), dpi=200)
plt.plot(result_exact, marker='x', color='red', label="exact")
plt.plot(result_first, marker='v', color='springgreen', label="first order")
plt.plot(result_zero, marker='^', color='blue', label="zero order")
plt.yscale('log')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('Cost (log scale)')
plt.title("Dubin's Car (Easy)")
plt.grid()
plt.savefig("dubin_car_easy.pdf", bbox_inches='tight', pad_inches=0.01)
plt.show()

#%%
result_exact = np.loadtxt(
    "examples/" + folder_name + "/analysis/" + folder_name + "_hard_exact.csv",
    delimiter=",")
result_first = np.loadtxt(
    "examples/" + folder_name + "/analysis/" + folder_name + "_hard_first.csv",
    delimiter=",")
result_zero = np.loadtxt(
    "examples/" + folder_name + "/analysis/" + folder_name + "_hard_zero.csv",
    delimiter=",")

plt.figure(figsize=(4, 3), dpi=200)
plt.plot(result_exact, marker='x', color='red', label="exact")
plt.plot(result_first, marker='v', color='springgreen', label="first order")
plt.plot(result_zero, marker='^', color='blue', label="zero order")
plt.yscale('log')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('Cost (logscale)')
plt.title("Dubin's Car (Hard)")
plt.grid()
plt.savefig("dubin_car_hard.pdf", bbox_inches='tight', pad_inches=0.01)
plt.show()

