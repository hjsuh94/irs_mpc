import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm

# Analysis code for box pushing.

xu_q = np.load("examples/box_pushing/analysis/xu_quasistatic.npy")
xu_s = np.load("examples/box_pushing/analysis/xu_second_order.npy")
dxdu_q = np.load("examples/box_pushing/analysis/dxdu_quasistatic.npy")
dxdu_s = np.load("examples/box_pushing/analysis/dxdu_second_order.npy")

print(xu_q.shape)
print(xu_s.shape)
offset = 5

# Plotting the positions.
plt.figure()
for index in range(5):
    plt.subplot(7,1,index+1)
    plt.plot(xu_q[:,index], '-', color="springgreen", label="quasistatic")
    plt.plot(xu_s[:,index], '-', color='red', label="second order")
    plt.legend()

plt.subplot(7,1,6)
plt.plot(xu_q[:,5], '-', color='springgreen', label='quasistatic')
plt.plot(xu_s[:,5 + offset], '-', color='red', label='second order')
plt.subplot(7,1,7)
plt.plot(xu_q[:,6], '-', color='springgreen', label='quasistatic')
plt.plot(xu_s[:,6 + offset], '-', color='red', label='second order')
plt.show()

plt.figure()

dim_lst = ["px_pusher", "px_box", "py_pusher", "py_box", "ptheta_box",
           "vx_pusher", "vx_box", "vy_pusher", "vy_box", "vtheta_box",]
"""
count = 1
for i in range(10):
    for j in range(10):
        plt.subplot(10,10,count)        
        if (i < 5) and (j < 5):
            plt.plot(dxdu_q[:,i,j], '-', color="springgreen")
        plt.plot(dxdu_s[:,i,j], '-', color='red')
        plt.plot(dxdu_s[:,i,j], '-', color='red')
        plt.plot(dxdu_s[:,i,j], '-', color='red')
        plt.title("d" + dim_lst[i] + "/d" + dim_lst[j])
        count += 1
plt.legend()
plt.show()
"""

count = 1
for i in [2,3,7,8]:
    for j in [2,3,7,8]:
        plt.subplot(4,5,count)
        if (i < 5) and (j < 5):
            plt.plot(dxdu_q[:,i,j], '-', color="springgreen")
        plt.plot(dxdu_s[:,i,j], '-', color='red')
        plt.title("d" + dim_lst[i] + "/d" + dim_lst[j])
        print(dxdu_s[-1,i,j] - dxdu_s[0,i,j])

        count += 1
    plt.subplot(4,5,count)
    if i < 5:
        plt.plot(dxdu_q[:,i,6], '-', color='springgreen')
    plt.plot(dxdu_s[:,i,11], color='red')
    plt.title("d" + dim_lst[i] + "/duy_pusher")
    count += 1

plt.show()        

plt.figure()
plt.plot(xu_s[:,3] - xu_s[:,2])
plt.show()

plt.figure()
count = 1
plt.subplot(2,1)
plt.legend()
plt.show()        
