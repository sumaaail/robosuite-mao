import math
import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
import time

initial_pos = np.array([1448.94, 17.4063, 1001.44, 179.75, 0.7, -1.33])
action = np.load('E:\THUME-SZIGS\RL//rl-robotic-assembly-mujoco-master-new//results//runs//real//tie_hrl_new_peg\TD3_dual-peg-in-hole_seed_0//test_im_actions_.npy')

state = np.load('E:\THUME-SZIGS\RL//rl-robotic-assembly-mujoco-master-new//results//runs//real//tie_hrl_new_peg\TD3_dual-peg-in-hole_seed_0//test_states_with_labels_init_.npy')
# plot force
YLABEL = ['Fx/N', 'Fy/N', 'Fz/N', 'Tx/Nm', 'Ty/Nm', 'Tz/Nm']
plt.figure(figsize=(14, 7), dpi=100)
plt.tight_layout(pad=3, w_pad=1., h_pad=0.5)
plt.subplots_adjust(left=0.12, bottom=0.10, right=0.95, top=0.96, wspace=0.32, hspace=0.26)
n = 130
for i in range(6):
    plt.subplot(2, 3, i + 1)
    Step = range(n)
    Fx = state[:n, i+1].transpose()
    # Fy = trajectory[:n, i].transpose()
    plt.plot(Step, Fx)
    # plt.plot(Step, Fy)
    plt.xlabel('Steps', fontsize=15)
    plt.ylabel(YLABEL[i], fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
plt.show()

# plot action
YLABEL = ['dX', 'dY', 'dZ', r'd$\theta$x', r'd$\theta$y', r'd$\theta$z']
plt.figure(figsize=(14, 7), dpi=100)
plt.tight_layout(pad=3, w_pad=1., h_pad=0.5)
plt.subplots_adjust(left=0.12, bottom=0.10, right=0.95, top=0.96, wspace=0.32, hspace=0.26)
n = 130
for i in range(6):
    plt.subplot(2, 3, i + 1)
    Step = range(n)
    Fx = action[:n, i].transpose()
    # Fy = trajectory[:n, i].transpose()
    plt.plot(Step, Fx)
    # plt.plot(Step, Fy)
    plt.xlabel('Steps', fontsize=15)
    plt.ylabel(YLABEL[i], fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
plt.show()

# plot pos
YLABEL = ['X/mm', 'Y/mm', 'Z/mm', r'$\theta$x/rad', r'$\theta$y/rad', r'$\theta$z/rad']
plt.figure(figsize=(14, 7), dpi=100)
plt.tight_layout(pad=3, w_pad=1., h_pad=0.5)
plt.subplots_adjust(left=0.12, bottom=0.10, right=0.95, top=0.96, wspace=0.32, hspace=0.26)
n = 130
for i in range(6):
    plt.subplot(2, 3, i + 1)
    Step = range(n)
    Fx = state[:n, i+7].transpose()
    # Fy = trajectory[:n, i].transpose()
    plt.plot(Step, Fx)
    # plt.plot(Step, Fy)
    plt.xlabel('Steps', fontsize=15)
    plt.ylabel(YLABEL[i], fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
plt.show()