import math
import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
import time


x_dim = 6
y_dim = 6
GM_conponemts = 3
initial_pos = np.array([1448.94, 17.4063, 1001.44, 179.75, 0.7, -1.33])

# GMM train
# load data
data = np.load('E:\THUME-SZIGS\RL//rl-robotic-assembly-mujoco-master-new//results//runs//real//test_im//test_im_08/test_states_kp_kd_noise4_.npy')[0]
force = data[:, :6]
pos = data[:, 6:] - initial_pos
data = np.hstack([pos, force])

# classification
buffer = [[], [], []]
for i in range(len(data)):
    if pos[i][2] >= -3.3629:
        buffer[0].append(data[i])
    elif -7.8629 <= pos[i][2] < -3.3629:
        buffer[1].append(data[i])
    else:
        buffer[2].append(data[i])
print('data size of phase 1:', len(buffer[0]))
print('data size of phase 2:', len(buffer[1]))
print('data size of phase 3:', len(buffer[2]))

# train model
GMM = []
for i in range(3):
    T = time.time()
    gmm = mixture.GaussianMixture(n_components=GM_conponemts, covariance_type='full').fit(buffer[i])
    T_train = time.time() - T
    print('phase', i, ' GMM training time:', T_train)

print(data[:10, 2])
