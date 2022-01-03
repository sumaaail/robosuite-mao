import math
import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
import time


x_dim = 6
y_dim = 6
initial_pos = np.array([1448.94, 17.4063, 1001.44, 179.75, 0.7, -1.33])

# GMM train
# load data
data = np.load('E:\THUME-SZIGS\RL//rl-robotic-assembly-mujoco-master-new//results//runs//real//tie_hrl\DDPG_dual-peg-in-hole_seed_0//test_states_.npy')[0][:2000]
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
print('data size of phase 1 training:', len(buffer[0]))
print('data size of phase 2 training:', len(buffer[1]))
print('data size of phase 3 training:', len(buffer[2]))

# train model
GMM = []
for i in range(3):
    T = time.time()
    if i >= 1:
        GM_conponemts = 4
    else:
        GM_conponemts = 2
    gmm = mixture.GaussianMixture(n_components=GM_conponemts, covariance_type='full').fit(buffer[i])
    GMM.append(gmm)
    T_train = time.time() - T
    print('phase', i, ' GMM training time:', T_train)


# GMM test
# load data
data2 = np.load('E:\THUME-SZIGS\RL//rl-robotic-assembly-mujoco-master-new//results//runs//real//tie_hrl\TD3_dual-peg-in-hole_seed_0//test_states_.npy')[0][:140]
force = data2[:, :6]
pos = data2[:, 6:] - initial_pos
data2 = np.hstack([pos, force])

# GMR: y = f(x)
T = time.time()
Y = []
SIGMA = []
for i in range(len(data2)):
    x_test = pos[i]
    # classification
    if pos[i][2] >= -3.3629:
        gmm = GMM[0]
        GM_conponemts = 2
    elif -7.8629 <= pos[i][2] < -3.3629:
        gmm = GMM[1]
        GM_conponemts = 4
    else:
        gmm = GMM[2]
        GM_conponemts = 4
    # Regression
    # Compute hl
    P = 0
    pl = []
    hl = []
    for i in range(GM_conponemts):
        ux = gmm.means_[i][: x_dim]
        covx = gmm.covariances_[i][: x_dim, : x_dim]
        p = (1/(np.linalg.det(covx)*(math.pi * 2)**x_dim)**0.5)\
            * np.exp(-0.5 * np.dot(np.dot(np.array([x_test - ux]), np.linalg.inv(covx)), np.transpose(np.array([x_test - ux]))))
        pl.append(gmm.weights_[i] * p[0][0])
        P += gmm.weights_[i] * p[0][0]
    if P == 0:
        print('GPR Err: very strange input, the likelihood P=0')
        y = np.array([0,0,0,0,0,0])
        Y.append(y)
        continue
    for i in range(GM_conponemts):
        hl.append(pl[i]/P)
    # print(hl)

    # Compute yl anc covl
    yl = []
    covl = []
    for i in range(GM_conponemts):
        ux = gmm.means_[i][: x_dim]
        uy = gmm.means_[i][-y_dim:]
        covx = gmm.covariances_[i][: x_dim, : x_dim]
        covy = gmm.covariances_[i][-y_dim:, -y_dim:]
        covyx = gmm.covariances_[i][-y_dim:, : x_dim]
        covxy = gmm.covariances_[i][: x_dim, -y_dim:]
        yi = uy + np.transpose(np.dot(np.dot(covyx, np.linalg.inv(covx)), np.transpose(np.array([x_test - ux]))))
        yl.append(yi)
        covi = covy - np.dot(np.dot(covyx, np.linalg.inv(covx)), covxy)
        covl.append(covi)
    # print(yl)

    # Compute y = f(x)
    y = np.zeros([1, y_dim])
    covy = np.zeros([y_dim, y_dim])
    for i in range(GM_conponemts):
        y += hl[i] * yl[i]
        covy += hl[i] * (covl[i] + np.dot(np.transpose(yl[i]), yl[i]))
    covy = covy - np.dot(np.transpose(y), y)

    # reshape y
    Y.append(y[0])
    SIGMA.append(covy)

Y = np.array(Y)
SIGMA = np.array(SIGMA)

T_test = time.time() - T
print('GMM testing time:', T_test)

# test and plot
m = 5  # plotting dimension
err = 0
sum = 0
sigma = []

for i in range(len(data2)):
    err = err + abs(Y[i][m] - force[i][m])
    sum = sum + abs(force[i][m])
    sigma.append([])
    for j in range(6):
        sigma[i].append(np.sqrt(SIGMA[i][j, j]))
sigma = np.array(sigma)
print('Overall accuracy', (1 - err / sum) * 100)

# plot reference data
Step = range(len(data2))
Fx = force[:, m].transpose()
Fy = Y[:, m].transpose()
plt.plot(Step, Fx)
plt.plot(Step, Fy)
miny = Fy - sigma[:, m].transpose()
maxy = Fy + sigma[:, m].transpose()
plt.fill_between(Step, miny, maxy, color=[0.83, 0.06, 0.06], alpha=0.3)
plt.xlabel('Steps', fontsize=15)
plt.ylabel('Value (force/torque)', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
