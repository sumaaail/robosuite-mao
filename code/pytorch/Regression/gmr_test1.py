import numpy as np
import math
from scipy import stats
from sklearn import mixture

x_dim = 4
y_dim = 3
GM_conponemts = 3

# Compute the training samples
X = np.array([[10, 20, 30, 40, 1.1, 1.2, 1.3],
              [20, 30, 40, 50, 1.2, 1.3, 1.4],
              [30, 40, 50, 60, 1.3, 1.4, 1.5],
              [40, 50, 60, 70, 1.4, 1.5, 1.6],
              [50, 60, 70, 80, 1.5, 1.6, 1.7],
              [60, 70, 80, 90, 1.6, 1.7, 1.8],
              [70, 80, 90, 100, 1.7, 1.8, 1.9],
              [80, 90, 100, 110, 1.8, 1.9, 2.0],
              [90, 100, 110, 120, 1.9, 2.0, 2.1],
              [100, 110, 120, 130, 2.0, 2.1, 1.3],
              [110, 120, 130, 140, 2.1, 2.2, 1.4],
              [120, 130, 140, 150, 2.2, 2.3, 1.5],
              [130, 140, 150, 160, 2.3, 2.4, 1.6],
              [140, 150, 160, 170, 2.4, 2.5, 1.7],
              [150, 160, 170, 180, 2.5, 2.6, 1.8],
              [160, 170, 180, 190, 2.6, 2.7, 2.8],
              [170, 180, 190, 200, 2.7, 2.8, 2.9],
              [180, 190, 200, 210, 2.8, 2.9, 3.0]])

X_noise = np.hstack([np.random.randn(18, x_dim)*10, np.random.randn(18, y_dim)/10])
X_train = X + X_noise
print(X_train)

x = np.array([55, 65, 75, 85])  # input x

# Train
clf = mixture.GaussianMixture(n_components=GM_conponemts, covariance_type='full').fit(X_train)

# GMR: y = f(x)
# Compute hl
P = 0
pl = []
hl = []
for i in range(GM_conponemts):
    ux = clf.means_[i][: x_dim]
    covx = clf.covariances_[i][: x_dim, : x_dim]
    p = (1/(np.linalg.det(covx)*(math.pi * 2)**x_dim)**0.5)\
        * np.exp(-0.5 * np.dot(np.dot(np.array([x - ux]), np.linalg.inv(covx)), np.transpose(np.array([x - ux]))))
    pl.append(clf.weights_[i] * p[0][0])
    P += clf.weights_[i] * p[0][0]
if P == 0:
    print('GPR Err: very strange input, the likelihood P=0')
for i in range(GM_conponemts):
    hl.append(pl[i]/P)
# print(hl)

# Compute yl anc covl
yl = []
covl = []
for i in range(GM_conponemts):
    ux = clf.means_[i][: x_dim]
    uy = clf.means_[i][-y_dim:]
    covx = clf.covariances_[i][: x_dim, : x_dim]
    covy = clf.covariances_[i][-y_dim:, -y_dim:]
    covyx = clf.covariances_[i][-y_dim:, : x_dim]
    covxy = clf.covariances_[i][: x_dim, -y_dim:]
    yi = uy + np.transpose(np.dot(np.dot(covyx, np.linalg.inv(covx)), np.transpose(np.array([x - ux]))))
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
print(y)
print(covy)