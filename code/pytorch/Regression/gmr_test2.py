import math
import numpy as np
from sklearn import mixture

x_dim = 1
y_dim = 1
cov = np.eye(2, 2)
u1 = np.array([2, 2])
u2 = np.array([-2, -2])
X1 = np.random.multivariate_normal(mean=u1, cov=cov, size=200)
X2 = np.random.multivariate_normal(mean=u2, cov=cov, size=200)
X = np.vstack([X1, X2])
GM_conponemts = 1
x_train = np.array([0.5])  # input x

gmm = mixture.GaussianMixture(n_components=GM_conponemts, covariance_type='full').fit(X)

# GMR: y = f(x)
# Compute hl
P = 0
pl = []
hl = []
for i in range(GM_conponemts):
    ux = gmm.means_[i][: x_dim]
    covx = gmm.covariances_[i][: x_dim, : x_dim]
    p = (1/(np.linalg.det(covx)*(math.pi * 2)**x_dim)**0.5)\
        * np.exp(-0.5 * np.dot(np.dot(np.array([x_train - ux]), np.linalg.inv(covx)), np.transpose(np.array([x_train - ux]))))
    pl.append(gmm.weights_[i] * p[0][0])
    P += gmm.weights_[i] * p[0][0]
if P == 0:
    print('GPR Err: very strange input, the likelihood P=0')
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
    yi = uy + np.transpose(np.dot(np.dot(covyx, np.linalg.inv(covx)), np.transpose(np.array([x_train - ux]))))
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