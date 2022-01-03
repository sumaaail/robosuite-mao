import time
import math
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import GPy
from utils.gmr import Gmr
from utils.gp_coregionalize_with_mean_regression import GPCoregionalizedWithMeanRegression
from utils.gmr_mean_mapping import GmrMeanMapping
from utils.gmr_kernels import Gmr_based_kernel

state_ = np.load('E:\THUME-SZIGS\RL//rl-robotic-assembly-mujoco-master-new//results//runs//real//tie_hrl_new_peg\TD3_dual-peg-in-hole_seed_0//test_states_with_labels_init_.npy')[2000:2300, :]
action_ = np.load('E:\THUME-SZIGS\RL//rl-robotic-assembly-mujoco-master-new//results//runs//real//tie_hrl_new_peg\TD3_dual-peg-in-hole_seed_0//test_im_actions_.npy')[2000:2300]
pos_ = state_[7:]
force_ = state_[1:7]

state = []
dpos = []
dpos1 = []
dpos2 = []
action = []
action1 = []
action2 = []
for i in range(len(state_) - 1):
    if state_[i+1][0] != 0:
        state.append(state_[i][1:])
        dpos.append(state_[i + 1][7:] - state_[i][7:])
        action.append(action_[i])
        if state_[i][9] >= -5.:
            dpos1.append(state_[i + 1][7:] - state_[i][7:])
            action1.append(action_[i])
        if state_[i][9] < -5.:
            dpos2.append(state_[i + 1][7:] - state_[i][7:])
            action2.append(action_[i])
    else:
        print(len(state))

dpos = np.array(dpos)
action = np.array(action)
state = np.array(state)
dpos1 = np.array(dpos1)
action1 = np.array(action1)
dpos2 = np.array(dpos2)
action2 = np.array(action2)

force = state[:, :6]
pos = state[:, 6:]
data_state_search = np.hstack([action1, dpos1])
data_state_insert = np.hstack([action2, dpos2])
data_force = np.hstack([pos, force])
print(len(data_state_search), len(data_state_insert), len(data_force))

X = pos
Y = force
state_t = state[128:254]
Xt = state_t[:, 6:]
Yt = state_t[:, :6]

# Parameters
nb_data_sup = 50
nb_samples = 2
dt = 0.01
input_dim = 6
output_dim = 6
in_idx = [0, 1, 2, 3, 4, 5]
out_idx = [6, 7, 8, 9, 10, 11]
nb_states = 10

nb_prior_samples = 10
nb_posterior_samples = 3

# Train data for GPR
X_list = [np.hstack((X, X)) for i in range(output_dim)]
Y_list = [Y[:, i][:, None] for i in range(output_dim)]

# Test data
nb_data_test = Xt.shape[0]
Xtest, _, output_index = GPy.util.multioutput.build_XY([np.hstack((Xt, Xt)) for i in range(output_dim)])

# Define via-points (new set of observations)
X_obs = np.array([[0, 0, 0, 0, 0, 0]])
Y_obs = np.array([[0, 0, 10, 0, 0, 0]])
X_obs_list = [np.hstack((X_obs, X_obs)) for i in range(output_dim)]
Y_obs_list = [Y_obs[:, i][:, None] for i in range(output_dim)]

# Train GMM
gmr_model = Gmr(nb_states=nb_states, nb_dim=input_dim + output_dim, in_idx=in_idx, out_idx=out_idx)
gmr_model.init_params_kbins(data_force.T, nb_samples=nb_samples)
gmr_model.gmm_em(data_force.T)

# Define GPR likelihood and kernels
likelihoods_list = [GPy.likelihoods.Gaussian(name="Gaussian_noise_%s" % j, variance=0.01) for j in
                    range(output_dim)]
# kernel_list = [GPy.kern.RBF(1, variance=1., lengthscale=0.1) for i in range(gmr_model.nb_states)]
kernel_list = [GPy.kern.Matern52(1, variance=1., lengthscale=5.) for i in range(gmr_model.nb_states)]

# Fix variance of kernels
for kernel in kernel_list:
    kernel.variance.fix(1.)
    kernel.lengthscale.constrain_bounded(0.01, 10.)

# Bound noise parameters
for likelihood in likelihoods_list:
    likelihood.variance.constrain_bounded(0.001, 0.05)

# GPR model
K = Gmr_based_kernel(gmr_model=gmr_model, kernel_list=kernel_list)
mf = GmrMeanMapping(2 * input_dim + 1, 1, gmr_model)

m = GPCoregionalizedWithMeanRegression(X_list, Y_list,
                                       kernel=K, likelihoods_list=likelihoods_list,
                                       mean_function=mf)

# Parameters optimization
m.optimize('bfgs', max_iters=100, messages=True)

# Print model parameters
print(m)

# GPR prior (no observations)
prior_traj = []
prior_mean = mf.f(Xtest)[:, 0]
prior_kernel = m.kern.K(Xtest)
for i in range(nb_prior_samples):
    prior_traj_tmp = np.random.multivariate_normal(prior_mean, prior_kernel)
    prior_traj.append(np.reshape(prior_traj_tmp, (output_dim, -1)))

prior_kernel_tmp = np.zeros((nb_data_test, nb_data_test, output_dim * output_dim))
for i in range(output_dim):
    for j in range(output_dim):
        prior_kernel_tmp[:, :, i * output_dim + j] = prior_kernel[i * nb_data_test:(i + 1) * nb_data_test,
                                                     j * nb_data_test:(j + 1) * nb_data_test]
prior_kernel_rshp = np.zeros((nb_data_test, output_dim, output_dim))
for i in range(nb_data_test):
    prior_kernel_rshp[i] = np.reshape(prior_kernel_tmp[i, i, :], (output_dim, output_dim))

# GPR posterior -> new points observed (the training points are discarded as they are "included" in the GMM)
m_obs = GPCoregionalizedWithMeanRegression(X_obs_list, Y_obs_list,
                                           kernel=K, likelihoods_list=likelihoods_list,
                                           mean_function=mf)
mu_posterior_tmp = m_obs.posterior_samples_f(Xtest, full_cov=True, size=nb_posterior_samples)

mu_posterior = []
for i in range(nb_posterior_samples):
    mu_posterior.append(np.reshape(mu_posterior_tmp[:, 0, i], (output_dim, -1)))

# GPR prediction
mu_gp, sigma_gp = m_obs.predict(Xtest, full_cov=True, Y_metadata={'output_index': output_index})

mu_gp_rshp = np.reshape(mu_gp, (output_dim, -1)).T

sigma_gp_tmp = np.zeros((nb_data_test, nb_data_test, output_dim * output_dim))
for i in range(output_dim):
    for j in range(output_dim):
        sigma_gp_tmp[:, :, i * output_dim + j] = sigma_gp[i * nb_data_test:(i + 1) * nb_data_test,
                                                 j * nb_data_test:(j + 1) * nb_data_test]
sigma_gp_rshp = np.zeros((nb_data_test, output_dim, output_dim))
for i in range(nb_data_test):
    sigma_gp_rshp[i] = np.reshape(sigma_gp_tmp[i, i, :], (output_dim, output_dim))

# test and plot
YLABEL = ['Fx/N', 'Fy/N', 'Fz/N', 'Tx/Nm', 'Ty/Nm', 'Tz/Nm']
plt.figure(figsize=(14, 7), dpi=100)
plt.tight_layout(pad=3, w_pad=1., h_pad=0.5)
plt.subplots_adjust(left=0.12, bottom=0.10, right=0.95, top=0.96, wspace=0.32, hspace=0.26)
for m in range(6):
    plt.subplot(2, 3, m + 1)
    err = 0
    sum = 0
    sigma = []
    for i in range(len(Xt)):
        mu_gp_rshp[i] = mu_gp_rshp[i]
        err = err + abs(mu_gp_rshp[i][m] - Yt[i][m])
        sum = sum + abs(Yt[i][m])
        sigma.append(np.sqrt(sigma_gp_rshp[i][m]))
    sigma = np.array(sigma)
    print('Overall accuracy', (1 - err / sum) * 100)
    # plot reference data
    Step = range(len(Xt))
    Fx = Yt[:, m].transpose()
    Fy = mu_gp_rshp[:, m].transpose()
    plt.plot(Step, Fx)
    plt.plot(Step, Fy)
    miny = Fy - sigma[:, m].transpose()
    maxy = Fy + sigma[:, m].transpose()
    plt.fill_between(Step, miny, maxy, color=[0.83, 0.06, 0.06], alpha=0.25)
    plt.xlabel('Steps', fontsize=15)
    plt.ylabel('Value (force/torque)', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
plt.show()