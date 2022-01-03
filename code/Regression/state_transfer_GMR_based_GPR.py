import numpy as np
import GPy
import matplotlib.pyplot as plt
from scipy.io import loadmat  # loading data from matlab
from utils.gmr import Gmr
from utils.gmr import plot_gmm
from utils.gp_coregionalize_with_mean_regression import GPCoregionalizedWithMeanRegression
from utils.gmr_mean_mapping import GmrMeanMapping
from utils.gmr_kernels import Gmr_based_kernel


# GMR-based GPR on 2D trajectories with time as input
if __name__ == '__main__':
    initial_pos = np.array([1448.94, 17.4063, 1001.44, 179.75, 0.7, -1.33])
    # pos_bound = np.array([0.8768, 0.5289, 30.3364, 0.1318, 0.2264, 0.7430])
    # force_bound = np.array([33.6377, 48.2733, 86.669, 4.0921, 4.4257, 3.0208])
    # initial_pos = np.array([0,0,0,0,0,0])
    action_bound = np.array([1,1,1,1,1,1])
    dpos_bound = np.array([1,1,1,1,1,1])

    # GMM train
    z = 5
    # train data
    state_ = np.load('E:\THUME-SZIGS\RL//rl-robotic-assembly-mujoco-master-new//results//runs//real//tie_hrl\DDPG_dual-peg-in-hole_seed_0//test_states_.npy')[0][:2000]
    force_ = state_[:, :6]
    pos_ = state_[:, 6:] - initial_pos
    dpos_ = []
    for i in range(len(pos_) - 1):
        dpos_.append(pos_[i + 1] - pos_[i])
    dpos_ = np.array(dpos_)
    dpos_ = dpos_[:, z:z+1] / dpos_bound[z]
    action_ = np.load('E:\THUME-SZIGS\RL//rl-robotic-assembly-mujoco-master-new//results//runs//real//tie_hrl\DDPG_dual-peg-in-hole_seed_0//test_im_actions_.npy')[0][:2000]
    action_ = action_[1:, :] / action_bound
    X = action_
    Y = dpos_
    data_ = np.hstack([action_, dpos_])

    # test data
    state = np.load('E:\THUME-SZIGS\RL//rl-robotic-assembly-mujoco-master-new//results//runs//real//tie_hrl\TD3_dual-peg-in-hole_seed_0//test_states_.npy')[0][:120]
    force = state[:, :6]
    pos = state[:, 6:] - initial_pos
    dpos = []
    for i in range(len(pos) - 1):
        dpos.append(pos[i + 1] - pos[i])
    dpos = np.array(dpos)
    dpos = dpos[:, z:z + 1] / dpos_bound[z]
    action = np.load('E:\THUME-SZIGS\RL//rl-robotic-assembly-mujoco-master-new//results//runs//real//tie_hrl\TD3_dual-peg-in-hole_seed_0//test_im_actions_.npy')[0][:120]
    action = action[1:, :] / action_bound
    data = np.hstack([action, dpos])
    print('testing data size:', len(data))

    # Parameters
    nb_data_sup = 50
    nb_samples = 2
    dt = 0.01
    input_dim = 6
    output_dim = 1
    in_idx = [0, 1, 2, 3, 4, 5]
    out_idx = [6]
    nb_states = 5

    nb_prior_samples = 10
    nb_posterior_samples = 3

    # Train data for GPR
    X_list = [np.hstack((X, X)) for i in range(output_dim)]
    Y_list = [Y[:, i][:, None] for i in range(output_dim)]

    # Test data
    Xt = action
    nb_data_test = Xt.shape[0]
    Xtest, _, output_index = GPy.util.multioutput.build_XY([np.hstack((Xt, Xt)) for i in range(output_dim)])

    # Define via-points (new set of observations)
    X_obs = np.array([[0, 0, 0, 0, 0, 0]])
    Y_obs = np.array([[0]])
    X_obs_list = [np.hstack((X_obs, X_obs)) for i in range(output_dim)]
    Y_obs_list = [Y_obs[:, i][:, None] for i in range(output_dim)]

    # GMM
    gmr_model = Gmr(nb_states=nb_states, nb_dim=input_dim + output_dim, in_idx=in_idx, out_idx=out_idx)
    gmr_model.init_params_kbins(data.T, nb_samples=nb_samples)
    gmr_model.gmm_em(data.T)

    # # GMR prediction
    # mu_gmr = []
    # sigma_gmr = []
    # for i in range(Xt.shape[0]):
    # 	mu_gmr_tmp, sigma_gmr_tmp, H_tmp = gmr_model.gmr_predict(Xt[i])
    # 	mu_gmr.append(mu_gmr_tmp)
    # 	sigma_gmr.append(sigma_gmr_tmp)
    #
    # mu_gmr = np.array(mu_gmr)
    # sigma_gmr = np.array(sigma_gmr)

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

    m = GPCoregionalizedWithMeanRegression(X_list, Y_list, kernel=K, likelihoods_list=likelihoods_list,
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
    m_obs = GPCoregionalizedWithMeanRegression(X_obs_list, Y_obs_list, kernel=K, likelihoods_list=likelihoods_list,
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
    m = 0  # plotting dimension
    err = 0
    sum = 0
    sigma = []
    for i in range(len(data)):
        mu_gp_rshp[i] = mu_gp_rshp[i] * dpos_bound[z]
        dpos = dpos * dpos_bound[z]
        err = err + abs(mu_gp_rshp[i][m] - dpos[i][m])
        sum = sum + abs(dpos[i][m])

        sigma.append(np.sqrt(sigma_gp_rshp[i][m]))
    sigma = np.array(sigma)
    print('Overall accuracy', (1-err / sum) * 100)
    # plot reference data
    Step = range(len(Xt))
    Fx = dpos[:, m].transpose()
    Fy = mu_gp_rshp[:, m].transpose()
    plt.plot(Step, Fx)
    plt.plot(Step, Fy)
    miny = Fy - sigma[:, m].transpose()
    maxy = Fy + sigma[:, m].transpose()
    plt.fill_between(Step, miny, maxy, color=[0.83, 0.06, 0.06], alpha=0.3)
    plt.xlabel('Steps', fontsize=15)
    plt.ylabel('Value (dpos)', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()