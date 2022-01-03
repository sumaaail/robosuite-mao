# import gym, assistive_gym
#
# # env = gym.make('DrinkingSawyer-v1')
# env = gym.make('FeedingSawyer-v1')
#
# env.render()
# observation = env.reset()
#
# while True:
#     env.render()
#     action = env.action_space.sample()
#     observation, reward, done, info = env.step(action)
#     print('info', info)

from sklearn import mixture
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

Xmoon, ymoon = make_moons(200, noise=.05, random_state=0)


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """用给定的位置和协方差画一个椭圆"""
    ax = ax or plt.gca()

    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


from sklearn import mixture
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

data = np.load('memory_realistic.npy', allow_pickle=True)
print("data_shape ::", data[:, 0].shape)

context_list = data[:, 0]
para_list = data[:, 1]
rewad_list = data[:, 2]

memory_realistic = data
Context_list = []
Param_list = []
Reward_list = []
Param_list_original = []
stiffness_high = np.array([4.0, 4.0, 4.0, 8.0, 8.0, 8.0])
stiffness_low = np.array([0.5, 0.5, 0.5, 4.0, 4.0, 4.0])
stiffness_initial = np.array([2.0, 2.0, 2.0, 6.0, 6.0, 6.0])
stiffness_range = stiffness_high - stiffness_low
print("range", stiffness_range)

from code.pytorch.REPS.GPREPS import GPREPS

gpreps = GPREPS(
    4,
    6,
    50,
    np.array([0.5, 0.5, 0.5, 4.0, 4.0, 4.0]),
    np.array([4.0, 4.0, 4.0, 8.0, 8.0, 8.0]),
    stiffness_initial,
    0.4
)


for i in range(900):
    context, param, reward = memory_realistic[-i]
    Context_list.append(np.array(context, copy=False))
    # Param_list_original.append(np.array(param[0], copy=False))
    Param_list.append((np.array(param[0], copy=False) - stiffness_low)/stiffness_range)
    Reward_list.append(np.array(reward, copy=False) * 10)

Context_list_test = []
Param_list_test = []
Reward_list_test = []

for j in range(650, 700):
    context, param, reward = memory_realistic[j]
    Context_list_test.append(np.array(context, copy=False))
    Param_list_test.append((np.array(param[0], copy=False) - stiffness_low)/stiffness_range)
    # print("sampled reward :::", reward, np.array(reward, copy=False) * 10)
    Reward_list_test.append(np.array(reward, copy=False) * 10)

print("Reward list test :::", Reward_list_test)
# Reward_list_test = Reward_list_test
# print("Context list test :::", Context_list_test)
# print("Params list test :::", Param_list_test)
kernel = DotProduct() + WhiteKernel()
input_list = np.hstack((Context_list, Param_list))
gp = GaussianProcessRegressor()
gp.fit(input_list, Reward_list)
# print("True reward :::", Reward_list_test)
print("score :::", gp.score(input_list, Reward_list))

input_list_test = np.hstack((Context_list_test, Param_list_test))
predict_reward, _ = gp.predict(input_list_test, return_std=True, return_cov=False)
# predict_reward = gp.sample_y(input_list_test)

print("predict reward :::", predict_reward)
# print("reward_list :::", predict_reward - Reward_list_test)
print("accuracy :::", np.mean(np.square((predict_reward - Reward_list_test)/Reward_list_test)))
print("accuracy :::", np.linalg.norm(predict_reward - Reward_list_test)/50)
# print("input_list :::", input_list.shape)
# print("reward_list :::", Reward_list)


# import GPy
# from code.pytorch.Regression.GMRbasedGP.utils.gmr import Gmr
# from code.pytorch.Regression.GMRbasedGP.utils.gp_coregionalize_with_mean_regression import GPCoregionalizedWithMeanRegression
# from code.pytorch.Regression.GMRbasedGP.utils.gmr_mean_mapping import GmrMeanMapping
# from code.pytorch.Regression.GMRbasedGP.utils.gmr_kernels import Gmr_based_kernel
#
# nb_states = 10
# nb_samples = 2
#
# input_dim = 10
# output_dim = 1
# in_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# out_idx = [10]
#
# data_input = np.hstack([input_list, Reward_list])
# data_test_input = np.hstack([input_list_test, Reward_list_test])
# print("data_input :::", data_input.shape)
#
# gmm16 = mixture.GaussianMixture(n_components=16, covariance_type='full', random_state=0)
# gmm16.fit(data_input[:, :4])
# # plot_gmm(gmm16, data_input[:, :4], label=False)
# # plt.show()
#
# Xnew = gmm16.sample(n_samples=50)
# print('Xnew :::', Xnew[0])
#
# z_list = Xnew[0]
# w_list = gpreps.choose_action(z_list)
# w_list = (w_list - stiffness_low)/stiffness_range
# print("w_list :::", w_list)
# test_input = np.hstack((z_list, w_list))
# print("test_input :::", test_input.shape)
#
# predict_reward_list, _ = gp.predict(test_input, return_std=True, return_cov=False)
# print("predict_reward_output :::", predict_reward_list)
#
# # ========================================== GMR =================================================
# gmr_model = Gmr(nb_states=nb_states, nb_dim=input_dim + output_dim, in_idx=in_idx, out_idx=out_idx)
# gmr_model.init_params_kbins(data_input.T, nb_samples=nb_samples)
# gmr_model.gmm_em(data_input.T)
#
# # GMR prediction
# mu_gmr = []
# sigma_gmr = []
# for i in range(input_list_test.shape[0]):
#     mu_gmr_tmp, sigma_gmr_tmp, H_tmp = gmr_model.gmr_predict(input_list_test[i])
#     mu_gmr.append(mu_gmr_tmp)
#     sigma_gmr.append(sigma_gmr_tmp)
#
# mu_gmr = np.array(mu_gmr)
# sigma_gmr = np.array(sigma_gmr)
#
# # Define GPR likelihood and kernels
# likelihoods_list = [GPy.likelihoods.Gaussian(name="Gaussian_noise_%s" % j, variance=0.01) for j in range(output_dim)]
# kernel_list = [GPy.kern.Matern52(1, variance=1., lengthscale=5.) for i in range(gmr_model.nb_states)]
#
# # Fix variance of kernels
# for kernel in kernel_list:
#     kernel.variance.fix(1.)
#     kernel.lengthscale.constrain_bounded(0.01, 10.)
#
# # Bound noise parameters
# for likelihood in likelihoods_list:
#     likelihood.variance.constrain_bounded(0.001, 0.05)
#
# # GPR model
# K = Gmr_based_kernel(gmr_model=gmr_model, kernel_list=kernel_list)
# mf = GmrMeanMapping(2 * input_dim + 1, 1, gmr_model)
#
# X = data_input[:, :10]
# Y = data_input[:, 10]
#
# print("X :::", X.shape)
# print("Y :::", Y.shape)
#
# # Train data for GPR
# X_list = [np.hstack((X, X)) for i in range(output_dim)]
# Y_list = [Y[:, None] for i in range(output_dim)]
#
# # Test data
# Xt = test_input[:, :10]
# nb_data_test = Xt.shape[0]
# Xtest, _, output_index = GPy.util.multioutput.build_XY([np.hstack((Xt, Xt)) for i in range(output_dim)])
#
# # Ytest = data_test_input[:, 10]
# # print("output index", output_index)
# # print("X_list :::", X_list)
# # print("Y_list :::", Y_list)
#
# # kernel = GPy.kern.Matern52(input_dim, variance=1., lengthscale=10.)
# # K = kernel.prod(GPy.kern.Coregionalize(1, output_dim, active_dims=[input_dim], name='B'))
# # m = GPy.models.GPCoregionalizedRegression(X_list=X_list, Y_list=Y_list)
# # m.randomize()
# # m.optimize('bfgs', max_iters=100, messages=True)
#
# m = GPCoregionalizedWithMeanRegression(X_list, Y_list,
#                                        kernel=K, likelihoods_list=likelihoods_list, mean_function=mf)
# m.randomize()
# m.optimize('bfgs', max_iters=500, messages=True)
#
# mu_gp_test, sigma_gp_test = m.predict(Xtest,
#                                       full_cov=True, Y_metadata={'output_index': output_index})
# print("mu_gp_test :::", mu_gp_test)
#
# for m in range(z_list.shape[0]):
#     gpreps.store_simulated_data(z_list[m], w_list[m], mu_gp_test[m])
# print("memory :::", gpreps.memory_simulated)
#
# context_list = []
# param_list = []
# reward_list = []
# for i in range(20):
#     context, param, reward = gpreps.memory_simulated[-i]
#     context_list.append(np.array(context, copy=False))
#     param_list.append((np.array(param[0], copy=False) - stiffness_low)/stiffness_range)
#     reward_list.append(np.array(reward[0], copy=False) * 10)
#
# data_input_second = np.hstack((context_list, param_list, reward_list))
#
# # gpreps.learn(training_type='Simulated', eps=0.4)
#
# # print("true reward :::", Ytest)
# # print("Error :::", (mu_gp_test - Ytest[:, None]))
# # print("error ::", np.linalg.norm(mu_gp_test - Ytest[:, None])/100)
#
# X_obs = data_input_second[:, :10]
# Y_obs = data_input_second[:, 10]
#
# # Train data for GPR
# X_obs_list = [np.hstack((X_obs, X_obs)) for i in range(output_dim)]
# Y_obs_list = [Y_obs[:, None] for i in range(output_dim)]
# nb_posterior_samples = 20
#
# # # GPR posterior -> new points observed (the training points are discarded as they are "included" in the GMM)
# m_obs = GPCoregionalizedWithMeanRegression(X_obs_list, Y_obs_list, kernel=K, likelihoods_list=likelihoods_list, mean_function=mf)
# mu_posterior_tmp = m_obs.posterior_samples_f(Xtest, full_cov=True, size=nb_posterior_samples)
#
# mu_posterior = []
# for i in range(nb_posterior_samples):
#     mu_posterior.append(np.reshape(mu_posterior_tmp[:, 0, i], (output_dim, -1)))
#
# # GPR prediction
# mu_gp, sigma_gp = m_obs.predict(Xtest, full_cov=True, Y_metadata={'output_index': output_index})
# mu_gp_rshp = np.reshape(mu_gp, (output_dim, -1)).T
#
# print("mu_gp :::", mu_gp)
#
# sigma_gp_tmp = np.zeros((nb_data_test, nb_data_test, output_dim * output_dim))
# for i in range(output_dim):
#     for j in range(output_dim):
#         sigma_gp_tmp[:, :, i * output_dim + j] = sigma_gp[i * nb_data_test:(i + 1) * nb_data_test,
#                                                  j * nb_data_test:(j + 1) * nb_data_test]
# sigma_gp_rshp = np.zeros((nb_data_test, output_dim, output_dim))
# for i in range(nb_data_test):
#     sigma_gp_rshp[i] = np.reshape(sigma_gp_tmp[i, i, :], (output_dim, output_dim))
