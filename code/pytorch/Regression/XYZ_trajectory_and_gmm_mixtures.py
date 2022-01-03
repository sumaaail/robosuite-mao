import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat  # loading data from matlab
from utils.gmr import Gmr, plot_gmm

# GMR on 2D trajectories with time as input
if __name__ == '__main__':
    initial_pos = np.array([1448.94, 17.4063, 1001.44, 179.75, 0.7, -1.33])
    # pos_bound = np.array([0.8768, 0.5289, 30.3364, 0.1318, 0.2264, 0.7430])
    # force_bound = np.array([33.6377, 48.2733, 86.669, 4.0921, 4.4257, 3.0208])
    # initial_pos = np.array([0,0,0,0,0,0])
    pos_bound = np.array([1,1,1,1,1,1])
    force_bound = np.array([1,1,1,1,1,1])

    # GMM train
    data = np.load('E:\THUME-SZIGS\RL//rl-robotic-assembly-mujoco-master-new//results//runs//real//tie_hrl_new_peg\TD3_dual-peg-in-hole_seed_0//test_states_with_labels_init_.npy')[2000:4000]
    force = data[:, 1:7] / force_bound
    pos = (data[:, 7:13]) / pos_bound
    step = data[:, 0:1]
    X = pos
    Y = force
    data = np.hstack([step, pos[:, :6]])

    # Parameters
    nb_data_sup = 50
    nb_samples = 2
    dt = 0.01
    input_dim = 1
    output_dim = 6
    in_idx = [0]
    out_idx = [1, 2, 3, 4, 5, 6]
    nb_states = 9

    # GMM
    gmr_model = Gmr(nb_states=nb_states, nb_dim=input_dim + output_dim, in_idx=in_idx, out_idx=out_idx)
    gmr_model.init_params_kbins(data.T, nb_samples=nb_samples)
    gmr_model.gmm_em(data.T)

    # GM Regression
    Xt = []
    for i in range(150):
        Xt.append(np.array([i]))
    Xt = np.array(Xt)
    print(np.shape(Xt))

    mu_gmr = []
    sigma_gmr = []
    for i in range(Xt.shape[0]):
        mu_gmr_tmp, sigma_gmr_tmp, H_tmp = gmr_model.gmr_predict(Xt[i])
        mu_gmr.append(mu_gmr_tmp)
        sigma_gmr.append(sigma_gmr_tmp)

    mu_gmr = np.array(mu_gmr)
    sigma_gmr = np.array(sigma_gmr)
    print(np.shape(sigma_gmr))

    # test and plot
    YLABEL = ['X/mm', 'Y/mm', 'Z/mm', r'$\theta$x/rad', r'$\theta$y/rad', r'$\theta$z/rad']
    # YLABEL = ['Fx/N', 'Fy/N', 'Fz/N', 'Tx/Nm', 'Ty/Nm', 'Tz/Nm']

    plt.figure(figsize=(14, 7), dpi=100)
    plt.tight_layout(pad=3, w_pad=1., h_pad=0.5)
    plt.subplots_adjust(left=0.12, bottom=0.10, right=0.95, top=0.96, wspace=0.32, hspace=0.26)
    for index in range(6):
        n = index
        plt.subplot(2, 3, index + 1)

        # plot testing trajectory
        s = 0
        for i in range(12):
            start_num = s
            s += 1
            while data[s][0] != 0:
                s += 1
            if s <= start_num + 150:
                end_num = s
            else:
                end_num = start_num + 150
            plt.plot(data[start_num: end_num, 0:1].transpose()[0], data[start_num: end_num, n+1:n+2].transpose()[0], linewidth=0.8, alpha=0.15, color='gray')


        # plot GMR trajectory
        plt.plot(Xt.transpose()[0], mu_gmr[:, n:n+1].transpose()[0], label='im_action --> pos', linewidth=2, alpha=0.6, color='r')

        # plot confidence zone
        sigma = []
        for i in range(len(Xt)):
            sigma.append([])
            for j in range(6):
                sigma[i].append(np.sqrt(sigma_gmr[i][j, j]))
        sigma = np.array(sigma)

        miny = mu_gmr[:, n].transpose() - sigma[:, n].transpose()
        maxy = mu_gmr[:, n].transpose() + sigma[:, n].transpose()
        plt.fill_between(Xt.transpose()[0], miny, maxy, color=[0.83, 0.06, 0.06], alpha=0.15)

        # plot gm mixture distribution
        mu = np.array(gmr_model.mu)
        mu = np.hstack([mu[:, :1], mu[:, n+1:n+2]])
        gmr_model.sigma = np.array(gmr_model.sigma)
        print(np.shape(gmr_model.sigma))
        SIGMA = []
        for i in range(nb_states):
            sigma_ = np.zeros([2, 2])
            sigma_[0][0] = gmr_model.sigma[i][0][0]
            sigma_[0][1] = gmr_model.sigma[i][0][n+1]
            sigma_[1][0] = gmr_model.sigma[i][n+1][0]
            sigma_[1][1] = gmr_model.sigma[i][n+1][n+1]
            SIGMA.append(sigma_)
        SIGMA = np.array(SIGMA)

        print(np.shape(SIGMA))
        print(np.shape(mu))
        plot_gmm(mu, SIGMA, linewidth=0.5, alpha=0.24, color=[0.1, 0.34, 0.76])
        plt.xlabel('Steps', fontsize=16)
        plt.ylabel(YLABEL[index], fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid()
    plt.show()