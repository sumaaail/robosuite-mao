import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat  # loading data from matlab
from utils.gmr import Gmr, plot_gmm

# GMR on 2D trajectories with time as input
if __name__ == '__main__':
    # initial_pos = np.array([1448.94, 17.4063, 1001.44, 179.75, 0.7, -1.33])
    # pos_bound = np.array([0.8768, 0.5289, 30.3364, 0.1318, 0.2264, 0.7430])
    # force_bound = np.array([33.6377, 48.2733, 86.669, 4.0921, 4.4257, 3.0208])
    initial_pos = np.array([0,0,0,0,0,0])
    pos_bound = np.array([1,1,1,1,1,1])
    force_bound = np.array([1,1,1,1,1,1])

    # GMM train
    data = np.load('E:\THUME-SZIGS\RL//rl-robotic-assembly-mujoco-master-new//results//runs//real//tie_hrl_new_peg\TD3_dual-peg-in-hole_seed_0//test_states_with_labels_.npy')[:2000, 1:]
    force = data[:, :6] / force_bound
    pos = (data[:, 6:] - initial_pos) / pos_bound
    X = pos
    Y = force
    data = np.hstack([pos, force])

    # test data
    data2 = np.load('E:\THUME-SZIGS\RL//rl-robotic-assembly-mujoco-master-new//results//runs//real//tie_hrl_new_peg\TD3_dual-peg-in-hole_seed_0//test_states_with_labels_.npy')[:140, 1:]
    force2 = data2[:, :6] / force_bound
    pos2 = (data2[:, 6:] - initial_pos) / pos_bound
    data2 = np.hstack([pos2, force2])

    # Parameters
    nb_data_sup = 50
    nb_samples = 2
    dt = 0.01
    input_dim = 6
    output_dim = 6
    in_idx = [0, 1, 2, 3, 4, 5]
    out_idx = [6, 7, 8, 9, 10, 11]
    nb_states = 10

    # Test data
    Xt = pos2

    # GMM
    gmr_model = Gmr(nb_states=nb_states, nb_dim=input_dim + output_dim, in_idx=in_idx, out_idx=out_idx)
    gmr_model.init_params_kbins(data.T, nb_samples=nb_samples)
    gmr_model.gmm_em(data.T)

    # GMR
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
    YLABEL = ['Fx/N', 'Fy/N', 'Fz/N', 'Tx/Nm', 'Ty/Nm', 'Tz/Nm']
    plt.figure(figsize=(14, 7), dpi=100)
    plt.tight_layout(pad=3, w_pad=1., h_pad=0.5)
    plt.subplots_adjust(left=0.12, bottom=0.10, right=0.95, top=0.96, wspace=0.32, hspace=0.26)
    for m in range(6):
        plt.subplot(2, 3, m + 1)
        err = 0
        sum = 0
        sigma = []
        for i in range(len(data2)):
            mu_gmr[i] = mu_gmr[i] * force_bound
            force2[i] = force2[i] * force_bound
            err = err + abs(mu_gmr[i][m] - force2[i][m])
            sum = sum + abs(force2[i][m])
            sigma.append([])
            for j in range(6):
                sigma[i].append(np.sqrt(sigma_gmr[i][j, j]))
        sigma = np.array(sigma) * force_bound
        print('Overall accuracy', (1-err / sum) * 100)

        # plot reference data
        Step = range(len(Xt))
        Fx = force2[:, m].transpose()
        Fy = mu_gmr[:, m].transpose()
        plt.plot(Step, Fx)
        plt.plot(Step, Fy)
        miny = Fy - sigma[:, m].transpose()
        maxy = Fy + sigma[:, m].transpose()
        plt.fill_between(Step, miny, maxy, color=[0.83, 0.06, 0.06], alpha=0.3)
        plt.xlabel('Steps', fontsize=15)
        plt.ylabel(YLABEL[m], fontsize=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
    plt.show()