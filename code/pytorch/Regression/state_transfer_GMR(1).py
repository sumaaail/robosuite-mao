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
    # action_bound = np.array([0.1416, 0.0956, 0.5962, 0.0723, 0.0796, 0.0492])
    # dpos_bound = np.array([0.0813, 0.0727, 0.4950, 0.0892, 0.0902, 0.0043])
    action_bound = np.array([1,1,1,1,1,1])
    dpos_bound = np.array([1,1,1,1,1,1])

    # GMM train
    # train data
    state_ = np.load('E:\THUME-SZIGS\RL//rl-robotic-assembly-mujoco-master-new//results//runs//real//tie_hrl_new_peg\TD3_dual-peg-in-hole_seed_0//test_states_with_labels_init_.npy')[:2000, 1:]
    force_ = state_[:, :6]
    pos_ = state_[:, 6:] - initial_pos
    dpos_ = []
    for i in range(len(pos_) - 1):
        dpos_.append(pos_[i + 1] - pos_[i])
    dpos_ = np.array(dpos_) / dpos_bound
    action_ = np.load('E:\THUME-SZIGS\RL//rl-robotic-assembly-mujoco-master-new//results//runs//real//tie_hrl_new_peg\TD3_dual-peg-in-hole_seed_0//test_im_actions_.npy')[:2000]
    action_ = action_[1:, :] / action_bound
    data_ = np.hstack([action_, dpos_])

    # test data
    state = np.load('E:\THUME-SZIGS\RL//rl-robotic-assembly-mujoco-master-new//results//runs//real//tie_hrl_new_peg\TD3_dual-peg-in-hole_seed_0//test_states_with_labels_init_.npy')[:150, 1:]
    force = state[:, :6]
    pos = state[:, 6:] - initial_pos
    dpos = []
    for i in range(len(pos) - 1):
        dpos.append(pos[i + 1] - pos[i])
    dpos = np.array(dpos) / dpos_bound
    action = np.load('E:\THUME-SZIGS\RL//rl-robotic-assembly-mujoco-master-new//results//runs//real//tie_hrl_new_peg\TD3_dual-peg-in-hole_seed_0//test_im_actions_.npy')[:150]
    action = action[1:, :] / action_bound
    data = np.hstack([action, dpos])
    print('testing data size:', len(data))

    # Parameters
    nb_data_sup = 50
    nb_samples = 2
    dt = 0.01
    input_dim = 6
    output_dim = 6
    in_idx = [0, 1, 2, 3, 4, 5]
    out_idx = [6, 7, 8, 9, 10, 11]
    nb_states = 12

    # Test data
    Xt = action

    # GMM
    gmr_model = Gmr(nb_states=nb_states, nb_dim=input_dim + output_dim, in_idx=in_idx, out_idx=out_idx)
    gmr_model.init_params_kbins(data_.T, nb_samples=nb_samples)
    gmr_model.gmm_em(data_.T)

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
    m = 0  # plotting dimension
    err = 0
    sum = 0
    sigma = []

    for i in range(len(data)):
        err = err + abs(mu_gmr[i][m] - dpos[i][m])
        sum = sum + abs(dpos[i][m])
        sigma.append([])
        for j in range(6):
            sigma[i].append(np.sqrt(sigma_gmr[i][j, j]))
    sigma = np.array(sigma) * dpos_bound
    dpos = dpos * dpos_bound
    mu_gmr = mu_gmr * dpos_bound
    print('Overall accuracy', (1 - err / sum) * 100)

    # plot reference data
    Step = range(len(data))
    Fx = dpos[:, m].transpose()
    Fy = mu_gmr[:, m].transpose()
    plt.plot(Step, Fx)
    plt.plot(Step, Fy)
    miny = Fy - sigma[:, m].transpose()
    maxy = Fy + sigma[:, m].transpose()
    plt.fill_between(Step, miny, maxy, color=[0.83, 0.06, 0.06], alpha=0.2)
    plt.xlabel('Steps', fontsize=15)
    plt.ylabel('Value (force/torque)', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    # test and plot 2: plot dpos trajectory and cumulative trajectory in all 6 dimensions
    n = len(data)
    # plot dpos results
    for i in range(6):
        Step = range(n)
        Fx = dpos[:n, i].transpose()
        Fy = mu_gmr[:n, i].transpose()
        plt.plot(Step, Fx)
        plt.plot(Step, Fy)
        plt.xlabel('Steps', fontsize=15)
        plt.ylabel('dPos', fontsize=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()

    # plot trajectory results
    trajectory = [pos[0]]
    for i in range(len(mu_gmr)):
        trajectory.append(trajectory[i] + mu_gmr[i])
    trajectory = np.array(trajectory)
    for i in range(6):
        Step = range(n)
        Fx = pos[:n, i].transpose()
        Fy = trajectory[:n, i].transpose()
        plt.plot(Step, Fx)
        plt.plot(Step, Fy)
        plt.xlabel('Steps', fontsize=15)
        plt.ylabel('POS', fontsize=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()