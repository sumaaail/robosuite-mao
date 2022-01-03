import math
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn import mixture
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

from code.pytorch.Regression.GMRbasedGP.utils.gmr import Gmr


class GPREPS(object):
    def __init__(self,
                 z_dim,
                 w_dim,
                 memory_dim,
                 w_lower_bound,
                 w_upper_bound,
                 initial_a,
                 eps
                 ):
        
        # initialize parameters
        self.memory = None
        self.memory_simulated = []
        self.memory_realistic = []
        
        # used for exploration
        self.eps = eps
        self.min_eta = 1e-10
        self.context_list = None
        self.R_list = None
        self.w_list = None
        self.P_list = None
        
        self.pointer = 0
        self.w_dim, self.z_dim, self.memory_dim, self.w_lower_bound, self.w_upper_bound = \
            w_dim, z_dim, memory_dim, w_lower_bound, w_upper_bound
        self.w_range = np.array(self.w_upper_bound) - np.array(self.w_lower_bound)

        self.reward_scale = 1.0
        self.a = initial_a
        
        # Parameters vector (n_context_features, w_dim)
        self.A = np.zeros((self.z_dim, self.w_dim), dtype=np.float32)
        
        # Covariance matrix (w_dim, w_dim)
        self.COV = np.eye((self.w_dim), dtype=np.float32)
        
    def choose_action(self, z):
        """
            choose impedance parameters
        """
        # Impedance parameter::: n_samples x impedance_weights
        w = np.zeros((z.shape[0], self.w_dim))
        
        # shape ::: n_samples x impedance_weights
        mean_w = self.a + np.dot(z, self.A)
        
        for i in range(z.shape[0]):
            w[i, :] = np.clip(np.random.multivariate_normal(mean=mean_w[i, :], cov=self.COV, size=1),
                              self.w_lower_bound, self.w_upper_bound)
        
        return w
    
    def dual_function(self, x):
        """
            Dual function
        """
        eta = x[0]
        theta = x[1:].reshape(-1, 1)
        
        context_mean = self.context_list.mean(0).reshape(1, -1)
        R_over_eta = (self.R_list - self.context_list.dot(theta))/eta
        R_over_eta_max = R_over_eta.max()
        Z = np.exp(R_over_eta - R_over_eta_max).T
        Z_sum = Z.sum()
        
        log_sum_exp = R_over_eta_max + np.log(Z_sum / self.context_list.shape[0])
        
        F = eta * (self.eps + log_sum_exp) + context_mean.dot(theta)
        
        d_eta = self.eps + log_sum_exp - (Z.dot(R_over_eta)/Z_sum)
        d_theta = context_mean - (Z.dot(self.context_list) / Z_sum)
        return F, np.append(d_eta, d_theta)
    
    def update(self):
        """
            update gaussian parameters
        """
        # context vector
        S = np.concatenate((np.ones((self.context_list.shape[0], 1)), self.context_list), axis=1)
        P = np.diag(self.P_list)
        
        # compute new mean
        new_A = np.linalg.inv(np.dot(np.dot(S.T, P), S) + 1e-6 * np.eye(S.shape[1])).dot(S.T).dot(P).dot(self.w_list)
        
        # update new contextual policy parameters
        self.a = new_A[0, :].reshape(1, -1)
        self.A = new_A[1:, :]
        self.COV = (P.dot(self.w_list - self.a)).T.dot(self.w_list - self.a)
        
    def learn(self, training_type='Realistic', N_samples=1000, eps=0.25):
        """
            optimize parameters
        """
        
        # reset exploration parameter
        self.eps = eps
        self.memory = self.memory_realistic
        
        # shape ::: n_samples X impedance_weights
        if training_type == 'Realistic':
            self.memory = self.memory_realistic
        
        if training_type == 'Simulated':
            # model-based learning with simulated data
            self.memory = self.memory_simulated
        
        if len(self.memory) < N_samples:
            N_samples = len(self.memory)
        
        print('Number of Training Samples :::', N_samples)
        
        self.context_list = np.zeros((N_samples, self.z_dim))
        self.w_list = np.zeros((N_samples, self.w_dim))
        self.R_list = np.zeros((N_samples, 1))
        
        for i in range(N_samples):
            self.context_list[i, :] = self.memory[-i][0]
            self.w_list[i, :] = self.memory[-i][1]
            self.R_list[i, :] = self.memory[-i][2]
        
        # print("context_list :::", self.context_list)
        # print("w_list :::", self.w_list)
        # print("R_list :::", self.R_list)
        
        # initial param
        x0 = np.ones(1 + self.context_list.shape[1])
        
        # min bounds
        bds = np.vstack(([[self.min_eta, None]], np.tile(None, (self.context_list.shape[1], 2))))
        x = fmin_l_bfgs_b(self.dual_function, x0=x0, bounds=bds)[0]

        # min eta and theta
        eta = x[0]
        theta = x[1:].reshape(-1, 1)
        
        R_baseline_eta = (self.R_list - self.context_list.dot(theta)) / eta
        P = np.exp(R_baseline_eta - R_baseline_eta.max())
        P /= P.sum()
        self.P_list = P.reshape(-1, )

        self.update()

    def store_simulated_data(self, z, w, r):
        """
            add simulated transition
        """
        transition = [z, w, r]
        if len(self.memory_simulated) == self.memory_dim:
            index = self.pointer % self.memory_dim
            self.memory_simulated[index] = transition
        else:
            self.memory_simulated.append(transition)
        self.pointer += 1

    def store_realistic_data(self, z, w, r):
        """
            add realistic transition
        """
        transition = [z, w, r]
        if len(self.memory_realistic) == self.memory_dim:
            index = self.pointer % self.memory_dim
            self.memory_realistic[index] = transition
        else:
            self.memory_realistic.append(transition)
        self.pointer += 1
    
    def train_reward_model(self, N_samples=50, type='GP'):
        """
            extract reward from replay buffer
        """
        print("===================== Fit Reward Model !!! ====================")
        context_list, param_list, reward_list = [], [], []
        if len(self.memory_realistic) < N_samples:
            N_samples = len(self.memory_realistic)
        
        for i in range(N_samples):
            context, param, reward = self.memory_realistic[-i]
            context_list.append(np.array(context, copy=False))
            param_list.append(np.array(param, copy=False))
            reward_list.append(np.array(reward, copy=False))
            # print("context_list ::::", context_list)
            # print("param_list ::::", param_list)
        # print("context_list ::::", context_list)
        # print("param_list ::::", param_list)
        # print("reward_list ::::", reward_list)
        input_state_list = np.hstack([context_list, param_list])

        # # ========================================== GMR =================================================
        # nb_states = 10
        # nb_samples = 2
        # input_dim = 10
        # output_dim = 1
        # in_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # out_idx = [10]
        # self.gmr_model = Gmr(nb_states=nb_states, nb_dim=input_dim + output_dim, in_idx=in_idx, out_idx=out_idx)
        # self.gmr_model.init_params_kbins(data_input.T, nb_samples=nb_samples)
        # self.gmr_model.gmm_em(data_input.T)
        
        if type == 'GP':
            self.gp = GaussianProcessRegressor()
        
        self.gp.fit(input_state_list.reshape(N_samples, -1), np.array(reward_list).reshape(N_samples, -1))
        # self.mu, self.sigma = gp.predict(self.X_grid, return_std=True)
        
    def train_context_model(self, N_samples=1000, N_components=16):
        """
            context model
        """
        print("===================== Fit Context Model !!! ====================")
        if len(self.memory_realistic) < N_samples:
            N_samples = len(self.memory_realistic)
        context_list = []
        for i in range(N_samples):
            context, param, reward = self.memory_realistic[-i]
            context_list.append(np.array(context, copy=False))

        input_state_list = context_list
        self.gmm_context = mixture.GaussianMixture(n_components=N_components,
                                                   covariance_type='full', random_state=0).fit(input_state_list)
        
        return self.gmm_context

    def train_dynamics_model(self, buffer, model_fit_size):
        print("===================== Fit Dynamic Model !!! ====================")
        # with the difference trick for output
    
        state_list, next_state_list, action_list, reward_list = buffer.sample_on_policy(model_fit_size, model_fit_size)
        input = np.concatenate((state_list, action_list))
        output = next_state_list - state_list
        kernel = DotProduct() + WhiteKernel()
        
        self.gpr_transition_model = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(input, output)
        self.gpr_reward_model = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(input, reward_list)
    
    def sample_context(self, N_samples=1):
        """
            sample context from context model
        """
        sampled_context = self.gmm_context.sample(n_samples=N_samples)
        return sampled_context[0]

    def generate_artificial_reward(self, z, w):
        """
            Simulate reward function
        """
        input_value = np.hstack([z, w])
        sample_reward = self.gp.sample_y(input_value, n_samples=1, random_state=0)
        return sample_reward
    
    def param_normalization(self, param_list):
        """
            Normalize the params
        """
        return (param_list - self.w_lower_bound)/self.w_range
    
    def context_normalization(self, context_list):
        """
            Normalize the params
        """
        return context_list
    
    def reward_normalization(self, reward_list):
        """
            Normalize the params
        """
        return reward_list * self.reward_scale

    def train_seperate_model(self):
        nb_samples = 2
        input_dim = 12
        output_dim = 6
        in_idx_state = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        out_idx_state = [12, 13, 14, 15, 16, 17]
        in_idx_force = [0, 1, 2, 3, 4, 5]
        out_idx_force = [6, 7, 8, 9, 10, 11]
        nb_states = 12
    
        'data'
        if len(state_) >= 6000:
            state_ = np.array(state_)[-6000:]
            action_ = np.array(action_)[-6000:]
        else:
            state_ = np.array(state_)
            action_ = np.array(action_)
        print('data size', np.shape(state_), np.shape(action_))
    
        'stage 1'
        state = []
        dpos = []
        action = []
        for i in range(len(state_) - 1):
            if state_[i + 1][0] != 0:
                if state_[i][9] >= -5:
                    state.append(state_[i][1:])
                    dpos.append(state_[i + 1][7:] - state_[i][7:])
                    action.append(action_[i])
    
        state = np.array(state)
        pos = state[:, 6:]
        force = state[:, :6]
    
        dpos = np.array(dpos)
        action = np.array(action)
    
        input = np.hstack([pos / self.pos_bound, action / self.action_bound])
        output = dpos / self.dpos_bound
        data_state = np.hstack([input, output])
        print(np.shape(data_state))
    
        "train gpr model stage 1"
        kernel = DotProduct() + WhiteKernel()
        self.gpr_force_model = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(pos / self.pos_bound,
                                                                                           force / self.force_bound)
        self.gpr_state_model = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(input, output)
    
        'stage 2'
        state = []
        dpos = []
        action = []
        for i in range(len(state_) - 1):
            if state_[i + 1][0] != 0:
                if -4 >= state_[i][9] >= -9:
                    state.append(state_[i][1:])
                    dpos.append(state_[i + 1][7:] - state_[i][7:])
                    action.append(action_[i])
    
        state = np.array(state)
        pos = state[:, 6:]
        force = state[:, :6]
    
        dpos = np.array(dpos)
        action = np.array(action)
        input = np.hstack([pos / self.pos_bound, action / self.action_bound])
        output = dpos / self.dpos_bound
        data_state = np.hstack([input, output])
    
        "train gpr model stage 2"
        kernel = DotProduct() + WhiteKernel()
        self.gpr_force_model_ = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(pos / self.pos_bound,
                                                                                            force / self.force_bound)
        self.gpr_state_model_ = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(input, output)
    
        'stage 3'
        state = []
        dpos = []
        action = []
        for i in range(len(state_) - 1):
            if state_[i + 1][0] != 0:
                if state_[i][9] <= -4:
                    state.append(state_[i][1:])
                    dpos.append(state_[i + 1][7:] - state_[i][7:])
                    action.append(action_[i])
        state = np.array(state)
        pos = state[:, 6:]
        force = state[:, :6]
        data_force = np.hstack([pos, force]) / np.hstack([self.pos_bound, self.force_bound])
    
        dpos = np.array(dpos)
        action = np.array(action)
        input = np.hstack([pos / self.pos_bound, action / self.action_bound])
        output = dpos / self.dpos_bound
        data_state = np.hstack([input, output])
    
        "train gpr model stage 3"
        self.gmr_force_model = Gmr(nb_states=nb_states, nb_dim=12, in_idx=in_idx_force, out_idx=out_idx_force)
        self.gmr_force_model.init_params_kbins(data_force.T, nb_samples=nb_samples)
        self.gmr_force_model.gmm_em(data_force.T)
        self.gmr_state_model = Gmr(nb_states=nb_states, nb_dim=18, in_idx=in_idx_state, out_idx=out_idx_state)
        self.gmr_state_model.init_params_kbins(data_state.T, nb_samples=nb_samples)
        self.gmr_state_model.gmm_em(data_state.T)
        
    def artificial_trajectory(self, L, w_bound):
        
        state = 0
        t = 0
        for i in range(len(State)):
            if State[i][0] == 0:
                state += State[i][1:]
                t += 1
        state_init = state / t
        
        print('Find initial state:', state_init)

        REWARD = []
        State = []
        cal = calculation()
        
        # contextual parameters
        z = np.array([0, np.random.uniform(-1, 1), 0])
        cal.set_context(z)
        w = np.clip(self.choose_action(z)[0], -0.0045, 0.0045)
        cal.set_w(w)
        
        l = 0
        while l < L:
            # initialize the observation
            done = False
            safe = True
            R = 0
            var = np.array([0, 0, 0, 0, 0, 0, 0.002, 0.002, 0, 0, 0, 0])
            state = np.random.normal(state_init, var)
            cal.reset()

            " start a trajecory "
            for i in range(200):
                "choose action"
                # action = self.policy.select_action(np.array(obs)) * ra
                action = np.array([0, 0, 0, 0, 0, 0])
                im_action = cal.step(state, action, i)

                "step"
                'calculate dpos'
                if state[8] >= -4.4:
                    dpos, _ = self.gpr_state_model.predict([np.hstack([state[6:] / self.pos_bound, im_action / self.action_bound])],
                                                      return_std=1, return_cov=0)  # GPR needed
                    dpos = dpos[0]  # GPR needed
                    dpos *= self.dpos_bound
                elif state[8] >= -6:
                    dpos, _ = self.gpr_state_model_.predict([np.hstack([state[6:] / self.pos_bound, im_action / self.action_bound])],
                                                       return_std=1, return_cov=0)  # GPR needed
                    dpos = dpos[0]  # GPR needed
                    dpos *= self.dpos_bound
                else:
                    'gmr state regression'
                    dpos, cov_dpos, _ = self.gmr_state_model.gmr_predict(
                        np.hstack([state[6:] / self.pos_bound, im_action / self.action_bound]))
                    dpos *= self.dpos_bound
                    cov_dpos = cov_dpos * self.dpos_bound ** 2 / 16
                    dpos = np.random.multivariate_normal(mean=dpos, cov=cov_dpos, size=1)[0]

                'problem check'
                for j in range(6):
                    if dpos[j] == 0:
                        safe = False
                if not safe:
                    break

                'calculate pos'
                pos = state[6:] + dpos

                'calculate force'
                if state[8] >= -4.4:
                    force, _ = self.gpr_force_model.predict([pos / self.pos_bound], return_std=1, return_cov=0)  # GPR needed
                    force = force[0]  # GPR needed
                    force *= self.force_bound
                elif state[8] >= -6:
                    force, _ = self.gpr_force_model_.predict([pos / self.pos_bound], return_std=1, return_cov=0)  # GPR needed
                    force = force[0]  # GPR needed
                    force *= self.force_bound
                else:
                    'gme force regression'
                    force, cov_force, _ = self.gmr_force_model.gmr_predict(pos / self.pos_bound)
                    force *= self.force_bound
                    cov_force = cov_force * self.force_bound ** 2 / 16
                    force = np.random.multivariate_normal(mean=force, cov=cov_force, size=1)[0]

                'calculate new state'
                new_state = np.hstack([force, pos])

                reward, done, safe = cal.get_reward(state, new_state, i)
                # print("-------------- step", i + 1, "----------------", 'done?', done, 'safe?', safe, 'reward', R)

                'update parameters'
                state = new_state.copy()
                R += reward

                if done or not safe:
                    if i == 199:
                        REWARD.append([z[1], w, i, R, 'Unfinished'])
                        print('trajectory episode', l, 'step', i, 'reward', R, 'Unfinished')
                    elif safe == False:
                        REWARD.append([z[1], w, i, R, 'Failed'])
                        print('trajectory episode', l, 'step', i, 'reward', R, 'Failed')
                    else:
                        REWARD.append([z[1], w, i, R, 'Successful'])
                        print('trajectory episode', l, 'step', i, 'reward', R, 'Successful')
                    self.store_data(z, w / w_bound, R)
                    l += 1
                    break


class R_MODEL(object):
    def __init__(self, policy, env, context_dim, pd_dim, observation_dim, action_dim, max_step):
        self.env = env
        self.policy = policy
        self.context_dim = context_dim
        self.pd_dim = pd_dim
        self.observation_dim = observation_dim  # single-hole assembly 中为21，分别为6+6+6+3
        self.action_dim = action_dim  # DDPG 输出动作的维度，此任务中为6，分别为Px, Py, Pz, Ox, Oy, Oz
        self.GM_conponemts_force = 13
        self.GM_conponemts_state = 5
        self.MAX_EP_STEPS = max_step
    
    def train_reward_model(self, replay_buffer):
        memory = replay_buffer.storage
        X, Y, A = [], [], []
        for i in range(len(memory)):
            x, y, a = memory[i]
            X.append(np.array(x, copy=False))
            Y.append(np.array(y, copy=False))
            A.append(np.array(a, copy=False))
        state1 = np.array(X)  # 前六位是力，后6-15位是位置
        state2 = np.array(Y)
        action = np.array(A)
        force1 = state1[:, :6]
        pos1 = state1[:, 6:]  # 一维6-15位
        force2 = state2[:, :6]
        pos2 = state2[:, 6:]
        
        # train force model
        data = np.hstack([pos1, force1])
        self.gmm_force = mixture.GaussianMixture(n_components=self.GM_conponemts_force, covariance_type='full').fit(
            data)
        
        # train state transfer model
        data = np.hstack([action, pos2 - pos1])
        self.gmm_state = mixture.GaussianMixture(n_components=self.GM_conponemts_state, covariance_type='full').fit(
            data)
    
    def gmm_regression(self, type, x):  # 输入一维数组state/action，计算返回估计值
        if type == 'force':
            x_dim = len(x)  # 位置为一维6-15位
            y_dim = 6  # 力为一维6位
            GM_conponets = self.GM_conponemts_force
            gmm = self.gmm_force
        else:  # type == 'state'
            x_dim = 6  # 动作action为6维
            y_dim = self.observation_dim - 6  # pos变化量dpos为6-15维
            GM_conponets = self.GM_conponemts_state
            gmm = self.gmm_state
        
        # Compute hl
        P = 0
        pl = []
        hl = []
        for i in range(GM_conponets):
            ux = gmm.means_[i][: x_dim]
            covx = gmm.covariances_[i][: x_dim, : x_dim]
            p = (1 / (np.linalg.det(covx) * (math.pi * 2) ** x_dim) ** 0.5) \
                * np.exp(-0.5 * np.dot(np.dot(np.array([x - ux]), np.linalg.inv(covx)),
                                       np.transpose(np.array([x - ux]))))
            pl.append(gmm.weights_[i] * p[0][0])
            P += gmm.weights_[i] * p[0][0]
        if P == 0:
            print('GMR Calculation Err: very strange input, P=0 when calculating pl[i] / P')
            y = np.array([0, 0, 0, 0, 0, 0])
            return y
        for i in range(GM_conponets):
            hl.append(pl[i] / P)
        # print(hl)
        
        # Compute yl anc covl
        yl = []
        covl = []
        for i in range(GM_conponets):
            ux = gmm.means_[i][: x_dim]
            uy = gmm.means_[i][-y_dim:]
            covx = gmm.covariances_[i][: x_dim, : x_dim]
            covy = gmm.covariances_[i][-y_dim:, -y_dim:]
            covyx = gmm.covariances_[i][-y_dim:, : x_dim]
            covxy = gmm.covariances_[i][: x_dim, -y_dim:]
            yi = uy + np.transpose(np.dot(np.dot(covyx, np.linalg.inv(covx)), np.transpose(np.array([x - ux]))))
            yl.append(yi)
            covi = covy - np.dot(np.dot(covyx, np.linalg.inv(covx)), covxy)
            covl.append(covi)
        
        # Compute y = f(x)
        y = np.zeros([1, y_dim])
        covy = np.zeros([y_dim, y_dim])
        for i in range(GM_conponets):
            y += hl[i] * yl[i]
            covy += hl[i] * (covl[i] + np.dot(np.transpose(yl[i]), yl[i]))
        covy = covy - np.dot(np.transpose(y), y)
        y = y[0]
        return y
    
    def trajectory(self, z, w):
        # initialize the observation
        done = False
        safe = True
        R = 0
        obs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0.0600, 0, 0.00011, 0])
        var = np.array([0, 0, 0, 0, 0, 0, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001])
        obs = np.random.normal(obs, var)
        
        # set the impedance parameters
        kp = w[:, :6][0]
        kd = w[:, 6:][0]
        cal.set_pd(kd, kp)
        
        # set the context parameters
        ra = z[0]  # racial for assembly speed
        rd = z[1]  # racial for assembly depth
        
        # start a trajecory
        for i in range(self.MAX_EP_STEPS):
            # choose action
            action = self.policy.select_action(np.array(obs)) * ra
            im_action = cal.actions(obs, action, True)
            
            # step
            dpos = self.gmm_regression('state', im_action)
            new_pos = dpos + obs[6:]
            new_force = self.gmm_regression('force', new_pos)
            new_obs = np.hstack([new_force, new_pos])
            
            # update parameters
            reward = 0.01
            obs = new_obs
            R += reward
            
            # done or safe？
            if new_obs[8] <= 0.060 - 0.058 * rd:
                done = True
            else:
                done = False
            for j in range(6):
                if obs[j + 6] > 100:
                    safe = False
            if done or not safe or i >= self.MAX_EP_STEPS - 1:
                print('trajectory episode ', 'step', i, 'done?', done, 'safe?', safe, 'reward', R)
                break
        return R
    
    # eta, theta = argmin(self.memory, z_, self.z_dim)
    # print("eta ::: first", eta)
    # print("theta ::: first", theta)
    #
    # p = 0.
    # P_ = []
    # Z = []
    # B = []
    # for i in range(len(self.memory)):
    #     z, w, r = self.memory[i]
    #     z = np.array([z])
    #     w = np.array(w)
    #     r = np.array(r)
    #     p = np.exp((r - np.dot(z, theta)) / eta)
    #     print("z * theta :::", np.dot(z, theta))
    #     print("z :::", z)
    #     print("p ::::", p)
    #     z_ = np.c_[np.array([1.]), z]
    #     print("z_ :::", z_[0])
    #     Z.append(z_[0])
    #     B.append(w)
    #     P_.append(p[0])
    #
    # P_, B, Z = np.array(P_), np.array(B), np.array(Z)
    # print("P_ shape :::", P_.shape)
    # print("B shape :::", B.shape)
    # print("Z shape :::", Z.shape)
    # P = P_ * np.eye(len(self.memory))
    # print("P matrix :::", P)
    #
    # """
    #     calculate mean action
    # """
    # shape = np.dot(np.dot(Z.transpose(), P), Z).shape
    # # print("matrix shape :::", np.dot(np.dot(Z.transpose(), P), Z).shape) + 1e-6 * np.eye(shape[0]
    # target1 = np.linalg.inv(np.dot(np.dot(Z.transpose(), P), Z) + 1e-6 * np.eye(shape[0]))
    # # print(np.shape(P), np.shape(Z.transpose()), np.shape(B))
    # target2 = np.dot(np.dot(Z.transpose(), P), B)
    # target = np.dot(target1, target2).transpose()
    # self.a = target[:, :1]
    # self.A = target[:, 1:]
    #
    # # calculate the COV
    # Err = 0
    # for i in range(len(self.memory)):
    #     z, w, r = self.memory[i]
    #     z = np.array([z])
    #     w = np.array([w])
    #     err = w - self.a - np.dot(self.A, z.transpose())
    #     Err += np.dot(err, err.transpose()) * P_[i]
    # self.COV = Err / np.sum(P_) / 50
    #
    # print("a :::", self.a)
    # print("A :::", self.A)
    # print("COV :::", self.COV)
    # return P


