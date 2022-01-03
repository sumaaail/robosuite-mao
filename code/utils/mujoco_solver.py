import copy as cp
import os
import numpy as np
import torch
from ..methods import DDPG, TD3, SAC, AAC

from ..REPS.GPREPS import GPREPS

from envs.abb.models import utils

import tensorflow as tf
from tensorflow.python.summary.writer.writer import FileWriter


class Solver(object):
    def __init__(self, args, env, result_path, info_keywords):

        self.args = args
        self.env = env
        self.result_path = result_path
        self.info_keywords = info_keywords

        self.render = self.args.render

        # ############################# REPS Parameters ##########################
        self.K = self.args.num_policy_update  # 上层policy训练循环总数
        self.N = self.args.num_real_episodes  # 在上层policy的一个训练周期中，下层RL训练，改变context参数的次数
        self.n = self.args.num_average_episodes  # 下层RL训练，每改变一次context参数，执行RL的episode数
        self.d = self.args.policy_freq  # 下层RL每d个step更新一次网络参数

        self.M = self.args.num_simulated_episodes  # 在上层policy的一个训练周期中，通过model获得artificial trajectory的过程中，改变context参数的次数
        self.L = self.args.num_simulated_average_episodes  # 每改变一次context参数，通过model获得artificial trajectory的次数

        self.max_episode_steps = self.env._max_episode_steps
        self.eps_min = self.args.eps[0]
        self.eps_max = self.args.eps[1]

        self.reward_scale = self.args.reward_scale

        # definition from environment
        self.context_dim = self.env.context_dim
        self.latent_parameter_dim = self.env.latent_parameter_dim

        self.latent_parameter_low = self.env.stiffness_low
        self.latent_parameter_high = self.env.stiffness_high
        self.latent_parameter_initial = self.env.stiffness_initial

        self.context = None
        self.impedance_params = None

        # set seeds
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # ============================= lower-level RL policy =========================
        self.done = False
        self.info = None
        state_dim = self.env.observation_space.shape[0]
        print("state_dim :::", state_dim)

        action_dim = self.env.action_space.shape[0]
        print("action_dim", action_dim)

        max_action = float(self.env.action_space.high[0])
        print('action_space_high', max_action)

        if 'TD3' == args.policy_name:
            policy = TD3.TD3(state_dim, action_dim, max_action)
        elif 'DDPG' == args.policy_name:
            policy = DDPG.DDPG(state_dim, action_dim, max_action)
        elif 'SAC' == args.policy_name:
            policy = SAC.SAC(args, state_dim, action_dim, max_action, self.env.action_space)
        elif 'AAC' == args.policy_name:
            policy = AAC.AAC(state_dim, action_dim, max_action)
        else:
            policy = DDPG.DDPG(state_dim, action_dim, max_action)
            print("Please give right control algorithm !!!")

        self.policy = policy

        # =========================== High-level contextual policy =========================
        # self.gpreps = GPREPS(
        #     self.context_dim,
        #     self.latent_parameter_dim,
        #     self.args.high_memory_size,
        #     self.latent_parameter_low,
        #     self.latent_parameter_high,
        #     self.latent_parameter_initial,
        #     self.eps_max
        # )

        # ============================= Model-based learning ===============================
        self.lower_replay_buffer = utils.ReplayBuffer(self.args.low_memory_size)
        self.higher_replay_buffer = utils.ReplayBuffer(self.args.high_memory_size)

        self.total_timesteps = 0
        self.episode_timesteps = 0
        self.episode_number = 0
        self.episode_reward = 0
        self.best_reward = 0

        # ============================== Data recording ===============================
        """ training performance """
        self.training_reward = []
        self.training_time = []
        self.training_states = []
        self.training_im_actions = []

        """ evaluation performance """
        self.evaluations_actions = []
        self.evaluations_im_actions = []
        self.evaluations_states = []
        self.evaluations_options = []

        """ evaluation cps """
        self.evaluations_info_value = []
        self.evaluations_reward = []

        """ save lower-level data """
        self.log_dir = '{}/{}/{}/{}/run_{}'.format(
            self.result_path,
            self.args.log_path,
            self.args.env_name,
            self.args.policy_name,
            self.args.seed
        )

        print("---------------------------------------")
        print("Settings: %s" % self.log_dir)
        print("---------------------------------------")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.writer = FileWriter(self.log_dir)

    def reset(self):
        """
            reset for each episode
        """
        self.obs = self.env.reset()
        self.done = False
        self.info = None

        self.episode_info = dict()
        self.episode_reward = 0
        self.episode_timesteps = 0

    # def context_reset(self):
    #     """
    #         reset context and impedance
    #     """
    #     # acquire context value z from environment
    #     # print(self.env.get_context())
    #     z = np.array(self.env.get_context())
    #     self.context = z.reshape(1, self.context_dim)
    #     print("context: ", self.context)
    #
    #     # impedance parameter
    #     self.impedance_params = self.gpreps.choose_action(self.context)
    #     self.env.controller.set_params(self.impedance_params[0])
    #     print("impedance_params: ", self.impedance_params)
    #
    #     self.episode_number = 0.
    #     self.average_reward = 0.

    def train(self):
        """
            train policy
        """
        self.reset()
        while self.total_timesteps < self.args.max_timesteps:
            if self.total_timesteps % self.d == 0:
                self.train_lower_once()

            action = self.policy.select_action(np.array(self.obs))

            # noise = np.random.normal(0, self.args.expl_noise,
            # 						 size=self.env.action_space.shape[0])

            # if self.args.expl_noise != 0:
            # 	action = (action + noise).clip(
            # 		self.env.action_space.low, self.env.action_space.high
            # 	)

            new_obs, reward, self.done, self.info = self.env.step(action)

            # done_bool = 0 if self.episode_timesteps + 1 == self.max_episode_steps else float(self.done)

            for key in self.info_keywords:
                if key not in self.info:
                    break
                if key in self.episode_info:
                    self.episode_info[key].append(self.info[key])
                else:
                    self.episode_info[key] = [self.info[key]]

            if self.render:
                self.env.render()

            self.lower_replay_buffer.add((self.obs, new_obs, action, reward, float(self.done), 0))

            self.episode_reward += reward
            self.obs = new_obs
            self.episode_timesteps += 1
            self.total_timesteps += 1

            # if self.done or self.episode_timesteps + 1 == self.max_episode_steps:
            if self.done:
                self.episode_number += 1
                # self.average_reward += self.episode_reward

                summary_train_value = [
                    tf.Summary.Value(
                        tag="episode_reward",
                        simple_value=self.episode_reward
                    )
                ]
                summary_train = tf.Summary(value=summary_train_value)
                self.writer.add_summary(summary_train, self.total_timesteps)

                # Compute data summaries.
                summary_values = []
                for key, value in self.episode_info.items():
                    mean = np.mean(value)
                    std = np.std(value)
                    minimum = np.min(value)
                    maximum = np.max(value)
                    total = np.sum(value)

                    summary_values.extend([
                        tf.Summary.Value(tag="eval/" + key + "/mean", simple_value=mean),
                        tf.Summary.Value(tag="eval/" + key + "/std", simple_value=std),
                        tf.Summary.Value(tag="eval/" + key + "/min", simple_value=minimum),
                        tf.Summary.Value(tag="eval/" + key + "/max", simple_value=maximum),
                        tf.Summary.Value(tag="eval/" + key + "/sum", simple_value=total),
                        tf.Summary.Value(tag="eval/" + key + "/initial", simple_value=value[0]),
                        tf.Summary.Value(tag="eval/" + key + "/final", simple_value=value[-1]),
                    ])

                summary = tf.Summary(value=summary_values)
                self.writer.add_summary(summary, self.total_timesteps)

                self.reset()

    def train_lower_once(self):
        """
            train lower-level policy once
        """
        if self.total_timesteps > self.args.start_timesteps:
            self.policy.train(
                self.lower_replay_buffer,
                self.args.batch_size,
                self.args.discount_low,
                self.args.tau,
                self.args.policy_noise,
                self.args.noise_clip,
                self.args.policy_freq
            )

    def train_higher_once(self, k=0, mode="model_free"):
        """
            train higher-level policy once
        """
        print("train higher policy once :")
        eps_k = self.eps_max - k * (self.eps_max - self.eps_min) / self.K

        if k > 2:
            if mode == "model_based":
                """ train reward model """
                self.gpreps.train_reward_model(N_samples=self.args.model_size, type='GP')

                """ Train context model """
                self.gpreps.train_context_model(N_samples=self.args.model_size, N_components=6)

                """ Sample context """
                z_list = self.gpreps.sample_context(self.M)
                z_list = z_list.reshape(-1, self.context_dim)
                w_list = self.gpreps.choose_action(z_list).reshape(-1, self.latent_parameter_dim)
                sample_reward = self.gpreps.generate_artificial_reward(z_list, w_list)

                for m in range(self.M):
                    self.gpreps.store_simulated_data(z_list[m], w_list[m], sample_reward[m])

                self.gpreps.learn(
                    training_type='Simulated',
                    N_samples=self.args.num_training_samples,
                    eps=eps_k
                )
            else:
                self.gpreps.learn(
                    training_type='Realistic',
                    N_samples=self.args.num_training_samples,
                    eps=eps_k
                )

    def eval_once(self):
        print('::::::::::::::::::::::::::::::: evaluations ::::::::::::::::::::::::::::')
        self.evaluation_reward_step = np.zeros((self.args.eval_max_context_pairs, self.args.max_eval_episode, 1))
        self.evaluation_info_step = np.zeros(
            (self.args.eval_max_context_pairs, self.args.max_eval_episode, len(self.info_keywords)))

        for i in range(self.args.eval_max_context_pairs):
            z = self.env.get_context()
            z = z.reshape(-1, self.context_dim)
            print("context z :::::", z)
            w = self.gpreps.choose_action(np.array(z))
            print('parameter w :::::', w)
            self.env.controller.set_params(w[0])

            for episode_number in range(self.args.max_eval_episode):
                self.reset()
                while self.episode_timesteps < self.max_episode_steps:
                    action = self.policy.select_action(np.array(self.obs))
                    action = action.clip(
                        self.env.action_space.low, self.env.action_space.high
                    )

                    new_obs, reward, self.done, self.info = self.env.step(action)

                    self.episode_reward += reward
                    self.obs = new_obs
                    self.episode_timesteps += 1
                    if self.done or self.episode_timesteps == self.max_episode_steps - 1:
                        print('RL episode\n', episode_number,
                              'Step\n', self.episode_timesteps,
                              'done?\n', self.done,
                              'reward\n', np.round(self.episode_reward * self.reward_scale, 4)
                              )
                        break

                self.evaluation_reward_step[i, episode_number, 0] = cp.deepcopy(
                    np.round(self.episode_reward * self.reward_scale, 4))
                for j in range(len(self.info_keywords)):
                    self.evaluation_info_step[i, episode_number, j] = cp.deepcopy(self.info[self.info_keywords[j]])

        self.evaluations_reward.append(cp.deepcopy(self.evaluation_reward_step))
        self.evaluations_info_value.append(cp.deepcopy(self.evaluation_info_step))

        # np.save(self.log_dir + "/evaluations_reward", self.evaluations_reward)
        # np.save(self.log_dir + "/evaluations_info_value", self.evaluations_info_value)
