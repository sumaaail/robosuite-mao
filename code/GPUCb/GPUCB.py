import copy as cp
import math
import os
import random
import numpy as np
import torch
from code.pytorch.REPS.GPREPS import *
from envs.mujoco.utils.quaternion import mat2Quat, subQuat
from results.result_analysis import *


def CGPUCB(
		task,
		num_waypoints=2,
		contextual_impedance_dim=12,
		num_rolluts=1000,
		random_rollut=200,
		beta=100,
):
	"""
		main function of GP-UCB for contextual policy training
	"""
	if num_waypoints == 3:
		mesh_grid_dist = np.array([0., 0.4, 0.4, 2.0, 2.0, 1.25, 2.0, 2.0, 1.0, 0.75, 0.75, 4.0])
		contextual_impedance_lower_bound = np.array([0.0, 0.2, 0.2, 1.0, 1.0, 0.5, 1.0, 1.0, 0.0, 0.5, 0.5, 2])
		contextual_impedance_upper_bound = np.array([0.0, 1.4, 1.4, 7.0, 7.0, 4.25, 7.0, 7.0, 3.0, 2.75, 2.75, 12])
		para_samples = [np.array([0., 0., 0.])]
	else:
		mesh_grid_dist = np.array([0.0, 0.4, 2.0, 2.0, 1.25, 2.0, 2.0, 1.0])
		contextual_impedance_lower_bound = np.array([0.0, 0.2, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0])
		contextual_impedance_upper_bound = np.array([0.0, 1.8, 9.0, 9.0, 5.5, 9.0, 9.0, 5.0])
		para_samples = [np.array([0., 0.])]
	
	for i in range(1, contextual_impedance_dim):
		sample_list = np.arange(contextual_impedance_lower_bound[i], contextual_impedance_upper_bound[i], mesh_grid_dist[i])
		para_samples.append(sample_list.copy())
	
	if num_waypoints == 3:
		meshgrid = np.array(
			np.meshgrid(para_samples[0], para_samples[1], para_samples[2], para_samples[3], para_samples[4],
			            para_samples[5], para_samples[6], para_samples[7], para_samples[8], para_samples[9],
			            para_samples[10], para_samples[11]
			))
	else:
		meshgrid = np.array(
			np.meshgrid(para_samples[0], para_samples[1], para_samples[2], para_samples[3], para_samples[4],
			            para_samples[5], para_samples[6], para_samples[7]
			))
	
	sample_grid = meshgrid.reshape(meshgrid.shape[0], -1).T
	print('sample grid shape :::', sample_grid.shape)
	
	realistic_params_list = []
	reward_list = []
	
	mu = np.array([-3.0 for _ in range(sample_grid.shape[0])])
	sigma = np.array([0.5 for _ in range(sample_grid.shape[0])])
	
	beta = beta
	
	print('data shape :::', np.argmax(mu + sigma * np.sqrt(0.0)))
	gp = GaussianProcessRegressor()
	
	epislon_start = 0.5
	epsilon_end = 0.99
	
	optimal_param = sample_grid[0]
	optimal_episode_reward = -3
	for k in range(num_rolluts):
		print(":::::::::::::::::::::: Num_Rollout ::::::::::::::::::::", k)
		epsilon = epislon_start + (epsilon_end - epislon_start)/200
		epsilon = np.clip(epsilon, 0.5, 0.99)

		if k < random_rollut:
			grid_idx = np.random.randint(mu.shape[0], size=1)[0]
		else:
			beta *= 0.8
			if np.random.uniform() < epsilon:
				grid_idx = argmax_ucb(mu, sigma, beta)
			else:
				grid_idx = np.random.randint(mu.shape[0], size=1)[0]
		
		params = sample_grid[grid_idx]
		reward, tot_reward = task.send_movement(params)
		realistic_params_list.append(cp.copy(params))
		reward_list.append(cp.copy(reward))

		if k % 2 == 0:
			input_list = np.array(realistic_params_list)
			output_list = np.array(reward_list).reshape(len(reward_list), -1)
			gp.fit(input_list, output_list)
			
			optimal_index = argmax_ucb(mu, sigma, 0.0)
			optimal_param = sample_grid[optimal_index]
			optimal_episode_reward, _ = task.send_movement(optimal_param)

		mu, sigma = gp.predict(sample_grid, return_std=True)
		print("mu :::", mu.shape)
		print("sigma :::", sigma.shape)
	# np.save(result_path + "/sample_list.npy", realistic_params_list)
	# np.save(result_path + "/successful_rate.npy", successful_rate_list)
	# np.save(result_path + "/episode_reward.npy", episode_reward_average_list)
	
	# Plot learning results
	# plot_single_reward(data=self.successful_rate_list, font_size=14, y_label_list=['Episode Successful Rate'])
	# plot_single_reward(data=self.episode_reward_average_list, font_size=14, y_label_list=['Episode Reward Average'])
	# plot_single_reward(data=evaluation_reward_list, font_size=14, y_label_list=['Evaluation Reward'])
	
	return optimal_param, optimal_episode_reward, realistic_params_list, reward_list


def argmax_ucb(mu, sigma, beta):
	return np.argmax(mu + sigma.reshape(mu.shape[0], 1) * np.sqrt(beta))