import argparse
import os

import commentjson
from code.pytorch.utils.recognition_solver import Solver
from envs.mujoco.envs import *
from envs.mujoco.utils.experiment_files import new_experiment_dir
import numpy as np
import time
import matplotlib.pyplot as plt

project_path = './'


def main(env, args):
	solver = Solver(args, env, project_path)
	solver.auto()


if __name__ == '__main__':
	import warnings
	
	parser = argparse.ArgumentParser(
		description='Runs a learning example on a registered gym environment.')
	parser.add_argument('--default_name',
						type=str, default='KukaMujoco-v0:PPO2',
						help='the name of the default entry to use')
	parser.add_argument('--param_file',
						type=str, default='manipulation/peg_insertion/ImpedanceV2PegInsertion:PPO2.json',
						help='the parameter file to use')
	parser.add_argument('--filter_warning', choices=['error', 'ignore', 'always', 'default', 'module', 'once'],
						default='default',
						help='the treatment of warnings')
	parser.add_argument(
		'--debug', action='store_true', help='enables useful debug settings')
	parser.add_argument(
		'--profile', action='store_true', help='runs in a profiler')
	parser.add_argument(
		'--final', action='store_true', help='puts the data in the final directory for easy tracking/plotting')
	parser.add_argument('--num_restarts', type=int, default=1, help='The number of trials to run.')
	
	parser.add_argument("--policy_name", default='TD3')  # Policy name
	parser.add_argument("--env_name", default="Peg-in-hole-single_assembly")  # OpenAI gym environment name
	parser.add_argument("--log_path", default='runs/single/f05_c1_a1_d30')  # transfer to f1_c1_a1_d30
	
	parser.add_argument("--eval_only", default=False)
	parser.add_argument("--eval_max_timesteps", default=5e4, type=int)  # Max time steps to run environment for
	
	parser.add_argument("--render", default=True)
	parser.add_argument("--save_video", default=False)
	parser.add_argument("--video_size", default=(600, 400))
	parser.add_argument("--save_all_policy", default=True)
	
	parser.add_argument("--load_policy", default=False)
	parser.add_argument("--load_policy_idx", default='100000')
	parser.add_argument("--evaluate_Q_value", default=False)
	parser.add_argument("--discount_low", default=0.99, type=float)  # Discount factor
	parser.add_argument("--discount_high", default=0.99, type=float)  # Discount factor
	
	parser.add_argument("--seq_len", default=2, type=int)
	parser.add_argument("--ini_seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--seed", default=10, type=int)  # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=1e3,
						type=int)  # How many time steps purely random policy is run for
	
	parser.add_argument("--auxiliary_reward", default=False)
	parser.add_argument("--option_num", default=4, type=int)
	
	parser.add_argument("--option_buffer_size", default=1000, type=int)  # Batch size for both actor and critic
	parser.add_argument("--option_batch_size", default=100, type=int)  # Batch size for both actor and critic
	parser.add_argument("--policy_batch_size", default=100, type=int)  # Batch size for both actor and critic
	parser.add_argument("--critic_batch_size", default=400, type=int)  # Batch size for both actor and critic
	parser.add_argument("--upper_critic_batch_size", default=200, type=int)  # Batch size for both actor and critic
	parser.add_argument("--batch_size", default=200, type=int)  # Batch size for both actor and critic
	
	parser.add_argument("--eval_freq", default=1e3, type=int)  # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e5, type=int)  # Max time steps to run environment for
	parser.add_argument("--max_episode_steps", default=200, type=int)
	
	parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
	parser.add_argument("--state_noise", default=0, type=float)  # Std of Gaussian exploration noise
	parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
	parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.2, type=float)  # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
	
	parser.add_argument("--average_steps", default=20, type=int)
	parser.add_argument("--learning_rate", default=1e-3, type=float)
	parser.add_argument("--option_change", default=50, type=int)  # How many time steps purely random policy is run for
	
	args = parser.parse_args()
	
	# Change the warning behavior for debugging.
	warnings.simplefilter(args.filter_warning, RuntimeWarning)
	
	# Load the learning parameters from a file.
	param_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'params')
	
	if args.param_file is None:
		default_path = os.path.join(param_dir, 'default_params.json')
		with open(default_path) as f:
			params = commentjson.load(f)[args.default_name]
	else:
		param_file = os.path.join(param_dir, args.param_file)
		print("param_dir :::", param_file)
		with open(param_file) as f:
			params = commentjson.load(f)
	
	# Override some arguments in debug mode
	if args.debug or args.profile:
		params['vectorized'] = False
	
	# Learn
	if args.final:
		save_path = new_experiment_dir(params, prefix='final', date=False, short_description=True)
	else:
		save_path = new_experiment_dir(params)
	
	print('save_path', save_path)
	
	policy_name_vec = ['TD3']
	
	# Visualize.
	env_cls = globals()[params['env']]
	print(env_cls)
	env = env_cls(**params['env_options'])
	average_steps = [1]
	
	# for policy_name in policy_name_vec:
	# 	for i in range(0, 1):
	# 		args.policy_name = policy_name
	# 		args.seed = i
	# 		main(env, args)
	
	def random_start(env):
		# x = np.random.uniform(-0.015, 0.015, 1)
		# y = np.random.uniform(-0.015, 0.015, 1)
		# x = np.clip(np.random.normal(0, 0.01, 1), -0.012, 0.012)
		# y = np.clip(np.random.normal(0, 0.01, 1), -0.012, 0.012)
		# z = np.clip(np.random.normal(-0.01, 0.002, 1), -0.012, -0.008)
		# rx = np.clip(np.random.normal(0, 0.015, 1), -0.015, 0.015)
		# ry = np.clip(np.random.normal(0, 0.015, 1), -0.015, 0.015)
		# rz = np.clip(np.random.normal(0, 0.015, 1), -0.015, 0.015)
		action = np.array([0.012, -0.012, 0.0023, 0.015, 0.015, 0.])
		# action = np.hstack([x, y, z, rx, ry, rz])
		print(action)
		for i in range(100):
			obs, reward, done, _ = env.step(action)
			env.render()
		time.sleep(0)
		# print('**************************** random init pos ******************************')
		return obs
	
	for i in range(1):
		print('episode', i + 1, ':')
		env.reset()
		obs = random_start(env)
		print("obs :::", obs)
		step = 0
		temporary_buffer = []
		safe = True
		
		# phase 1: approach
		while obs[20] >= 0.02:
			action = np.array([0, 0, 0.03, 0, 0, 0])
			obs, reward, done, _ = env.step(action)
			temporary_buffer.append(np.hstack([1, obs]))
			env.render()
			print('phase 1, step', step, ':', obs[:6])
			step += 1
			# safe?
			if obs[18] > 0.1 or obs[18] < -0.1 or obs[19] > 0.1 or obs[19] < -0.1:
				safe = False
		
		# phase 2: contact
		while obs[20] >= 0.005:
			action = np.array([0, 0, 0.01, 0, 0, 0])
			obs, reward, done, _ = env.step(action)
			temporary_buffer.append(np.hstack([2, obs]))
			env.render()
			print('phase 2, step', step, ':', obs[:6])
			step += 1
			# safe?
			if obs[18] > 0.1 or obs[18] < -0.1 or obs[19] > 0.1 or obs[19] < -0.1:
				safe = False

		# phase 3: fit
		while np.abs(obs[-3]) >= 0.001 or np.abs(obs[-2]) >= 0.001:
			e = obs[-3:-1]
			le = (e[0] ** 2 + e[1] ** 2) ** 0.5
			action = 0.01 * e / le
			action_z = np.random.normal(0.005, 0.01)
			action = np.hstack([action, np.array([action_z, 0, 0, 0])])
			if le <= 0.01:
				obs, reward, done, _ = env.step(action / 2)
			else:
				obs, reward, done, _ = env.step(action)
			temporary_buffer.append(np.hstack([3, obs]))
			print('phase 3, step', step, ':', obs[:6])
			step += 1
			# safe?
			if obs[18] > 0.1 or obs[18] < -0.1 or obs[19] > 0.1 or obs[19] < -0.1:
				safe = False
			env.render()
			if step > 500:
				break
		
		# # phase 4: align
		# while np.abs(obs[9]) >= 0.001 or np.abs(obs[10]) >= 0.001:
		# 	w = 0.2
		# 	a1 = 0
		# 	a2 = 0
		# 	if np.abs(obs[9]) >= 0.001:
		# 		if obs[9] >= 0:
		# 			a1 = w
		# 		else:
		# 			a1 = -w
		# 	if np.abs(obs[10]) >= 0.001:
		# 		if obs[10] >= 0:
		# 			a2 = w
		# 		else:
		# 			a2 = -w
		# 	action = np.array([0, 0, 0, a1, a2, 0])
		# 	obs, reward, done, _ = env.step(action / 5)
		# 	temporary_buffer.append(np.hstack([4, obs]))
		# 	env.render()
		# 	print('phase 4, step', step, ':', obs[:6])
		# 	step += 1
		# 	# safe?
		# 	if obs[18] > 0.1 or obs[18] < -0.1 or obs[19] > 0.1 or obs[19] < -0.1:
		# 		safe = False
		#
		# # phase 5: insertion
		# while obs[8] >= 0:
		# 	action = np.array([0, 0, 0.01, 0, 0, 0])
		# 	obs, reward, done, _ = env.step(action)
		# 	temporary_buffer.append(np.hstack([5, obs]))
		# 	env.render()
		# 	print('phase 5, step', step, ':', obs[:6])
		# 	step += 1
		# 	# safe?
		# 	if obs[18] > 0.1 or obs[18] < -0.1 or obs[19] > 0.1 or obs[19] < -0.1:
		# 		safe = False
		
		# plot the trajectory of force
		fx = []
		fy = []
		fz = []
		s = []
		for t in range(step):
		    fx.append(temporary_buffer[t][0])
		    fy.append(temporary_buffer[t][1])
		    fz.append(temporary_buffer[t][2])
		    s .append(t+1)
		plt.plot(s, fx, c='r')
		plt.plot(s, fy, c='b')
		plt.plot(s, fz, c='g')
		plt.show()