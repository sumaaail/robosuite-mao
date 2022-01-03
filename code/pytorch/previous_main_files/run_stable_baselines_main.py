import argparse
import os
import sys
import commentjson

# ================================ stable_baseline algorithms ====================================
from stable_baselines import PPO2, SAC
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from envs.mujoco.utils.experiment_files import new_experiment_dir

# ================================================================================================

project_path = './'
sys.path.insert(0, project_path + 'code')

# Other RL algorithm
from code.pytorch.utils.RLVIC_solver import Solver
from envs.mujoco.utils.quaternion import *
from envs.mujoco.wrappers import TBVecEnvWrapper
from envs.gym_kuka_mujoco.envs import *


def PPO_callback(_locals, _globals, log_dir):
	"""
		Callback called at each gradient update.
	"""
	# Get the current update step.
	n_update = _locals['n_updates']
	
	# Save on the first update and every 10 updates after that.
	if (n_update == 1) or (n_update % 10 == 0):
		checkpoint_save_path = os.path.join(
			log_dir, 'model_checkpoint_{}.pkl'.format(n_update))
		_locals['self'].save(checkpoint_save_path)

PPO_callback.n_update = 0


def SAC_callback(_locals, _globals, log_dir):
	"""
		Callback called at each gradient update.
	"""
	# Get the current update step.
	# print('in callback')
	new_update = SAC_callback.n_updates < _locals['n_updates']
	if new_update:
		SAC_callback.n_updates = _locals['n_updates']
	
	# Save on the first update and every 10 updates after that.
	if new_update and ((SAC_callback.n_updates == 1) or
					   (SAC_callback.n_updates % 1000 == 0)):
		checkpoint_save_path = os.path.join(
			log_dir, 'model_checkpoint_{}.pkl'.format(SAC_callback.n_updates))
		_locals['self'].save(checkpoint_save_path)


SAC_callback.n_updates = 0


def make_env(env_cls,
			 rank,
			 save_path,
			 seed=0,
			 info_keywords=None,
			 **env_options):
	"""
		Utility function for vectorized env.
	
		:param env_cls: (str) the environment class to instantiate
		:param rank: (int) index of the subprocess
		:param seed: (int) the inital seed for RNG
		:param info_keywords: (tuple) optional, the keywords to record
		:param **env_options: additional arguments to pass to the environment
	"""
	
	def _init():
		env = env_cls(**env_options)
		env.seed(seed + rank)
		if info_keywords:
			# import pdb; pdb.set_trace()
			monitor_path = os.path.join(save_path, "proc_{}".format(rank))
			env = Monitor(env, monitor_path, info_keywords=tuple(info_keywords))
		return env
	
	set_global_seeds(seed)
	return _init


def replay_model(env, model, deterministic=True, num_episodes=None, record=False, render=True):
	# Don't record data forever.
	assert (not record) or (num_episodes is not None), \
		"there must be a finite number of episodes to record the data"
	
	# Initialize counts and data.
	num_episodes = num_episodes if num_episodes else np.inf
	episode_count = 0
	infos = []
	
	# Simulate forward.
	obs = env.reset()
	while episode_count < num_episodes:
		# import pdb; pdb.set_trace()
		action, _states = model.predict(obs, deterministic=deterministic)
		clipped_action = np.clip(action, env.action_space.low, env.action_space.high)
		obs, reward, done, info = env.step(clipped_action, render=render)
		if record:
			infos.append(info)
		if done:
			obs = env.reset()
			episode_count += 1
	 
	return infos


def run_learn(params, save_path, model=None, run_count=0):
	'''
	Runs the learning experiment defined by the params dictionary.

	:param params: (dict) the parameters for the learning experiment
	'''
	# Unpack options
	learning_options = params['learning_options']
	actor_options = params.get('actor_options', None)
	run_save_path = os.path.join(save_path, 'run_{}'.format(run_count))
	os.makedirs(run_save_path, exist_ok=True)
	
	# Save the parameters that will generate the model
	params_save_path = os.path.join(run_save_path, 'params.json')
	with open(params_save_path, 'w') as f:
		commentjson.dump(
			params, f, sort_keys=True, indent=4, ensure_ascii=False)
	
	# Generate vectorized environment.
	env_cls = globals()[params['env']]
	envs = [
		make_env(
			env_cls,
			i,
			run_save_path,
			info_keywords=params.get('info_keywords', None),
			**params['env_options']) for i in range(params['n_env'])
	]
	# envs = [make_env(params['env'], i, save_path) for i in range(params['n_env'])]
	
	if params.get('vectorized', True):
		env = SubprocVecEnv(envs)
	else:
		env = DummyVecEnv(envs)
	
	env = TBVecEnvWrapper(
		env, save_path, info_keywords=params.get('info_keywords', tuple()))
	
	# Create the actor and learn
	if model is not None:
		model.set_env(env)
	elif params['alg'] == 'PPO2':
		model = PPO2(
			params['policy_type'],
			env,
			tensorboard_log=save_path,
			**actor_options)
	elif params['alg'] == 'SAC':
		model = SAC(
			params['policy_type'],
			env,
			tensorboard_log=save_path,
			**actor_options)
	else:
		raise NotImplementedError
	
	# Create the callback
	if isinstance(model, PPO2):
		learn_callback = lambda l, g: PPO_callback(l, g, run_save_path)
	elif isinstance(model, SAC):
		learn_callback = lambda l, g: SAC_callback(l, g, run_save_path)
	else:
		raise NotImplementedError
	
	print("Learning and recording to:\n{}".format(run_save_path))
	model.learn(callback=learn_callback, **learning_options)
	
	# Save the model
	model_save_path = os.path.join(run_save_path, 'model')
	model.save(model_save_path)
	
	return model


if __name__ == '__main__':
	import warnings
	
	# Setup command line arguments.
	parser = argparse.ArgumentParser(
		description='Runs a learning example on a registered gym environment.')
	parser.add_argument(
		'--default_name',
		type=str,
		default='KukaMujoco-v0:PPO2',
		help='the name of the default entry to use')
	parser.add_argument(
		# '--param_file', default='manipulation/peg_insertion/ImpedanceV2PegInsertion:PPO2.json',
		'--param_file', default='manipulation/pushing/ImpedanceV2Pushing:PPO2.json',
		type=str,
		help='the parameter file to use')
	parser.add_argument(
		'--filter_warning',
		choices=['error', 'ignore', 'always', 'default', 'module', 'once'],
		default='default',
		help='the treatment of warnings')
	parser.add_argument(
		'--debug', action='store_true', help='enables useful debug settings')
	parser.add_argument(
		'--profile', action='store_true', help='runs in a profiler')
	parser.add_argument(
		'--final', action='store_true', help='puts the data in the final directory for easy tracking/plotting')
	parser.add_argument(
		'--num_restarts',
		type=int,
		default=1,
		help='The number of trials to run.')
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
		with open(param_file) as f:
			params = commentjson.load(f)
	
	# Override some arguments in debug mode
	if args.debug or args.profile:
		params['vectorized'] = False
	
	# Learn.
	# if args.final:
	# 	save_path = new_experiment_dir(params, prefix='final', date=False, short_description=True)
	# else:
	# 	save_path = new_experiment_dir(params)
	
	save_path = './results/4-kuka/'
		
	print("save_path :::", save_path)
	
	if args.profile:
		import cProfile

		for i in range(args.num_restarts):
			cProfile.run('run_learn(params, save_path, run_count=i)')
	else:
		for i in range(args.num_restarts):
			model = run_learn(params, save_path, run_count=i)
	
	# Visualize.
	# print("env params :::", params['env'])
	# env_cls = globals()[params['env']]
	# env = env_cls(**params['env_options'])
	# # replay_model(env, model)
	# env.reset()
	# while True:
	# 	env.render()
