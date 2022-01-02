import argparse
import json
import os.path
import numpy as np
import commentjson
import torch
import rlkit.torch.pytorch_util as ptu
import robosuite as suite
from robosuite.wrappers import GymWrapper
from utils.rlkit_custom import rollout

from rlkit.torch.pytorch_util import set_gpu_mode

import torch
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.td3.td3 import TD3Trainer
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from utils.rlkit_custom import CustomTorchBatchRLAlgorithm
import abc

from rlkit.core import logger, eval_util
import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm, _get_epoch_timings
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import DataCollector
from rlkit.samplers.data_collector import PathCollector

from collections import OrderedDict

import numpy as np

import rlkit.pythonplusplus as ppp
from rlkit.core import logger
import robosuite as suite
from robosuite.wrappers import GymWrapper

from robosuite.controllers import load_controller_config, ALL_CONTROLLERS

import numpy as np
from robosuite.controllers import load_controller_config, ALL_CONTROLLERS
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from utils.arguments import *
import robosuite as suite
from run_robosuite_main import PPO_callback, SAC_callback, TD3_callback
# Global vars
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

add_robosuite_args()
add_agent_args()
add_training_args()

def experiment(variant, agent="SAC"):
    # Make sure agent is a valid choice
    assert agent in AGENTS, "Invalid agent selected. Selected: {}. Valid options: {}".format(agent, AGENTS)

    controller_name = args.controller
    controller_path = os.path.join(os.path.dirname(__file__),
                                   'robosuite',
                                   'controllers/config/{}.json'.format(controller_name.lower()))
    with open(controller_path) as f:
        controller_config = json.load(f)
    controller_config["impedance_mode"] = args.impedance_mode
    controller_config["kp_limits"] = [0, 300]
    controller_config["damping_limits"] = [0, 10]

    # Get environment configs for expl and eval envs and create the appropriate envs
    # suites[0] is expl and suites[1] is eval
    suites = []
    for env_config in (variant["expl_environment_kwargs"], variant["eval_environment_kwargs"]):
        # Load controller
        controller = env_config.pop("controller")
        if controller in set(ALL_CONTROLLERS):
            # This is a default controller
            controller_config = load_controller_config(default_controller=controller)
        else:
            # This is a string to the custom controller
            controller_config = load_controller_config(custom_fpath=controller)
        # Create robosuite env and append to our list
        suites.append(suite.make(**env_config,
                                 has_renderer=False,
                                 has_offscreen_renderer=False,
                                 use_object_obs=True,
                                 use_camera_obs=False,
                                 reward_shaping=True,
                                 controller_configs=controller_config,
                                 ))
    # Create gym-compatible envs
    expl_env = NormalizedBoxEnv(GymWrapper(suites[0]))
    eval_env = NormalizedBoxEnv(GymWrapper(suites[1]))

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )

    # Define references to variables that are agent-specific
    trainer = None
    eval_policy = None
    expl_policy = None

    # Instantiate trainer with appropriate agent
    if agent == "SAC":
        expl_policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **variant['policy_kwargs'],
        )
        eval_policy = MakeDeterministic(expl_policy)
        trainer = SACTrainer(
            env=eval_env,
            policy=expl_policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            **variant['trainer_kwargs']
        )
    elif agent == "TD3":
        eval_policy = TanhMlpPolicy(
            input_size=obs_dim,
            output_size=action_dim,
            **variant['policy_kwargs']
        )
        target_policy = TanhMlpPolicy(
            input_size=obs_dim,
            output_size=action_dim,
            **variant['policy_kwargs']
        )
        es = GaussianStrategy(
            action_space=expl_env.action_space,
            max_sigma=0.1,
            min_sigma=0.1,  # Constant sigma
        )
        expl_policy = PolicyWrappedWithExplorationStrategy(
            exploration_strategy=es,
            policy=eval_policy,
        )
        trainer = TD3Trainer(
            policy=eval_policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            target_policy=target_policy,
            **variant['trainer_kwargs']
        )
    else:
        print("Error: No valid agent chosen!")

    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        expl_policy,
    )

    # Define algorithm
    algorithm = CustomTorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


def run_learn(args, params, save_path='', run_count=1):
    trainer_kwargs = None
    if args.agent == "SAC":
        trainer_kwargs = dict(
            discount=args.gamma,
            soft_target_tau=args.soft_target_tau,
            target_update_period=args.target_update_period,
            policy_lr=args.policy_lr,
            qf_lr=args.qf_lr,
            reward_scale=args.reward_scale,
            use_automatic_entropy_tuning=(not args.no_auto_entropy_tuning),
        )
    elif args.agent == "TD3":
        trainer_kwargs = dict(
            target_policy_noise=args.target_policy_noise,
            discount=0.99,
            reward_scale=args.reward_scale,
            policy_learning_rate=args.policy_lr,
            qf_learning_rate=args.qf_lr,
            policy_and_target_update_period=args.policy_and_target_update_period,
            tau=args.tau,
        )
    else:
        pass

    # Construct variant to train
    if args.variant is None:
        variant = dict(
            algorithm=args.agent,
            seed=args.seed,
            version="normal",
            replay_buffer_size=int(1E6),
            qf_kwargs=dict(
                hidden_sizes=args.qf_hidden_sizes,
            ),
            policy_kwargs=dict(
                hidden_sizes=args.policy_hidden_sizes,
            ),
            algorithm_kwargs=dict(
                num_epochs=args.n_epochs,
                num_eval_steps_per_epoch=args.eval_horizon * args.num_eval,
                num_trains_per_train_loop=args.trains_per_train_loop,
                num_expl_steps_per_train_loop=args.expl_horizon * args.expl_ep_per_train_loop,
                min_num_steps_before_training=args.steps_before_training,
                expl_max_path_length=args.expl_horizon,
                eval_max_path_length=args.eval_horizon,
                batch_size=args.batch_size,
            ),
            trainer_kwargs=trainer_kwargs,
            expl_environment_kwargs=get_expl_env_kwargs(args),
            eval_environment_kwargs=get_eval_env_kwargs(args),
        )
        # Set logging
        tmp_file_prefix = "{}_{}_{}_SEED{}".format(args.env, "".join(args.robots), args.controller, args.seed)
    else:
        # This is a variant we want to load
        # Attempt to load the json file
        try:
            with open(args.variant) as f:
                variant = json.load(f)
        except FileNotFoundError:
            print("Error opening specified variant json at: {}. "
                  "Please check filepath and try again.".format(variant))

        # Set logging
        tmp_file_prefix = "{}_{}_{}_SEED{}".format(variant["expl_environment_kwargs"]["env_name"],
                                                   "".join(variant["expl_environment_kwargs"]["robots"]),
                                                   variant["expl_environment_kwargs"]["controller"],
                                                   args.seed)
        # Set agent
        args.agent = variant["algorithm"]

    # Setup logger
    abs_root_dir = os.path.join(THIS_DIR, args.log_dir)
    # tmp_dir = setup_logger(tmp_file_prefix, variant=variant, base_log_dir=abs_root_dir)
    ptu.set_gpu_mode(torch.cuda.is_available())  # optionally set the GPU (default=False

    # Run experiment
    experiment(variant, agent=args.agent)

if __name__ == '__main__':
    import warnings

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    warnings.simplefilter('default', RuntimeWarning)

    # load parameters
    param_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'params/params')
    print("param_dir: {}".format(param_dir))
    param_file_name = 'ImpedanceV2PegInsertion:SAC.json'
    param_file = os.path.join(param_dir, param_file_name)
    print("param_file: {}".format(param_file))
    with open(param_file) as f:
        params_loaded = commentjson.load(f)

    # save path
    save_path_env_name = 'results/mao/' + args.env + '/'
    save_path = os.path.join(save_path_env_name, params_loaded['alg'])
    save_path = os.path.join(save_path, args.impedance_mode)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("created result_path: {}".format(save_path))

    for i in range(1):
        # Notify user we're starting run
        print('\n\n')
        print('------------- Running {} --------------'.format(args.agent))

        print('  Params: ')
        if args.variant is None:
            for key, value in args.__dict__.items():
                if key.startswith('__') or key.startswith('_'):
                    continue
                print('    {}: {}'.format(key, value))
        else:
            print('    variant: {}'.format(args.variant))

        print('\n\n')
        model = run_learn(args, params_loaded, save_path, run_count=i)
        print('Finished run!')