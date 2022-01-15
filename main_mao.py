import argparse
import json
import os.path
import random
import numpy as np
import commentjson
import torch

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise

import robosuite as suite
from run_robosuite_main import PPO_callback, SAC_callback, TD3_callback
# from mujoco_py import GlfwContext
#
# GlfwContext(offscreen=True)

def run_learn(args, params, save_path=''):
    total_timesteps = args.total_timesteps

    actor_options = params['alg_params'].get('actor_options', None)

    # if args.impedance_mode == 'variable':
    #     run_save_path = os.path.join(save_path, args.alg + '_kp-limits[{},{}]'.format(args.kp_min, args.kp_max))
    # elif args.impedance_mode == 'fixed':
    #     run_save_path = os.path.join(save_path, args.alg + '_kp{}'.format(args.kp, args.damping_ratio))
    run_save_path = os.path.join(save_path, args.alg)
    # run_save_path = os.path.join(run_save_path, 'batch_size_'+str(args.batch_size))
    run_save_path = os.path.join(run_save_path, 'seed_'+str(args.seed))

    os.makedirs(run_save_path, exist_ok=True)

    # save parameters in params to params_save_path
    params_save_path = os.path.join(run_save_path, 'params.json')
    with open(params_save_path, 'w') as f:
        commentjson.dump(params, f, sort_keys=True, indent=4, ensure_ascii=False)

    set_seed(args.seed)
    # create env
    env = suite.make(
        args.env_name,
        robots=args.robot,

        has_renderer=False,
        render_camera="frontview",
        has_offscreen_renderer=False,

        use_camera_obs=False,
        use_object_obs=True,
        horizon=args.horizon,
        control_freq=args.control_freq,
        reward_shaping=True,
        controller_configs=params
    )

    # Setup printing options for numbers
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    from robosuite.wrappers.gym_wrapper_new import GymWrapper
    from monitor4wrapper import Monitor4wrapper
    env = GymWrapper(env, logdir=run_save_path)
    env = Monitor4wrapper(env, run_save_path, extra_print_key=('action_space',))
    # if need this reset?
    # obs = env.reset()
    # print("obs: {}".format(len(obs)))

    # Create the actor and learn
    if args.alg == 'PPO':
        model = PPO(
            params['alg_params']['policy_type'],
            env,
            batch_size=args.batch_size,
            seed=args.ppo_seed,
            n_epochs=args.n_epochs,
            # tensorboard_log=save_path,
            verbose=1,
            **actor_options)
    elif args.alg == 'SAC':
        model = SAC(
            params['alg_params']['policy_type'],
            env,
            # tensorboard_log=save_path,
            **actor_options)
    elif args.alg == 'TD3':
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = TD3(
            params['alg_params']['policy_type'],
            env,
            action_noise=action_noise,
            # verbose=1,
            # tensorboard_log=save_path,
            **actor_options)
    else:
        raise NotImplementedError

    # Create the callback
    if isinstance(model, PPO):
        learn_callback = lambda l, g: PPO_callback(l, g, run_save_path)
    elif isinstance(model, SAC):
        learn_callback = lambda l, g: SAC_callback(l, g, run_save_path)
    elif isinstance(model, TD3):
        learn_callback = lambda l, g: TD3_callback(l, g, run_save_path)
    else:
        raise NotImplementedError

    print("Learning and recording to: {}".format(run_save_path))
    from callback_for_save_best import SaveOnBestEpisodeRewardCallback
    callback = SaveOnBestEpisodeRewardCallback(check_freq=1000, log_dir=run_save_path)
    model.learn(callback=callback, total_timesteps=total_timesteps)

    model_save_path = os.path.join(run_save_path, 'model')
    model.save(model_save_path)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    import warnings
    parser = argparse.ArgumentParser(
        description='Runs a learning example on a registered gym environment.'
    )
    parser.add_argument(
        '--env_name',
        default='Wipe',
        type=str
    )
    parser.add_argument(
        '--alg',
        default='PPO',
        type=str
    )
    parser.add_argument(
        '--impedance_mode',
        default='variable',
        type=str
    )
    parser.add_argument(
        '--total_timesteps',
        default=1000000,
        type=int
    )
    parser.add_argument(
        '--controller_name',
        default='OSC_POSE',
        type=str
    )
    parser.add_argument(
        '--seed',
        default=17,
        type=int
    )
    parser.add_argument(
        '--robot',
        default='Panda',
        type=str
    )
    parser.add_argument(
        '--kp_max',
        default=300,
        type=int
    )
    parser.add_argument(
        '--kp_min',
        default=0,
        type=int
    )
    parser.add_argument(
        '--kp',
        default=150,
        type=int
    )
    parser.add_argument(
        '--damping_ratio',
        default=1,
        type=int
    )
    parser.add_argument(
        '--horizon',
        default=10000,
        type=int
    )
    parser.add_argument(
        '--control_freq',
        default=20,
        type=int
    )

    # PPO params
    parser.add_argument(
        '--batch_size',
        default=64,
        type=int
    )
    parser.add_argument(
        '--n_epochs',
        default=10,
        type=int
    )
    parser.add_argument(
        '--ppo_seed',
        default=None,
        type=int
    )



    args = parser.parse_args()
    warnings.simplefilter('default', RuntimeWarning)

    # load parameters
    param_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'params/params')
    print("param_dir: {}".format(param_dir))
    param_file_name = 'osc_pose.json'
    param_file = os.path.join(param_dir, param_file_name)

    print("param_file: {}".format(param_file))
    with open(param_file) as f:
        params_loaded = commentjson.load(f)
    params_loaded['impedance_mode'] = args.impedance_mode
    params_loaded['seed'] = args.seed
    params_loaded['horizon'] = args.horizon
    params_loaded['control_freq'] = args.control_freq
    if args.impedance_mode == 'variable':
        params_loaded['kp_limits'] = [args.kp_min, args.kp_max]
    elif args.impedance_mode == 'fixed':
        params_loaded['kp'] = args.kp
        params_loaded['damping_ratio'] = args.damping_ratio
    print("params :::", params_loaded)

    # save path
    save_path_env_name = 'new_results/v6/'+args.env_name+'/'
    # save_path = os.path.join(save_path_env_name, args.alg)
    # save_path = os.path.join(save_path_env_name, args.robot)
    save_path = os.path.join(save_path_env_name, args.impedance_mode)
    if args.impedance_mode == 'fixed':
        save_path = os.path.join(save_path, 'kp_{}'.format(args.kp))
    elif args.impedance_mode == 'variable':
        save_path = os.path.join(save_path, 'kp_limits{}_{}'.format(args.kp_min, args.kp_max))
    save_path = os.path.join(save_path, 'horizon_{}'.format(args.horizon))
    # save_path = os.path.join(save_path, 'control_freq_{}'.format(args.control_freq))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("created result_path: {}".format(save_path))

    model = run_learn(args, params_loaded, save_path)