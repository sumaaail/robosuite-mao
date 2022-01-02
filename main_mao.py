import argparse
import json
import os.path
import numpy as np
import commentjson

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise

import robosuite as suite
from run_robosuite_main import PPO_callback, SAC_callback, TD3_callback


def run_learn(args, params, save_path='', run_count=1):
    total_timesteps = args.total_timesteps

    actor_options = params['alg_params'].get('actor_options', None)
    run_save_path = os.path.join(save_path, 'run_{}'.format(run_count))
    os.makedirs(run_save_path, exist_ok=True)

    # save parameters in params to params_save_path
    params_save_path = os.path.join(run_save_path, 'params.json')
    with open(params_save_path, 'w') as f:
        commentjson.dump(params, f, sort_keys=True, indent=4, ensure_ascii=False)

    controller_name = args.controller_name
    np.random.seed(5)

    # load controller from its path
    controller_path = os.path.join(os.path.dirname(__file__),
                                   'robosuite',
                                   'controllers/config/{}.json'.format(controller_name.lower()))
    with open(controller_path) as f:
        controller_config = json.load(f)

    controller_config["impedance_mode"] = args.impedance_mode
    controller_config["kp_limits"] = [0, 300]
    controller_config["damping_limits"] = [0, 10]

    # create env
    env = suite.make(
        args.env_name,
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        horizon=10000,
        control_freq=20,
        controller_configs=controller_config
    )

    # Setup printing options for numbers
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    from robosuite.wrappers.gym_wrapper import GymWrapper
    env = GymWrapper(env)

    obs = env.reset()
    print("obs: {}".format(len(obs)))

    # Create the actor and learn
    if args.alg == 'PPO':
        model = PPO(
            params['alg_params']['policy_type'],
            env,
            tensorboard_log=save_path,
            **actor_options)
    elif args.alg == 'SAC':
        model = SAC(
            params['policy_type'],
            env,
            tensorboard_log=save_path,
            **actor_options)
    elif args.alg == 'TD3':
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = TD3(
            params['policy_type'],
            env,
            action_noise=action_noise,
            # verbose=1,
            tensorboard_log=save_path,
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
    model.learn(callback=learn_callback, total_timesteps=total_timesteps)

    model_save_path = os.path.join(run_save_path, 'model')
    model.save(model_save_path)


if __name__ == '__main__':
    import warnings
    parser = argparse.ArgumentParser(
        description='Runs a learning example on a registered gym environment.'
    )
    parser.add_argument(
        '--env_name',
        default='Door',
        type=str
    )
    parser.add_argument(
        '--alg',
        default='PPO',
        type=str
    )
    parser.add_argument(
        '--impedance_mode',
        default='fixed',
        type=str
    )
    parser.add_argument(
        '--total_timesteps',
        default=700000,
        type=int
    )
    parser.add_argument(
        '--controller_name',
        default='OSC_POSE',
        type=str
    )
    parser.add_argument(
        '--seed'
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

    # save path
    save_path_env_name = 'results/mao/'+args.env_name+'/'
    save_path = os.path.join(save_path_env_name, args.alg)
    save_path = os.path.join(save_path, args.impedance_mode)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("created result_path: {}".format(save_path))

    model = run_learn(args, params_loaded, save_path)