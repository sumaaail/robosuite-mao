#!/usr/bin/env python3
from code.tensorflow.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from code.tensorflow.common import tf_util as U
import code.tensorflow.common.logger as logger


def train(env_id, num_timesteps, seed):
    from code.tensorflow.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = make_mujoco_env(env_id, seed)
    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    env.close()


def main(args):
    logger.configure()
    train(args.env,
          num_timesteps=args.num_timesteps,
          seed=args.seed)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env', help='environment ID', type=str, default='RoboschoolHopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--alg', help='Algorithm', type=str, default='ppo2')
    parser.add_argument('--num_timesteps', type=float, default=1e6),
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
    parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    parser.add_argument('--num_env',
                        help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco',
                        default=None, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--play', default=False, action='store_true')

    args = parser.parse_args()

    env_name_vec = [
        # 'Walker2d-v2',
        # 'Hopper-v2',
        # 'Ant-v2',
        # 'HalfCheetah-v2',
        # 'RoboschoolWalker2d-v1',
        # 'RoboschoolHalfCheetah-v1',
        'RoboschoolHopper-v1',
        # 'RoboschoolAnt-v1',
        # 'RoboschoolHumanoid-v1',
        # 'RoboschoolInvertedPendulum-v1',
        # 'RoboschoolInvertedPendulumSwingup-v1',
        # 'RoboschoolInvertedDoublePendulum-v1',
        # 'RoboschoolAtlasForwardWalk-v1'
    ]

    # print(args)
    main(args)
