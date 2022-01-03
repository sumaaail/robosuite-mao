import os
print(os.getcwd())
import sys
project_path = './'
sys.path.append("/usr/local/webots/lib")
sys.path.insert(0, project_path + 'pytorch')
print(sys.path)

from envs.abb.env_abb_assembly import env_assembly_search

import argparse
import numpy as np

# using pytorch code base
from code.pytorch.utils.real_solver import Assembly_solver


def test_env(env):
    env.reset()
    state = np.random.rand(22)
    print(env.set_robot(state) - state)
    while True:
        env.render()


def main(env, args):
    solver = Assembly_solver(args, env, project_path)
    if not args.eval_only:
        solver.train()
    else:
        solver.eval_only()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--policy_name", default='TD3')  # Policy name
    parser.add_argument("--env_name", default="dual-peg-in-hole")  # OpenAI gym environment name
    parser.add_argument("--log_path", default='hrl')
    parser.add_argument("--seed", default=10, type=int)  # Sets Gym, PyTorch and Numpy seeds
    
    parser.add_argument("--eval_only", default=False)
    parser.add_argument("--load_path", default='transfer/single_assembly')
    parser.add_argument("--save_all_policy", default=True)
    parser.add_argument("--load_policy_idx", default=1e5, type=int)

    parser.add_argument("--eval_freq", default=1e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e5, type=int)  # Max time steps to run environment for
    parser.add_argument("--max_episode_steps", default=150, type=int)
    parser.add_argument("--start_timesteps", default=1e3, type=int)  # How many time steps purely random policy is run for
    
    # parameters for DDPG && TD3
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--state_noise", default=0, type=float)
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.2, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    
    # parameters for HRL
    parser.add_argument("--option_num", default=4, type=int)
    parser.add_argument("--option_change", default=20, type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--auxiliary_reward", default=False)

    parser.add_argument("--option_buffer_size", default=1000, type=int)  # Batch size for both actor and critic
    parser.add_argument("--option_batch_size", default=32, type=int)
    parser.add_argument("--policy_batch_size", default=32, type=int)
    parser.add_argument("--critic_batch_size", default=64, type=int)
    parser.add_argument("--upper_critic_batch_size", default=32, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    
    args = parser.parse_args()

    # connect to abb robot
    env = env_assembly_search()
    policy_name_vec = ['HRLACOP']

    for policy_name in policy_name_vec:
        args.policy_name = policy_name
        for i in range(0, 2):
            args.seed = i
            main(env, args)
