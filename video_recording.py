import json
import robosuite as suite
import imageio
from mujoco_py import GlfwContext

GlfwContext(offscreen=True)
import argparse
import robosuite.utils.macros as macros

macros.IMAGE_CONVENTION = "opencv"
import numpy as np
import os
from robosuite import make
from run_robosuite_main import set_seed
from stable_baselines3 import PPO, SAC, TD3


def list_all_subdir(root_dir):
    for root, ds, fs in os.walk(root_dir):
        if fs != [] and ds == []:
            if 'Door' not in str.split(root, os.sep):
                for f in fs:
                    if f.endswith('.json'):
                        yield root + '/'


def video_recording(path_for_record):
    path = path_for_record
    # load controller from its path0000000000000000000000h
    controller_path = os.path.join(os.path.dirname(__file__), path + 'params.json')
    with open(controller_path) as f:
        controller_config = json.load(f)
    if 'seed' not in controller_config.keys():
        controller_config['seed'] = args.seed
    if 'horizon' not in controller_config.keys():
        controller_config['horizon'] = args.horizon
    if 'control_freq' not in controller_config.keys():
        controller_config['control_freq'] = args.control_freq
    # print(controller_config['control_freq'])
    # set_seed(controller_config['seed'])
    import quantumrandom
    seed = int(quantumrandom.randint(0, 1000))
    print("random seed: {}".format(seed))
    set_seed(seed)
    env = suite.make(
        args.env,
        robots=args.robots,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        # use_object_obs=False,
        camera_names=args.camera,
        # camera_heights=args.height,
        # camera_widths=args.width,
        horizon=controller_config['horizon'],
        control_freq=controller_config['control_freq'],
        controller_configs=controller_config
    )
    # print("after make env============================================================================================")
    from robosuite.wrappers.gym_wrapper_new import GymWrapper

    # print("before wrapper-------------------------------------------------------------------------------------------")
    env = GymWrapper(env)
    # print("after wrapper==============================================================================================")

    model_path = os.path.join(os.path.dirname(__file__), path + 'best_model.zip')
    if args.alg == "SAC":
        model = SAC.load(model_path)
    elif args.alg == "PPO":
        model = PPO.load(model_path)
    elif args.alg == "TD3":
        model = TD3.load(model_path)
    else:
        raise NotImplementedError
    # print("before reset------------------------------------------------------------------------------------------")
    obs = env.reset()
    # print("after reset=============================================================================================")
    print("obs: {}".format(len(obs)))

    # video path
    # video_path = os.path.join(os.path.dirname(__file__), 'recorded_video')
    # if controller_config['impedance_mode'] == 'fixed':
    #     video_path = os.path.join(video_path, '{}_{}_{}_{}_{}_{}_video.mp4'.format(args.env, args.alg,
    #                                                                                controller_config['impedance_mode'],
    #                                                                                controller_config['kp'],
    #                                                                                controller_config['seed'],
    #                                                                                args.camera))
    # elif controller_config['impedance_mode'] == 'variable':
    #     video_path = os.path.join(video_path, '{}_{}_{}_{}_{}_{}_video.mp4'.format(args.env, args.alg,
    #                                                                                controller_config['impedance_mode'],
    #                                                                                controller_config['kp_limits'],
    #                                                                                controller_config['seed'],
    #                                                                                args.camera))
    video_path = os.path.join(path, 'result_video.mp4')

    writer = imageio.get_writer(video_path, fps=60)

    frames = []
    if env.use_camera_obs:
        # print( "video with camera
        # recording-----------------------------------------------------------------------------")
        for i in range(args.record_timesteps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            print("action: {}".format(action))
            print("info: {}".format(info))
            # if i % args.skip_frame == 0:
            #     frame = np.array(env.get_imginfo())
            #     writer.append_data(frame)
                # print("saving frame #{}".format(i))
            if done:
                break
    print("saved to {}".format(path))
    writer.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Wipe")  # Door, Lift, NutAssembly, NutAssemblyRound,
    # NutAssemblySingle, NutAssemblySquare, PickPlace, PickPlaceBread, PickPlaceCan, PickPlaceCereal, PickPlaceMilk,
    # PickPlaceSingle, Stack, TwoArmHandover, TwoArmLift, TwoArmPegInHole, Wipe
    parser.add_argument("--robots", nargs="+", type=str, default="Panda")  # Panda, Sawyer, Baxter
    parser.add_argument("--alg", type=str, default="PPO")
    parser.add_argument("--controller_name", type=str, default="OSC_POSE")  # see in robosuite/controllers/config
    parser.add_argument("--impedance_mode", type=str, default="fixed")  # fixed, variable, variable_kp
    parser.add_argument("--camera", type=str, default="frontview")  # frontview, birdview, agentview, sideview,
    # robot0_robotview, robot0_eye_in_hand
    parser.add_argument("--video_path", type=str, default="video.mp4")
    parser.add_argument("--record_timesteps", type=str, default=1000)
    parser.add_argument("--skip_frame", type=int, default=1)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--control_freq", type=int, default=20)
    args = parser.parse_args()
    for dir in list_all_subdir('new_results/v6/Wipe/variable/kp_limits0_300/horizon_1000/'):
        video_recording(dir)
    # video_recording('new_results/v2/Wipe/Panda/fixed/kp_150/PPO/seed_17_1000_20/')
