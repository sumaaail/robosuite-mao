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
from stable_baselines3 import PPO, SAC, TD3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Door")  # Door, Lift, NutAssembly, NutAssemblyRound,
    # NutAssemblySingle, NutAssemblySquare, PickPlace, PickPlaceBread, PickPlaceCan, PickPlaceCereal, PickPlaceMilk,
    # PickPlaceSingle, Stack, TwoArmHandover, TwoArmLift, TwoArmPegInHole, Wipe
    parser.add_argument("--robots", nargs="+", type=str, default="Panda")  # Panda, Sawyer, Baxter
    parser.add_argument("--alg", type=str, default="PPO")
    parser.add_argument("--controller_name", type=str, default="OSC_POSE")  # see in robosuite/controllers/config
    parser.add_argument("--impedance_mode", type=str, default="fixed")  # fixed, variable, variable_kp
    parser.add_argument("--camera", type=str, default="frontview")  # frontview, birdview, agentview, sideview,
    # robot0_robotview, robot0_eye_in_hand
    parser.add_argument("--video_path", type=str, default="video.mp4")
    parser.add_argument("--record_timesteps", type=str, default=300)
    parser.add_argument("--skip_frame", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # load controller
    controller_name = args.controller_name
    np.random.seed(args.seed)

    # load controller from its path0000000000000000000000h
    controller_path = os.path.join(os.path.dirname(__file__),
                                   'results/2022/{}/{}/{}/{}/seed_{}/params.json'.format(args.env, args.alg,
                                                                                         args.robots,
                                                                                         args.impedance_mode,
                                                                                         args.seed))
    with open(controller_path) as f:
        controller_config = json.load(f)

    print("before make env--------------------------------------------------------------------------------------------")
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
        horizon=10000,
        control_freq=20,
        controller_configs=controller_config
    )
    print("after make env============================================================================================")
    from robosuite.wrappers.gym_wrapper import GymWrapper

    print("before wrapper-------------------------------------------------------------------------------------------")
    env = GymWrapper(env)
    print("after wrapper==============================================================================================")
    result_path = os.path.join(os.path.dirname(__file__),
                               'results/2022/{}/{}/{}/{}/seed_{}'.format(args.env, args.alg, args.robots,
                                                                                     args.impedance_mode, args.seed))
    model_path = os.path.join(result_path, 'model_1e4.zip')
    if args.alg == "SAC":
        model = SAC.load(model_path)
    elif args.alg == "PPO":
        model = PPO.load(model_path)
    elif args.alg == "TD3":
        model = TD3.load(model_path)
    else:
        raise NotImplementedError
    print("before reset------------------------------------------------------------------------------------------")
    obs = env.reset()
    print("after reset=============================================================================================")
    print("obs: {}".format(len(obs)))

    # video path
    video_path = "{}_{}_{}_{}_video.mp4".format(args.env, args.alg, args.impedance_mode, args.camera)
    writer = imageio.get_writer(video_path, fps=20)

    frames = []
    if env.use_camera_obs:
        print(
            "video with camera recording-----------------------------------------------------------------------------")
        for i in range(args.record_timesteps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            if i % args.skip_frame == 0:
                frame = np.array(env.get_imginfo())
                writer.append_data(frame)
                print("saving frame #{}".format(i))
            if done:
                break
    writer.close()
