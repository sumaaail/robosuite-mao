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

if __name__ == '__main__':
    path = 'new_results/v1/Wipe/Panda/fixed/kp_150/PPO/seed_3/'
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
    parser.add_argument("--record_timesteps", type=str, default=1500)
    parser.add_argument("--skip_frame", type=int, default=1)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    # load controller from its path0000000000000000000000h
    controller_path = os.path.join(os.path.dirname(__file__), path+'params.json')
    with open(controller_path) as f:
        controller_config = json.load(f)

    set_seed(controller_config['seed'])
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
        horizon=1000,
        control_freq=60,
        controller_configs=controller_config
    )
    print("after make env============================================================================================")
    from robosuite.wrappers.gym_wrapper_new import GymWrapper

    print("before wrapper-------------------------------------------------------------------------------------------")
    env = GymWrapper(env)
    print("after wrapper==============================================================================================")

    model_path = os.path.join(os.path.dirname(__file__), path+'model_checkpoint_0.pkl')
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
    video_path = os.path.join(os.path.dirname(__file__), 'recorded_video')
    if controller_config['impedance_mode'] == 'fixed':
        video_path = os.path.join(video_path, '{}_{}_{}_{}_{}_{}_video.mp4'.format(args.env, args.alg, controller_config['impedance_mode'], controller_config['kp'], controller_config['seed'], args.camera))
    elif controller_config['impedance_mode'] == 'variable':
        video_path = os.path.join(video_path, '{}_{}_{}_{}_{}_{}_video.mp4'.format(args.env, args.alg, controller_config['impedance_mode'], controller_config['kp_limits'], controller_config['seed'], args.camera))

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
