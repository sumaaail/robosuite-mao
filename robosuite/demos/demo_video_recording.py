"""
Record video of agent episodes with the imageio library.
This script uses offscreen rendering.

Example:
    $ python demo_video_recording.py --environment Lift --robots Panda
"""

import argparse
import json
import os

import cv2
import gym
gym.logger.set_level(40)
import imageio
import numpy as np
from stable_baselines3 import SAC

import robosuite as suite
import robosuite.utils.macros as macros
from robosuite import make

# Set the image convention to opencv so that the images are automatically rendered "right side up" when using imageio
# (which uses opencv convention)
macros.IMAGE_CONVENTION = "opencv"
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Stack")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--camera", type=str, default="sideview", help="Name of camera to render")
    parser.add_argument("--video_path", type=str, default="video.mp4")
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--skip_frame", type=int, default=1)
    args = parser.parse_args()

    # initialize an environment with offscreen renderer
    # env = make(
    #     # args.environment,
    #     # args.robots,
    #     "Wipe",
    #     robots="Sawyer",
    #     has_renderer=False,
    #     ignore_done=True,
    #     use_camera_obs=True,
    #     use_object_obs=False,
    #     camera_names=args.camera,
    #     camera_heights=args.height,
    #     camera_widths=args.width,
    # )


    # Load the controller
    with open("/home/sumail/robosuite/robosuite/controllers/config/osc_pose.json") as f:
        controller_config = json.load(f)
    env = make(
        "Wipe",
        robots="Sawyer",
        has_renderer=False,
        ignore_done=True,
        use_camera_obs=True,
        use_object_obs=False,
        camera_names=args.camera,
        camera_heights=args.height,
        camera_widths=args.width,
        horizon=10000,
        control_freq=20,
        controller_configs=controller_config
    )
    obs = env.reset()

    print("obs len: {}\nobs after raw reset: {} ".format(len(obs), obs))
    print("img in raw obs: {}".format(type(obs["robot0_joint_pos_cos"])))

    from robosuite.wrappers.gym_wrapper import GymWrapper
    env = GymWrapper(env)
    # env = gym.make("Pendulum-v0")


    # model = SAC.load("/home/sumail/robosuite/results/mao_tmech_v1/Wipe/SAC/variable/run_0/model_no_seed.zip")
    # model = SAC("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=50000, log_interval=4)
    # model.save("sac_withimage1")
    # print("learnt")
    # del model  # remove to demonstrate saving and loading
    #
    model = SAC.load("sac_withimage1")
    print("-----------------------------------------------------------------------------------")
    obs = env.reset()
    # while True:
    #     new_image = np.ndarray((3, 256, 256), dtype=int)
    #     action, _states = model.predict(obs, deterministic=True)
    #     ob, _, _, _ = env.step(action)
    #     for m in range(256):
    #         for n in range(256):
    #             for p in range(3):
    #                 new_image[p][m][n] = ob['sideview_image'][m][n][p]
    #
    #     new_image = new_image.astype(np.uint8)
    #
    #     # Separated the channels in my new image
    #     new_image_red, new_image_green, new_image_blue = new_image
    #
    #     # Stacked the channels
    #     new_rgb = np.dstack([new_image_red, new_image_green, new_image_blue])
    #
    #     # Displayed the image
    #     cv2.imshow("WindowNameHere", new_rgb)
    # img = env.getimginfo()
    # print("image len: {}\nimage: {}".format(len(img), img))
    print("obs.shape after wrapper: {}\nobs after wrapper: {}".format(obs.shape, obs))


    # ndim = env.action_dim
    # print(ndim)
    # create a video writer with imageio
    writer = imageio.get_writer(args.video_path, fps=20)

    frames = []
    if env.use_camera_obs:
        print("video with camera recording----------------------------------------------------------------------------")
        for i in range(args.timesteps):

            # run a uniformly random agent
            # action = 0.5 * np.random.randn(ndim)
            # obs, reward, done, info = env.step(action)

            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            # dump a frame from every K frames
            if i % args.skip_frame == 0:
                # frame = obs[args.camera + "_image"]
                frame = np.array(env.get_imginfo())
                # print("frame type: {}".format(frame))
                writer.append_data(frame)
                print("Saving frame #{}".format(i))

            if done:
                break
    else:
        print(
            "video recording------------------------------------------------------------------------------------------")
        for i in range(args.timesteps):
            # run a uniformly random agent
            # action = 0.5 * np.random.randn(ndim)
            # obs, reward, done, info = env.step(action)

            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            # dump a frame from every K frames
            if i % args.skip_frame == 0:
                frame = obs[args.camera + "_image"]
                # frame = obs
                writer.append_data(frame)
                print("Saving frame #{}".format(i))

            if done:
                break
    writer.close()
