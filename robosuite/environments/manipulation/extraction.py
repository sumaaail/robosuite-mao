from collections import OrderedDict
import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from mujoco_py import load_model_from_path, MjSim

from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
import multiprocessing

FIXED_ENV = False
GT_STATE = True
EARLY_TERMINATION = True

class obstract(SingleArmEnv):
    def __init__(self,
        robots,
        env_configuration="default",
        controller_configs = None,
        gripper_types = "default",
        initialization_noise = "default",
        use_latch = True,
        use_camera_obs = True,
        use_object_obs = True,
        reward_scale = 1.0,
        reward_shaping = False,
        placement_initializer = None,
        has_renderer = False,
        has_offscreen_renderer = True,
        render_camera = "frontview",
        render_collision_mesh = False,
        render_visual_mesh = True,
        render_gpu_device_id = -1,
        control_freq = 20,
        horizon = 1000,
        ignore_done = False,
        hard_reset = True,
        camera_names = "agentview",
        camera_heights = 256,
        camera_widths = 256,
        camera_depths = False,
    ):
        parent_params = super()._default_hparams()
        self.reset_xml = '/home/sumail/robosuite/robosuite/models/assets/shelf.xml'
        self._frame_height = parent_params.viewer_image_height
        self._frame_width = parent_params.viewer_image_width

        self._reset_sim(self.reset_xml)

        self._base_adim, self._base_sdim = None, None  # state/action dimension of Mujoco control
        self._adim, self._sdim = None, None  # state/action dimension presented to agent
        self.num_objects, self._n_joints = None, None
        self._goal_obj_pose = None
        self._goaldistances = []

        self._ncam = parent_params.ncam
        if self._ncam == 2:
            self.cameras = ['maincam', 'leftcam']
        elif self._ncam == 1:
            self.cameras = ['maincam']
        else:
            raise ValueError
        self._last_obs = None
        self._hp = parent_params

        self._adim = 4
        self.substeps = 500
        self.low_bound = np.array([-0.4, -0.4, -0.05])
        self.high_bound = np.array([0.4, 0.4, 0.15])
        self.ac_high = np.array([0.02, 0.02, 0.02, 0.1])
        self.ac_low = -self.ac_high
        self.action_space = Box(self.ac_low, self.ac_high)
        self._previous_target_qpos = None
        self.target_height_thresh = 0.03
        self.object_fall_thresh = -0.03
        self.obj_y_dist_range = np.array([0.05, 0.05])
        self.obj_x_range = np.array([-0.2, -0.05])
        self.randomize_objects = not FIXED_ENV
        self.gt_state = GT_STATE
        self._max_episode_steps = 25

        if self.gt_state:
            self.observation_space = Box(low=-np.inf,
                                         high=np.inf,
                                         shape=(27,))
        else:
            self.observation_space = (48, 64, 3)
        self.reset()    # ?

        # reward configuration
        self.use_latch = use_latch
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def _reset_sim(self, model_path):
        """
        Creates a MjSim from passed in model_path
        :param model_path: Absolute path to model file
        :return: None
        """
        self._model_path = model_path
        self.sim = MjSim(load_model_from_path(self._model_path))

    def reward(self, action):
        reward = 0

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # else, we consider only the case if we're using shaped rewards
        elif self.reward_shaping:
            # Add reaching component
            dist = np.linalg.norm(self._gripper_to_handle)
            reaching_reward = 0.25 * (1 - np.tanh(10.0 * dist))
            reward += reaching_reward
            # Add rotating component if we're using a locked door
            if self.use_latch:
                handle_qpos = self.sim.data.qpos[self.handle_qpos_addr]
                reward += np.clip(0.25 * np.abs(handle_qpos / (0.5 * np.pi)), -0.25, 0.25)

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 1.0

        return reward

    def _load_model(self):
        super._load_model()
        self.model = self.sim.model

    def _setup_references(self):
        super()._setup_references()
        self.object_body_ids = dict()
        # ?

    def _setup_observables(self):
        observables = super()._setup_observables()
        if self.use_camera_obs:
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # sensor?

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            # ?

    def _check_success(self):
        pass

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
        if vis_settings["grippers"]:
            # ?
