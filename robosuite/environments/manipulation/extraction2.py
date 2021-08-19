from collections import OrderedDict
import numpy as np

from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import Observable, sensor


class Stack(SingleArmEnv):
    def __init__(
        self,
        robots,
            env_configuration="default",
            controller_configs=None,
            gripper_types="default",
            initialization_noise="default",
            table_full_size=(0.8, 0.8, 0.05),
            table_friction=(1., 5e-3, 1e-4),
            use_camera_obs=True,
            use_object_obs=True,
            reward_scale=1.0,
            reward_shaping=False,
            placement_initializer=None,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="frontview",
            render_collision_mesh=False,
            render_visual_mesh=True,
            render_gpu_device_id=-1,
            control_freq=20,
            horizon=1000,
            ignore_done=False,
            hard_reset=True,
            camera_names="agentview",
            camera_heights=256,
            camera_widths=256,
            camera_depths=False,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
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

    def reward(self, action):
        r_reach, r_lift, r_place = self.staged_rewards()
        if self.reward_shaping:
            reward = max(r_reach, r_lift, r_place)
        else:
            reward = 2.0 if r_place > 0 else 0.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.0
        return reward

    def staged_rewards(self):
        # reach is successful when gripper site close to center of cubeB
        cubeA_pos = self.sim.data.body_xpos[self.cubaA_body_id]
        cubeB_pos = self.sim.data.body_xpos[self.cubaB_body_id]
        cubeC_pos = self.sim.data.body_xpos[self.cubaC_body_id]
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - cubeB_pos)
        r_reach = (1-np.tanh(10.0 * dist)) * 0.25

        # grasp reward
        grasping_cubeB = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeB)
        if grasping_cubeB:
            r_reach += 0.25

        # lift is successful when cubeB is above the table top by a margin
        cubeB_height = cubeB_pos[2]
        table_height = self.table_offset[2]
        cubeB_lifted = cubeB_height > table_height + 0.08
        r_lift = 1.0 if cubeB_lifted else 0.0

        if cubeB_lifted:
            horiz_dist = np.linalg.norm(
                np.array(cubeB_pos[:2]) - np.array(cubeA_pos[:2])
            )
            r_lift += 0.5 * (1 - np.tanh(horiz_dist))

        # place is successful when B is placed and gripper is not holding the object
        r_place = 0
        cubeB_touching_cubaA = self.check_contact(self.cubeA, self.cubeB)
        if not grasping_cubeB and r_lift > 0 and cubeB_touching_cubaA:
            r_place = 2.0
        return r_reach, r_lift, r_place


    def _load_model(self):
        super()._load_model()
        print("I am extraction===================================================================================")
        # adjust base pose
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        mujoco_arena.set_origin([0, 0, 0])
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }

        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="bluewood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cubeA = BoxObject(
            name="cubeA",
            size_min=[0.01, 0.02, 0.03],
            size_max=[0.01, 0.02, 0.03],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=[0.025, 0.025, 0.05],
            size_max=[0.025, 0.025, 0.05],
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )
        self.cubeC = BoxObject(
            name="cubeC",
            size_min=[0.02, 0.025, 0.04],
            size_max=[0.02, 0.025, 0.04],
            rgba=[0, 0, 1, 1],
            material=bluewood,
        )
        cubes = [self.cubeA, self.cubeB, self.cubeC]
        # placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_object(cubes)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=cubes,
                x_range=[-0.12, 0.12],
                y_range=[-0.1, 0.1],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
            self.model = ManipulationTask(
                mujoco_arena=mujoco_arena,
                mujoco_robots=[robot.robot_model for robot in self.robots],
                mujoco_objects=cubes,
            )

    def _setup_references(self):
        super()._setup_references()
        self.cubaA_body_id = self.sim.model.body_name2id(self.cubeA.root_body)
        self.cubaB_body_id = self.sim.model.body_name2id(self.cubeB.root_body)
        self.cubaC_body_id = self.sim.model.body_name2id(self.cubeC.root_body)

    def _reset_internal(self):
        super()._reset_internal()
        # if not loading from xml, reset all object positions
        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def _setup_observables(self):
        observables = super()._setup_observables()

        # low-level object info
        if self.use_object_obs:
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # position and rotation of the first cube
            @sensor(modality=modality)
            def cubeA_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cubaA_body_id])

            @sensor(modality=modality)
            def cubeA_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cubaA_body_id]), to="xyzw")

            @sensor(modality=modality)
            def cubeB_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cubaB_body_id])

            @sensor(modality=modality)
            def cubeB_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cubaB_body_id]), to="xyzw")

            @sensor(modality=modality)
            def cubeC_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cubaC_body_id])

            @sensor(modality=modality)
            def cubeC_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cubaC_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_cubeA(obs_cache):
                return obs_cache["cubeA_pos"] - obs_cache[f"{pf}eef_pos"] if \
                    "cubeA_pos" in obs_cache and f"{pf}eef_pos" in obs_cache else np.zeros(3)

            @sensor(modality=modality)
            def gripper_to_cubeB(obs_cache):
                return obs_cache["cubeB_pos"] - obs_cache[f"{pf}eef_pos"] if \
                    "cubeB_pos" in obs_cache and f"{pf}eef_pos" in obs_cache else np.zeros(3)

            @sensor(modality=modality)
            def gripper_to_cubeC(obs_cache):
                return obs_cache["cubeC_pos"] - obs_cache[f"{pf}eef_pos"] if \
                    "cubeC_pos" in obs_cache and f"{pf}eef_pos" in obs_cache else np.zeros(3)

            @sensor(modality=modality)
            def cubeA_to_cubeB(obs_cache):
                return obs_cache["cubeB_pos"] - obs_cache["cubeA_pos"] if \
                    "cubeA_pos" in obs_cache and "cubeB_pos" in obs_cache else np.zeros(3)

            @sensor(modality=modality)
            def cubeB_to_cubeC(obs_cache):
                return obs_cache["cubeC_pos"] - obs_cache["cubeB_pos"] if \
                    "cubeB_pos" in obs_cache and "cubeC_pos" in obs_cache else np.zeros(3)

            @sensor(modality=modality)
            def cubeA_to_cubeC(obs_cache):
                return obs_cache["cubeC_pos"] - obs_cache["cubeA_pos"] if \
                    "cubeA_pos" in obs_cache and "cubeC_pos" in obs_cache else np.zeros(3)

            sensors = [cubeA_pos, cubeA_quat, cubeB_pos, cubeB_quat, cubeC_pos, cubeC_quat, gripper_to_cubeA, gripper_to_cubeB, gripper_to_cubeC, cubeA_to_cubeB, cubeA_to_cubeC, cubeB_to_cubeC]
            names = [s.__name__ for s in sensors]

            # create ovservables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )
        return observables

    def _check_success(self):
        _, _, r_place = self.staged_rewards()
        return r_place > 0

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cubeA)