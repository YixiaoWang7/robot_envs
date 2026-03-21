from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import CrossObject, BoxObject, CylinderObject, Bin, Cup, Plate, Mug
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, UniformApartRandomSampler
from robosuite.utils.transform_utils import convert_quat

object_A_index = {"cross": 0, "cube": 1, "cylinder": 2}
object_B_index = {"bin": 0, "cup": 1, "plate": 2}
one_hot_dict = {"place": 0, "the": 1, "into": 2, "cross": 3, "cube": 4, "cylinder": 5, "bin": 6, "cup": 7, "plate": 8}

class CG_L2(ManipulationEnv):
    """
    This class corresponds to the L2 task of CG benchmark, modified from Lift.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        base_types (None or str or list of str): type of base, used to instantiate base models from base factory.
            Default is "default", which is the default base associated with the robot(s) the 'robots' specification.
            None results in no base, and any other (valid) model overrides the default base. Should either be
            single str if same base type is to be used for all robots or else it should be a list of the same
            length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        lite_physics (bool): Whether to optimize for mujoco forward and step calls to reduce total simulation overhead.
            Set to False to preserve backward compatibility with datasets collected in robosuite <= 1.4.1.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.
        
        task (str): Language to use for task tasks.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        strategy,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer_A=None,
        placement_initializer_B=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        task=None,
        initial_qpos=None,
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
        self.placement_initializer_A = placement_initializer_A
        self.placement_initializer_B = placement_initializer_B
        self.object_A_init_pos = None
        self.object_B_init_pos = None

        # parse language input
        self.strategy = strategy
        self.task = task
        if self.strategy == "fixed":
            if self.task is None:
                raise ValueError("Task is required for fixed strategy")
        else:
            self.task = self.random_task()
        
        
        self.language_vector = np.zeros(one_hot_dict.__len__())
        self.parse_task()

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
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
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            initial_qpos=initial_qpos,
        )
    
    def parse_task(self):
        """
        Parse the task string and set the task configuration accordingly
        """
        self.object_A_index = 0
        self.object_B_index = 0
        words = self.task.split()

        self.language_vector = np.zeros(one_hot_dict.__len__())
        for word in words:
            self.language_vector[one_hot_dict[word]] = 1
        
        i = 0
        while i < len(words):
            if words[i] in object_A_index:
                self.object_A_index = object_A_index[words[i]]
                i += 1
                break
            i += 1
        while i < len(words):
            if words[i] in object_B_index:
                self.object_B_index = object_B_index[words[i]]
                i += 1
                break
            i += 1

    def update_task(self, new_task):
        """
        Update the language task and reset task configuration accordingly
        
        Args:
            new_task (str): New task to set ("place", "push", or "stack")
        """
        # Validate new task
        new_task = new_task.lower()
        
        # Only proceed if task actually changed
        if new_task != self.task:
            self.task = new_task
            self.parse_task()
    
    def random_task(self):
        """
        Generate a random task string based on the available objects

        Returns:
            str: Randomly generated task
        """
        
        # # "L2": [[0,0],[0,2],[2,0],[2,2]]
        # "Sfull":[[0,2],[1,0],[2,1]]
        # "diagmid":[[0,2],[1,0],[1,2],[2,0],[2,1]]
        # "diagcorner":[[0,1],[1,0],[1,2],[2,0],[2,1]]
        
        from random import choice
        if self.strategy == "all":
            random_list = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)] # all
        elif self.strategy == "L":
            random_list = [(1, 1), (1, 2), (2, 1), (2, 2)] # L-unseen
        elif self.strategy == "only-00":
            random_list = [(0, 0)] # only-00
        elif self.strategy == "S":
            random_list = [(0, 2), (1, 0), (2, 0), (2, 1)] # S-unseen
        elif self.strategy == "diag":
            random_list = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)] # diag-unseen
        elif self.strategy == "L2":
            random_list = [(0, 0), (0, 2), (2, 0), (2, 2)] # L2-unseen
        elif self.strategy == "Sfull":
            random_list = [(0, 2), (1, 0), (2, 1)] # Sfull-unseen
        elif self.strategy == "diagmid":
            random_list = [(0, 2), (1, 0), (1, 2), (2, 0), (2, 1)] # diagmid-unseen
        elif self.strategy == "diagcorner":
            random_list = [(0, 1), (1, 0), (1, 2), (2, 0), (2, 1)] # diagcorner-unseen
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")
        
        
        object_A_idx, object_B_idx = choice(random_list)
        object_A = list(object_A_index.keys())[object_A_idx]
        object_B = list(object_B_index.keys())[object_B_idx]
        print(f"Random task: {object_A} into {object_B}")
        
        
        return f"place the {object_A} into the {object_B}"
    
    def reset(self):
        """
        Reset the environment and handle task changes, generating a new task each time.
        """
        # Generate and parse a new task
        if self.strategy != "fixed":
            self.task = self.random_task()
            
            
        self.parse_task()

        # Call parent reset
        obs = super().reset()
        
        # # If task was changed since last reset, reconfigure
        # if hasattr(self, '_last_task') and self._last_task != self.task:
        #     self.parse_task()
        #     # Need to reset again after reconfiguration
        #     obs = super().reset()
        
        # # Store current task for next reset comparison
        # self._last_task = self.task
        
        self.object_A_init_pos = self.sim.data.body_xpos[self.object_A_body_id].copy()
        self.object_B_init_pos = self.sim.data.body_xpos[self.object_B_body_id].copy()

        return obs
    
    def reward(self, action=None):
        """
        Reward function for the task.

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        return self.place_reward(action)

    def place_reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 3.00 is provided if object A is placed to object B

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arm to reach object A
            - Grasping: in {0, 0.5}, non-zero if arm is grasping object A
            - Placing: in [0, 1], to encourage the arm to place object A into object B

        The sparse reward only consists of the lifting component.

        Note that the final reward is normalized and scaled by
        reward_scale / 3.00 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0
        success_reward = 3.00

        # sparse completion reward
        if self._check_success():
            reward = success_reward

        # use a shaping reward
        elif self.reward_shaping:

            # reaching reward
            reaching_dist = self._gripper_to_target(
                gripper=self.robots[0].gripper, target=self.object_A.root_body, target_type="body", return_distance=True
            )
            reaching_reward = 1 - np.tanh(10.0 * reaching_dist)

            # grasping reward
            is_grasping = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.object_A)
            grasping_reward = 0.5 if is_grasping else 0.0

            # placing reward
            object_A_pos = self.sim.data.body_xpos[self.object_A_body_id]
            object_B_pos = self.sim.data.body_xpos[self.object_B_body_id]
            placing_dist = np.linalg.norm(object_A_pos - object_B_pos)
            placing_reward = 1 - np.tanh(10.0 * placing_dist) if is_grasping else 0.0
            
            
            reward = reaching_reward + grasping_reward + placing_reward

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / success_reward

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize object A and B
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
        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="bluewood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        # ----------------------------
        # Contact friction tuning
        # ----------------------------
        # MuJoCo geom friction is (sliding, torsional, rolling).
        # Typical stable defaults in robosuite-style setups keep torsional / rolling small
        # to avoid "sticky / glued" contacts.
        # OBJECT_FRICTION = [1.0, 0.01, 0.001]
        # CONTAINER_FRICTION = [1.0, 0.005, 0.0001]
        self.cross = CrossObject(
            name="cross",
            arm_length=0.03,
            arm_width=0.01,
            height=0.02,
            # friction=OBJECT_FRICTION,
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cube = BoxObject(
            name="cube",
            size=[0.015, 0.015, 0.015],
            # friction=OBJECT_FRICTION,
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cylinder = CylinderObject(
            name="cylinder",
            size=[0.015, 0.015],
            friction=[1,1,1],
            # friction=OBJECT_FRICTION,
            rgba=[1, 0, 0, 1],
            material=redwood,
        )

        self.bin = Bin(
            name="bin",
            transparent_walls=False,
            bin_size=[0.06, 0.06, 0.04],
            wall_thickness=0.005,
            # friction=CONTAINER_FRICTION,
            rgba=[0, 0, 1, 1],
            use_texture=False,
            material=bluewood,
        )
        # self.cup = Cup(
        #     name="cup",
        #     outer_radius=0.035,
        #     inner_radius=0.03,
        #     height=0.04,
        #     thickness=0.005,
        #     rgba=[0, 0, 1, 1],
        #     material=bluewood,
        # )
        self.cup = Mug(
            name="cup",
            outer_radius=0.035,
            inner_radius=0.03,
            mug_height=0.04,
            add_handle=True,
            handle_outer_radius=0.02,
            handle_inner_radius=0.015,
            # friction=CONTAINER_FRICTION,
            rgba=[0, 0, 1, 1],
            material=bluewood,
        )
        self.plate = Plate(
            name="plate",
            radius=0.03,
            # friction=CONTAINER_FRICTION,
            rgba=[0, 0, 1, 1],
            material=bluewood,
        )
        if self.object_A_index == 0:
            self.object_A = self.cross
        elif self.object_A_index == 1:
            self.object_A = self.cube
        elif self.object_A_index == 2:
            self.object_A = self.cylinder

        if self.object_B_index == 0:
            self.object_B = self.bin
        elif self.object_B_index == 1:
            self.object_B = self.cup
        elif self.object_B_index == 2:
            self.object_B = self.plate
        
        self.object_A_list = [self.cross, self.cube, self.cylinder]
        self.object_B_list = [self.bin, self.cup, self.plate]
        self.objects = [self.cross, self.cube, self.cylinder, self.bin, self.cup, self.plate]

        # Create placement initializer
        if self.placement_initializer_A is not None:
            self.placement_initializer_A.reset()
            self.placement_initializer_A.add_objects(self.object_A_list)
        else:
            self.placement_initializer_A = UniformApartRandomSampler(
                name="Object_A_Sampler",
                mujoco_objects=self.object_A_list,
                x_range=[-0.15, 0.15],
                y_range=[-0.25, -0.02],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                min_distance=0.07,
            )
        if self.placement_initializer_B is not None:
            self.placement_initializer_B.reset()
            self.placement_initializer_B.add_objects(self.object_B_list)
        else:
            self.placement_initializer_B = UniformApartRandomSampler(
                name="Object_B_Sampler",
                mujoco_objects=self.object_B_list,
                x_range=[-0.15, 0.15],
                y_range=[0.02, 0.25],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                min_distance=0.00,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects,
        )


    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.object_A_body_id = self.sim.model.body_name2id(self.object_A.root_body)
        self.object_B_body_id = self.sim.model.body_name2id(self.object_B.root_body)
        if self.object_A_init_pos is None:
            self.object_A_init_pos = self.sim.data.body_xpos[self.object_A_body_id].copy()
        if self.object_B_init_pos is None:
            self.object_B_init_pos = self.sim.data.body_xpos[self.object_B_body_id].copy()

        self.cross_body_id = self.sim.model.body_name2id(self.cross.root_body)
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)
        self.cylinder_body_id = self.sim.model.body_name2id(self.cylinder.root_body)
        self.bin_body_id = self.sim.model.body_name2id(self.bin.root_body)
        self.cup_body_id = self.sim.model.body_name2id(self.cup.root_body)
        self.plate_body_id = self.sim.model.body_name2id(self.plate.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # define observables modality
            modality = "object"

            # object-related observables            
            @sensor(modality=modality)
            def cross_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cross_body_id])

            @sensor(modality=modality)
            def cross_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cross_body_id]), to="xyzw")

            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")

            @sensor(modality=modality)
            def cylinder_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cylinder_body_id])

            @sensor(modality=modality)
            def cylinder_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cylinder_body_id]), to="xyzw")

            @sensor(modality=modality)
            def bin_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.bin_body_id])

            @sensor(modality=modality)
            def bin_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.bin_body_id]), to="xyzw")

            @sensor(modality=modality)
            def cup_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cup_body_id])

            @sensor(modality=modality)
            def cup_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cup_body_id]), to="xyzw")

            @sensor(modality=modality)
            def plate_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.plate_body_id])

            @sensor(modality=modality)
            def plate_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.plate_body_id]), to="xyzw")

            sensors = [
                cross_pos, cross_quat, cube_pos, cube_quat, cylinder_pos, cylinder_quat,
                bin_pos, bin_quat, cup_pos, cup_quat, plate_pos, plate_quat
            ]

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])

            # gripper to cube position sensor; one for each arm
            sensors += [
                self._get_obj_eef_sensor(full_pf, "object_A_pos", f"{arm_pf}gripper_to_object_A_pos", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]

            @sensor(modality="object_A")
            def object_A_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.object_A_body_id])
            @sensor(modality="object_A")
            def object_A_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.object_A_body_id]), to="xyzw")
            @sensor(modality="object_B")
            def object_B_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.object_B_body_id])
            @sensor(modality="object_B")
            def object_B_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.object_B_body_id]), to="xyzw")
            sensors += [object_A_pos, object_A_quat, object_B_pos, object_B_quat]

            # @sensor(modality="language")
            # def language_task(obs_cache):
            #     return self.task
            # sensors += [language_task]
            @sensor(modality="language")
            def language_vector(obs_cache):
                return self.language_vector
            sensors += [language_vector]

            # TODO: the language condition may be replaced by masked image input

            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )
        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements_A = self.placement_initializer_A.sample(trial_times=100000)
            object_placements_B = self.placement_initializer_B.sample(trial_times=100000)
            self.object_A_init_pos = None
            self.object_B_init_pos = None


            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements_A.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
            for obj_pos, obj_quat, obj in object_placements_B.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to object A.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to object A
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.object_A)
    
    def _check_success(self):
        """
        Check if the task has been successfully completed.

        Returns:
            bool: True if task is successfully completed
        """
        gripper_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]
        table_height = self.model.mujoco_arena.table_offset[2]
        lift_check = gripper_pos[2] - table_height > 0.20

        # print(self.place_success(), lift_check)
        # lift_check = True

        return self.place_success() and lift_check

    def place_success(self):
        """
        Check if object_A have been placed into object_B.

        Returns:
            bool: True if object_A is successfully placed inside object_B
        """
        # Get positions of both objects
        obj_a_pos = self.sim.data.body_xpos[self.object_A_body_id]
        obj_b_pos = self.sim.data.body_xpos[self.object_B_body_id]
        
        # Get relative position of object_A with respect to object_B
        rel_pos = obj_a_pos - obj_b_pos
        
        if self.object_B_index == 0:  # Bin case
            bin_size = self.object_B.bin_size
            
            x_check = abs(rel_pos[0]) < bin_size[0] / 2
            y_check = abs(rel_pos[1]) < bin_size[1] / 2
            z_check = abs(rel_pos[2]) < 0.06
                        
            return x_check and y_check and z_check

        elif self.object_B_index == 1:  # Cup case
            cup_inner_r = self.object_B.r1
            cup_outer_r = self.object_B.r2
            cup_height = self.object_B.mug_height
            horizontal_dist = np.linalg.norm(rel_pos[:2])
            
            radius_check = horizontal_dist < (cup_inner_r + cup_outer_r) / 2
            height_check = (rel_pos[2] > -cup_height / 1) and (rel_pos[2] < cup_height / 1)
            
            return radius_check and height_check
        
        elif self.object_B_index == 2: # Plate case
            plate_radius = self.object_B.radius
            plate_rim_width = self.object_B.rim_width
            plate_height = self.object_B.height
            horizontal_dist = np.linalg.norm(rel_pos[:2])

            radius_check = horizontal_dist < (plate_radius - plate_rim_width)
            height_check = abs(rel_pos[2]) < 0.06

            return radius_check and height_check
        
        return False
        
