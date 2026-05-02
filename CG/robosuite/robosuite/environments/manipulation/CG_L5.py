from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject, BoxObject, CompositeBodyObject, CrossObject, CylinderObject, MilkObject, Bin, mug, Plate, Mug
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, UniformApartRandomSampler
from robosuite.utils.transform_utils import convert_quat

object_A_index = {"cross": 0, "cube": 1, "cylinder": 2, "milk": 3}
object_B_index = {"bin": 0, "mug": 1, "plate": 2, "mug_no_handle": 3}
button_color_index = {"red": 0, "green": 1, "blue": 2, "yellow": 3}
one_hot_dict = {
    "place": 0,
    "the": 1,
    "into": 2,
    "cross": 3,
    "cube": 4,
    "cylinder": 5,
    "milk": 6,
    "bin": 7,
    "mug": 8,
    "plate": 9,
    "mug_no_handle": 10,
    "then": 11,
    "press": 12,
    "button": 13,
    "red": 14,
    "green": 15,
    "blue": 16,
    "yellow": 17,
}

class CG_L5(ManipulationEnv):
    """
    This class corresponds to the L5 task of CG benchmark, modified from Lift.

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
        task: str | None = None,
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
        self.has_completed_place = False

        # Parse language input (single-task env; task can be changed via `set_task()`).
        if task is None:
            raise ValueError(
                "CG_L5 requires an explicit task string "
                "(e.g. 'place the cross into the bin then press the red button')."
            )
        self.task = str(task).lower()
        
        
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
        self.target_button_index = None
        words = self.task.split()

        self.language_vector = np.zeros(one_hot_dict.__len__())
        for word in words:
            if word not in one_hot_dict:
                raise ValueError(f"Unknown word '{word}' in CG_L5 task: {self.task}")
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
        for word in words:
            if word in button_color_index:
                self.target_button_index = button_color_index[word]
                break
        if self.target_button_index is None:
            raise ValueError(
                "CG_L5 task must specify a target button color "
                "(red, green, blue, or yellow), e.g. "
                "'place the cube into the bin then press the red button'."
            )

    def _refresh_task_pointers(self) -> None:
        """
        Refresh `object_A/object_B` and their MuJoCo body ids from the current indices.

        Safe to call before the model/sim exists.
        """
        if not hasattr(self, "object_A_list") or not hasattr(self, "object_B_list"):
            return
        self.object_A = self.object_A_list[int(self.object_A_index)]
        self.object_B = self.object_B_list[int(self.object_B_index)]
        if hasattr(self, "button_list"):
            self.target_button = self.button_list[int(self.target_button_index)]
        if hasattr(self, "sim") and self.sim is not None:
            self.object_A_body_id = self.sim.model.body_name2id(self.object_A.root_body)
            self.object_B_body_id = self.sim.model.body_name2id(self.object_B.root_body)
            if hasattr(self, "button_list"):
                self.button_body_ids = [
                    self.sim.model.body_name2id(button.root_body) for button in self.button_list
                ]
                self.button_qpos_addrs = [
                    self.sim.model.get_joint_qpos_addr(button.slide_joint_name) for button in self.button_list
                ]
                self.target_button_body_id = self.button_body_ids[int(self.target_button_index)]
                self.target_button_qpos_addr = self.button_qpos_addrs[int(self.target_button_index)]

    def set_task(self, task: str) -> None:
        """Set the current task string and refresh internal pointers."""
        self.task = str(task).lower()
        self.parse_task()
        self._refresh_task_pointers()

    def update_task(self, new_task):
        """
        Update the language task and reset task configuration accordingly
        
        Args:
            new_task (str): New task to set ("place", "push", or "stack")
        """
        self.set_task(new_task)
    
    def reset(self):
        """
        Reset the environment for the current task.
        """
        self.has_completed_place = False
        self.parse_task()
        self._refresh_task_pointers()

        # Call parent reset
        obs = super().reset()
        self._refresh_task_pointers()
        
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
            if self.has_completed_place:
                gripper_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]
                button_pos = self.sim.data.body_xpos[self.target_button_body_id]
                button_xy_dist = np.linalg.norm(gripper_pos[:2] - button_pos[:2])
                button_reaching_reward = 1 - np.tanh(10.0 * button_xy_dist)
                button_qpos = self.sim.data.qpos[int(self.target_button_qpos_addr)]
                press_reward = np.clip(button_qpos / self.button_press_joint_threshold, 0.0, 1.0)
                reward = 2.0 + button_reaching_reward + press_reward

            else:
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

        # initialize object A and B — procedural rgba textures (distinct hues) + low specular to limit glare.
        tex_attrib_base = {"type": "cube"}

        def make_solid_mat(rgba, tex_name, mat_name, specular="0.18", shininess="0.05"):
            return CustomMaterial(
                texture=rgba,
                tex_name=tex_name,
                mat_name=mat_name,
                tex_attrib=tex_attrib_base,
                mat_attrib={
                    "texrepeat": "1 1",
                    "specular": specular,
                    "shininess": shininess,
                },
            )

        # Maximize hue separation (rose / green / blue / amber / purple / cyan / orange). Milk stays its own mesh.
        col_cross = [0.93, 0.38, 0.48, 1.0]  # rose
        col_cube = [0.30, 0.72, 0.42, 1.0]  # green
        col_cylinder = [0.28, 0.42, 0.92, 1.0]  # blue
        col_bin = [0.90, 0.62, 0.18, 1.0]  # amber
        col_mug = [0.58, 0.32, 0.82, 1.0]  # purple
        col_plate = [0.20, 0.78, 0.85, 1.0]  # cyan
        col_mug_nh = [0.95, 0.48, 0.20, 1.0]  # orange
        col_button_red = [0.92, 0.12, 0.12, 1.0]
        col_button_green = [0.08, 0.70, 0.20, 1.0]
        col_button_blue = [0.10, 0.25, 0.95, 1.0]
        col_button_yellow = [0.95, 0.82, 0.10, 1.0]

        mat_cross = make_solid_mat(col_cross, "cg_cross_tex", "cg_cross_mat", "0.16", "0.04")
        mat_cube = make_solid_mat(col_cube, "cg_cube_tex", "cg_cube_mat", "0.17", "0.05")
        mat_cylinder = make_solid_mat(col_cylinder, "cg_cyl_tex", "cg_cyl_mat", "0.17", "0.05")
        # Bin geoms reference mat name "dark_wood_mat" when use_texture=True.
        mat_bin = make_solid_mat(col_bin, "cg_bin_tex", "dark_wood_mat", "0.15", "0.04")
        mat_mug = make_solid_mat(col_mug, "cg_mug_tex", "cg_mug_mat", "0.18", "0.05")
        mat_plate = make_solid_mat(col_plate, "cg_plate_tex", "cg_plate_mat", "0.17", "0.05")
        mat_mug_nh = make_solid_mat(col_mug_nh, "cg_mug_nh_tex", "cg_mug_nh_mat", "0.18", "0.05")
        mat_button_red = make_solid_mat(col_button_red, "cg_button_red_tex", "cg_button_red_mat", "0.12", "0.03")
        mat_button_green = make_solid_mat(col_button_green, "cg_button_green_tex", "cg_button_green_mat", "0.12", "0.03")
        mat_button_blue = make_solid_mat(col_button_blue, "cg_button_blue_tex", "cg_button_blue_mat", "0.12", "0.03")
        mat_button_yellow = make_solid_mat(col_button_yellow, "cg_button_yellow_tex", "cg_button_yellow_mat", "0.12", "0.03")

        # ----------------------------
        # Contact friction tuning
        # ----------------------------
        # MuJoCo geom friction is (sliding, torsional, rolling).
        # Typical stable defaults in robosuite-style setups keep torsional / rolling small
        # to avoid "sticky / glued" contacts.
        # OBJECT_FRICTION = [1.0, 0.01, 0.001]
        # CONTAINER_FRICTION = [1.0, 0.005, 0.0001]
        # Manipulatable objects (primitives): rgba matches procedural texture above.
        self.cross = CrossObject(
            name="cross",
            arm_length=0.03,
            arm_width=0.01,
            height=0.03,
            rgba=col_cross,
            material=mat_cross,
        )
        self.cube = BoxObject(
            name="cube",
            size=[0.015, 0.015, 0.015],
            rgba=col_cube,
            material=mat_cube,
        )
        self.cylinder = CylinderObject(
            name="cylinder",
            size=[0.015, 0.015],
            friction=[1, 1, 1],
            rgba=col_cylinder,
            material=mat_cylinder,
        )
        self.milk = MilkObject(name="milk")
        # Resize milk to be roughly cube-sized (~3cm): non-uniform scale (XY vs Z) based on milk.xml sites.
        # milk.xml: horizontal_radius_site ~= (0.025, 0.025, 0) => r_xy ~= 0.035
        #           bottom_site z=-0.085, top_site z=+0.075 => half-height ~= 0.085
        scale_xy = 0.015 / 0.025
        scale_z = 0.015 / 0.025
        self.milk.set_scale([scale_xy, scale_xy, scale_z])

        self.bin = Bin(
            name="bin",
            transparent_walls=False,
            bin_size=[0.06, 0.06, 0.04],
            wall_thickness=0.005,
            rgba=col_bin,
            use_texture=True,
            material=mat_bin,
        )
        # self.mug = mug(
        #     name="mug",
        #     outer_radius=0.035,
        #     inner_radius=0.03,
        #     height=0.04,
        #     thickness=0.005,
        #     rgba=[0, 0, 1, 1],
        #     material=mat_mug,
        # )
        # Note: robosuite Mug's internal sub-objects are named "mug_*".
        # If we also name this object "mug", robosuite's prefixing logic will double-prefix
        # names at lookup time (e.g. "mug_mug_body_*"), causing geom-id mapping failures.
        self.mug = Mug(
            name="mug_container",
            outer_radius=0.035,
            inner_radius=0.03,
            mug_height=0.04,
            add_handle=True,
            handle_outer_radius=0.02,
            handle_inner_radius=0.015,
            rgba=col_mug,
            material=mat_mug,
        )
        self.plate = Plate(
            name="plate",
            radius=0.03,
            rgba=col_plate,
            material=mat_plate,
        )
        # Handle-less mug (still a Mug object, but with add_handle=False)
        self.mug_no_handle = Mug(
            name="mug_no_handle_container",
            outer_radius=0.035,
            inner_radius=0.03,
            mug_height=0.04,
            add_handle=False,
            rgba=col_mug_nh,
            material=mat_mug_nh,
        )
        def make_button(name, color_rgba):
            base = CylinderObject(
                name="base",
                size=[0.024, 0.004],
                rgba=[0.08, 0.08, 0.08, 1.0],
                joints=None,
                obj_type="visual",
            )
            cap = CylinderObject(
                name="cap",
                size=[0.016, 0.005],
                rgba=color_rgba,
                joints=None,
            )
            slide_joint = {
                "name": "button_slide",
                "type": "slide",
                "axis": "0 0 1",
                "limited": "true",
                "range": "-0.020 0",
                "damping": "1.0",
                "stiffness": "18.0",
                "springref": "0",
            }
            button = CompositeBodyObject(
                name=name,
                objects=[base, cap],
                object_locations=[np.zeros(3), np.array([0.0, 0.0, 0.009])],
                object_parents=[None, base.root_body],
                body_joints={cap.root_body: [slide_joint]},
                joints=None,
                total_size=[0.024, 0.024, 0.009],
            )
            button.slide_joint_name = f"{button.name}_button_slide"
            return button

        self.button_red = make_button("button_red", col_button_red)
        self.button_green = make_button("button_green", col_button_green)
        self.button_blue = make_button("button_blue", col_button_blue)
        self.button_yellow = make_button("button_yellow", col_button_yellow)

        self.button_list = [self.button_red, self.button_green, self.button_blue, self.button_yellow]
        self.button_colors = ["red", "green", "blue", "yellow"]
        self.button_rgba_colors = [col_button_red, col_button_green, col_button_blue, col_button_yellow]
        self.button_radius = 0.024
        self.button_height = 0.018
        self.button_cap_top_offset = 0.014
        self.button_press_depth = 0.026
        self.button_press_xy_threshold = 0.020
        self.button_press_z_threshold = 0.045
        self.button_press_joint_threshold = -0.006
        self.lamp_off_rgba = [0.04, 0.04, 0.04, 1.0]
        self.lamp_base_position = np.array([-0.30, -0.3, self.table_offset[2] + 0.008])
        self.lamp_pole_position = np.array([-0.30, -0.3, self.table_offset[2] + 0.080])
        self.lamp_shade_position = np.array([-0.30, -0.3, self.table_offset[2] + 0.170])
        self.lamp_bulb_position = np.array([-0.30, -0.3, self.table_offset[2] + 0.145])
        self.lamp_base = CylinderObject(
            name="desk_lamp_base",
            size=[0.050, 0.008],
            rgba=[0.08, 0.08, 0.08, 1.0],
            joints=None,
            obj_type="visual",
        )
        self.lamp_pole = CylinderObject(
            name="desk_lamp_pole",
            size=[0.008, 0.080],
            rgba=[0.12, 0.12, 0.12, 1.0],
            joints=None,
            obj_type="visual",
        )
        shade_radii = [0.022, 0.031, 0.039, 0.048, 0.056]
        shade_ring_height = 0.004
        self.lamp_shade_off_rgba = [0.18, 0.18, 0.18, 0.85]
        self.lamp_shade_alpha = 0.85
        shade_rings = [
            CylinderObject(
                name=f"shade_ring_{idx}",
                size=[radius, shade_ring_height],
                rgba=self.lamp_shade_off_rgba,
                joints=None,
                obj_type="visual",
            )
            for idx, radius in enumerate(shade_radii)
        ]
        shade_z_offsets = np.linspace(0.019, -0.019, len(shade_rings))
        self.lamp_shade = CompositeBodyObject(
            name="desk_lamp_shade",
            objects=shade_rings,
            object_locations=[np.array([0.0, 0.0, z]) for z in shade_z_offsets],
            object_parents=[None] * len(shade_rings),
            joints=None,
            total_size=[0.056, 0.056, 0.022],
        )
        self.lamp = BallObject(
            name="desk_lamp_bulb",
            size=[0.02],
            rgba=self.lamp_off_rgba,
            joints=None,
            obj_type="visual",
        )
        self.lamp_base.get_obj().set("pos", array_to_string(self.lamp_base_position))
        self.lamp_pole.get_obj().set("pos", array_to_string(self.lamp_pole_position))
        self.lamp_shade.get_obj().set("pos", array_to_string(self.lamp_shade_position))
        self.lamp.get_obj().set("pos", array_to_string(self.lamp_bulb_position))
        button_center_z = self.table_offset[2] + 0.004
        button_cluster_center = np.array([-0.22, -0.10, button_center_z])
        self.button_positions = [
            button_cluster_center + np.array([0.03, 0.03, 0.0]),
            button_cluster_center + np.array([-0.03, 0.03, 0.0]),
            button_cluster_center + np.array([0.03, -0.03, 0.0]),
            button_cluster_center + np.array([-0.03, -0.03, 0.0]),
        ]
        for button, button_pos in zip(self.button_list, self.button_positions):
            button.get_obj().set("pos", array_to_string(button_pos))

        if self.object_A_index == 0:
            self.object_A = self.cross
        elif self.object_A_index == 1:
            self.object_A = self.cube
        elif self.object_A_index == 2:
            self.object_A = self.cylinder
        elif self.object_A_index == 3:
            self.object_A = self.milk

        if self.object_B_index == 0:
            self.object_B = self.bin
        elif self.object_B_index == 1:
            self.object_B = self.mug
        elif self.object_B_index == 2:
            self.object_B = self.plate
        elif self.object_B_index == 3:
            self.object_B = self.mug_no_handle
        
        self.object_A_list = [self.cross, self.cube, self.cylinder, self.milk]
        self.object_B_list = [self.bin, self.mug, self.plate, self.mug_no_handle]
        self.target_button = self.button_list[int(self.target_button_index)]
        self.objects = [
            self.cross,
            self.cube,
            self.cylinder,
            self.milk,
            self.bin,
            self.mug,
            self.plate,
            self.mug_no_handle,
            *self.button_list,
            self.lamp_base,
            self.lamp_pole,
            self.lamp_shade,
            self.lamp,
        ]


        # Create placement initializer
        if self.placement_initializer_A is not None:
            self.placement_initializer_A.reset()
            self.placement_initializer_A.add_objects(self.object_A_list)
        else:
            self.placement_initializer_A = UniformApartRandomSampler(
                name="Object_A_Sampler",
                mujoco_objects=self.object_A_list,
                x_range=[-0.05, 0.15],
                y_range=[-0.25, -0.05],
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
                x_range=[-0.05, 0.15],
                y_range=[0.05, 0.25],
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
        self.mug_body_id = self.sim.model.body_name2id(self.mug.root_body)
        self.plate_body_id = self.sim.model.body_name2id(self.plate.root_body)
        self.milk_body_id = self.sim.model.body_name2id(self.milk.root_body)
        self.mug_no_handle_body_id = self.sim.model.body_name2id(self.mug_no_handle.root_body)
        self.button_body_ids = [
            self.sim.model.body_name2id(button.root_body) for button in self.button_list
        ]
        self.button_qpos_addrs = [
            self.sim.model.get_joint_qpos_addr(button.slide_joint_name) for button in self.button_list
        ]
        self.target_button = self.button_list[int(self.target_button_index)]
        self.target_button_body_id = self.button_body_ids[int(self.target_button_index)]
        self.target_button_qpos_addr = self.button_qpos_addrs[int(self.target_button_index)]
        self.lamp_body_id = self.sim.model.body_name2id(self.lamp.root_body)
        self.lamp_geom_id = self.sim.model.geom_name2id(self.lamp.visual_geoms[0])
        self.lamp_shade_geom_ids = [
            self.sim.model.geom_name2id(name)
            for name in self.sim.model.geom_names
            if "shade_ring" in name
        ]
        self.lamp_pressed_index = None

    def _set_button_positions(self, positions: list[np.ndarray]) -> None:
        self.button_positions = [np.asarray(pos, dtype=float) for pos in positions]
        for body_id, button_pos in zip(self.button_body_ids, self.button_positions):
            self.sim.model.body_pos[int(body_id)] = np.asarray(button_pos, dtype=float)
        for qpos_addr in self.button_qpos_addrs:
            self.sim.data.qpos[int(qpos_addr)] = 0.0
        self.lamp_pressed_index = None
        self.sim.model.geom_rgba[int(self.lamp_geom_id)] = np.asarray(self.lamp_off_rgba, dtype=float)
        for geom_id in self.lamp_shade_geom_ids:
            self.sim.model.geom_rgba[int(geom_id)] = np.asarray(self.lamp_shade_off_rgba, dtype=float)
        self.target_button_body_id = self.button_body_ids[int(self.target_button_index)]
        self.target_button_qpos_addr = self.button_qpos_addrs[int(self.target_button_index)]
        self.sim.forward()

    def _pressed_button_index(self) -> int | None:
        qpos = np.array([self.sim.data.qpos[int(addr)] for addr in self.button_qpos_addrs])
        if qpos.size == 0 or float(np.min(qpos)) > self.button_press_joint_threshold:
            return None
        return int(np.argmin(qpos))

    def _update_lamp_color(self) -> None:
        pressed_index = self._pressed_button_index()
        if pressed_index is not None:
            self.lamp_pressed_index = pressed_index
        if self.lamp_pressed_index is None:
            bulb_rgba = self.lamp_off_rgba
            shade_rgba = self.lamp_shade_off_rgba
        else:
            button_rgb = self.button_rgba_colors[int(self.lamp_pressed_index)][:3]
            bulb_rgba = button_rgb + [1.0]
            shade_rgba = button_rgb + [self.lamp_shade_alpha]
        self.sim.model.geom_rgba[int(self.lamp_geom_id)] = np.asarray(bulb_rgba, dtype=float)
        for geom_id in self.lamp_shade_geom_ids:
            self.sim.model.geom_rgba[int(geom_id)] = np.asarray(shade_rgba, dtype=float)

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
            def milk_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.milk_body_id])

            @sensor(modality=modality)
            def milk_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.milk_body_id]), to="xyzw")

            @sensor(modality=modality)
            def bin_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.bin_body_id])

            @sensor(modality=modality)
            def bin_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.bin_body_id]), to="xyzw")

            @sensor(modality=modality)
            def mug_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.mug_body_id])

            @sensor(modality=modality)
            def mug_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.mug_body_id]), to="xyzw")

            @sensor(modality=modality)
            def plate_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.plate_body_id])

            @sensor(modality=modality)
            def plate_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.plate_body_id]), to="xyzw")

            @sensor(modality=modality)
            def mug_no_handle_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.mug_no_handle_body_id])

            @sensor(modality=modality)
            def mug_no_handle_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.mug_no_handle_body_id]), to="xyzw")

            @sensor(modality=modality)
            def button_red_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.button_body_ids[0]])

            @sensor(modality=modality)
            def button_green_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.button_body_ids[1]])

            @sensor(modality=modality)
            def button_blue_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.button_body_ids[2]])

            @sensor(modality=modality)
            def button_yellow_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.button_body_ids[3]])

            sensors = [
                cross_pos, cross_quat, cube_pos, cube_quat, cylinder_pos, cylinder_quat, milk_pos, milk_quat,
                bin_pos, bin_quat, mug_pos, mug_quat, plate_pos, plate_quat, mug_no_handle_pos, mug_no_handle_quat,
                button_red_pos, button_green_pos, button_blue_pos, button_yellow_pos,
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
            @sensor(modality="target_button")
            def target_button_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.target_button_body_id])
            sensors += [object_A_pos, object_A_quat, object_B_pos, object_B_quat, target_button_pos]

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
        self.has_completed_place = False

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
            self.sim.forward()
            self._set_button_positions(self.button_positions)

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
    
    def _place_stage_success(self):
        """
        Check whether the pick-and-place stage is complete before button pressing.

        Returns:
            bool: True if object_A is placed and the gripper has lifted away
        """
        gripper_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]
        table_height = self.model.mujoco_arena.table_offset[2]
        lift_check = gripper_pos[2] - table_height > 0.20
        return self.place_success() and lift_check

    def _check_success(self):
        """
        Check if the task has been successfully completed.

        Returns:
            bool: True if task is successfully completed
        """
        if not self.has_completed_place and self._place_stage_success():
            self.has_completed_place = True

        self._update_lamp_color()
        return self.has_completed_place and self.place_success() and self.button_press_success()

    def button_press_success(self):
        """
        Check whether the requested colored button is physically pressed down.

        Returns:
            bool: True if the target button's slide joint is depressed past the press threshold
        """
        self._update_lamp_color()
        button_qpos = self.sim.data.qpos[int(self.target_button_qpos_addr)]
        return button_qpos <= self.button_press_joint_threshold

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

        elif self.object_B_index == 1:  # mug case
            mug_inner_r = self.object_B.r1
            mug_outer_r = self.object_B.r2
            mug_height = self.object_B.mug_height
            horizontal_dist = np.linalg.norm(rel_pos[:2])
            
            radius_check = horizontal_dist < (mug_inner_r + mug_outer_r) / 2
            height_check = (rel_pos[2] > -mug_height) and (rel_pos[2] < mug_height)
            
            return radius_check and height_check
        
        elif self.object_B_index == 2: # Plate case
            plate_radius = self.object_B.radius
            plate_rim_width = self.object_B.rim_width
            plate_height = self.object_B.height
            horizontal_dist = np.linalg.norm(rel_pos[:2])

            radius_check = horizontal_dist < (plate_radius - plate_rim_width)
            height_check = abs(rel_pos[2]) < 0.06

            return radius_check and height_check
        
        elif self.object_B_index == 3:  # Handle-less mug case
            cup_inner_r = self.object_B.r1
            cup_outer_r = self.object_B.r2
            cup_height = self.object_B.mug_height
            horizontal_dist = np.linalg.norm(rel_pos[:2])
            
            radius_check = horizontal_dist < (cup_inner_r + cup_outer_r) / 2
            height_check = (rel_pos[2] > -cup_height) and (rel_pos[2] < cup_height)
            return radius_check and height_check

        return False
        