import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import CustomMaterial, add_to_dict


class CrossObject(CompositeObject):
    """
    A center-symmetric cross object composed of 3 perpendicular boxes.
    
    Args:
        name (str): Name of this Cross object
        arm_length (float): Length of each cross arm from center to end
        arm_width (float): Width/thickness of the cross arms
        height (float): Height/depth of the cross (z-dimension)
        arm_length_max (float): Max length of arms for randomization
        arm_length_min (float): Min length of arms for randomization
        arm_width_max (float): Max width of arms for randomization
        arm_width_min (float): Min width of arms for randomization
        height_max (float): Max height for randomization
        height_min (float): Min height for randomization
        density (float): Density value for all geoms
        friction (3-array or None): Friction values for the cross
        use_texture (bool): Whether to use textures
        material (CustomMaterial or None): Custom material to use
        rgba (4-array or None): Color and transparency
    """

    def __init__(
        self,
        name,
        arm_length=0.1,
        arm_width=0.02,
        height=0.03,
        density=1000.0,
        friction=None,
        material=None,
        rgba=None,
    ):
        # Set name
        self._name = name

        # Set dimensions with randomization ranges
        self.arm_length = arm_length
        self.arm_width = arm_width
        self.height = height
        
        # Physical properties
        self.friction = friction if friction is None else np.array(friction)
        self.density = density
        self.rgba = rgba
        
        # Material setup
        self.has_material = material is not None
        if self.has_material:
            assert isinstance(material, CustomMaterial)
            self.material = material

        # Create dictionary of values to create geoms for composite object
        self._important_sites = {}
        super().__init__(**self._get_geom_attrs())
        
        if self.has_material:
            self.append_material(self.material)

    def _get_geom_attrs(self):
        """
        Creates geom elements that will be passed to superclass CompositeObject constructor
        Returns:
            dict: args to be used by CompositeObject to generate geoms
        """
        # Initialize dict of obj args that we'll pass to the CompositeObject constructor
        base_args = {
            "total_size": np.array([self.arm_length + self.arm_width, 
                                   self.arm_length + self.arm_width, 
                                   self.height]) / 2,
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
            "density": self.density,
        }
        obj_args = {}

        # Horizontal arm (x-axis)
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0, 0, 0),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=(self.arm_length/2, self.arm_width/2, self.height/2),
            geom_names="horizontal_arm",
            geom_rgbas=self.rgba,
            geom_materials=self.material.mat_attrib["name"] if self.has_material else None,
            geom_frictions=self.friction,
        )

        # Vertical arm (y-axis)
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0, 0, 0),
            geom_quats=T.convert_quat(T.axisangle2quat(np.array([0, 0, np.pi/2])), to="wxyz"),
            geom_sizes=(self.arm_length/2, self.arm_width/2, self.height/2),
            geom_names="vertical_arm",
            geom_rgbas=self.rgba,
            geom_materials=self.material.mat_attrib["name"] if self.has_material else None,
            geom_frictions=self.friction,
        )

        # Add base args
        obj_args.update(base_args)

        return obj_args

    @property
    def bottom_offset(self):
        """Offset from object center to bottom surface."""
        return np.array([0, 0, -self.height/2])

    @property
    def top_offset(self):
        """Offset from object center to top surface."""
        return np.array([0, 0, self.height/2])

    @property
    def horizontal_radius(self):
        """Maximum horizontal radius for grasping purposes."""
        return np.sqrt(self.arm_length**2 + self.arm_width**2)

    def get_bounding_box_half_size(self):
        """Get half sizes of the full bounding box."""
        return np.array([
            self.arm_length + self.arm_width,
            self.arm_length + self.arm_width,
            self.height
        ]) / 2

    @property
    def horizontal_geoms(self):
        """Returns geom names corresponding to horizontal arm."""
        return [self.correct_naming("horizontal_arm")]

    @property
    def vertical_geoms(self):
        """Returns geom names corresponding to vertical arm."""
        return [self.correct_naming("vertical_arm")]