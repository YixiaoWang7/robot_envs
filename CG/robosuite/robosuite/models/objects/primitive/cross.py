import numpy as np
from robosuite.models.objects import PrimitiveObject
from robosuite.utils.mjcf_utils import get_size, new_geom, new_body, new_site, new_joint


class CrossObject(PrimitiveObject):
    """
    A cross object (centered symmetric cross) composed of 3 perpendicular boxes.
    
    Args:
        arm_length (float): length of each cross arm from center to end
        arm_width (float): width/thickness of the cross arms
        height (float): height/depth of the cross (z-dimension)
    """

    def __init__(
        self,
        name,
        arm_length=None,
        arm_width=None,
        height=None,
        arm_length_max=None,
        arm_length_min=None,
        arm_width_max=None,
        arm_width_min=None,
        height_max=None,
        height_min=None,
        density=None,
        friction=None,
        rgba=None,
        solref=None,
        solimp=None,
        material=None,
        joints="default",
        obj_type="all",
        duplicate_collision_geoms=True,
    ):
        # Set default dimensions
        arm_length = get_size(arm_length, arm_length_max, arm_length_min, [0.1], [0.05])
        arm_width = get_size(arm_width, arm_width_max, arm_width_min, [0.02], [0.01])
        height = get_size(height, height_max, height_min, [0.03], [0.01])
        
        # Store cross-specific dimensions
        self.arm_length = arm_length
        self.arm_width = arm_width
        self.height = height
        
        # The 'size' parameter for PrimitiveObject will represent the full bounding box
        super().__init__(
            name=name,
            size=np.array([arm_length + arm_width, arm_length + arm_width, height]),
            rgba=rgba,
            density=density,
            friction=friction,
            solref=solref,
            solimp=solimp,
            material=material,
            joints=joints,
            obj_type=obj_type,
            duplicate_collision_geoms=duplicate_collision_geoms,
        )

    def sanity_check(self):
        """Check dimension parameters."""
        assert self.arm_length > 0, "arm_length must be positive"
        assert self.arm_width > 0, "arm_width must be positive"
        assert self.height > 0, "height must be positive"

    def _get_object_subtree(self):
        """Create MJCF subtree for cross object."""
        # Main body that will hold all pieces
        main_body = new_body(name=self.name)
        
        # Horizontal arm (x-axis)
        breakpoint()
        horizontal_geom = new_geom(
            name=f"{self.name}_horizontal",
            type="box",
            size=[self.arm_length, self.arm_width, self.height/2],
            pos=[0, 0, 0],
            rgba=self.rgba,
            material=self.material,
            density=self.density,
            friction=self.friction,
            solref=self.solref,
            solimp=self.solimp,
        )
        
        # Vertical arm (y-axis)
        vertical_geom = new_geom(
            name=f"{self.name}_vertical",
            type="box",
            size=[self.arm_width, self.arm_length, self.height/2],
            pos=[0, 0, 0],
            rgba=self.rgba,
            material=self.material,
            density=self.density,
            friction=self.friction,
            solref=self.solref,
            solimp=self.solimp,
        )
        
        # Add all components to main body
        main_body.append(horizontal_geom)
        main_body.append(vertical_geom)
        
        # Add joint if specified
        if self.joints == "default":
            main_body.append(new_joint(name=f"{self.name}_joint", type="free"))
        
        return main_body

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
            self.height/2
        ])