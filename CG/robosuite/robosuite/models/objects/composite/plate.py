import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import RED, CustomMaterial, add_to_dict


class Plate(CompositeObject):
    """
    Generates a plate-like object with a flat base and raised rim.
    Args:
        name (str): Name of this Plate object
        radius (float): Radius of the plate
        height (float): Height of the plate (including rim)
        rim_width (float): Width of the rim
        rim_height (float): Height of the rim (how much it raises from base)
        ngeoms (int): Number of box geoms used to approximate the circular rim. Use
            more geoms to make the approximation better.
        thickness (float): Thickness of the base plate
    """

    def __init__(
        self,
        name,
        radius=0.1,
        height=0.02,
        rim_width=0.003,
        rim_height=0.02,
        ngeoms=16,
        thickness=0.005,
        rgba=None,
        material=None,
        density=1000.0,
        solref=(0.02, 1.0),
        solimp=(0.9, 0.95, 0.001),
        friction=None,
    ):

        # Set object attributes
        self._name = name
        self.rgba = rgba
        self.density = density
        self.friction = friction if friction is None else np.array(friction)
        self.solref = solref
        self.solimp = solimp
        self.thickness = thickness  # thickness of the base plate

        self.has_material = material is not None
        if self.has_material:
            assert isinstance(material, CustomMaterial)
            self.material = material

        # Other private attributes
        self._important_sites = {}

        # plate dimensions
        self.radius = radius
        self.height = height
        self.rim_width = rim_width
        self.rim_height = rim_height

        # number of geoms used to approximate the circular rim
        self.n = ngeoms

        # half-width of each box in the rim
        self.unit_box_width = (self.radius + self.rim_width) * np.sin(np.pi / self.n)

        # half-height of each box in the rim
        self.unit_box_height = self.rim_height / 2.0

        # each box geom depth will be the rim width
        self.unit_box_depth = self.rim_width

        # radius of circle that connects all box centers
        self.int_r = self.radius + (self.rim_width / 2.0)

        # Create dictionary of values to create geoms for composite object and run super init
        super().__init__(**self._get_geom_attrs())

        # Optionally add material
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
            "total_size": [self.radius + self.rim_width, self.radius + self.rim_width, self.height],
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
            "density": self.density,
            "solref": self.solref,
            "solimp": self.solimp,
        }
        obj_args = {}

        # Create the circular rim
        angle_step = 2.0 * np.pi / self.n
        for i in range(self.n):
            # we start with the top-most box object and proceed clockwise (thus an offset of np.pi)
            geom_angle = np.pi - i * angle_step
            geom_center = np.array([
                self.int_r * np.cos(geom_angle), 
                self.int_r * np.sin(geom_angle), 
                0
            ])
            geom_quat = np.array([np.cos(geom_angle / 2.0), 0.0, 0.0, np.sin(geom_angle / 2.0)])
            geom_size = np.array([self.unit_box_depth, self.unit_box_width, self.unit_box_height])

            add_to_dict(
                dic=obj_args,
                geom_types="box",
                geom_locations=tuple(geom_center),
                geom_quats=tuple(geom_quat),
                geom_sizes=tuple(geom_size),
                geom_names="rim_{}".format(i),
                geom_rgbas=self.rgba,
                geom_materials=self.material.mat_attrib["name"] if self.has_material else None,
                geom_frictions=self.friction,
                geom_condims=4,
            )

        # Create the base plate (a flat cylinder)
        add_to_dict(
            dic=obj_args,
            geom_types="cylinder",
            geom_locations=(0, 0, self.thickness*0.5 - self.rim_height*0.5),  # position at the bottom
            geom_quats=(1, 0, 0, 0),  # no rotation
            geom_sizes=(self.int_r, self.thickness*0.5),  # radius and half-height
            geom_names="base",
            geom_rgbas=self.rgba,
            geom_materials=self.material.mat_attrib["name"] if self.has_material else None,
            geom_frictions=self.friction,
            geom_condims=4,
        )

        # Sites
        obj_args["sites"] = [
            {
                "name": "center",
                "pos": (0, 0, 0),
                "size": "0.002",
                "rgba": RED,
                "type": "sphere",
            },
            {
                "name": "bottom",
                "pos": (0, 0, -self.height/2),
                "size": "0.002",
                "rgba": RED,
                "type": "sphere",
            }
        ]

        # Add back in base args and site args
        obj_args.update(base_args)

        # Return this dict
        return obj_args