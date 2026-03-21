# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.utils.mjcf_utils import RED, BLUE, CustomMaterial

from robosuite.models.objects import CompositeBodyObject, BoxObject, CylinderObject
from mimicgen.models.robosuite.objects import HollowCylinderObject


class Mug(CompositeBodyObject):
    """
    Mug object with optional handle.
    """
    def __init__(
        self,
        name,
        outer_radius=0.0425,
        inner_radius=0.03,
        mug_height=0.05,
        mug_ngeoms=8,
        mug_base_height=0.01,
        mug_base_offset=0.005,
        add_handle=False,
        handle_outer_radius=0.03,
        handle_inner_radius=0.015,
        handle_thickness=0.005,
        handle_ngeoms=8,
        joints="default",
        rgba=None,
        material=None,
        density=100.,
        friction=None,
    ):

        # Object properties

        # radius of the inner mug hole and entire mug
        self.r1 = inner_radius
        self.r2 = outer_radius

        # number of geoms used to approximate the cylindrical shell
        self.n = mug_ngeoms

        # mug half-height
        self.mug_height = mug_height

        # mug base args
        self.mug_base_height = mug_base_height
        self.mug_base_offset = mug_base_offset

        # handle args
        self.add_handle = add_handle
        self.handle_outer_radius = handle_outer_radius
        self.handle_inner_radius = handle_inner_radius
        self.handle_thickness = handle_thickness
        self.handle_ngeoms = handle_ngeoms

        # Create objects
        objects = []
        object_locations = []
        object_quats = []
        object_parents = []

        # mug body
        self.mug_body = HollowCylinderObject(
            name="mug_body",
            outer_radius=self.r2,
            inner_radius=self.r1,
            height=self.mug_height,
            ngeoms=self.n,
            rgba=rgba,
            material=material,
            density=density,
            friction=friction,
        )
        objects.append(self.mug_body)
        object_locations.append([0., 0., 0.])
        object_quats.append([1., 0., 0., 0.])
        object_parents.append(None)

        # mug base
        self.mug_base = CylinderObject(
            name="mug_base",
            size=[self.mug_body.int_r, self.mug_base_height],
            rgba=rgba,
            material=material,
            density=density,
            solref=[0.02, 1.],
            solimp=[0.998, 0.998, 0.001],
            joints=None,
        )
        objects.append(self.mug_base)
        object_locations.append([0., 0., -self.mug_height + self.mug_base_height + self.mug_base_offset])
        object_quats.append([1., 0., 0., 0.])
        object_parents.append(None)

        if self.add_handle:
            # mug handle is a hollow half-cylinder
            self.mug_handle = HollowCylinderObject(
                name="mug_handle",
                outer_radius=self.handle_outer_radius,
                inner_radius=self.handle_inner_radius,
                height=self.handle_thickness,
                ngeoms=self.handle_ngeoms,
                rgba=rgba,
                material=material,
                density=density,
                make_half=True,
                friction=friction,
            )
            # translate handle to right side of mug body, and rotate by +90 degrees about y-axis 
            # to orient the handle geoms on the mug body
            objects.append(self.mug_handle)
            object_locations.append([0., (self.mug_body.r2 + self.mug_handle.unit_box_width), 0.])
            object_quats.append(
                T.convert_quat(
                    T.mat2quat(T.rotation_matrix(angle=np.pi / 2., direction=[0., 1., 0.])[:3, :3]),
                    to="wxyz",
                )
            )
            object_parents.append(None)

        # total size of mug
        body_total_size = [self.r2, self.r2, self.mug_height]
        if self.add_handle:
            body_total_size[1] += self.handle_outer_radius

        # Run super init
        super().__init__(
            name=name,
            objects=objects,
            object_locations=object_locations,
            object_quats=object_quats,
            object_parents=object_parents,
            joints=joints,
            total_size=body_total_size,
            # locations_relative_to_corner=True,
        )
