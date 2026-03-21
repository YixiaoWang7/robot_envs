from .objects import MujocoObject, MujocoXMLObject, MujocoGeneratedObject
from .generated_objects import CompositeBodyObject, CompositeObject, PrimitiveObject
from .object_groups import ObjectGroup

from .xml_objects import (
    BottleObject,
    CanObject,
    LemonObject,
    MilkObject,
    BreadObject,
    CerealObject,
    SquareNutObject,
    RoundNutObject,
    MilkVisualObject,
    BreadVisualObject,
    CerealVisualObject,
    CanVisualObject,
    PlateWithHoleObject,
    DoorObject,
)

# # Explicit primitive objects
# from .primitive.box import BoxObject
# from .primitive.ball import BallObject
# from .primitive.capsule import CapsuleObject
# from .primitive.cross import CrossObject
# from .primitive.cylinder import CylinderObject

# # Explicit composite objects
# from .composite.bin import Bin
# from .composite.box_pattern_object import BoxPatternObject
# from .composite.cone import ConeObject
# from .composite.cross import CrossObject
# from .composite.cup import Cup
# from .composite.hammer import HammerObject
# from .composite.hollow_cylinder import HollowCylinderObject
# from .composite.hollow_cylinder_ring import HollowCylinderRingObject
# from .composite.hook_frame import HookFrameObject
# from .composite.lid import LidObject
# from .composite.needle import NeedleObject
# from .composite.plate import Plate


# # Explicit composite body objects
# from .composite_body.coffee_machine import CoffeeMachineObject
# from .composite_body.hinged_box import HingedBoxObject
# from .composite_body.mug import MugObject
# from .composite_body.ratcheting_wrench import RatchetingWrenchObject

# # Grouped transport object
# from .group.transport import TransportObject

from .primitive import *
from .composite import *
from .composite_body import *
from .group import *
