from enum import Enum
from dataclasses import dataclass
import numpy as np

class CameraMode(Enum):

    VELOCITY_FOLLOWING = 1
    VELOCITY_FACING = 2
    NORMAL_FACING = 3

@dataclass(frozen=True)
class FixedCameraOrientation:
    """An inertially fixed camera orientation.

    ``facing`` points from the camera towards the spacecraft. ``up`` sets
    the camera roll and must not be parallel to ``facing``.
    """

    facing: np.ndarray
    up: np.ndarray
