"""Modular approach: the column is taken from the depth image, not the true pole pose.

When the column is outside the image, the vehicle yaws at the maximum rate.
The scan direction is fixed, so the command does not depend on the true bearing.
"""

import numpy as np

import expert
from task import (
    CAMERA_WIDTH,
    KP_VERTICAL,
    TARGET_ALTITUDE_M,
    V_MAX_M_S,
    YAW_RATE_MAX_RAD_S,
    camera_intrinsics,
    column_observation,
)


def hold_altitude(position):
    up = float(np.clip(KP_VERTICAL * (TARGET_ALTITUDE_M - float(position[2])), -V_MAX_M_S, V_MAX_M_S))
    return np.array([0.0, 0.0, up, 0.0], dtype=np.float32)


def estimate_pole_xy(depth, position, yaw):
    """Pole position implied by the nearest column in the depth image, or None."""
    found = column_observation(depth)
    if found is None:
        return None
    column, distance = found
    width = int(np.asarray(depth).shape[1])
    fx = float(camera_intrinsics()[0, 0]) * (width / CAMERA_WIDTH)
    bearing_right = float(np.arctan((column - (width / 2.0)) / fx))
    forward = distance * float(np.cos(bearing_right))
    left = -distance * float(np.sin(bearing_right))
    yaw = float(yaw)
    forward_axis = np.array([np.cos(yaw), np.sin(yaw)])
    left_axis = np.array([-np.sin(yaw), np.cos(yaw)])
    origin = np.asarray(position[:2], dtype=np.float64)
    return origin + forward * forward_axis + left * left_axis


def command(depth, position, yaw):
    """Body velocity from the depth image. A missing column starts a leftward scan."""
    position = np.asarray(position, dtype=np.float64)
    if depth is None:
        return hold_altitude(position)
    pole_xy = estimate_pole_xy(depth, position, yaw)
    if pole_xy is None:
        up = hold_altitude(position)[2]
        return np.array([0.0, 0.0, up, YAW_RATE_MAX_RAD_S], dtype=np.float32)
    return expert.command(position, yaw, pole_xy)
