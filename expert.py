"""Privileged velocity controller for the approach.

The expert is given the true pole position. Its output is a body-frame
velocity and yaw rate in the forward-left-up frame, which is the action the
student policy has to imitate.
"""

import numpy as np

from task import (
    KP_HORIZONTAL,
    KP_VERTICAL,
    KP_YAW,
    STANDOFF_M,
    SUCCESS_ALTITUDE_M,
    SUCCESS_RADIUS_M,
    SUCCESS_SPEED_M_S,
    TARGET_ALTITUDE_M,
    V_MAX_M_S,
    YAW_RATE_MAX_RAD_S,
    HIT_RADIUS_M,
    wrap_angle,
)


def _clip_horizontal(forward, left):
    speed = float(np.hypot(forward, left))
    if speed > V_MAX_M_S:
        scale = V_MAX_M_S / speed
        return forward * scale, left * scale
    return forward, left


def command(position, yaw, pole_xy):
    """Return forward, left, up velocity (m/s) and CCW yaw rate (rad/s)."""
    position = np.asarray(position, dtype=np.float64)
    pole_xy = np.asarray(pole_xy, dtype=np.float64)
    to_pole = pole_xy - position[:2]
    distance = float(np.linalg.norm(to_pole))
    if distance < 1e-3:
        direction = np.array([1.0, 0.0])
    else:
        direction = to_pole / distance

    desired_xy = pole_xy - STANDOFF_M * direction
    error_xy = desired_xy - position[:2]
    forward_axis = np.array([np.cos(yaw), np.sin(yaw)])
    left_axis = np.array([-np.sin(yaw), np.cos(yaw)])

    forward, left = _clip_horizontal(
        KP_HORIZONTAL * float(np.dot(error_xy, forward_axis)),
        KP_HORIZONTAL * float(np.dot(error_xy, left_axis)),
    )
    up = float(np.clip(KP_VERTICAL * (TARGET_ALTITUDE_M - position[2]), -V_MAX_M_S, V_MAX_M_S))
    yaw_rate = float(
        np.clip(KP_YAW * wrap_angle(np.arctan2(to_pole[1], to_pole[0]) - yaw), -YAW_RATE_MAX_RAD_S, YAW_RATE_MAX_RAD_S)
    )
    return np.array([forward, left, up, yaw_rate], dtype=np.float32)


def reached_staging(position, pole_xy, speed):
    position = np.asarray(position, dtype=np.float64)
    pole_xy = np.asarray(pole_xy, dtype=np.float64)
    to_pole = pole_xy - position[:2]
    distance = float(np.linalg.norm(to_pole))
    if distance < 1e-3:
        return False
    staging = pole_xy - STANDOFF_M * (to_pole / distance)
    horizontal = float(np.linalg.norm(position[:2] - staging))
    return (
        horizontal < SUCCESS_RADIUS_M
        and abs(float(position[2]) - TARGET_ALTITUDE_M) < SUCCESS_ALTITUDE_M
        and speed < SUCCESS_SPEED_M_S
    )


def hit_pole(position, pole_xy):
    position = np.asarray(position, dtype=np.float64)
    offset = np.asarray(pole_xy, dtype=np.float64) - position[:2]
    return float(np.linalg.norm(offset)) < HIT_RADIUS_M
