"""Shared definition of the pole-approach task."""

import numpy as np

POLE_DIAMETER_M = 0.076
POLE_HEIGHT_M = 4.0
POLE_RADIUS_M = POLE_DIAMETER_M / 2.0

# The learned policy stops at this distance from the pole. It does not make contact.
STANDOFF_M = 1.0
TARGET_ALTITUDE_M = 1.5

DISTANCE_MIN_M = 3.0
DISTANCE_MAX_M = 5.0
LATERAL_MAX_M = 1.0

V_MAX_M_S = 0.5
YAW_RATE_MAX_RAD_S = 0.5
KP_HORIZONTAL = 0.6
KP_VERTICAL = 0.8
KP_YAW = 1.0

SUCCESS_RADIUS_M = 0.3
SUCCESS_ALTITUDE_M = 0.3
SUCCESS_SPEED_M_S = 0.3
HIT_RADIUS_M = POLE_RADIUS_M + 0.15
EPISODE_TIMEOUT_S = 40.0

# Depth camera sized for training. Intrinsics use the D435i depth field of view.
CAMERA_WIDTH = 160
CAMERA_HEIGHT = 120
CAMERA_HFOV_DEG = 87.0
CAMERA_VFOV_DEG = 58.0
DEPTH_MIN_M = 0.2
DEPTH_MAX_M = 10.0

OBS_HEIGHT = 48
OBS_WIDTH = 64
STATE_DIM = 4
ACTION_DIM = 4

CONTROL_PERIOD_S = 0.1


def camera_intrinsics():
    """Pinhole intrinsics for the training camera, in pixel units."""
    # Isaac Sim forces square pixels, so fy has to equal fx or the camera bridge warns and overwrites it.
    fx = 0.5 * CAMERA_WIDTH / np.tan(np.radians(CAMERA_HFOV_DEG) / 2.0)
    fy = fx
    return np.array(
        [
            [fx, 0.0, CAMERA_WIDTH / 2.0],
            [0.0, fy, CAMERA_HEIGHT / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def downsample_depth(depth):
    """Convert a depth image in metres to the fixed observation size."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth = np.clip(depth, 0.0, DEPTH_MAX_M)

    src_h, src_w = depth.shape
    ys = np.linspace(0, src_h, OBS_HEIGHT + 1).astype(int)
    xs = np.linspace(0, src_w, OBS_WIDTH + 1).astype(int)
    out = np.zeros((OBS_HEIGHT, OBS_WIDTH), dtype=np.float32)
    for row in range(OBS_HEIGHT):
        for col in range(OBS_WIDTH):
            block = depth[ys[row] : ys[row + 1], xs[col] : xs[col + 1]]
            out[row, col] = 0.0 if block.size == 0 else float(block.mean())
    return out / DEPTH_MAX_M
