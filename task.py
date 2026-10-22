"""Shared definition of the pole-approach task."""

import numpy as np

POLE_DIAMETER_M = 0.076
POLE_HEIGHT_M = 4.0
POLE_RADIUS_M = POLE_DIAMETER_M / 2.0

# Stop short of the column. 0.5 m from the centre is about 0.46 m from the surface.
STANDOFF_M = 0.5
TARGET_ALTITUDE_M = 1.5

DISTANCE_MIN_M = 3.0
DISTANCE_MAX_M = 5.0
LATERAL_MAX_M = 1.0

V_MAX_M_S = 0.5
YAW_RATE_MAX_RAD_S = 0.5
KP_HORIZONTAL = 0.6
KP_VERTICAL = 0.8
KP_YAW = 1.0

# Keep the success disk outside the hit volume around the 0.5 m staging point.
SUCCESS_RADIUS_M = 0.15
SUCCESS_ALTITUDE_M = 0.3
SUCCESS_SPEED_M_S = 0.3
HIT_RADIUS_M = POLE_RADIUS_M + 0.15
EPISODE_TIMEOUT_S = 40.0

# The column is 76 mm wide. At 160x120 it is about two pixels across at 4 m,
# and block averaging then mixes it into the background. 640x480 keeps it visible.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
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
    """Convert a depth image in metres to the fixed observation size.

    Each block keeps the nearest return. Averaging would hide the column,
    which is only a few pixels wide. Missing returns are treated as far.
    """
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = np.nan_to_num(depth, nan=DEPTH_MAX_M, posinf=DEPTH_MAX_M, neginf=DEPTH_MAX_M)
    depth = np.clip(depth, 0.0, DEPTH_MAX_M)
    depth[depth < DEPTH_MIN_M] = DEPTH_MAX_M

    src_h, src_w = depth.shape
    ys = np.linspace(0, src_h, OBS_HEIGHT + 1).astype(int)
    xs = np.linspace(0, src_w, OBS_WIDTH + 1).astype(int)
    out = np.full((OBS_HEIGHT, OBS_WIDTH), DEPTH_MAX_M, dtype=np.float32)
    for row in range(OBS_HEIGHT):
        for col in range(OBS_WIDTH):
            block = depth[ys[row] : ys[row + 1], xs[col] : xs[col + 1]]
            if block.size:
                out[row, col] = float(block.min())
    return out / DEPTH_MAX_M
