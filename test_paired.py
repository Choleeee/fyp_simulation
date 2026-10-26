#!/usr/bin/env python3
"""Checks for the paired-start draw, the modular depth command, and the comparison."""

import os
import tempfile

import numpy as np

from compare_paired import command_smoothness, load_method, report
from modular import command, estimate_pole_xy
from starts import generate_starts, pole_from_vehicle
from task import CAMERA_HEIGHT, CAMERA_WIDTH, YAW_RATE_MAX_RAD_S, column_observation, pole_visible


def _column_image(column_x, distance):
    depth = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 8.0, dtype=np.float32)
    half = 3
    depth[:, column_x - half : column_x + half + 1] = distance
    return depth


def main():
    first = generate_starts(200, 0)
    second = generate_starts(200, 0)
    assert np.allclose(first["pole_xy"], second["pole_xy"])
    assert np.allclose(first["yaw"], second["yaw"])
    other = generate_starts(200, 1)
    assert not np.allclose(first["pole_xy"], other["pole_xy"])
    offset = first["pole_xy"] - first["vehicle_xy"]
    distance = np.linalg.norm(offset, axis=1)
    assert np.all((distance >= 3.0 - 1e-6) & (distance <= np.hypot(5.0, 1.0) + 1e-6))
    print("pass  shared starts are reproducible")

    moved = pole_from_vehicle([0.0, 0.0], 0.0, [4.0, 0.5], [1.0, 2.0], np.pi / 2.0)
    assert np.allclose(moved, [0.5, 6.0])
    assert abs(np.linalg.norm(moved - [1.0, 2.0]) - np.hypot(4.0, 0.5)) < 1e-6
    print("pass  pole follows the vehicle without moving it")

    depth = _column_image(CAMERA_WIDTH // 2, 4.0)
    assert pole_visible(depth)
    found = column_observation(depth)
    assert found is not None and abs(found[0] - CAMERA_WIDTH // 2) <= 3
    pole = estimate_pole_xy(depth, np.array([0.0, 0.0, 3.0]), 0.0)
    assert abs(pole[0] - 4.0) < 0.3 and abs(pole[1]) < 0.3
    action = command(depth, np.array([0.0, 0.0, 3.0]), 0.0)
    assert action.shape == (4,)
    assert float(np.hypot(action[0], action[1])) <= 0.5 + 1e-6
    print("pass  modular estimate from a centred column")

    empty = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 8.0, dtype=np.float32)
    assert column_observation(empty) is None
    scan = command(empty, np.array([0.0, 0.0, 3.0]), 0.0)
    assert abs(float(scan[3]) - YAW_RATE_MAX_RAD_S) < 1e-6
    assert abs(float(scan[0])) < 1e-6
    print("pass  missing column yaws without using the true bearing")

    with tempfile.TemporaryDirectory() as directory:
        for method, success, reason in (("policy", True, "success"), ("modular", False, "timeout")):
            action = np.zeros((5, 4), dtype=np.float32)
            action[:, 0] = 0.2
            np.savez_compressed(
                os.path.join(directory, f"paired_{method}_0003.npz"),
                depth=np.zeros((5, 48, 64), dtype=np.float16),
                state=np.zeros((5, 4), dtype=np.float32),
                action=action,
                success=np.array([success]),
                reason=np.array(reason),
            )
        loaded = load_method(directory, "policy")
        assert set(loaded) == {3}
        assert loaded[3]["success"]
        assert abs(loaded[3]["time_s"] - 0.5) < 1e-6
        assert np.isfinite(command_smoothness(action))
        summary = report(directory)
        assert summary["shared"] == [3]
    print("pass  paired files match on the pair index")
    print("5 checks passed")


if __name__ == "__main__":
    main()
