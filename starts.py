#!/usr/bin/env python3
"""Draw the shared starts for a paired comparison.

    python3 starts.py --episodes 200 --seed 0

The vehicle begins at the origin on the ground. The heading and the pole
position are drawn once. Both the policy run and the modular run load this
file, so pair 0 is the same approach for each method.
"""

import argparse
import os

import numpy as np

from task import TARGET_ALTITUDE_M, random_pole

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(HERE, "data", "paired_starts.npz")


def generate_starts(n, seed):
    rng = np.random.default_rng(int(seed))
    yaw = rng.uniform(-np.pi, np.pi, size=int(n))
    vehicle_xy = np.zeros((int(n), 2), dtype=np.float64)
    pole_xy = np.stack([random_pole(vehicle_xy[index], yaw[index], rng) for index in range(int(n))])
    altitude = np.full(int(n), TARGET_ALTITUDE_M, dtype=np.float64)
    return {
        "vehicle_xy": vehicle_xy,
        "yaw": yaw.astype(np.float64),
        "pole_xy": pole_xy.astype(np.float64),
        "altitude": altitude,
        "seed": np.array(int(seed)),
    }


def save_starts(path, table):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **table)


def pole_from_vehicle(start_vehicle_xy, start_yaw, start_pole_xy, vehicle_xy, yaw):
    """Place the saved pole relative to the vehicle, without moving the vehicle."""
    delta = np.asarray(start_pole_xy, dtype=np.float64) - np.asarray(start_vehicle_xy, dtype=np.float64)
    start_yaw = float(start_yaw)
    forward = float(np.cos(start_yaw) * delta[0] + np.sin(start_yaw) * delta[1])
    left = float(-np.sin(start_yaw) * delta[0] + np.cos(start_yaw) * delta[1])
    yaw = float(yaw)
    forward_axis = np.array([np.cos(yaw), np.sin(yaw)])
    left_axis = np.array([-np.sin(yaw), np.cos(yaw)])
    return np.asarray(vehicle_xy, dtype=np.float64) + forward * forward_axis + left * left_axis


def load_starts(path):
    if not os.path.isfile(path):
        raise SystemExit(
            f"Missing {path}. Draw the shared starts first:\n"
            "    python3 starts.py --episodes 200 --seed 0"
        )
    stored = np.load(path)
    table = {key: stored[key] for key in ("vehicle_xy", "yaw", "pole_xy", "altitude", "seed")}
    count = len(table["yaw"])
    if table["vehicle_xy"].shape != (count, 2) or table["pole_xy"].shape != (count, 2):
        raise SystemExit(f"{path} does not contain one vehicle pose and one pole per start")
    return table


def main():
    parser = argparse.ArgumentParser(description="Draw shared starts for the paired comparison.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=DEFAULT_PATH)
    args = parser.parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1")
    table = generate_starts(args.episodes, args.seed)
    save_starts(args.out, table)
    print(f"Saved {args.episodes} starts, seed {args.seed}, to {args.out}")


if __name__ == "__main__":
    main()
