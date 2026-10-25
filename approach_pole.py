#!/usr/bin/env python3
"""Fly the simulated Iris to a hover 0.5 m in front of the 76 mm pole.

Start iris_pole.py first and wait until PX4 prints "Ready for takeoff".
Close the MAVLink shell before running this; both listen on UDP port 14550.

    python3 approach_pole.py

The pole in iris_pole.py is 3 m along world +X. PX4's local frame is
north-east-down, and that direction is east, so the target is
north = 0, east = 2.5 m, 3 m above the origin. If the vehicle flies
the wrong way, swap TARGET_NORTH_M and TARGET_EAST_M.
"""

import math
import time

from pymavlink import mavutil

# 3 m pole, 0.5 m standoff, so the vehicle stops before contact.
TARGET_NORTH_M = 0.0
TARGET_EAST_M = 2.5
TARGET_ALTITUDE_M = 3.0
TARGET_YAW_RAD = math.pi / 2.0  # face east, toward the pole

# Position and yaw are used. Velocity, acceleration, and yaw-rate are ignored.
POSITION_YAW_MASK = 0b0000101111111000
OFFBOARD_MODE = 6


def send_target(master):
    master.mav.set_position_target_local_ned_send(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        POSITION_YAW_MASK,
        TARGET_NORTH_M,
        TARGET_EAST_M,
        -TARGET_ALTITUDE_M,
        0,
        0,
        0,
        0,
        0,
        0,
        TARGET_YAW_RAD,
        0,
    )


def set_offboard(master):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        OFFBOARD_MODE,
        0,
        0,
        0,
        0,
        0,
    )


def arm(master):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
    )


def main():
    master = mavutil.mavlink_connection("udpin:0.0.0.0:14550")
    master.wait_heartbeat()
    print(f"Connected to system {master.target_system}")

    # PX4 rejects offboard mode until setpoints are already arriving.
    for _ in range(20):
        send_target(master)
        time.sleep(0.05)

    set_offboard(master)
    arm(master)
    print("Offboard armed. Streaming the approach point. Ctrl+C to stop.")

    while True:
        send_target(master)
        time.sleep(0.05)


if __name__ == "__main__":
    main()
