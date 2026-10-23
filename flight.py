"""Run privileged-expert or learned-policy approaches inside iris_pole.py.

The vehicle is commanded in PX4 offboard mode. It climbs at the start
without recording, then the approach is recorded until the staging point,
half a metre short of the pole. If the column is outside the camera, the
vehicle yaws in place until the depth image shows it, then approaches. It
lands on that spot. The next pole is placed 3–5 m from the landing on a new
heading, at least 60 degrees off the direction to the pole just reached.
"""

import os
import time

import numpy as np
from pymavlink import mavutil
from scipy.spatial.transform import Rotation

import expert
from task import (
    CAMERA_HFOV_DEG,
    CONTROL_PERIOD_S,
    DISTANCE_MAX_M,
    DISTANCE_MIN_M,
    EPISODE_TIMEOUT_S,
    KP_VERTICAL,
    LATERAL_MAX_M,
    POLE_HEIGHT_M,
    TARGET_ALTITUDE_M,
    V_MAX_M_S,
    YAW_RATE_MAX_RAD_S,
    downsample_depth,
    pole_visible,
    wrap_angle,
)

OFFBOARD_MODE = 6
AUTO_MODE = 4
AUTO_LAND = 6
# Ignore position, acceleration, and yaw. Use body velocity and yaw rate.
VELOCITY_MASK = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 6) | (1 << 7) | (1 << 8) | (1 << 10)
BODY_NED = 8


def _yaw(attitude_xyzw):
    return float(Rotation.from_quat(attitude_xyzw).as_euler("ZYX")[0])


def _body_state(vehicle_state):
    velocity = np.asarray(vehicle_state.linear_body_velocity, dtype=np.float32)
    yaw_rate = float(vehicle_state.angular_velocity[2])
    return np.array([velocity[0], velocity[1], velocity[2], yaw_rate], dtype=np.float32)


def _read_depth(camera_sensor):
    if not getattr(camera_sensor, "_camera_full_set", False):
        return None
    frame = camera_sensor._camera.get_current_frame()
    if not frame:
        return None
    depth = frame.get("distance_to_image_plane")
    if depth is None:
        return None
    return np.asarray(depth)


class ApproachSession:
    def __init__(self, vehicle, pole, camera_sensor, mode, episodes, policy_path, data_dir):
        self.vehicle = vehicle
        self.pole = pole
        self.camera_sensor = camera_sensor
        self.mode = mode
        self.episodes_requested = episodes
        self.policy_path = policy_path
        self.data_dir = data_dir
        self.policy = None
        self.master = None
        self.phase = "connect"
        self.episode = 0
        self.pole_xy = np.array([3.0, 0.0])
        self.phase_started = time.monotonic()
        self.last_command = 0.0
        self.samples = []
        self.missed_frames = 0
        self.searching = False
        self.done = False
        self._status_time = 0.0

        if mode in ("rollout", "dagger"):
            if not os.path.isfile(policy_path):
                raise SystemExit(f"Missing {policy_path}. Run train_policy.py after collecting episodes.")
            from policy_net import load_policy

            self.policy = load_policy(policy_path)

    def finished(self):
        return self.done

    def on_step(self):
        if self.done:
            return
        now = time.monotonic()
        if self.phase == "connect":
            self._connect()
            return
        if self.phase == "configure":
            if now - self.last_command >= 0.5:
                self.last_command = now
                self._recharge_battery()
            if now - self.phase_started > 3.0:
                self._begin_episode()
            return
        if self.phase == "recharge":
            if now - self.last_command >= 0.2:
                self.last_command = now
                self._recharge_battery()
            if now - self.phase_started > 1.0:
                self._release_battery_floor()
                print(f"Episode {self.episode + 1}: battery recharged")
                self.phase = "stream"
                self.phase_started = now
            return
        if now - self.last_command < CONTROL_PERIOD_S:
            return
        self.last_command = now

        position = np.asarray(self.vehicle.state.position, dtype=np.float64)
        yaw = _yaw(self.vehicle.state.attitude)
        expert_action = expert.command(position, yaw, self.pole_xy)
        state = _body_state(self.vehicle.state)
        action = self._select_action(expert_action, state)

        if self.phase == "stream":
            self._send(action)
            if now - self.phase_started > 1.0:
                self._set_offboard()
                self._arm(force=False)
                self.phase = "arm"
                self.phase_started = now
            return

        if self.phase == "arm":
            self._send(action)
            self._arm(force=now - self.phase_started > 3.0)
            if self._armed():
                self._set_offboard()
                self.phase = "takeoff"
                self.phase_started = now
            elif now - self.phase_started > 12.0:
                print(f"Episode {self.episode + 1} did not arm")
                self._finish_episode(False, "no arm")
            return

        if self.phase == "takeoff":
            # Climb at the landing spot. The approach recording starts at altitude.
            climb = float(np.clip(0.8 * (TARGET_ALTITUDE_M - position[2]), -0.5, 0.5))
            self._send(np.array([0.0, 0.0, climb, 0.0], dtype=np.float32))
            vertical_speed = abs(float(self.vehicle.state.linear_velocity[2]))
            if abs(position[2] - TARGET_ALTITUDE_M) < 0.3 and vertical_speed < 0.3:
                self.samples = []
                self.missed_frames = 0
                self.searching = False
                self.phase = "fly"
                self.phase_started = now
                print(f"Episode {self.episode + 1}: pole at x={self.pole_xy[0]:.2f} y={self.pole_xy[1]:.2f}")
            elif now - self.phase_started > 20.0:
                print(f"Episode {self.episode + 1} did not reach approach altitude")
                self._finish_episode(False, "no climb")
            return

        if self.phase == "fly":
            depth = _read_depth(self.camera_sensor)
            if not _pole_detected(position, yaw, self.pole_xy, depth):
                action = _yaw_search(position, yaw, self.pole_xy)
                if not self.searching:
                    self.searching = True
                    print(f"Episode {self.episode + 1}: pole outside the camera, yawing")
            elif self.searching:
                self.searching = False
                print(f"Episode {self.episode + 1}: pole in the camera view")
            self._send(action)
            self._record(state, action, expert_action)
            speed = float(np.linalg.norm(self.vehicle.state.linear_velocity))
            elapsed = now - self.phase_started
            if expert.reached_staging(position, self.pole_xy, speed):
                self._finish_episode(True, "success")
            elif expert.hit_pole(position, self.pole_xy):
                self._finish_episode(False, "hit")
            elif elapsed > EPISODE_TIMEOUT_S:
                self._finish_episode(False, "timeout")
            return

        if self.phase == "land":
            self._set_land_mode()
            landed = position[2] < 0.25 and not self._armed()
            if landed or now - self.phase_started > 25.0:
                self.phase = "between"
                self.phase_started = now
            return

        if self.phase == "between" and now - self.phase_started > 2.0:
            self._begin_episode()

    def _connect(self):
        now = time.monotonic()
        if self.master is None:
            self.master = mavutil.mavlink_connection("udpin:0.0.0.0:14550", source_system=255)
            self.phase_started = now
        message = self.master.recv_match(type="HEARTBEAT", blocking=False)
        if message is not None:
            self.master.target_system = message.get_srcSystem()
            self.master.target_component = message.get_srcComponent()
            print(f"PX4 connected, system {self.master.target_system}")
            # PX4 is still loading the airframe here. Repeat the battery writes
            # for a few seconds so the boot defaults do not replace them.
            self.phase = "configure"
            self.phase_started = now
            self.last_command = 0.0
        elif now - self._status_time > 5.0:
            self._status_time = now
            print("Waiting for PX4 heartbeat on UDP 14550")

    def _begin_episode(self):
        if self.episode >= self.episodes_requested:
            self._disarm_and_stop()
            return
        self.samples = []
        self.missed_frames = 0
        position = np.asarray(self.vehicle.state.position, dtype=np.float64)
        yaw = _yaw(self.vehicle.state.attitude)
        self.pole_xy = _next_pole(position[:2], self.pole_xy, yaw, np.random.default_rng())
        self.pole.set_world_pose(position=np.array([self.pole_xy[0], self.pole_xy[1], POLE_HEIGHT_M / 2.0]))
        self.phase = "recharge"
        self.phase_started = time.monotonic()
        self.last_command = 0.0

    def _select_action(self, expert_action, state):
        if self.mode == "collect" or self.policy is None:
            return expert_action
        depth = _read_depth(self.camera_sensor)
        if depth is None:
            return expert_action
        from policy_net import predict

        return predict(self.policy, downsample_depth(depth), state)

    def _record(self, state, executed_action, expert_action):
        depth = _read_depth(self.camera_sensor)
        if depth is None:
            self.missed_frames += 1
            return
        observation = downsample_depth(depth)
        if self.mode == "collect":
            noise = np.random.normal(0.0, 0.01, observation.shape).astype(np.float32)
            observation = np.clip(observation + noise, 0.0, 1.0)
        # Collect and DAgger store the expert command as the label.
        # Rollout stores the command the policy actually sent.
        label = executed_action if self.mode == "rollout" else expert_action
        self.samples.append((observation, state, label))

    def _finish_episode(self, success, reason):
        self._save(success, reason)
        self.episode += 1
        # Descend at the staging point. The landing is not part of the demonstration.
        self._set_land_mode()
        self.phase = "land"
        self.phase_started = time.monotonic()

    def _save(self, success, reason):
        if not self.samples:
            print(
                f"Episode {self.episode + 1} recorded no camera frames ({reason}, "
                f"{self.missed_frames} reads empty)"
            )
            return
        os.makedirs(self.data_dir, exist_ok=True)
        prefix = {"dagger": "dagger", "rollout": "rollout"}.get(self.mode, "episode")
        path = os.path.join(self.data_dir, f"{prefix}_{int(time.time())}_{self.episode}.npz")
        depth, state, action = map(np.stack, zip(*self.samples))
        np.savez_compressed(
            path,
            depth=depth.astype(np.float16),
            state=state.astype(np.float32),
            action=action.astype(np.float32),
            success=np.array([success]),
            reason=np.array(reason),
        )
        missed = f", camera missed {self.missed_frames} steps" if self.missed_frames else ""
        print(f"Saved {len(self.samples)} samples to {path} ({reason}{missed})")

    def _send(self, flu_action):
        # PX4 body frame is forward-right-down, and its yaw rate is clockwise.
        forward, left, up, yaw_rate_ccw = [float(value) for value in flu_action]
        self.master.mav.set_position_target_local_ned_send(
            0,
            self.master.target_system,
            self.master.target_component,
            BODY_NED,
            VELOCITY_MASK,
            0,
            0,
            0,
            forward,
            -left,
            -up,
            0,
            0,
            0,
            0,
            -yaw_rate_ccw,
        )

    def _set_param(self, name, value, ptype):
        self.master.mav.param_set_send(
            self.master.target_system,
            self.master.target_component,
            name.encode("utf-8"),
            float(value),
            ptype,
        )

    def _recharge_battery(self):
        # PX4 raises the simulated charge up to SIM_BAT_MIN_PCT. Holding that
        # floor at 100% for a moment fills the battery. It does not recharge
        # on its own between episodes.
        self._set_param("SIM_BAT_MIN_PCT", 100, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        self._set_param("COM_LOW_BAT_ACT", 0, mavutil.mavlink.MAV_PARAM_TYPE_INT32)

    def _release_battery_floor(self):
        # The charge is already at 100%. Restore the normal SITL floor so this
        # episode can discharge from full instead of staying locked at 100%.
        self._set_param("SIM_BAT_MIN_PCT", 50, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)

    def _set_land_mode(self):
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            AUTO_MODE,
            AUTO_LAND,
            0,
            0,
            0,
            0,
        )

    def _set_offboard(self):
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
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

    def _arm(self, force=False, disarm=False):
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0 if disarm else 1,
            21196 if force and not disarm else 0,
            0,
            0,
            0,
            0,
            0,
        )

    def _armed(self):
        heartbeat = self.master.recv_match(type="HEARTBEAT", blocking=False)
        if heartbeat is None:
            return False
        return bool(heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    def _disarm_and_stop(self):
        if self.master is not None:
            self._arm(disarm=True)
        self.done = True
        print(f"Finished {self.episode} episodes")


def _view_error(position, yaw, pole_xy):
    to_pole = np.asarray(pole_xy, dtype=np.float64) - np.asarray(position[:2], dtype=np.float64)
    if float(np.linalg.norm(to_pole)) < 1e-3:
        return 0.0
    return wrap_angle(float(np.arctan2(to_pole[1], to_pole[0])) - yaw)


def _pole_detected(position, yaw, pole_xy, depth):
    """The column is in the camera when its bearing is inside the field of view and the depth image shows it."""
    error = abs(_view_error(position, yaw, pole_xy))
    half_fov = np.radians(CAMERA_HFOV_DEG) / 2.0
    if error > half_fov * 0.85:
        return False
    if error < np.radians(8.0):
        return True
    if depth is None:
        return False
    return pole_visible(depth)


def _yaw_search(position, yaw, pole_xy):
    """Yaw toward the pole. Hold the horizontal position and the approach altitude."""
    error = _view_error(position, yaw, pole_xy)
    direction = 1.0 if error >= 0.0 else -1.0
    up = float(np.clip(KP_VERTICAL * (TARGET_ALTITUDE_M - float(position[2])), -V_MAX_M_S, V_MAX_M_S))
    return np.array([0.0, 0.0, up, direction * YAW_RATE_MAX_RAD_S], dtype=np.float32)


def _random_pole(vehicle_xy, yaw, rng):
    distance = float(rng.uniform(DISTANCE_MIN_M, DISTANCE_MAX_M))
    lateral = float(rng.uniform(-LATERAL_MAX_M, LATERAL_MAX_M))
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    left = np.array([-np.sin(yaw), np.cos(yaw)])
    return np.asarray(vehicle_xy, dtype=np.float64) + distance * forward + lateral * left


def _next_pole(vehicle_xy, pole_xy, yaw, rng):
    """Place the next pole from the landing, on a heading that is not straight ahead."""
    vehicle_xy = np.asarray(vehicle_xy, dtype=np.float64)
    to_pole = np.asarray(pole_xy, dtype=np.float64) - vehicle_xy
    if float(np.linalg.norm(to_pole)) < 1e-3:
        previous = yaw
    else:
        previous = float(np.arctan2(to_pole[1], to_pole[0]))
    turn = float(rng.choice([-1.0, 1.0]) * rng.uniform(np.pi / 3, np.pi))
    return _random_pole(vehicle_xy, previous + turn, rng)
