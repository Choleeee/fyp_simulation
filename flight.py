"""Run privileged-expert or learned-policy approaches inside iris_pole.py.

The vehicle is commanded in PX4 offboard mode. It climbs at the start
without recording, then the approach is recorded until the staging point,
half a metre short of the pole. If the column is outside the camera, the
vehicle yaws in place until the depth image shows it, then approaches. It
lands on that spot. The next pole is placed 3–5 m from the landing on a new
heading, at least 60 degrees off the direction to the pole just reached.

After each landing the vehicle flies back to the origin and lands there. The
next pole is placed on a 3–5 m radius from that point, so a long run stays
inside the geofence. Paired episodes keep the saved bearing.

A timed-out or aborted episode descends in offboard and the next episode
starts only after the vehicle is disarmed on the ground. Return-to-launch is
not used: that mode climbs far above the room and the following episodes then
time out while it is still active.
"""

import os
import time

import numpy as np
from pymavlink import mavutil
from scipy.spatial.transform import Rotation

import expert
from supervisor import DROPOUT_S, ApproachSupervisor, Command
from task import (
    ACTION_DIM,
    CAMERA_HFOV_DEG,
    CONTROL_PERIOD_S,
    EPISODE_TIMEOUT_S,
    KP_VERTICAL,
    OBS_HEIGHT,
    OBS_WIDTH,
    POLE_HEIGHT_M,
    STATE_DIM,
    TARGET_ALTITUDE_M,
    V_MAX_M_S,
    YAW_RATE_MAX_RAD_S,
    downsample_depth,
    pole_visible,
    random_pole,
    wrap_angle,
)

OFFBOARD_MODE = 6
# Ignore position, acceleration, and yaw. Use body velocity and yaw rate.
VELOCITY_MASK = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 6) | (1 << 7) | (1 << 8) | (1 << 10)
# Ignore velocity, acceleration, yaw, and yaw rate. Use local position.
POSITION_MASK = (
    (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8) | (1 << 10) | (1 << 11)
)
BODY_NED = 8
LOCAL_NED = 1


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
    def __init__(self, vehicle, pole, camera_sensor, mode, episodes, policy_path, data_dir, starts=None, method=None):
        self.vehicle = vehicle
        self.pole = pole
        self.camera_sensor = camera_sensor
        self.mode = mode
        self.method = method
        self.starts = starts
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
        self.supervisor = ApproachSupervisor()
        self.done = False
        self._status_time = 0.0
        self.takeoff_ned = None
        self._last_fly_tick = None
        self._return_step = None
        self._armed_state = True

        if mode == "paired":
            if starts is None or method not in ("policy", "modular"):
                raise SystemExit("Paired mode needs a start table and method 'policy' or 'modular'.")
            self.episodes_requested = len(starts["yaw"]) if episodes is None else min(int(episodes), len(starts["yaw"]))
            self.episode = self._first_missing_pair()
            seed = int(np.asarray(starts["seed"]).reshape(-1)[0])
            print(
                f"Paired {method}, {self.episodes_requested} starts, seed {seed}, "
                f"resuming at pair {self.episode + 1}"
            )
        if mode in ("rollout", "dagger") or (mode == "paired" and method == "policy"):
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
        action = self._select_action(expert_action, state, position, yaw)

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
                self.takeoff_ned = None
                self.master.mav.request_data_stream_send(
                    self.master.target_system,
                    self.master.target_component,
                    mavutil.mavlink.MAV_DATA_STREAM_POSITION,
                    10,
                    1,
                )
                self.phase = "takeoff"
                self.phase_started = now
            elif now - self.phase_started > 12.0:
                print(f"Episode {self.episode + 1} did not arm")
                self._finish_episode(False, "no arm")
            return

        if self.phase == "takeoff":
            # A body-up velocity never settles: it is still climbing when the
            # height check becomes true. A position setpoint holds the approach altitude.
            self._set_offboard()
            self._command_climb(position)
            height_error = abs(float(position[2]) - TARGET_ALTITUDE_M)
            vertical_speed = abs(float(self.vehicle.state.linear_velocity[2]))
            if height_error < 0.3 and vertical_speed < 0.35:
                self.samples = []
                self.missed_frames = 0
                self.searching = False
                self.phase = "fly"
                self.phase_started = now
                print(f"Episode {self.episode + 1}: pole at x={self.pole_xy[0]:.2f} y={self.pole_xy[1]:.2f}")
            elif now - self.phase_started > 30.0:
                print(
                    f"Episode {self.episode + 1} did not reach approach altitude "
                    f"(z={float(position[2]):.2f})"
                )
                self._finish_episode(False, "no climb")
            return

        if self.phase == "fly":
            depth = _read_depth(self.camera_sensor)
            if self.mode != "paired" and not _pole_detected(position, yaw, self.pole_xy, depth):
                action = _yaw_search(position, yaw, self.pole_xy)
                if not self.searching:
                    self.searching = True
                    print(f"Episode {self.episode + 1}: pole outside the camera, yawing")
            elif self.searching:
                self.searching = False
                print(f"Episode {self.episode + 1}: pole in the camera view")
            # A gap longer than one missed setpoint means Isaac stopped
            # stepping. PX4 then prints "simulator_mavlink poll timeout".
            # That pause is not a command dropout.
            if self._last_fly_tick is not None and now - self._last_fly_tick > DROPOUT_S:
                self.supervisor.resume(now)
            self._last_fly_tick = now
            if not self.supervisor.poll(now):
                self._reject(self.supervisor.reason)
                return
            allowed = self.supervisor.evaluate(
                Command.from_action(action, now),
                position,
                yaw,
                self.pole_xy,
                now,
            )
            if not allowed:
                self._reject(self.supervisor.reason)
                return
            # Re-enter offboard after a stall. Otherwise PX4 stays in the
            # failsafe it selected when the setpoint stream paused.
            self._send(action)
            self._set_offboard()
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
            if self._descend_and_disarm(position, now):
                self._start_return(position, now)
            return

        if self.phase == "return":
            self._return_home(position, now)
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
        self.supervisor.reset()
        self.takeoff_ned = None
        self._last_fly_tick = None
        if self.mode == "paired":
            from starts import pole_at_radius

            position = np.asarray(self.vehicle.state.position, dtype=np.float64)
            yaw = _yaw(self.vehicle.state.attitude)
            self.pole_xy = pole_at_radius(
                self.starts["vehicle_xy"][self.episode],
                self.starts["yaw"][self.episode],
                self.starts["pole_xy"][self.episode],
                position[:2],
                yaw,
            )
            self.pole.set_world_pose(position=np.array([self.pole_xy[0], self.pole_xy[1], POLE_HEIGHT_M / 2.0]))
            distance = float(np.linalg.norm(self.pole_xy - position[:2]))
            print(
                f"Pair {self.episode + 1}: vehicle at x={float(position[0]):.2f} "
                f"y={float(position[1]):.2f} yaw={yaw:.2f}, "
                f"pole at x={self.pole_xy[0]:.2f} y={self.pole_xy[1]:.2f} ({distance:.2f} m)"
            )
            self.phase = "recharge"
            self.phase_started = time.monotonic()
            self.last_command = 0.0
            return
        position = np.asarray(self.vehicle.state.position, dtype=np.float64)
        yaw = _yaw(self.vehicle.state.attitude)
        self.pole_xy = _next_pole(position[:2], self.pole_xy, yaw, np.random.default_rng())
        self.pole.set_world_pose(position=np.array([self.pole_xy[0], self.pole_xy[1], POLE_HEIGHT_M / 2.0]))
        self.phase = "recharge"
        self.phase_started = time.monotonic()
        self.last_command = 0.0

    def _select_action(self, expert_action, state, position, yaw):
        if self.mode == "paired" and self.method == "modular":
            import modular

            return modular.command(_read_depth(self.camera_sensor), position, yaw)
        if self.mode == "collect" or self.policy is None:
            return expert_action
        depth = _read_depth(self.camera_sensor)
        if depth is None:
            if self.mode == "paired":
                import modular

                return modular.hold_altitude(position)
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
        label = executed_action if self.mode in ("rollout", "paired") else expert_action
        self.samples.append((observation, state, label))

    def _finish_episode(self, success, reason):
        self._save(success, reason)
        self.episode += 1
        # The landing is not part of the demonstration. Stay in offboard and
        # descend; the next episode waits until the vehicle is disarmed.
        self.phase = "land"
        self.phase_started = time.monotonic()

    def _save(self, success, reason):
        if self.mode == "paired":
            self._save_pair(success, reason)
            return
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

    def _save_pair(self, success, reason):
        os.makedirs(self.data_dir, exist_ok=True)
        path = self._pair_path(self.episode)
        if self.samples:
            depth, state, action = map(np.stack, zip(*self.samples))
        else:
            depth = np.zeros((0, OBS_HEIGHT, OBS_WIDTH), dtype=np.float32)
            state = np.zeros((0, STATE_DIM), dtype=np.float32)
            action = np.zeros((0, ACTION_DIM), dtype=np.float32)
        smoothness = float("nan")
        if len(action) >= 2:
            smoothness = float(np.abs(np.diff(np.asarray(action, dtype=np.float64), axis=0)).mean())
        np.savez_compressed(
            path,
            depth=np.asarray(depth).astype(np.float16),
            state=np.asarray(state).astype(np.float32),
            action=np.asarray(action).astype(np.float32),
            success=np.array([success]),
            reason=np.array(reason),
            pair=np.array(self.episode),
            method=np.array(self.method),
            time_s=np.array(len(action) * CONTROL_PERIOD_S),
            smoothness=np.array(smoothness),
        )
        print(f"Saved pair {self.episode + 1} to {path} ({reason}, {len(action) * CONTROL_PERIOD_S:.1f} s)")

    def _pair_path(self, index):
        return os.path.join(self.data_dir, f"paired_{self.method}_{int(index):04d}.npz")

    def _first_missing_pair(self):
        for index in range(self.episodes_requested):
            if not os.path.isfile(self._pair_path(index)):
                return index
        return self.episodes_requested

    def _start_return(self, position, now):
        if float(np.hypot(position[0], position[1])) <= 0.8:
            self.phase = "between"
            self.phase_started = now
            return
        print(
            f"Returning to the origin from x={float(position[0]):.2f} y={float(position[1]):.2f}"
        )
        self.phase = "return"
        self._return_step = "stream"
        self.takeoff_ned = None
        self.phase_started = now
        self.last_command = now

    def _return_home(self, position, now):
        """Fly back to the origin and land. The estimator is not teleported."""
        horizontal = float(np.hypot(position[0], position[1]))
        if self._return_step == "stream":
            self._send(np.zeros(4, dtype=np.float32))
            if now - self.phase_started > 1.0:
                self._set_offboard()
                self._arm(force=False)
                self._return_step = "arm"
                self.phase_started = now
            return
        if self._return_step == "arm":
            self._send(np.zeros(4, dtype=np.float32))
            self._arm(force=now - self.phase_started > 3.0)
            if self._armed():
                self._set_offboard()
                self.takeoff_ned = None
                self._return_step = "climb"
                self.phase_started = now
            elif now - self.phase_started > 12.0:
                print(
                    f"Could not arm for the return. Next pole is placed from "
                    f"x={float(position[0]):.2f} y={float(position[1]):.2f}"
                )
                self.phase = "between"
                self.phase_started = now
            return
        if self._return_step == "climb":
            self._set_offboard()
            self._command_climb(position)
            if abs(float(position[2]) - TARGET_ALTITUDE_M) < 0.4 or now - self.phase_started > 30.0:
                self._return_step = "fly"
                self.phase_started = now
            return
        if self._return_step == "fly":
            self._set_offboard()
            self._command_origin(position)
            speed = float(np.linalg.norm(self.vehicle.state.linear_velocity))
            if (horizontal < 0.8 and speed < 0.4) or now - self.phase_started > 90.0:
                if horizontal >= 0.8:
                    print(
                        f"Return stopped at x={float(position[0]):.2f} y={float(position[1]):.2f}"
                    )
                self._return_step = "land"
                self.phase_started = now
            return
        if self._descend_and_disarm(position, now):
            self.phase = "between"
            self.phase_started = now

    def _command_origin(self, position):
        """Position setpoint for Isaac (0, 0) at the approach altitude."""
        local = self.master.recv_match(type="LOCAL_POSITION_NED", blocking=False)
        if local is None:
            direction = -np.asarray(position[:2], dtype=np.float64)
            norm = float(np.linalg.norm(direction))
            yaw = _yaw(self.vehicle.state.attitude)
            if norm < 1e-3:
                body = np.zeros(2)
            else:
                # Isaac x is east and y is north. Body forward is the yaw axis.
                world = 0.5 * direction / norm
                forward = float(world[0] * np.cos(yaw) + world[1] * np.sin(yaw))
                left = float(-world[0] * np.sin(yaw) + world[1] * np.cos(yaw))
                body = np.array([forward, left])
            self._send(np.array([body[0], body[1], 0.0, 0.0], dtype=np.float32))
            return
        # PX4 local x is north, which is Isaac y. Local y is east, which is Isaac x.
        north = float(local.x) + (0.0 - float(position[1]))
        east = float(local.y) + (0.0 - float(position[0]))
        down = float(local.z) - (TARGET_ALTITUDE_M - float(position[2]))
        self.master.mav.set_position_target_local_ned_send(
            0,
            self.master.target_system,
            self.master.target_component,
            LOCAL_NED,
            POSITION_MASK,
            north,
            east,
            down,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def _command_climb(self, position):
        if self.takeoff_ned is None:
            local = self.master.recv_match(type="LOCAL_POSITION_NED", blocking=False)
            if local is None:
                climb = float(np.clip(0.5 * (TARGET_ALTITUDE_M - position[2]), -0.4, 0.4))
                self._send(np.array([0.0, 0.0, climb, 0.0], dtype=np.float32))
                return
            # PX4 down is negative above the origin. Shift by the height still to climb.
            self.takeoff_ned = (
                float(local.x),
                float(local.y),
                float(local.z) - (TARGET_ALTITUDE_M - float(position[2])),
            )
        north, east, down = self.takeoff_ned
        self.master.mav.set_position_target_local_ned_send(
            0,
            self.master.target_system,
            self.master.target_component,
            LOCAL_NED,
            POSITION_MASK,
            north,
            east,
            down,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )

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
        # Offboard loss lands in place. Action 3 is return-to-launch, which
        # climbs to a home altitude above this room. 4 is land.
        self._set_param("COM_OBL_RC_ACT", 4, mavutil.mavlink.MAV_PARAM_TYPE_INT32)
        self._set_param("COM_OF_LOSS_T", 5.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)

    def _release_battery_floor(self):
        # The charge is already at 100%. Restore the normal SITL floor so this
        # episode can discharge from full instead of staying locked at 100%.
        self._set_param("SIM_BAT_MIN_PCT", 50, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)

    def _reject(self, reason):
        print(f"Episode {self.episode + 1}: supervisor abort ({reason})")
        self._finish_episode(False, reason)

    def _descend_and_disarm(self, position, now):
        """True once the vehicle is disarmed on the ground.

        Setpoints keep streaming, so a paused simulator is not followed by a
        return-to-launch climb. The next episode is not started in the air.
        """
        height = float(position[2])
        downward = -0.4 if height > 0.35 else -0.15
        self._send(np.array([0.0, 0.0, downward, 0.0], dtype=np.float32))
        self._set_offboard()
        if height < 0.35:
            self._arm(force=True, disarm=True)
        landed = height < 0.25 and not self._armed()
        if not landed and now - self.phase_started > 45.0 and now - self._status_time > 10.0:
            self._status_time = now
            print(f"Still landing from z={height:.2f}")
        return landed

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
            21196 if force else 0,
            0,
            0,
            0,
            0,
            0,
        )

    def _armed(self):
        heartbeat = self.master.recv_match(type="HEARTBEAT", blocking=False)
        if heartbeat is not None:
            self._armed_state = bool(heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        return self._armed_state

    def _disarm_and_stop(self):
        if self.master is not None:
            self._arm(disarm=True)
        self.done = True
        print(f"Finished {self.episode} episodes")
        if self.mode == "paired":
            from compare_paired import report

            report(self.data_dir)


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


def _next_pole(vehicle_xy, pole_xy, yaw, rng):
    """Place the next pole from the landing, on a heading that is not straight ahead."""
    vehicle_xy = np.asarray(vehicle_xy, dtype=np.float64)
    to_pole = np.asarray(pole_xy, dtype=np.float64) - vehicle_xy
    if float(np.linalg.norm(to_pole)) < 1e-3:
        previous = yaw
    else:
        previous = float(np.arctan2(to_pole[1], to_pole[0]))
    turn = float(rng.choice([-1.0, 1.0]) * rng.uniform(np.pi / 3, np.pi))
    return random_pole(vehicle_xy, previous + turn, rng)
