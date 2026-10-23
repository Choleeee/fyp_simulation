"""Independent command supervisor between the approach and PX4.

The policy is not trusted. A command is forwarded only when it stays inside
the approach limits. Anything else is dropped, and the caller must abort.
The same checks apply whether the command came from the learned network or
from a test generator.
"""

import numpy as np

from task import STANDOFF_M, V_MAX_M_S, YAW_RATE_MAX_RAD_S

# A one-second hold of the command is long enough to see a standoff breach
# and short enough that a legal 0.5 m/s approach from 3–5 m still passes.
LOOKAHEAD_S = 1.0
STALE_S = 0.1
# One missed 10 Hz setpoint is 0.2 s since the previous accept.
DROPOUT_S = 0.2
GEOFENCE_HALF_M = 30.0
# Inside this range, a command aimed through the staging circle is a pole hit.
POLE_GUARD_M = 1.0


class Command:
    def __init__(self, forward, left, up, yaw_rate, timestamp):
        self.forward = float(forward)
        self.left = float(left)
        self.up = float(up)
        self.yaw_rate = float(yaw_rate)
        self.timestamp = float(timestamp)

    @classmethod
    def from_action(cls, action, timestamp):
        forward, left, up, yaw_rate = [float(value) for value in action]
        return cls(forward, left, up, yaw_rate, timestamp)


class ApproachSupervisor:
    def __init__(self, geofence_half_m=GEOFENCE_HALF_M):
        self.geofence_half_m = float(geofence_half_m)
        self.reason = None
        self._last_accept = None
        self._streaming = False

    @property
    def aborted(self):
        return self.reason is not None

    def reset(self):
        self.reason = None
        self._last_accept = None
        self._streaming = False

    def evaluate(self, command, position, yaw, pole_xy, now):
        """Return True when the command may be sent to PX4."""
        if self.aborted:
            return False
        reason = _violation(
            command,
            position,
            yaw,
            pole_xy,
            now,
            self.geofence_half_m,
        )
        if reason is not None:
            self.reason = reason
            return False
        self._last_accept = float(now)
        self._streaming = True
        return True

    def poll(self, now):
        """Abort when a started stream has stopped."""
        if self.aborted or not self._streaming or self._last_accept is None:
            return not self.aborted
        if float(now) - self._last_accept > DROPOUT_S:
            self.reason = "dropout"
            return False
        return True


def _violation(command, position, yaw, pole_xy, now, geofence_half_m):
    if float(now) - command.timestamp > STALE_S:
        return "stale"
    if _excess_speed(command):
        return "velocity"
    position = np.asarray(position, dtype=np.float64)
    velocity = _world_velocity(command, yaw)
    if _outside_geofence(position[:2], velocity, geofence_half_m):
        return "geofence"
    return _range_violation(position[:2], velocity, pole_xy)


def _excess_speed(command):
    horizontal = float(np.hypot(command.forward, command.left))
    return (
        horizontal > V_MAX_M_S + 1e-6
        or abs(command.up) > V_MAX_M_S + 1e-6
        or abs(command.yaw_rate) > YAW_RATE_MAX_RAD_S + 1e-6
    )


def _world_velocity(command, yaw):
    yaw = float(yaw)
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    left = np.array([-np.sin(yaw), np.cos(yaw)])
    return command.forward * forward + command.left * left


def _outside_geofence(position_xy, velocity, half):
    predicted = np.asarray(position_xy, dtype=np.float64) + velocity * LOOKAHEAD_S
    for point in (position_xy, predicted):
        if max(abs(float(point[0])), abs(float(point[1]))) > half:
            return True
    return False


def _range_violation(position_xy, velocity, pole_xy):
    position_xy = np.asarray(position_xy, dtype=np.float64)
    to_pole = np.asarray(pole_xy, dtype=np.float64) - position_xy
    distance = float(np.linalg.norm(to_pole))
    speed = float(np.linalg.norm(velocity))
    if distance < 1e-3:
        return "pole_motion" if speed > 0.05 else None
    closing = float(np.dot(velocity, to_pole / distance))
    if closing <= 0.0:
        return None
    predicted = position_xy + velocity * LOOKAHEAD_S
    predicted_distance = float(np.linalg.norm(np.asarray(pole_xy, dtype=np.float64) - predicted))
    if predicted_distance > STANDOFF_M + 1e-6:
        return None
    # A residual at the staging hover is not a new command into the column.
    if distance <= STANDOFF_M + 0.15 and closing <= 0.35:
        return None
    aimed_at_pole = speed > 1e-6 and closing / speed > 0.95
    if distance <= POLE_GUARD_M and aimed_at_pole:
        return "pole_motion"
    return "standoff"
