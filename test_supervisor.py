#!/usr/bin/env python3
"""Fault matrix for the approach supervisor. No simulator required.

    python3 test_supervisor.py
"""

import numpy as np

from supervisor import POLE_GUARD_M, ApproachSupervisor, Command
from task import STANDOFF_M, V_MAX_M_S

POLE = np.array([4.0, 0.0])


def check(name, ok, reason, expected_ok, expected_reason=None):
    passed = ok is expected_ok and (expected_reason is None or reason == expected_reason)
    print(f"{'pass' if passed else 'FAIL'}  {name}  reason={reason}")
    return passed


def main():
    results = []
    now = 10.0

    legal = ApproachSupervisor()
    approach = Command(0.4, 0.0, 0.0, 0.2, now)
    ok = legal.evaluate(approach, np.array([0.0, 0.0, 1.5]), 0.0, POLE, now)
    results.append(check("approach forwarded", ok, legal.reason, True))

    yaw = ApproachSupervisor()
    search = Command(0.0, 0.0, 0.0, 0.5, now)
    ok = yaw.evaluate(search, np.array([0.0, 0.0, 1.5]), 0.0, POLE, now)
    results.append(check("search yaw forwarded", ok, yaw.reason, True))

    hover = ApproachSupervisor()
    staging = np.array([4.0 - STANDOFF_M, 0.0, 1.5])
    hold = Command(0.1, 0.0, 0.0, 0.0, now)
    ok = hover.evaluate(hold, staging, 0.0, POLE, now)
    results.append(check("staging hover forwarded", ok, hover.reason, True))

    fast = ApproachSupervisor()
    ok = fast.evaluate(Command(2.0, 0.0, 0.0, 0.0, now), np.array([0.0, 0.0, 1.5]), 0.0, POLE, now)
    results.append(check("excessive velocity", ok, fast.reason, False, "velocity"))

    into_pole = ApproachSupervisor()
    ok = into_pole.evaluate(
        Command(V_MAX_M_S, 0.0, 0.0, 0.0, now),
        np.array([POLE[0] - POLE_GUARD_M, 0.0, 1.5]),
        0.0,
        POLE,
        now,
    )
    results.append(check("motion toward the pole", ok, into_pole.reason, False, "pole_motion"))

    fence = ApproachSupervisor(geofence_half_m=5.0)
    ok = fence.evaluate(Command(V_MAX_M_S, 0.0, 0.0, 0.0, now), np.array([4.8, 0.0, 1.5]), 0.0, POLE, now)
    results.append(check("geofence", ok, fence.reason, False, "geofence"))

    # Cross the staging circle from the side without aiming straight at the centre.
    clip = ApproachSupervisor()
    ok = clip.evaluate(
        Command(V_MAX_M_S, 0.0, 0.0, 0.0, now),
        np.array([3.55, 0.4, 1.5]),
        0.0,
        POLE,
        now,
    )
    results.append(check("standoff", ok, clip.reason, False, "standoff"))

    stale = ApproachSupervisor()
    ok = stale.evaluate(Command(0.2, 0.0, 0.0, 0.0, now - 0.15), np.array([0.0, 0.0, 1.5]), 0.0, POLE, now)
    results.append(check("stale input", ok, stale.reason, False, "stale"))

    dropout = ApproachSupervisor()
    dropout.evaluate(Command(0.2, 0.0, 0.0, 0.0, now), np.array([0.0, 0.0, 1.5]), 0.0, POLE, now)
    alive = dropout.poll(now + 0.1)
    stopped = dropout.poll(now + 0.25)
    results.append(check("dropout", stopped, dropout.reason, False, "dropout"))
    results.append(check("stream still alive at 0.1 s", alive, None, True))

    if not all(results):
        raise SystemExit(1)
    print(f"{len(results)} checks passed")


if __name__ == "__main__":
    main()
