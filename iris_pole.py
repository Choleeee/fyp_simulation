#!/usr/bin/env python
"""Iris, a 76 mm pole, and a forward depth camera in Isaac Sim.

Scene only, for a manual takeoff:

    isaac_run /path/to/iris_pole.py

Collect expert demonstrations. The pole is moved to a new 3–5 m start
between episodes. Close any MAVLink shell first; this listens on UDP 14550.

    isaac_run /path/to/iris_pole.py --collect --episodes 5

Train on the Ubuntu machine, outside Isaac, after collection:

    python3 train_policy.py

Fly the trained network, or record DAgger labels while it flies:

    isaac_run /path/to/iris_pole.py --rollout --episodes 5
    isaac_run /path/to/iris_pole.py --dagger --episodes 5
"""

import os
import sys


def _consume_args(argv):
    mode = "scene"
    episodes = 5
    policy = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.pt")
    kept = [argv[0]]
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg == "--collect":
            mode = "collect"
        elif arg == "--rollout":
            mode = "rollout"
        elif arg == "--dagger":
            mode = "dagger"
        elif arg == "--episodes" and index + 1 < len(argv):
            index += 1
            episodes = int(argv[index])
        elif arg == "--policy" and index + 1 < len(argv):
            index += 1
            policy = argv[index]
        else:
            kept.append(arg)
        index += 1
    return mode, episodes, policy, kept


MODE, EPISODES, POLICY_PATH, _KEPT = _consume_args(sys.argv)
sys.argv = _KEPT

import carb
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import omni.timeline
from omni.isaac.core.objects import FixedCylinder
from omni.isaac.core.world import World
from scipy.spatial.transform import Rotation

from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend,
    PX4MavlinkBackendConfig,
)
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task import (  # noqa: E402
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    DEPTH_MAX_M,
    DEPTH_MIN_M,
    POLE_DIAMETER_M,
    POLE_HEIGHT_M,
    camera_intrinsics,
)

POLE_DISTANCE_M = 3.0


def _enable_ros_camera():
    try:
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.ros2.bridge")
        from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend

        return ROS2Backend
    except Exception as exc:
        carb.log_warn(f"ROS 2 camera bridge is not available: {exc}")
        return None


class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Box Room"])

        self.pole = self.world.scene.add(
            FixedCylinder(
                prim_path="/World/pole",
                name="pole",
                position=np.array([POLE_DISTANCE_M, 0.0, POLE_HEIGHT_M / 2.0]),
                radius=POLE_DIAMETER_M / 2.0,
                height=POLE_HEIGHT_M,
                color=np.array([180, 180, 180]),
            )
        )

        config = MultirotorConfig()
        config.backends = [
            PX4MavlinkBackend(
                PX4MavlinkBackendConfig(
                    {
                        "vehicle_id": 0,
                        "px4_autolaunch": True,
                        "px4_dir": self.pg.px4_path,
                        "px4_vehicle_model": self.pg.px4_default_airframe,
                    }
                )
            )
        ]
        ros_backend = _enable_ros_camera()
        if ros_backend is not None:
            config.backends.append(
                ros_backend(
                    vehicle_id=1,
                    config={
                        "namespace": "drone",
                        "pub_sensors": False,
                        "pub_graphical_sensors": True,
                        "pub_state": False,
                        "sub_control": False,
                    },
                )
            )

        self.camera = MonocularCamera(
            "camera",
            config={
                "depth": True,
                "position": np.array([0.10, 0.0, 0.05]),
                "resolution": (CAMERA_WIDTH, CAMERA_HEIGHT),
                "frequency": 10,
                "clipping_range": (DEPTH_MIN_M, DEPTH_MAX_M),
                "intrinsics": camera_intrinsics(),
            },
        )
        config.graphical_sensors = [self.camera]

        self.vehicle = Multirotor(
            "/World/quadrotor",
            ROBOTS["Iris"],
            0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config,
        )
        self.world.reset()
        self.session = None
        if MODE != "scene":
            from flight import ApproachSession

            self.session = ApproachSession(
                vehicle=self.vehicle,
                pole=self.pole,
                camera_sensor=self.camera,
                mode=MODE,
                episodes=EPISODES,
                policy_path=POLICY_PATH,
                data_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
            )

    def run(self):
        self.timeline.play()
        while simulation_app.is_running():
            self.world.step(render=True)
            if self.session is not None:
                self.session.on_step()
                if self.session.finished():
                    break
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()


if __name__ == "__main__":
    PegasusApp().run()
