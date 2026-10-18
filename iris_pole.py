#!/usr/bin/env python
"""Iris quadrotor and a 76 mm lighting column in Isaac Sim.

Copy this file to the Pegasus examples folder and run it with isaac_run:

    cp iris_pole.py ~/PegasusSimulator/examples/
    cd ~/PegasusSimulator
    isaac_run examples/iris_pole.py
"""

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
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS

# 76 mm shaft. The cylinder position is its centre, so a 4 m column stands on z = 0.
POLE_DIAMETER_M = 0.076
POLE_HEIGHT_M = 4.0
POLE_DISTANCE_M = 3.0


class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()

        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Box Room"])

        self.world.scene.add(
            FixedCylinder(
                prim_path="/World/pole",
                name="pole",
                position=np.array([POLE_DISTANCE_M, 0.0, POLE_HEIGHT_M / 2.0]),
                radius=POLE_DIAMETER_M / 2.0,
                height=POLE_HEIGHT_M,
                color=np.array([180, 180, 180]),
            )
        )

        config_multirotor = MultirotorConfig()
        mavlink_config = PX4MavlinkBackendConfig(
            {
                "vehicle_id": 0,
                "px4_autolaunch": True,
                "px4_dir": self.pg.px4_path,
                "px4_vehicle_model": self.pg.px4_default_airframe,
            }
        )
        config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]

        Multirotor(
            "/World/quadrotor",
            ROBOTS["Iris"],
            0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

        self.world.reset()
        self.stop_sim = False

    def run(self):
        self.timeline.play()

        while simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)

        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()


def main():
    pg_app = PegasusApp()
    pg_app.run()


if __name__ == "__main__":
    main()
