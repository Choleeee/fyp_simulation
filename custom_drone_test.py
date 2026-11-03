#!/usr/bin/env python
"""
Standalone Pegasus Simulator script to launch a single custom drone.
Based on the official Pegasus example (1_px4_single_vehicle.py).
"""

import carb
from isaacsim import SimulationApp

# Start Isaac Sim's simulation environment
# Note: This must be instantiated right after the SimulationApp import
simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
from omni.isaac.core.world import World

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from scipy.spatial.transform import Rotation

class PegasusApp:
    """
    A Template class that serves as an example on how to build a simple Isaac Sim standalone App.
    """

    def __init__(self):
        """
        Method that initializes the PegasusApp and is used to setup the simulation environment.
        """
        # Acquire the timeline that will be used to start/stop the simulation
        self.timeline = omni.timeline.get_timeline_interface()

        # Start the Pegasus Interface
        self.pg = PegasusInterface()

        # Acquire the World
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Launch one of the worlds provided by NVIDIA
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Box Room"])

        # ---------------------------------------------------------
        # CUSTOM DRONE CONFIGURATION
        # ---------------------------------------------------------
        config_multirotor = MultirotorConfig()

        # --- IMPORTANT: SET YOUR CUSTOM USD PATH HERE ---
        # Replace the string below with the absolute path to your modified USD file
        config_multirotor.usd_file = "/home/fyp/fyp_simulation/Drone_Model/usd/flattened-version.usd" 

        # Configure the PX4 backend
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": True,
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": self.pg.px4_default_airframe
        })
        config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]

        # Spawn the vehicle with your custom model
        Multirotor(
            stage_prefix="/World/quadrotor",
            usd_file=config_multirotor.usd_file, # Use the custom file
            vehicle_id=0,
            init_pos=[0.0, 0.0, 0.5], # Start at 0.5m height
            init_orientation=Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

        # Reset the simulation environment so that all articulations are initialized
        self.world.reset()

        # Auxiliary variable for the timeline callback example
        self.stop_sim = False

    def run_simulation(self):
        """
        Method that implements the simulation loop.
        """
        # Start the simulation
        self.timeline.play()

        # The "infinite" simulation loop
        while simulation_app.is_running() and not self.stop_sim:
            # Run the simulation for a step
            self.world.step(render=True)

        # Cleanup
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()

def main():
    # Instantiate the template app
    pg_app = PegasusApp()

    # Run the application loop
    pg_app.run_simulation()

if __name__ == "__main__":
    main()