#!/usr/bin/env python
"""Iris, a lamppost on a 76 mm shaft, and a forward depth camera in Isaac Sim.

Scene only, for a manual takeoff:

    isaac_run /path/to/iris_pole.py

Collect expert demonstrations. Recording stops at the staging point, the
vehicle lands there, and the next pole is placed from that landing on a new
heading. Close any MAVLink shell first; this listens on UDP 14550.

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
from pxr import Gf, Sdf, UsdGeom, UsdShade
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
# Charcoal shaft. The lamp and the wider base are parented to this cylinder,
# so moving the pole between episodes carries them with it.
POLE_COLOR = np.array([32, 34, 38])
# Lower section of the column: 165 mm across and 1.655 m tall, then a short
# shoulder back to the 76 mm shaft. The overall column stays 4 m.
BASE_DIAMETER_M = 0.165
BASE_HEIGHT_M = 1.655
BASE_TAPER_M = 0.08


def _bind_color(stage, prim, path, color, emissive=None):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.6 if emissive is None else 0.0)
    if emissive is not None:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _add_box(stage, path, translate, scale, color, emissive=None):
    """Unit cube, scaled and then moved in the pole's local frame."""
    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(box.GetPrim())
    scale_op = xform.AddScaleOp()
    scale_op.Set(Gf.Vec3f(*scale))
    move_op = xform.AddTranslateOp()
    move_op.Set(Gf.Vec3d(*translate))
    xform.SetXformOpOrder([scale_op, move_op])
    _bind_color(stage, box.GetPrim(), path + "/Looks", color, emissive)
    return box


def _add_frustum(stage, path, z_bottom, z_top, radius_bottom, radius_top, color, segments=24):
    """Taper from the wide base up to the shaft. Points are in the pole frame."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    points = []
    for radius, z in ((radius_bottom, z_bottom), (radius_top, z_top)):
        for angle in angles:
            points.append(Gf.Vec3f(float(radius * np.cos(angle)), float(radius * np.sin(angle)), float(z)))
    counts = []
    indices = []
    for index in range(segments):
        nxt = (index + 1) % segments
        counts.append(4)
        indices.extend([index, nxt, segments + nxt, segments + index])
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    _bind_color(stage, mesh.GetPrim(), path + "/Looks", color)
    return mesh


def _add_base(stage, pole_path):
    """Wider bottom section, 165 mm across and 1.655 m above the ground."""
    bottom = -POLE_HEIGHT_M / 2.0
    shoulder_bottom = bottom + BASE_HEIGHT_M - BASE_TAPER_M
    shoulder_top = bottom + BASE_HEIGHT_M
    barrel_height = shoulder_bottom - bottom
    color = (0.12, 0.13, 0.14)

    barrel = UsdGeom.Cylinder.Define(stage, pole_path + "/base")
    barrel.CreateRadiusAttr(BASE_DIAMETER_M / 2.0)
    barrel.CreateHeightAttr(barrel_height)
    barrel.CreateAxisAttr("Z")
    move = UsdGeom.Xformable(barrel.GetPrim()).AddTranslateOp()
    move.Set(Gf.Vec3d(0.0, 0.0, bottom + barrel_height / 2.0))
    _bind_color(stage, barrel.GetPrim(), pole_path + "/base/Looks", color)
    _add_frustum(
        stage,
        pole_path + "/base_taper",
        shoulder_bottom,
        shoulder_top,
        BASE_DIAMETER_M / 2.0,
        POLE_DIAMETER_M / 2.0,
        color,
    )


def _dress_as_lamppost(pole_path):
    """Add the lamp head and the wider base. Local z is centred on the shaft."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    top = POLE_HEIGHT_M / 2.0
    shaft = stage.GetPrimAtPath(pole_path)
    _bind_color(stage, shaft, pole_path + "/Looks", (0.12, 0.13, 0.14))
    _add_base(stage, pole_path)
    # Arm starts at the shaft surface and reaches out along world +X.
    _add_box(
        stage,
        pole_path + "/arm",
        translate=(0.32, 0.0, top - 0.05),
        scale=(0.56, 0.04, 0.04),
        color=(0.16, 0.16, 0.17),
    )
    _add_box(
        stage,
        pole_path + "/head",
        translate=(0.52, 0.0, top - 0.16),
        scale=(0.22, 0.14, 0.10),
        color=(0.18, 0.18, 0.19),
    )
    _add_box(
        stage,
        pole_path + "/lamp",
        translate=(0.52, 0.0, top - 0.22),
        scale=(0.14, 0.08, 0.02),
        color=(1.0, 0.78, 0.35),
        emissive=(1.0, 0.72, 0.25),
    )


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
                color=POLE_COLOR,
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
        _dress_as_lamppost(self.pole.prim_path)
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
