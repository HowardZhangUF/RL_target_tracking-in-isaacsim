from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import sys
import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path

# Preparing scene
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()
set_camera_view(
    eye=[5.0, 0.0, 1.5], target=[0.00, 0.00, 1.00], camera_prim_path="/OmniverseKit_Persp"
)

# Add Jetbot (Follower)
asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/Jetbot")
jetbot = Articulation(prim_paths_expr="/World/Jetbot", name="my_jetbot")

# Add Carter (Leader)
asset_path = assets_root_path + "/Isaac/Robots/NVIDIA/Carter/nova_carter/nova_carter.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/Car")
car = Articulation(prim_paths_expr="/World/Car", name="my_car")

# Initial positions
jetbot.set_world_poses(positions=np.array([[0.0, 1.0, 0.0]]) / get_stage_units())
car.set_world_poses(positions=np.array([[0.0, -1.0, 0.0]]) / get_stage_units())

my_world.reset()

# Leader path: move forward in +X
leader_speed = 0.5  # m/s
follower_gain = 1.0  # proportional gain

for step in range(300):
    # --- Move leader ---
    leader_pos, leader_rot = car.get_world_poses()
    leader_pos[0][0] += leader_speed * my_world.get_physics_dt()
    car.set_world_poses(positions=leader_pos, orientations=leader_rot)

    # --- Follower control ---
    follower_pos, follower_rot = jetbot.get_world_poses()
    error_vec = leader_pos[0] - follower_pos[0]  # vector from follower to leader
    follower_pos[0] += follower_gain * error_vec * my_world.get_physics_dt()
    jetbot.set_world_poses(positions=follower_pos, orientations=follower_rot)

    # Step simulation
    my_world.step(render=True)

simulation_app.close()
