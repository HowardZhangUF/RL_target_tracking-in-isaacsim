from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import sys
import carb
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path


class LeaderFollowerEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        # ===== Initialize Isaac Sim World =====
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            simulation_app.close()
            sys.exit()

        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        set_camera_view(
            eye=[5.0, 0.0, 1.5],
            target=[0.00, 0.00, 1.00],
            camera_prim_path="/OmniverseKit_Persp",
        )

        # --- Add Jetbot (Follower) ---
        jb_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        add_reference_to_stage(usd_path=jb_path, prim_path="/World/Jetbot")
        self.jetbot = Articulation(prim_paths_expr="/World/Jetbot", name="my_jetbot")

        # --- Add Carter (Leader / Static Target) ---
        car_path = assets_root_path + "/Isaac/Robots/NVIDIA/Carter/nova_carter/nova_carter.usd"
        add_reference_to_stage(usd_path=car_path, prim_path="/World/Car")
        self.car = Articulation(prim_paths_expr="/World/Car", name="my_car")

        self.stage_units = get_stage_units()

        # Observation: relative x, y
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )

        # Action: follower velocity in x, y
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset follower and place leader at fixed target
        self.jetbot.set_world_poses(
            positions=np.array([[0.0, 0.0, 0.0]]) / self.stage_units
        )
        self.car.set_world_poses(
            positions=np.array([[2.0, 0.0, 0.0]]) / self.stage_units  # leader stays at (2, 0)
        )

        self.world.reset()

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        dt = self.world.get_physics_dt()

        # --- Leader stays static, no motion ---
        # Do nothing to leader position

        # --- Follower motion from RL agent ---
        follower_pos, follower_rot = self.jetbot.get_world_poses()
        follower_pos[0][0] += action[0] * dt
        follower_pos[0][1] += action[1] * dt
        self.jetbot.set_world_poses(follower_pos, follower_rot)

        # --- Step simulation ---
        self.world.step(render=True)

        # --- Reward: negative distance to static leader ---
        obs = self._get_obs()
        distance = np.linalg.norm(obs)
        reward = -distance

        # Done if very close or too far
        done = distance < 0.2 or distance > 10.0

        return obs, reward, done, False, {}

    def _get_obs(self):
        leader_pos, _ = self.car.get_world_poses()
        follower_pos, _ = self.jetbot.get_world_poses()
        rel_pos = (leader_pos[0][:2] - follower_pos[0][:2]).astype(np.float32)
        return rel_pos

    def render(self, mode="human"):
        pass

    def close(self):
        simulation_app.close()


# ===== Create environment and train PPO =====
env = LeaderFollowerEnv()
Kp = 10.0
Kd = 1.0
Ki = 0.1

integral = 0.0
prev_error = None
total_reward = 0.0
rewards = []
observation, info = env.reset()
for step in range(1000):
    env.render()
    rel_x, rel_y = observation
    distance = np.linalg.norm([rel_x, rel_y])
    error = distance
    integral += error
    derivative = 0.0 if prev_error is None else (error - prev_error)
    prev_error = error
    speed = Kp * error + Ki * integral + Kd * derivative
    direction = np.array([rel_x, rel_y]) / (distance + 1e-6)
    action = direction * np.clip(speed, env.action_space.low[0], env.action_space.high[0])
    action = np.clip(action, env.action_space.low, env.action_space.high)
    observation, reward, done, _, info = env.step(action)
    total_reward += reward
    rewards.append(total_reward)
    print(f"Step: {step}, Reward: {reward}, Total Reward: {total_reward}")
    if done:
        print("Reached goal or failed.")
        print(f"Final distance to goal: {error}")
        if error < 0.2:
            print("Reached goal!")
        else:
            print("Failed to reach goal.")
        observation, info = env.reset()
        integral = 0.0
        prev_error = None

env.close()
simulation_app.close()