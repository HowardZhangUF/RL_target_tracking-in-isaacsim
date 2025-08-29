from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import sys
import carb
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path

# USD helpers to draw a red goal dot
import omni.usd
from pxr import UsdGeom, Gf


class LeaderFollowerEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

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
        jb_path = assets_root_path + "/Isaac/Robots/Crazyflie/crazyflie.usd"
        add_reference_to_stage(usd_path=jb_path, prim_path="/World/Crazyflie")
        self.jetbot = Articulation(prim_paths_expr="/World/Crazyflie", name="my_jetbot")

        # --- Add Carter (Leader) ---
        car_path = assets_root_path + "/Isaac/Robots/NVIDIA/Carter/nova_carter/nova_carter.usd"
        add_reference_to_stage(usd_path=car_path, prim_path="/World/Car")
        self.car = Articulation(prim_paths_expr="/World/Car", name="my_car")

        self.stage_units = get_stage_units()

        # ===== Speed caps & episode params =====
        self.jetbot_speed_max = 1.0   # m/s (scales RL action)
        self.car_speed_max    = 1.2   # m/s (leader speed toward goal)
        self.max_steps        = 600   # episode cap
        self.capture_radius   = 0.20  # follower "captures" leader (meters)
        self.far_reset_dist   = 12.0  # optional: fail if too far (meters)
        self.goal_radius      = 0.25  # leader reaches goal then respawn (meters)
        self.goal_xy          = np.array([2.0, 0.0], dtype=np.float32)  # initial goal
        self.goal_prim_path   = "/World/GoalDot"

        # ===== Reward shaping (Pygame-style) =====
        # Pygame env used: distance_reward = -0.0001 * dist  (pixels)
        # Here units are meters; set coeff to -1e-4 to replicate tiny magnitude,
        # or use -1.0 to make distance matter more in meters.
        self.rew_distance_coeff = -1.0      # set to -1e-4 to mimic Pygame scaling
        self.rew_approach_gain  = 2.0
        self.rew_retreat_gain   = 5.0       # penalty magnitude when moving away
        self.rew_capture_bonus  = 10.0

        # ===== Expert / PID toggle for DAgger/BC =====
        self.expert_enabled = False   # if True, compute expert PID action each step
        self.expert_override = False  # if True, override policy action with expert action
        self.expert_mix_prob = 0.0    # if >0, with prob p use expert (DAgger-style mixing)

        # PID gains (m/s per meter of error)
        self.pid_kp = 2.0
        self.pid_ki = 0.0
        self.pid_kd = 0.3
        self._pid_integral = np.zeros(2, dtype=np.float32)
        self._pid_prev_err = np.zeros(2, dtype=np.float32)

        # ===== Spaces =====
        # Observation: [dx, dy, distance]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        # Direct velocity control [-1,1] for each component (normalized)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Book-keeping
        self.t = 0.0
        self.steps = 0
        self.prev_distance = 0.0

        # Create the red goal dot once
        self._ensure_goal_prim()

        self.reset()

    # ---------- Public controls for expert / curriculum etc. ----------
    def set_expert(self, enabled: bool, override: bool = False, mix_prob: float = 0.0):
        """
        Enable/disable expert PID.
        - enabled: compute expert action and store in info["expert_action"].
        - override: if True, replace policy action with expert action (imitation).
        - mix_prob ∈ [0,1]: with probability p, use expert instead of policy (DAgger).
        """
        self.expert_enabled = bool(enabled)
        self.expert_override = bool(override)
        self.expert_mix_prob = float(np.clip(mix_prob, 0.0, 1.0))

    # ---------- USD helpers ----------
    def _ensure_goal_prim(self):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.goal_prim_path)
        if not prim.IsValid():
            sphere = UsdGeom.Sphere.Define(stage, self.goal_prim_path)
            sphere.CreateRadiusAttr(0.06)  # ~6 cm
            sphere.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.0, 0.0)])  # red
        self._set_goal_prim_pose(self.goal_xy)

    def _set_goal_prim_pose(self, xy):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.goal_prim_path)
        xform_api = UsdGeom.XformCommonAPI(prim)
        # NOTE: XformCommonAPI.SetTranslate expects Gf.Vec3d
        xform_api.SetTranslate(Gf.Vec3d(float(xy[0]), float(xy[1]), 0.0))

    def _sample_new_goal(self):
        gx = np.random.uniform(1.0, 4.0)
        gy = np.random.uniform(-2.0, 2.0)
        self.goal_xy = np.array([gx, gy], dtype=np.float32)
        self._set_goal_prim_pose(self.goal_xy)

    # ---------- Gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0.0
        self.steps = 0
        self._pid_integral[:] = 0.0
        self._pid_prev_err[:] = 0.0

        # Place follower (origin) and leader (ahead on x)
        self.jetbot.set_world_poses(
            positions=np.array([[0.0, 0.0, 0.0]]) / self.stage_units
        )
        self.car.set_world_poses(
            positions=np.array([[2.0, 0.0, 0.0]]) / self.stage_units
        )

        # New random goal for leader to chase
        self._sample_new_goal()

        # Commit transforms
        self.world.reset()
        self.world.step(render=False)

        # Initialize prev_distance for reward shaping
        leader_pos, _ = self.car.get_world_poses()
        follower_pos, _ = self.jetbot.get_world_poses()
        rel = leader_pos[0][:2] - follower_pos[0][:2]
        self.prev_distance = float(np.linalg.norm(rel))

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        """Direct velocity control:
           action in [-1,1]^2 -> (vx,vy) = action * jetbot_speed_max
        """
        dt = self.world.get_physics_dt()
        self.t += dt
        self.steps += 1

        # ===== Leader: chase goal at capped speed =====
        car_pos, car_rot = self.car.get_world_poses()
        car_xy = car_pos[0][:2]
        to_goal = self.goal_xy - car_xy
        dist_to_goal = float(np.linalg.norm(to_goal))
        if dist_to_goal < self.goal_radius:
            self._sample_new_goal()
        else:
            dir_vec = to_goal / (dist_to_goal + 1e-8)
            v_leader = dir_vec * self.car_speed_max
            car_pos[0][0] += float(v_leader[0]) * dt
            car_pos[0][1] += float(v_leader[1]) * dt
            self.car.set_world_poses(car_pos, car_rot)

        # ===== Compute expert PID action (normalized), if enabled =====
        expert_action = None
        if self.expert_enabled:
            expert_action = self._pid_expert_action(dt)  # normalized in [-1,1]^2

        # ===== Follower: map selected action to velocity =====
        a = np.clip(action, -1.0, 1.0).astype(np.float32)
        if self.expert_enabled:
            use_expert = self.expert_override or (np.random.rand() < self.expert_mix_prob)
            if use_expert and expert_action is not None:
                a = expert_action  # override/mix for DAgger/IL

        vx = float(a[0]) * self.jetbot_speed_max
        vy = float(a[1]) * self.jetbot_speed_max

        fol_pos, fol_rot = self.jetbot.get_world_poses()
        fol_pos[0][0] += vx * dt
        fol_pos[0][1] += vy * dt
        self.jetbot.set_world_poses(fol_pos, fol_rot)

        # Step sim
        self.world.step(render=True)

        # Observation / reward / termination
        obs = self._get_obs()
        distance = float(obs[2])

        reward = self._compute_reward(distance)

        captured  = distance < self.capture_radius
        too_far   = distance > self.far_reset_dist
        timeout   = self.steps >= self.max_steps

        terminated = captured or too_far
        truncated  = (not terminated) and timeout

        info = {
            "distance": distance,
            "steps": self.steps,
            "captured": bool(captured),
            "goal": self.goal_xy.copy(),
        }
        # Always export expert label for BC/DAgger collection
        if expert_action is not None:
            info["expert_action"] = expert_action.copy()

        return obs, reward, terminated, truncated, info

    # ---------- Reward shaping (ported from Pygame) ----------
    def _compute_reward(self, current_distance: float) -> float:
        """
        Pygame-style reward:
          - distance_term = coeff * distance
          - approach_bonus = max(0, prev - current) * gain
          - retreat_penalty = min(0, prev - current) * gain_retreat  (negative)
          - capture_bonus if within capture_radius
        """
        # distance component
        distance_term = self.rew_distance_coeff * current_distance

        # approach/retreat shaping
        delta = self.prev_distance - current_distance
        approach_bonus = max(0.0, delta) * self.rew_approach_gain
        retreat_penalty = min(0.0, delta) * self.rew_retreat_gain  # negative

        # capture bonus
        capture_bonus = self.rew_capture_bonus if current_distance < self.capture_radius else 0.0

        # update prev_distance
        self.prev_distance = current_distance

        return distance_term + approach_bonus + retreat_penalty + capture_bonus

    # ---------- Expert PID (returns normalized action in [-1,1]^2) ----------
    def _pid_expert_action(self, dt: float) -> np.ndarray:
        """
        Compute a normalized velocity command [-1,1]^2 via PID towards the leader.
        This mimics your Pygame 'autopilot' mode but outputs the *normalized*
        action that a policy would produce (for BC/DAgger labels).
        """
        leader_pos, _ = self.car.get_world_poses()
        follower_pos, _ = self.jetbot.get_world_poses()
        err = (leader_pos[0][:2] - follower_pos[0][:2]).astype(np.float32)

        # PID on position error to produce velocity (m/s)
        self._pid_integral += err * dt
        deriv = (err - self._pid_prev_err) / max(dt, 1e-6)
        self._pid_prev_err = err.copy()

        v_cmd = self.pid_kp * err + self.pid_ki * self._pid_integral + self.pid_kd * deriv

        # Normalize to action space by speed cap
        a = v_cmd / max(self.jetbot_speed_max, 1e-6)
        a = np.clip(a, -1.0, 1.0).astype(np.float32)
        return a

    # ---------- Obs ----------
    def _get_obs(self):
        leader_pos, _ = self.car.get_world_poses()
        follower_pos, _ = self.jetbot.get_world_poses()
        rel = (leader_pos[0][:2] - follower_pos[0][:2]).astype(np.float32)
        d = float(np.linalg.norm(rel))
        return np.array([rel[0], rel[1], d], dtype=np.float32)

    def render(self, mode="human"):
        pass

    def close(self):
        simulation_app.close()
