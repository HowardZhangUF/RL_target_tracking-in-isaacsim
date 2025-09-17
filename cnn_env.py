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
from isaacsim.core.prims import XFormPrim
from pxr import UsdGeom, Gf, Sdf
from pxr import UsdGeom, Gf, Sdf, UsdShade
from pxr import UsdLux, Sdf
import time,math
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")  # IMPORTANT: non-interactive backend (no plt.show)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pf import ParticleFilter2D  # ← use the module version
import torch
import omni.replicator.core as rep
from pxr import UsdGeom, Gf


class _RgbListener:
    """
    Works with PytorchWriter that calls listener.write_data({...})
    and also with older call-style listeners (listener({...})).
    """
    def __init__(self):
        self.rgb = None
        self.frames_seen = 0

    def write_data(self, data):          # <-- required by your PytorchWriter
        self._consume(data)

    def __call__(self, data):            # <-- keep compatibility
        self._consume(data)

    def close(self):
        pass

    def get(self):
        return self.rgb

    def _consume(self, data):
        # Try standard keys first
        t = None
        if isinstance(data, dict):
            t = data.get("pytorch_rgb", data.get("rgb"))
            # If not found, search one level deeper (some payloads are nested)
            if t is None:
                for v in data.values():
                    if isinstance(v, dict):
                        t = v.get("pytorch_rgb", v.get("rgb"))
                        if t is not None:
                            break

        if t is not None:
            # Remove batch dim if present (1, H, W, C)
            if hasattr(t, "dim") and t.dim() == 4 and t.shape[0] == 1:
                t = t[0]
            self.rgb = t
            self.frames_seen += 1


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
        # === Camera / image obs config ===
        self.img_w = 640
        self.img_h = 480
        self.img_c = 3               # RGB
        self.channel_first = True    # (C,H,W) for PyTorch/Stable-Baselines3

        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        set_camera_view(
            eye=[5.0, 0.0, 1.5],
            target=[0.00, 0.00, 1.00],
            camera_prim_path="/OmniverseKit_Persp",
        )
        self._init_fixed_camera(ortho=False, height=3.0, focal_len=10.0)
        self._init_replicator()          # NEW
        if self.channel_first:
            obs_shape = (self.img_c, self.img_h, self.img_w)
        else:
            obs_shape = (self.img_h, self.img_w, self.img_c)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )
        # --- Add Jetbot (Follower) ---
        drone_usd = assets_root_path + "/Isaac/Robots/Crazyflie/cf2x.usd"
        add_reference_to_stage(usd_path=drone_usd, prim_path="/World/Crazyflie")
        #self.drone = Articulation(prim_paths_expr="/World/Crazyflie", name="my_drone")
        self.drone = XFormPrim("/World/Crazyflie", name="my_drone")

        # --- Add Carter (Leader) ---
        car_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        add_reference_to_stage(usd_path=car_path, prim_path="/World/Jetbot")
        self.car = Articulation(prim_paths_expr="/World/Jetbot", name="Jetbot")

        self.stage_units = get_stage_units()

        # ===== Speed caps & episode params =====
        # CHANGED: separate XY and Z caps for the drone
        self.drone_speed_max_xy = 1.0  # m/s in XY
        self.drone_speed_max_z  = 0.6  # m/s in Z
        self.z_min, self.z_max  = 0.2, 10.0  # altitude bounds (meters)
        self.car_speed_max    = 1.2   # m/s (leader speed toward goal)
        self.max_steps        = 6000   # episode cap
        self.capture_radius   = 0.20  # follower "captures" leader (meters)
        self.far_reset_dist   = 12.0  # optional: fail if too far (meters)
        self.goal_radius      = 0.25  # leader reaches goal then respawn (meters)
        self.goal_xy          = np.array([2.0, 0.0], dtype=np.float32)  # initial goal
        self.goal_prim_path   = "/World/GoalDot"
        self.z_target = 4.0
        # ----- Blue visual area (non-physical) -----
        # Visual (non-physical) marked area
        # ----- Visual occlusion area -----
        self.red_area_rect = (1.0, 2.0, -1.0, 2.0)
        self._ensure_colored_area(rect=self.red_area_rect, color=(1.0, 0.0, 0.0), opacity=0.35)
        self._ensure_bright_lighting()

        # >>> Occlusion settings (logic lives in env)
        self.occlusion_enabled  = True                 # set False to disable
        self.occlusion_mode     = "leader_inside"       # or "los" for line-of-sight
        self.missing_obs_policy = "zeros"               # or "last"
        self._last_visible_obs  = np.zeros(3, dtype=np.float32)  # cache [dx,dy,d]
        # inside LeaderFollowerEnv.__init__
        self._pf = ParticleFilter2D(
            N=600,
            proc_pos_std=0.06,
            proc_vel_std=0.5,
            meas_std=0.05,
            resample_frac=0.5,
        )

        # ---- Observation space now includes visibility bit ----
        # self.observation_space = spaces.Box(
        #     low=np.array([-np.inf, -np.inf, 0.0, 0.0], dtype=np.float32),
        #     high=np.array([ np.inf,  np.inf, np.inf, 1.0], dtype=np.float32),
        #     dtype=np.float32,
        # )
       

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

       
        # Direct velocity control [-1,1] for each component (normalized)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Book-keeping
        self.t = 0.0
        self.steps = 0
        self.prev_distance = 0.0

        # --- particle snapshot config/state for info emission ---
        self.particle_emit        = True     # turn on/off
        self.particle_stride      = 2        # emit every N steps
        self.particle_max_show    = 250      # at most M particles
        self._particle_idx        = None     # fixed subset indices
        self._rng_particles       = np.random.default_rng(123)
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
            sphere.CreateRadiusAttr(0.006)  # ~6 cm
            sphere.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.0, 0.0)])  # red
        self._set_goal_prim_pose(self.goal_xy)


    def _ensure_colored_area(self, rect=None, color=(1.0, 0.0, 0.0), opacity=0.35):
        """
        Create a thin, semi-transparent colored quad on the floor to mark a region.
        Uses USD Preview Surface material with correct output connection.
        Non-physical (collisions disabled).
        """
        if rect is None:
            rect = (-1.0, 1.5, -1.0, 0.5)  # (xmin, xmax, ymin, ymax)

        stage = omni.usd.get_context().get_stage()
        area_path = "/World/RedArea"
        mat_path  = "/World/RedArea_Mat"

        # ---- Prim (cube scaled to a flat rectangle) ----
        prim = stage.GetPrimAtPath(area_path)
        if not prim.IsValid():
            UsdGeom.Cube.Define(stage, area_path)
            prim = stage.GetPrimAtPath(area_path)
            # optional display color (material will dominate)
            gprim = UsdGeom.Gprim(prim)
            gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            # explicitly disable collisions
            prim.CreateAttribute(
                "physics:collisionEnabled", Sdf.ValueTypeNames.Bool
            ).Set(False)

        xmin, xmax, ymin, ymax = rect
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        sx = max(xmax - xmin, 1e-3)
        sy = max(ymax - ymin, 1e-3)

        xform = UsdGeom.XformCommonAPI(prim)
        # Lift to avoid z-fighting with ground
        xform.SetTranslate(Gf.Vec3d(float(cx), float(cy), 0.0001))
        xform.SetScale(Gf.Vec3f(float(sx*0.5), float(sy*0.5), 0.002))

        # ---- Material with opacity (USD Preview Surface) ----
        mat_prim = stage.GetPrimAtPath(mat_path)
        if not mat_prim.IsValid():
            material = UsdShade.Material.Define(stage, mat_path)
            shader   = UsdShade.Shader.Define(stage, mat_path + "/Preview")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("opacity",      Sdf.ValueTypeNames.Float).Set(float(opacity))
            shader.CreateInput("specular",     Sdf.ValueTypeNames.Float).Set(0.05)
            shader.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(0.9)

            # CORRECT connection: create a shader *output* named "surface", then connect to it
            surf_out = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            material.CreateSurfaceOutput().ConnectToSource(surf_out)
        else:
            material = UsdShade.Material(mat_prim)

        # Bind material to the quad
        UsdShade.MaterialBindingAPI(prim).Bind(material)

    def _ensure_bright_lighting(self):
        stage = omni.usd.get_context().get_stage()

        # 1) Dome light – soft ambient illumination
        dome_path = "/World/Lighting/DomeLight"
        dome = stage.GetPrimAtPath(dome_path)
        if not dome.IsValid():
            UsdGeom.Xform.Define(stage, "/World/Lighting")
            dome = UsdLux.DomeLight.Define(stage, dome_path).GetPrim()
            dome.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(0.5)  # brighter ambient
            dome.CreateAttribute("inputs:exposure",  Sdf.ValueTypeNames.Float).Set(0.5)
            # optional: neutral white
            UsdLux.DomeLight(dome).CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

        # 2) Distant light – directional “sun”
        sun_path = "/World/Lighting/SunLight"
        sun = stage.GetPrimAtPath(sun_path)
        if not sun.IsValid():
            sun = UsdLux.DistantLight.Define(stage, sun_path)
            sun.CreateIntensityAttr(10.0)  # strong key light
            sun.CreateAngleAttr(0.3)          # sharper shadows
            # tilt the sun a bit
            xform = UsdGeom.XformCommonAPI(sun.GetPrim())
            xform.SetRotate(Gf.Vec3f(-45.0, 15.0, 0.0))  # pitch, yaw, roll

    # >>> Geometry + occlusion helpers
    def _point_in_rect(self, p, rect):
        x, y = float(p[0]), float(p[1])
        xmin, xmax, ymin, ymax = rect
        return (xmin <= x <= xmax) and (ymin <= y <= ymax)

    def _segments_intersect(self, p, q, a, b):
        p, q, a, b = map(lambda v: np.asarray(v, dtype=float), (p, q, a, b))
        def orient(u, v, w):
            return (v[0]-u[0])*(w[1]-u[1]) - (v[1]-u[1])*(w[0]-u[0])
        def on_seg(u, v, w):
            return (min(u[0], v[0])-1e-9 <= w[0] <= max(u[0], v[0])+1e-9) and \
                (min(u[1], v[1])-1e-9 <= w[1] <= max(u[1], v[1])+1e-9)
        o1, o2 = orient(p, q, a), orient(p, q, b)
        o3, o4 = orient(a, b, p), orient(a, b, q)
        if (o1*o2 < 0) and (o3*o4 < 0): return True
        if abs(o1) < 1e-12 and on_seg(p, q, a): return True
        if abs(o2) < 1e-12 and on_seg(p, q, b): return True
        if abs(o3) < 1e-12 and on_seg(a, b, p): return True
        if abs(o4) < 1e-12 and on_seg(a, b, q): return True
        return False

    def _line_intersects_rect(self, p, q, rect):
        xmin, xmax, ymin, ymax = rect
        bl, br = np.array([xmin, ymin]), np.array([xmax, ymin])
        tr, tl = np.array([xmax, ymax]), np.array([xmin, ymax])
        if self._point_in_rect(p, rect) or self._point_in_rect(q, rect):
            return True
        for a, b in [(bl, br), (br, tr), (tr, tl), (tl, bl)]:
            if self._segments_intersect(p, q, a, b):
                return True
        return False

    def _is_occluded(self, leader_xy, follower_xy):
        if not self.occlusion_enabled:
            return False
        if self.occlusion_mode == "leader_inside":
            return self._point_in_rect(leader_xy, self.red_area_rect)
        elif self.occlusion_mode == "los":
            return self._line_intersects_rect(follower_xy, leader_xy, self.red_area_rect)
        return False


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
    
    def _init_replicator(self):
        # 1) render product from your fixed camera
        self._render_product = rep.create.render_product(
            self._cam_prim_path, resolution=(self.img_w, self.img_h)
        )

        # 2) pytorch listener/writer
        self._rgb_listener = _RgbListener()
        self._rp_writer = rep.WriterRegistry.get("PytorchWriter")
        self._rp_writer.initialize(listener=self._rgb_listener, device="cuda")
        self._rp_writer.attach([self._render_product])

        # 3) warm up a couple renders so the first read isn't None
        for _ in range(2):
            self.world.step(render=True)

    # --- NEW: fetch the latest RGB and convert to numpy uint8
    def _get_image_obs(self):
        rgb = self._rgb_listener.get()
        # If no frame yet, advance one render step and try again
        if rgb is None:
            self.world.step(render=True)
            rgb = self._rgb_listener.get()
            if rgb is None:
                # Still nothing: return a black frame of the right shape
                if self.channel_first:
                    return np.zeros((self.img_c, self.img_h, self.img_w), dtype=np.uint8)
                else:
                    return np.zeros((self.img_h, self.img_w, self.img_c), dtype=np.uint8)

        # Remove batch dim if present
        if rgb.dim() == 4:
            rgb = rgb[0]  # (H, W, C)
        # Drop alpha if RGBA
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        # Channels-first if requested
        if self.channel_first:
            rgb = rgb.permute(2, 0, 1).contiguous()

        return rgb.to(torch.uint8).cpu().numpy()

    
    def _init_fixed_camera(
        self,
        center_xy=(0.0, 0.0),
        height=4.0,
        roll_deg=0.0,
        ortho=True,
        focal_len=24.0,           # used only in perspective
        world_width_m=8.0,        # used only in ortho (smaller => closer)
    ):
        """
        Create a top-down camera.
        - center_xy: camera XY center over the ground.
        - height:    Z height in meters.
        - roll_deg:  rotate image about viewing axis.
        - ortho:     True for orthographic map view; False for perspective.
        - focal_len: perspective lens (mm). Larger = tighter view.
        - world_width_m: orthographic width of the view in meters (smaller = closer).
        """
        stage = omni.usd.get_context().get_stage()

        cam_xform_path = "/World/TopDownCam_Xform"
        cam_prim_path  = f"{cam_xform_path}/Camera"

        # Xform that holds the camera, pitched straight down
        cam_xf = UsdGeom.Xform.Define(stage, cam_xform_path)
        xform = UsdGeom.XformCommonAPI(cam_xf)
        xform.SetTranslate(Gf.Vec3d(float(center_xy[0]), float(center_xy[1]), float(height)))
        xform.SetRotate(Gf.Vec3f(-0.0, 0.0, float(roll_deg)))  # top-down: -90° about X

        # Create the camera and remember its path
        cam = UsdGeom.Camera.Define(stage, cam_prim_path)
        self._cam_prim_path = cam_prim_path

        if ortho:
            # Orthographic: "zoom" via aperture size (world width/height), not focal length
            cam.CreateProjectionAttr(UsdGeom.Tokens.orthographic)
            aspect = self.img_h / self.img_w
            cam.CreateHorizontalApertureAttr(float(world_width_m) * 10.0)
            cam.CreateVerticalApertureAttr(float(world_width_m * aspect) * 10.0)
        else:
            # Perspective: use focal length for zoom
            cam.CreateProjectionAttr(UsdGeom.Tokens.perspective)
            cam.CreateHorizontalApertureAttr(36.0)  # 36x24 "full-frame" back
            cam.CreateVerticalApertureAttr(24.0)
            cam.CreateFocalLengthAttr(float(focal_len))  # larger => tighter/closer

        cam.CreateClippingRangeAttr(Gf.Vec2f(0.001, 1000.0))
        return cam


    # ---------- Gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0.0
        self.steps = 0
        self._pid_integral[:] = 0.0
        self._pid_prev_err[:] = 0.0

        # Place follower (origin) and leader (ahead on x)
        self.drone.set_world_poses(
            positions=np.array([[0.0, 0.0, self.z_target]]) / self.stage_units
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
        follower_pos, _ = self.drone.get_world_poses()
        leader_xy   = leader_pos[0][:2].astype(np.float32)
        follower_xy = follower_pos[0][:2].astype(np.float32)
        rel_xy = leader_xy - follower_xy
        self.prev_distance = float(np.linalg.norm(rel_xy))

        occluded = self._is_occluded(leader_xy, follower_xy)
        if not occluded:
            self._pf.init(mean_xy=rel_xy, std_pos=0.05, std_vel=0.05)
        else:
            # wide prior if you start inside the red zone
            self._pf.init(mean_xy=np.array([0.0, 0.0], dtype=np.float32), std_pos=0.5, std_vel=0.5)
        
        # Seed prev_distance from PF so reward uses estimate consistently
        mean, cov, neff = self._pf.estimate()
        self.prev_distance = float(np.linalg.norm(mean[:2]))
        # if not occluded:
        #     obs = self._get_obs(dt=0.0,pf_mean=mean, pf_cov=cov, pf_neff=neff)
        # else:
        #     obs = np.array([mean[0], mean[1], self.prev_distance, 0.0 if occluded else 1.0], dtype=np.float32)
        img = self._get_image_obs()    # << return image instead of [dx,dy,d,vis]
        return img, {}
        

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

        vx = float(a[0]) * self.drone_speed_max_xy
        vy = float(a[1]) * self.drone_speed_max_xy

        drone_pos, drone_rot = self.drone.get_world_poses()
        # update XY
        drone_pos[0][0] += vx * dt
        drone_pos[0][1] += vy * dt
        self.drone.set_world_poses(drone_pos, drone_rot)

        # Step sim
        self.world.step(render=True)

       

        # >>> Use ground truth distance for reward/termination (stable learning)
        car_pos, _   = self.car.get_world_poses()
        drone_pos, _ = self.drone.get_world_poses()
        leader_xy = car_pos[0][:2].astype(np.float32)
        follower_xy = drone_pos[0][:2].astype(np.float32)
        rel_xy = leader_xy - follower_xy
        gt_distance = float(np.linalg.norm(rel_xy))
        # === Occlusion check ===
        was_visible = getattr(self, "visible", False)  # Get previous state, default False if first call
        self.visible = not self._is_occluded(leader_xy, follower_xy)
        just_reappeared = self.visible and not was_visible  # Target just became visible again

        # === Always update the particle filter ===
        self._pf.predict(dt)
        self._pf.update(None if not self.visible else rel_xy, was_occluded=just_reappeared)
        mean, pf_cov, pf_neff = self._pf.estimate()
        rel_est = mean[:2]
        d_est = float(np.linalg.norm(rel_est))
        # === Get PF estimate ===
         # Observation / reward / termination
        # Observation (masked)
        obs = self._get_obs(dt,mean, pf_cov, pf_neff)

        reward = self._compute_reward(gt_distance)

        captured  = gt_distance < self.capture_radius
        too_far   = gt_distance > self.far_reset_dist
        timeout   = self.steps >= self.max_steps

        terminated =  too_far #or captured 
        truncated  = (not terminated) and timeout

       
        
        info = {
            "distance": gt_distance,
            "steps": self.steps,
            "captured": bool(captured),
            "goal": self.goal_xy.copy(),
            "pf_mean": mean.copy() if mean is not None else None,
            "pf_cov": pf_cov.copy() if pf_cov is not None else None,
            "pf_neff": float(pf_neff),
            "occluded": not self.visible
        }
        # Always export expert label for BC/DAgger collection
        if expert_action is not None:
            info["expert_action"] = expert_action.copy()
        
          # --- Emit particle snapshot through `info` (so ppo_test can log NPZ) ---
        info["pf_particles_emitted"] = False
        try:
            if (
                self.particle_emit
                and (self.steps % max(1, int(self.particle_stride)) == 0)
                and hasattr(self._pf, "p") and (self._pf.p is not None)
                and self._pf.p.shape[0] > 0
            ):
                P = self._pf.p  # shape (N,4): [dx, dy, vx, vy] in stage units
                # lazily choose a fixed subset of particles to visualize
                if self._particle_idx is None:
                    N = P.shape[0]
                    k = min(int(self.particle_max_show), N)
                    self._particle_idx = self._rng_particles.choice(N, size=k, replace=False)

                # follower pose (absolute) in meters
                su = float(self.stage_units)
                follower_pos, _ = self.drone.get_world_poses()
                follower_xy_m = (follower_pos[0][:2].astype(np.float32)) * su

                # particle relative positions -> absolute in meters
                cloud_rel = P[self._particle_idx, :2].astype(np.float32)   # [dx, dy] (stage units)
                cloud_abs_m = follower_xy_m[None, :] + cloud_rel * su

                # weights for the chosen subset
                w = self._pf.w[self._particle_idx].astype(np.float32) if hasattr(self._pf, "w") else None

                info["pf_particles_xy"] = cloud_abs_m         # (M,2) in meters
                if w is not None:
                    info["pf_particles_w"] = w                 # (M,)
                info["pf_particles_emitted"] = True
        except Exception as e:
            # keep sim robust even if particle emission fails
            carb.log_warn(f"particle emission skipped: {e}")

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
        follower_pos, _ = self.drone.get_world_poses()
        err = (leader_pos[0][:2] - follower_pos[0][:2]).astype(np.float32)

        # PID on position error to produce velocity (m/s)
        self._pid_integral += err * dt
        deriv = (err - self._pid_prev_err) / max(dt, 1e-6)
        self._pid_prev_err = err.copy()

        v_cmd = self.pid_kp * err + self.pid_ki * self._pid_integral + self.pid_kd * deriv

        # Normalize to action space by speed cap
        a = v_cmd / max(self.drone_speed_max_xy, 1e-6)
        a = np.clip(a, -1.0, 1.0).astype(np.float32)
        return a

    # ---------- Obs ----------
    # >>> REPLACE your _get_obs with this version
    def _get_obs(self, dt: float,pf_mean: float, pf_cov: float, pf_neff: float) -> np.ndarray:
        # inside _get_obs(self, dt)
        leader_pos, _ = self.car.get_world_poses()
        follower_pos, _ = self.drone.get_world_poses()
        leader_xy   = leader_pos[0][:2].astype(np.float32)
        follower_xy = follower_pos[0][:2].astype(np.float32)
        rel_xy = leader_xy - follower_xy
        d_true = float(np.linalg.norm(rel_xy))
        occluded = self._is_occluded(leader_xy, follower_xy)
        if not occluded:
            # visible → serve ground truth (and cache if you also support "last")
            self._last_visible_obs = np.array([rel_xy[0], rel_xy[1], d_true], dtype=np.float32)
            return np.array([rel_xy[0], rel_xy[1], d_true, 1.0], dtype=np.float32)
        else:
            # occluded → serve PF
            pf_mean = pf_mean if pf_mean is not None else np.array([0.0, 0.0], dtype=np.float32)
            rel_est = pf_mean[:2]
            d_est = float(np.linalg.norm(rel_est))
            return np.array([rel_est[0], rel_est[1], d_est, 0.0], dtype=np.float32)


        


    def render(self, mode="human"):
        pass

    def close(self):
        simulation_app.close()


def test_pf_switch(env, steps=400, sleep=0.0, tol=1e-4, show_vel=False, print_every=1):
    """
    Runs the sim, verifies switching (GT when visible, PF when occluded),
    and logs errors for plotting later.

    Returns:
        log: dict with arrays: steps, err_gt, err_pf, vis, gt_d, pf_d, obs_d
    """
    obs, info = env.reset()
    prev_leader_xy = None
    # --- particle cloud logging (for animation) ---
    PARTICLE_SNAPSHOT_STRIDE = 2     # take a snapshot every N sim steps
    PARTICLE_MAX_SHOW        = 250   # sample at most this many particles
    PARTICLE_SAMPLE_SEED     = 123   # for repeatable sampling

    su = float(env.stage_units)  # stage-units -> meters
    log = dict(
        steps=[], err_gt=[], err_pf=[], vis=[],
        gt_d=[], pf_d=[], obs_d=[],
        leader_x=[], leader_y=[],
        follower_x=[], follower_y=[],
        pf_leader_x=[], pf_leader_y=[],
        pf_neff=[], pf_cov_xx=[], pf_cov_xy=[], pf_cov_yy=[],
        pf_cov_vxvx=[], pf_cov_vyvy=[],
    )
    # particle snapshot state
    particle_idx = None
    part_frames = []   # list of arrays (M,2) absolute XY per snapshot
    part_steps  = []   # sim step index for each snapshot
    rng_sample  = np.random.default_rng(PARTICLE_SAMPLE_SEED)

    for t in range(steps):
        action = np.array([0.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        # Ground truth (absolute)
        car_pos, _   = env.car.get_world_poses()
        drone_pos, _ = env.drone.get_world_poses()
        leader_xy   = (car_pos[0][:2]   * su).astype(float)
        follower_xy = (drone_pos[0][:2] * su).astype(float)

        # GT / PF / OBS in *meters*
        gt_rel = leader_xy - follower_xy
        gt_d   = float(np.linalg.norm(gt_rel))

        mean, cov, neff = env._pf.estimate()
        pf_rel = (mean[:2] * su).astype(float)
        pf_d   = float(np.linalg.norm(pf_rel))
        pf_leader_xy =  follower_xy + pf_rel
        obs_rel = np.array([float(obs[0]) * su, float(obs[1]) * su])
        obs_d   = float(obs[2]) * su
        pf_cov4 = (cov[:4, :4] * (su*su)).astype(float)
        pf_neff = float(neff)
        visible = bool(obs[3] > 0.5)
        err_vs_gt = float(np.linalg.norm(obs_rel - gt_rel))
        err_vs_pf = float(np.linalg.norm(pf_rel - gt_rel))

        # --- particle snapshot (absolute/world XY) ---
        if (t % PARTICLE_SNAPSHOT_STRIDE) == 0 and hasattr(env._pf, "p") and env._pf.p is not None:
            P = env._pf.p  # shape (N,4) with [dx, dy, vx, vy]
            if particle_idx is None:
                N = P.shape[0]
                k = min(PARTICLE_MAX_SHOW, N)
                particle_idx = rng_sample.choice(N, size=k, replace=False)
            cloud_rel = P[particle_idx, :2].astype(float)          # [dx,dy]
            cloud_abs = follower_xy + cloud_rel                    # world XY for plotting
            part_frames.append(cloud_abs)
            part_steps.append(t)

        # Log
        log["steps"].append(t)
        log["err_gt"].append(err_vs_gt)
        log["err_pf"].append(err_vs_pf)
        log["vis"].append(1 if visible else 0)
        log["gt_d"].append(gt_d)
        log["pf_d"].append(pf_d)
        log["obs_d"].append(obs_d)
        # Log trajectories (all in meters)
        log["leader_x"].append(float(leader_xy[0]))
        log["leader_y"].append(float(leader_xy[1]))
        log["follower_x"].append(float(follower_xy[0]))
        log["follower_y"].append(float(follower_xy[1]))
        log["pf_leader_x"].append(float(pf_leader_xy[0]))
        log["pf_leader_y"].append(float(pf_leader_xy[1]))
        log["pf_neff"].append(float(pf_neff))
        log["pf_cov_xx"].append(float(pf_cov4[0, 0]))
        log["pf_cov_xy"].append(float(pf_cov4[0, 1]))
        log["pf_cov_yy"].append(float(pf_cov4[1, 1]))
        log["pf_cov_vxvx"].append(float(pf_cov4[2, 2]))
        log["pf_cov_vyvy"].append(float(pf_cov4[3, 3]))
        # Optional prints
        if (t % print_every) == 0:
            src = "GT" if visible else "PF"
            print(
                f"[{t:04d}] vis={int(visible)} src={src} "
                f"GT_REL=({gt_rel[0]:+.3f},{gt_rel[1]:+.3f}, d={gt_d:.3f}) "
                f"OBS_REL=({obs_rel[0]:+.3f},{obs_rel[1]:+.3f}, d={obs_d:.3f}) "
                f"PF_REL=({pf_rel[0]:+.3f},{pf_rel[1]:+.3f}, d={pf_d:.3f}) "
                f"err(obs,GT)={err_vs_gt:.3e} err(obs,PF)={err_vs_pf:.3e} N_eff={neff:.1f}"
            )
            if visible and err_vs_gt > tol:
                print("   WARNING: visible but obs != ground-truth beyond tol.")
            if (not visible) and err_vs_pf > tol:
                print("   WARNING: occluded but obs != PF beyond tol.")

        if show_vel:
            # quick finite-diff leader velocity
            if prev_leader_xy is not None:
                dt = env.world.get_physics_dt()
                vx, vy = (leader_xy - prev_leader_xy) / max(dt, 1e-6)
                print(f"         leader_v=({vx:+.3f},{vy:+.3f}) m/s")
            prev_leader_xy = leader_xy.copy()

        if terminated or truncated:
            print("Episode end:", "terminated" if terminated else "truncated")
            break

        if sleep > 0:
            time.sleep(sleep)

    return log, part_frames, part_steps

def save_particle_snapshots(frames, steps, out_dir=None, stem="pf_switch"):
    out_dir = Path(out_dir) if out_dir else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.stack(frames, axis=0) if len(frames) else np.zeros((0, 0, 2))
    steps_arr = np.asarray(steps, dtype=np.int32)
    npz_path = out_dir / f"{stem}_particles.npz"
    np.savez(npz_path, XY=arr, steps=steps_arr)
    print(f"[saved] {npz_path}")
    return str(npz_path)

def _default_out_dir():
    # Same folder as this file; fall back to CWD if __file__ is unavailable
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()

def save_log_csv(log, out_dir=None, stem="pf_switch"):
    out_dir = Path(out_dir) if out_dir else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}_log.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "step","visible","err_obs_gt","err_obs_pf","gt_d","pf_d","obs_d",
            "leader_x","leader_y","follower_x","follower_y","pf_leader_x","pf_leader_y",
            "pf_neff","pf_cov_xx","pf_cov_xy","pf_cov_yy","pf_cov_vxvx","pf_cov_vyvy"
        ])
        for i in range(len(log["steps"])):
            w.writerow([
            log["steps"][i], log["vis"][i], log["err_gt"][i], log["err_pf"][i],
            log["gt_d"][i], log["pf_d"][i], log["obs_d"][i],
            log["leader_x"][i], log["leader_y"][i],
            log["follower_x"][i], log["follower_y"][i],
            log["pf_leader_x"][i], log["pf_leader_y"][i],
            log["pf_neff"][i], log["pf_cov_xx"][i], log["pf_cov_xy"][i], log["pf_cov_yy"][i],
            log["pf_cov_vxvx"][i], log["pf_cov_vyvy"][i],
        ])
    print(f"[saved] {csv_path}")
    return str(csv_path)

def plot_errors(log, out_dir=None, stem="pf_switch"):
    out_dir = Path(out_dir) if out_dir else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: ||obs - GT||
    fig1 = plt.figure()
    plt.plot(log["steps"], log["err_gt"])
    plt.title("Observation error vs Ground Truth")
    plt.xlabel("Step")
    plt.ylabel("||obs - GT|| (m)")
    plt.grid(True)
    plt.tight_layout()
    png1 = out_dir / f"{stem}_err_obs_gt.png"
    fig1.savefig(png1, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: ||obs - PF||
    fig2 = plt.figure()
    plt.plot(log["steps"], log["err_pf"])
    plt.title("Observation error vs Particle Filter")
    plt.xlabel("Step")
    plt.ylabel("||obs - PF|| (m)")
    plt.grid(True)
    plt.tight_layout()
    png2 = out_dir / f"{stem}_err_obs_pf.png"
    fig2.savefig(png2, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"[saved] {png1}")
    print(f"[saved] {png2}")
    return str(png1), str(png2)

def plot_trajectories(log, red_rect, out_dir=None, stem="pf_switch"):
    """
    Plots leader & follower trajectories, PF leader estimate, and occlusion rectangle.
    Saves: <stem>_traj.png (and a visibility-highlight version <stem>_traj_vis.png)
    """
    out_dir = Path(out_dir) if out_dir else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    Lx = np.array(log["leader_x"])
    Ly = np.array(log["leader_y"])
    Fx = np.array(log["follower_x"])
    Fy = np.array(log["follower_y"])
    Px = np.array(log["pf_leader_x"])
    Py = np.array(log["pf_leader_y"])
    Vis = np.array(log["vis"], dtype=int)  # 1=visible, 0=occluded

    # --- Base figure ---
    fig, ax = plt.subplots(figsize=(7, 6))

    # Occlusion rectangle
    xmin, xmax, ymin, ymax = red_rect
    rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                     facecolor="red", alpha=0.20, edgecolor="red", linewidth=1.5)
    ax.add_patch(rect)

    # Paths
    ax.plot(Lx, Ly, label="Leader (GT)", linewidth=2)
    ax.plot(Fx, Fy, label="Follower (GT)", linewidth=2)
    ax.plot(Px, Py, "--", label="Leader (PF est)", linewidth=1.6)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Trajectories with Occlusion Area")
    ax.legend(loc="best")

    # Tight bounds with some padding
    all_x = np.concatenate([Lx, Fx, Px])
    all_y = np.concatenate([Ly, Fy, Py])
    if all_x.size > 0 and all_y.size > 0:
        pad = 0.5
        ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
        ax.set_ylim(all_y.min() - pad, all_y.max() + pad)

    png1 = out_dir / f"{stem}_traj.png"
    fig.savefig(png1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Optional: visibility-highlighted leader path ---
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    rect2 = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                      facecolor="red", alpha=0.2, edgecolor="red", linewidth=1.5)
    ax2.add_patch(rect2)

    # Segment the leader path by visibility
    # Plot visible steps as one color and occluded steps as another
    vis_idx = np.where(Vis == 1)[0]
    occ_idx = np.where(Vis == 0)[0]
    ax2.plot(Fx, Fy, label="Follower (GT)", linewidth=1.8)
    if vis_idx.size:
        ax2.scatter(Lx[vis_idx], Ly[vis_idx], s=8, label="Leader (visible)")
    if occ_idx.size:
        ax2.scatter(Lx[occ_idx], Ly[occ_idx], s=8, label="Leader (occluded)")

    ax2.set_aspect("equal", adjustable="box")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title("Leader Visibility Along Trajectory")
    ax2.legend(loc="best")

    if all_x.size > 0 and all_y.size > 0:
        pad = 0.5
        ax2.set_xlim(all_x.min() - pad, all_x.max() + pad)
        ax2.set_ylim(all_y.min() - pad, all_y.max() + pad)

    png2 = out_dir / f"{stem}_traj_vis.png"
    fig2.savefig(png2, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"[saved] {png1}")
    print(f"[saved] {png2}")
    return str(png1), str(png2)





# --------------- run it ---------------
if __name__ == "__main__":
    env = LeaderFollowerEnv()
    # quick occlusion toggling on the leader path
    env.red_area_rect = (1.0, 2.0, -1.0, 2.0)
    env.occlusion_mode = "leader_inside"
    env._ensure_colored_area(rect=env.red_area_rect, color=(1.0, 0.0, 0.0), opacity=0.4)
    
    try:
        log, part_frames, part_steps = test_pf_switch(env, steps=1200, sleep=0.0, tol=1e-4, show_vel=False, print_every=10)
        img = env._get_image_obs()
        import numpy as np, imageio
        img_hwc = np.moveaxis(img, 0, 2) if img.ndim == 3 and img.shape[0] in (1,3) else img
        out_path = Path(__file__).resolve().parent / "camera_debug.png"
        imageio.imwrite(str(out_path), img_hwc)
        print("[cam] saved to:", out_path)       
    finally:
        # Save CSV + PNGs right next to ppo_env.py (or CWD)
        save_log_csv(log, stem="pf_switch")
        plot_errors(log, stem="pf_switch")
        plot_trajectories(log, env.red_area_rect, stem="pf_switch")
        parts_npz = save_particle_snapshots(part_frames, part_steps, stem="pf_switch")
        env.close()
        

