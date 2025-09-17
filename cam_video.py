# record_topdown_standalone.py
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from pathlib import Path
import carb
import torch
import omni.usd
import omni.replicator.core as rep
from pxr import UsdGeom, UsdLux, Gf, Sdf

# ---------- tiny frame listener (robust to payload variations) ----------
class _RgbListener:
    def __init__(self):
        self.rgb = None  # torch tensor (H,W,C) or (1,H,W,C)
    def write_data(self, data):
        self._consume(data)
    def __call__(self, data):
        self._consume(data)
    def _consume(self, data):
        t = None
        if isinstance(data, dict):
            t = data.get("pytorch_rgb", data.get("rgb"))
            if t is None:
                for v in data.values():
                    if isinstance(v, dict):
                        t = v.get("pytorch_rgb", v.get("rgb"))
                        if t is not None:
                            break
        if t is not None:
            if hasattr(t, "dim") and t.dim() == 4 and t.shape[0] == 1:
                t = t[0]
            self.rgb = t

# ---------- helpers ----------
def make_topdown_camera(center_xy=(0.0, 0.0), height=5.0, roll_deg=0.0, res=(640, 480)):
    """
    Perspective top-down camera (like your UI 'Top' view, but with perspective).
    """
    stage = omni.usd.get_context().get_stage()

    cam_xform = "/World/TopDownCam_Xform"
    cam_prim  = f"{cam_xform}/Camera"

    UsdGeom.Xform.Define(stage, cam_xform)
    xform = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(cam_xform))
    xform.SetTranslate(Gf.Vec3d(float(center_xy[0]), float(center_xy[1]), float(height)))
    # Look straight down: -90° pitch, optional roll
    xform.SetRotate(Gf.Vec3f(0.0, 0.0, float(roll_deg)))

    cam = UsdGeom.Camera.Define(stage, cam_prim)
    cam.CreateProjectionAttr(UsdGeom.Tokens.perspective)
    cam.CreateFocalLengthAttr(18.0)          # wider lens to see more
    cam.CreateHorizontalApertureAttr(36.0)
    cam.CreateVerticalApertureAttr(24.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))

    return cam_prim, tuple(int(x) for x in res)

def ensure_bright_lighting():
    stage = omni.usd.get_context().get_stage()
    dome_path = "/World/Lighting/Dome"
    sun_path  = "/World/Lighting/Sun"
    if not stage.GetPrimAtPath("/World/Lighting").IsValid():
        UsdGeom.Xform.Define(stage, "/World/Lighting")
    if not stage.GetPrimAtPath(dome_path).IsValid():
        dome = UsdLux.DomeLight.Define(stage, dome_path)
        dome.CreateIntensityAttr(0.6)
        UsdLux.DomeLight(stage.GetPrimAtPath(dome_path)).CreateColorAttr(Gf.Vec3f(1,1,1))
    if not stage.GetPrimAtPath(sun_path).IsValid():
        sun = UsdLux.DistantLight.Define(stage, sun_path)
        sun.CreateIntensityAttr(10.0)
        sun.CreateAngleAttr(0.3)
        UsdGeom.XformCommonAPI(sun.GetPrim()).SetRotate(Gf.Vec3f(-45.0, 15.0, 0.0))

def chw_to_hwc_u8(t):
    # Input torch tensor (H,W,3 or H,W,4). Return numpy HWC uint8 (drops alpha).
    if t.shape[-1] == 4:
        t = t[..., :3]
    arr = t.to(torch.uint8).cpu().numpy()
    return arr

# ---------- main ----------
if __name__ == "__main__":
    from isaacsim.core.api import World

    # Scene + lighting (add your own assets if you want)
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    ensure_bright_lighting()

    # Camera
    CAM_CENTER = (0.0, 0.0)
    CAM_HEIGHT = 14.0
    cam_prim_path, RES = make_topdown_camera(center_xy=CAM_CENTER, height=CAM_HEIGHT,
                                             roll_deg=0.0, res=(1024, 768))

    # Replicator render product + listener
    render_product = rep.create.render_product(cam_prim_path, resolution=RES)
    listener = _RgbListener()
    writer = rep.WriterRegistry.get("PytorchWriter")
    writer.initialize(listener=listener, device="cuda")  # "cuda" or "cpu"
    writer.attach([render_product])

    # Warmup a couple frames so listener has data
    for _ in range(2):
        world.step(render=True)

    # Output setup
    out_dir = Path(__file__).resolve().parent / "captures"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / "topdown.mp4"
    seq_dir  = out_dir / "topdown_frames"

    # FPS from physics dt (fallback 30)
    try:
        dt = world.get_physics_dt()
        FPS = int(round(1.0 / dt)) if dt and dt > 0 else 30
    except Exception:
        FPS = 30

    # Prefer MP4 via imageio+ffmpeg; fall back to PNG sequence
    use_video = True
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg  # ensure backend present
        vw = imageio.get_writer(
            mp4_path.as_posix(),
            format="FFMPEG",
            fps=FPS,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=None,
        )
    except Exception as e:
        carb.log_warn(f"[record] MP4 backend unavailable → saving PNG sequence ({e})")
        use_video = False
        seq_dir.mkdir(parents=True, exist_ok=True)

    # Record loop
    STEPS = 1200
    frames = 0
    try:
        world.reset()
        for i in range(STEPS):
            world.step(render=True)

            rgb = listener.rgb
            if rgb is None:
                continue  # nothing yet

            frame = chw_to_hwc_u8(rgb)  # HWC uint8
            if use_video:
                vw.append_data(frame)
            else:
                # Save numbered PNGs
                (seq_dir / f"{i:05d}.png").write_bytes(
                    frame.tobytes()
                )  # super fast but raw; comment this line and use imageio below if you prefer PNG encoding

                # If you prefer real PNG files, uncomment:
                # import imageio.v2 as imageio
                # imageio.imwrite(seq_dir / f"{i:05d}.png", frame)

            frames += 1
    finally:
        if use_video:
            vw.close()
            print(f"[video] {frames} frames → {mp4_path}")
        else:
            print(f"[frames] {frames} frames → {seq_dir}")
        simulation_app.close()
