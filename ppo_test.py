from isaacsim import SimulationApp 
from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np
import time
from pathlib import Path
import csv
import matplotlib.pyplot as plt

from ppo_env import LeaderFollowerEnv

# Register the environment with Gym
gym.register(
    id='LeaderFollower-v0',
    entry_point='ppo_env:LeaderFollowerEnv',
    max_episode_steps=1200,
)

def _default_out_dir():
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()

# ---------- Logging helpers ----------
def make_log():
    return dict(
        steps=[], err_gt=[], err_pf=[], err_gt_log=[], err_pf_log=[],
        vis=[], gt_d=[], pf_d=[], obs_d=[],
        leader_x=[], leader_y=[],
        follower_x=[], follower_y=[],
        pf_leader_x=[], pf_leader_y=[],
        pf_neff=[], pf_cov_xx=[], pf_cov_xy=[], pf_cov_yy=[],
        pf_cov_vxvx=[], pf_cov_vyvy=[],
        particles_frames=[],   # list of np.ndarray (M,2) absolute XY
        particles_steps=[],    # list of int step indices
        particles_weights=[],  # optional list of (M,) arrays
    )

def log_step_metrics(env, obs, info, t, log, tol=1e-4, show_vel=False, print_every=1, _state_cache=dict()):
    """
    Compute GT / PF / OBS metrics for the current env state and append to `log`.

    - obs: current observation from env.step(...)
    - t:   current step index (int)
    """
    su = float(env.stage_units)  # stage-units -> meters

    # Ground truth poses
    car_pos, _   = env.car.get_world_poses()
    drone_pos, _ = env.drone.get_world_poses()
    leader_xy   = (car_pos[0][:2]   * su).astype(float)
    follower_xy = (drone_pos[0][:2] * su).astype(float)

    gt_rel = leader_xy - follower_xy
    gt_d   = float(np.linalg.norm(gt_rel))

     # --- PF internals from info ---
    pf_mean = info.get("pf_mean", None)
    pf_cov  = info.get("pf_cov",  None)
    pf_neff = float(info.get("pf_neff", np.nan))

    if pf_mean is not None:
        pf_rel = (np.asarray(pf_mean[:2]) * su).astype(float)  # relative [dx,dy] in meters
        pf_d   = float(np.linalg.norm(pf_rel))
        pf_leader_xy = follower_xy + pf_rel                    # absolute leader estimate
    else:
        pf_rel = np.array([np.nan, np.nan], dtype=float)
        pf_d   = np.nan
        pf_leader_xy = np.array([np.nan, np.nan], dtype=float)

    # normalize covariance to a 4x4 view for logging convenience
    if pf_cov is not None:
        pf_cov = np.asarray(pf_cov)
        if pf_cov.shape == (4, 4):
            pf_cov4 = pf_cov
        elif pf_cov.shape == (2, 2):
            pf_cov4 = np.full((4, 4), np.nan, dtype=float)
            pf_cov4[:2, :2] = pf_cov
        else:
            pf_cov4 = np.full((4, 4), np.nan, dtype=float)
    else:
        pf_cov4 = np.full((4, 4), np.nan, dtype=float)

    

    # Observation (already relative); obs layout assumed [rel_x, rel_y, d, vis]
    obs_rel = np.array([float(obs[0]) * su, float(obs[1]) * su])
    obs_d   = float(obs[2]) * su
    visible = bool(obs[3] > 0.5)

    # Errors
    err_vs_gt = float(np.linalg.norm(obs_rel - gt_rel))
    err_vs_pf = float(np.linalg.norm(pf_rel - gt_rel))
    log_err_gt = float(np.log1p(err_vs_gt))
    log_err_pf = float(np.log1p(err_vs_pf)) if np.isfinite(err_vs_pf) else float(np.log1p(0.0))

    # Append raw
    # Log
    log["steps"].append(t)
    log["err_gt"].append(err_vs_gt)
    log["err_pf"].append(err_vs_pf)
    log["err_gt_log"].append(log_err_gt)
    log["err_pf_log"].append(log_err_pf)
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

    # PF quality / uncertainty
    log["pf_neff"].append(float(pf_neff))
    log["pf_cov_xx"].append(float(pf_cov4[0, 0]))
    log["pf_cov_xy"].append(float(pf_cov4[0, 1]))
    log["pf_cov_yy"].append(float(pf_cov4[1, 1]))
    log["pf_cov_vxvx"].append(float(pf_cov4[2, 2]))
    log["pf_cov_vyvy"].append(float(pf_cov4[3, 3]))

    # --- NEW: stash particle snapshot if present in info ---
    if info.get("pf_particles_emitted", False):
        Pxy = info.get("pf_particles_xy", None)
        if isinstance(Pxy, np.ndarray) and Pxy.ndim == 2 and Pxy.shape[1] == 2 and Pxy.size > 0:
            log["particles_frames"].append(Pxy.copy())
            log["particles_steps"].append(int(t))
            Pw = info.get("pf_particles_w", None)
            if isinstance(Pw, np.ndarray) and Pw.ndim == 1 and Pw.size == Pxy.shape[0]:
                log["particles_weights"].append(Pw.copy())
            else:
                log["particles_weights"].append(None)

    # Optional prints
    if (t % print_every) == 0:
        src = "GT" if visible else "PF"
        print(
            f"[{t:04d}] vis={int(visible)} src={src} "
            f"GT_REL=({gt_rel[0]:+.3f},{gt_rel[1]:+.3f}, d={gt_d:.3f}) "
            f"OBS_REL=({obs_rel[0]:+.3f},{obs_rel[1]:+.3f}, d={obs_d:.3f}) "
            f"PF_REL=({pf_rel[0]:+.3f},{pf_rel[1]:+.3f}, d={pf_d:.3f}) "
            f"err(obs,GT)={err_vs_gt:.3e} err(obs,PF)={err_vs_pf:.3e} N_eff={pf_neff:.1f}"
        )
        if visible and err_vs_gt > tol:
            print("   WARNING: visible but obs != ground-truth beyond tol.")
        if (not visible) and err_vs_pf > tol:
            print("   WARNING: occluded but obs != PF beyond tol.")

    # Optional leader velocity (finite diff)
    if show_vel:
        prev = _state_cache.get("prev_leader_xy")
        if prev is not None:
            dt = env.world.get_physics_dt()
            vx, vy = (leader_xy - prev) / max(dt, 1e-6)
            print(f"         leader_v=({vx:+.3f},{vy:+.3f}) m/s")
        _state_cache["prev_leader_xy"] = leader_xy.copy()

def save_log_csv(log, out_dir=None, stem="pf_switch"):
    out_dir = Path(out_dir) if out_dir else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}_test_log_.csv"
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

    # 1) ||obs - GT||
    fig1 = plt.figure()
    plt.plot(log["steps"], log["err_gt"])
    plt.title("Observation error vs Ground Truth")
    plt.xlabel("Step")
    plt.ylabel("||obs - GT|| (m)")
    plt.grid(True); plt.tight_layout()
    png1 = out_dir / f"{stem}_err_obs_gt.png"
    fig1.savefig(png1, dpi=150, bbox_inches="tight"); plt.close(fig1)

    # 2) ||obs - PF||
    fig2 = plt.figure()
    plt.plot(log["steps"], log["err_pf"])
    plt.title("Observation error vs Particle Filter")
    plt.xlabel("Step")
    plt.ylabel("||obs - PF|| (m)")
    plt.grid(True); plt.tight_layout()
    png2 = out_dir / f"{stem}_err_obs_pf.png"
    fig2.savefig(png2, dpi=150, bbox_inches="tight"); plt.close(fig2)

    # 3) (optional) log-scaled versions
    fig3 = plt.figure()
    plt.plot(log["steps"], log["err_gt_log"], label="log1p(||obs-GT||)")
    plt.plot(log["steps"], log["err_pf_log"], label="log1p(||obs-PF||)")
    plt.title("Log-Scaled Errors (log1p)")
    plt.xlabel("Step"); plt.ylabel("log(1+error)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    png3 = out_dir / f"{stem}_errors_logscale.png"
    fig3.savefig(png3, dpi=150, bbox_inches="tight"); plt.close(fig3)

    print(f"[saved] {png1}")
    print(f"[saved] {png2}")
    print(f"[saved] {png3}")
    return str(png1), str(png2), str(png3)

def save_particle_npz(log, out_dir=None, stem="pf_switch"):
    out_dir = Path(out_dir) if out_dir else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(log["particles_frames"]) == 0:
        print("[particles] nothing to save"); return None
    XY = np.stack(log["particles_frames"], axis=0)  # (T_snap, M, 2)
    steps = np.asarray(log["particles_steps"], dtype=np.int32)
    # weights are optional/mixed; pack only if all exist
    if all(isinstance(w, np.ndarray) for w in log["particles_weights"]):
        W = np.stack(log["particles_weights"], axis=0)
        path = out_dir / f"{stem}_test_particles.npz"
        np.savez_compressed(path, XY=XY, steps=steps, W=W)
    else:
        path = out_dir / f"{stem}_test_particles.npz"
        np.savez_compressed(path, XY=XY, steps=steps)
    print(f"[saved] {path}")
    return str(path)

# ---------- PPO test (drives the sim loop) ----------
def test_ppo_model():
    """Run one PPO episode, logging PF/GT/OBS metrics each step."""
    env = gym.make('LeaderFollower-v0')
    env.red_area_rect = (-1.0, 2.5, -2.0, 3.5)   # xmin, xmax, ymin, ymax
    env.occlusion_mode = "leader_inside"

    # Windows path: use raw string or escape backslashes
    model = PPO.load(r"C:\isaacsim\standalone_examples\custom_env\leader_follower_ppo_model_pf")

    obs, info = env.reset()
    total_reward = 0.0
    done = False
    t = 0

    log = make_log()

    while not done:
        # PPO action
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        # Per-step logging (no internal loop)
        log_step_metrics(env, obs, info, t, log, tol=1e-4, show_vel=False, print_every=10)

        # Distance readout (from obs channel 2)
        distance_m = float(obs[2]) * float(env.stage_units)
        if (t % 10) == 0:
            print(f"[{t:04d}] Distance: {distance_m:.2f} m, Reward: {reward:.2f}, Total: {total_reward:.2f}")

        if terminated or truncated:
            done = True
            print(f"Episode finished at step {t} with total reward: {total_reward:.2f}")
            print(f"Final distance: {distance_m:.2f} m")
        # Now you can access PF info directly
        if "pf_mean" in info and info["pf_mean"] is not None:
            pf_pos = info["pf_mean"][:2]  # First two elements are x,y
            pf_vel = info["pf_mean"][2:4]  # Elements 2,3 are vx,vy
            pf_neff = info["pf_neff"]
            
            # Example usage
            print(f"PF position: {pf_pos}, velocity: {pf_vel}, N_eff: {pf_neff:.1f}")

        t += 1

    # Persist results
    save_log_csv(log, stem="pf_switch")
    plot_errors(log, stem="pf_switch")
    parts_npz = save_particle_npz(log, stem="pf_switch")
    env.close()

if __name__ == "__main__":
    test_ppo_model()
