from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Rectangle

def _occluded_spans(visible_array):
    """Return list of (start_step, end_step) intervals where visible==0 (occluded)."""
    v = np.asarray(visible_array, dtype=int)
    if v.size == 0:
        return []
    spans = []
    occl = (v == 0)
    # Find changes
    diff = np.diff(occl.astype(int), prepend=occl[0])
    # indices where occlusion starts/ends
    starts = np.where((diff == 1))[0]
    ends   = np.where((diff == -1))[0]
    # If starts after time 0 and we began occluded, prepend 0
    if occl[0] and (len(starts) == 0 or starts[0] != 0):
        starts = np.r_[0, starts]
    # If we end occluded, append last index+1
    if occl[-1]:
        ends = np.r_[ends, len(occl)]
    # Pair up
    for s, e in zip(starts, ends):
        if e > s:
            spans.append((s, e))
    return spans

def _shade_occlusions(ax, steps, visible, label_once=True):
    """Shade occluded regions (visible==0) behind the plot."""
    spans = _occluded_spans(visible)
    first = True
    for s_idx, e_idx in spans:
        # Map indices to step values
        x0 = steps[s_idx]
        # extend one step to the right for cleaner coverage
        x1 = steps[min(e_idx-1, len(steps)-1)] + 1
        ax.axvspan(x0, x1, color="0.90", alpha=0.6,
                   label=("occluded" if (label_once and first) else None),
                   zorder=0)
        first = False

def plot_pf_csv(csv_path, stem=None):
    """
    Reads pf_switch_log.csv with columns:
      step, visible, err_obs_gt, err_obs_pf, gt_d, pf_d, obs_d
    Creates two figures (errors & distances) shaded where visible==0,
    and saves them next to the CSV.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    required = ["step","visible","err_obs_gt","err_obs_pf","gt_d","pf_d","obs_d"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    steps   = df["step"].to_numpy()
    visible = df["visible"].to_numpy()
    out_dir = csv_path.parent
    base    = stem if stem else csv_path.stem

    # --- Figure 1: errors ---
    fig1, ax1 = plt.subplots()
    ax1.plot(steps, df["err_obs_gt"].to_numpy(), label="||obs - GT||")
    ax1.plot(steps, df["err_obs_pf"].to_numpy(), label="||obs - PF||")
    _shade_occlusions(ax1, steps, visible)
    ax1.set_title("Observation Errors")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Error (m)")
    ax1.grid(True)
    ax1.legend()
    fig1.tight_layout()
    png1 = out_dir / f"{base}_errors.png"
    fig1.savefig(png1, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # --- Figure 2: distances ---
    fig2, ax2 = plt.subplots()
    ax2.plot(steps, df["gt_d"].to_numpy(),  label="GT distance")
    ax2.plot(steps, df["pf_d"].to_numpy(),  label="PF distance")
    ax2.plot(steps, df["obs_d"].to_numpy(), label="Obs distance")
    _shade_occlusions(ax2, steps, visible)
    ax2.set_title("Distances (GT vs PF vs Obs)")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Distance (m)")
    ax2.grid(True)
    ax2.legend()
    fig2.tight_layout()
    png2 = out_dir / f"{base}_distances.png"
    fig2.savefig(png2, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"[saved] {png1}")
    print(f"[saved] {png2}")
    return str(png1), str(png2)

def animate_traj_from_csv(
    csv_path, red_rect, out_dir=None, stem=None,
    fps=30, step_stride=1, fmt="mp4",
    particles_npz=None,          # <<< NEW: NPZ path with particle snapshots
    particle_size=10,            # points^2
    particle_alpha=0.35          # transparency
    ):
    """
    Build an animated trajectory plot from a CSV with columns:
      step, visible, ..., leader_x, leader_y, follower_x, follower_y, pf_leader_x, pf_leader_y
    red_rect: (xmin, xmax, ymin, ymax)
    fmt: "mp4" (preferred) or "gif"
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    need = ["step","visible","leader_x","leader_y","follower_x","follower_y","pf_leader_x","pf_leader_y"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"CSV missing columns: {miss}. Re-run sim with position logging enabled.")

    steps = df["step"].to_numpy()[::step_stride]
    vis   = df["visible"].to_numpy(dtype=int)[::step_stride]
    Lx, Ly = df["leader_x"].to_numpy()[::step_stride], df["leader_y"].to_numpy()[::step_stride]
    Fx, Fy = df["follower_x"].to_numpy()[::step_stride], df["follower_y"].to_numpy()[::step_stride]
    Px, Py = df["pf_leader_x"].to_numpy()[::step_stride], df["pf_leader_y"].to_numpy()[::step_stride]

     # --- Load particle snapshots (optional) ---
    PXY, Psteps = None, None
    if particles_npz:
        data = np.load(particles_npz)
        PXY = data["XY"] if "XY" in data else None           # (T_snap, M, 2)
        Psteps = data["steps"] if "steps" in data else None   # (T_snap,)
        if (PXY is not None) and (PXY.size == 0):
            PXY, Psteps = None, None

    out_dir = Path(out_dir) if out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base = stem if stem else csv_path.stem
    outfile = out_dir / f"{base}_traj.{fmt}"

    fig, ax = plt.subplots(figsize=(7, 6))
    xmin, xmax, ymin, ymax = red_rect
    rect = Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, facecolor="red", alpha=0.20, edgecolor="red")
    ax.add_patch(rect)

    # lines & head markers
    lnL, = ax.plot([], [], lw=2, label="Leader (GT)")
    lnF, = ax.plot([], [], lw=2, label="Follower (GT)")
    lnP, = ax.plot([], [], lw=1.6, ls="--", label="Leader (PF est)")
    ptL, = ax.plot([], [], marker="o", ms=6, linestyle="None")  # leader head
    ptF, = ax.plot([], [], marker="s", ms=6, linestyle="None")  # follower head
    # <<< NEW: particle scatter
    scat = ax.scatter([], [], s=particle_size, alpha=particle_alpha, label="Particles")
    txt  = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")



    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Trajectories ")
    ax.legend(loc="best")

    # reasonable bounds with padding
    all_x = np.concatenate([Lx, Fx, Px])
    all_y = np.concatenate([Ly, Fy, Py])
    if all_x.size and all_y.size:
        pad = 0.5
        ax.set_xlim(all_x.min()-pad, all_x.max()+pad)
        ax.set_ylim(all_y.min()-pad, all_y.max()+pad)

    def init():
            lnL.set_data([], []); lnF.set_data([], []); lnP.set_data([], [])
            ptL.set_data([], []); ptF.set_data([], [])
            scat.set_offsets(np.empty((0, 2)))  # empty particle cloud
            txt.set_text("")
            return lnL, lnF, lnP, ptL, ptF, scat, txt

    def update(i):
        # Safeguard against index errors
        if i >= len(steps) or i < 0 or len(Lx) <= i:
            return lnL, lnF, lnP, ptL, ptF, txt
            
        lnL.set_data(Lx[:i+1], Ly[:i+1])
        lnF.set_data(Fx[:i+1], Fy[:i+1])
        lnP.set_data(Px[:i+1], Py[:i+1])
        
        # For single-point plotting, wrap in list if not array-like
        if isinstance(Lx[i], (int, float)):
            ptL.set_data([Lx[i]], [Ly[i]])
        else:
            ptL.set_data(Lx[i], Ly[i])
            
        if isinstance(Fx[i], (int, float)):
            ptF.set_data([Fx[i]], [Fy[i]])
        else:
            ptF.set_data(Fx[i], Fy[i])

        # color leader head by visibility
        if vis[i] == 1:
            ptL.set_markerfacecolor("C2")  # visible
        else:
            ptL.set_markerfacecolor("0.5") # occluded
        ptL.set_markeredgecolor("k")
        ptF.set_markerfacecolor("C0")
        ptF.set_markeredgecolor("k")
        # <<< NEW: pick the particle snapshot whose step <= current step
        if (PXY is not None) and (Psteps is not None):
            idx = np.searchsorted(Psteps, steps[i], side="right") - 1
            if idx >= 0:
                scat.set_offsets(PXY[idx])     # (M,2)
            else:
                scat.set_offsets(np.empty((0, 2)))

        txt.set_text(f"step: {int(steps[i])}   visible: {int(vis[i])}")
        return lnL, lnF, lnP, ptL, ptF, txt

    ani = FuncAnimation(fig, update, frames=len(steps), init_func=init,
                        interval=1000.0/max(1, fps), blit=True)

    try:
        if fmt.lower() == "mp4":
            writer = FFMpegWriter(fps=fps, bitrate=2400)
        else:
            writer = PillowWriter(fps=fps)
        ani.save(outfile, writer=writer, dpi=150)
        print(f"[saved] {outfile}")
    finally:
        plt.close(fig)

    return str(outfile)

def plot_pf_cov(csv_path, stem=None):
    csv_path = Path(csv_path); df = pd.read_csv(csv_path)
    need = ["step","pf_cov_xx","pf_cov_yy","pf_neff"]
    miss = [c for c in need if c not in df.columns]
    if miss: raise ValueError(f"CSV missing columns: {miss}")

    steps = df["step"].to_numpy()
    sx = np.sqrt(np.maximum(0.0, df["pf_cov_xx"].to_numpy()))
    sy = np.sqrt(np.maximum(0.0, df["pf_cov_yy"].to_numpy()))
    neff = df["pf_neff"].to_numpy()

    out_dir = csv_path.parent; base = stem if stem else csv_path.stem
    fig, ax1 = plt.subplots()
    ax1.plot(steps, sx, label="σx (m)")
    ax1.plot(steps, sy, label="σy (m)")
    ax1.set_xlabel("Step"); ax1.set_ylabel("Position std (m)")
    ax1.grid(True); ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(steps, neff, linestyle="--", label="N_eff")
    ax2.set_ylabel("Effective particles")
    # combine legends
    h1,l1 = ax1.get_legend_handles_labels()
    h2,l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="upper right")

    fig.tight_layout()
    out = out_dir / f"{base}_pf_cov.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out}")
    return str(out)


if __name__ == "__main__":
    csv = r"C:\isaacsim\standalone_examples\custom_env\pf_switch_test_log_.csv"
    plot_pf_csv(r"C:\isaacsim\standalone_examples\custom_env\pf_switch_test_log_.csv")
    animate_traj_from_csv(
        csv,
        red_rect=(-1.0, 2.5, -2.0, 3.5),
        fps=30,
        fmt="gif",
        step_stride=1,
        particles_npz=r"C:\isaacsim\standalone_examples\custom_env\pf_switch_test_particles.npz",   # << include particles
        particle_size=8,
        particle_alpha=0.35
    )
    plot_pf_cov(csv)
