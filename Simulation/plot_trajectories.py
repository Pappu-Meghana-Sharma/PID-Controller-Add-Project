import numpy as np
import matplotlib.pyplot as plt
import os  # Added to fix the 'os' not defined error
from traj import traj_hover,traj_circle,traj_figure8
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

# ── colours ──────────────────────────────────────────────────────
COLORS = {"PID": "#ff6b4a", "MPC": "#4c9fff", "PID+MPC": "#3ddba6"}
REF_COLOR = "#ffffff"
 
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor":   "#13161d",
    "axes.edgecolor":   "#2a2f40",
    "axes.labelcolor":  "#8b93b0",
    "axes.titlecolor":  "#c8cfe8",
    "axes.grid":        True,
    "grid.color":       "#1e2333",
    "grid.linewidth":   0.6,
    "xtick.color":      "#4a5070",
    "ytick.color":      "#4a5070",
    "text.color":       "#c8cfe8",
    "legend.facecolor": "#13161d",
    "legend.edgecolor": "#2a2f40",
    "lines.linewidth":  1.8,
    "font.family":      "monospace",
    "font.size":        9,
})
 
# ── trajectory generators (mirrors main script) ───────────────────
CTRL_DT  = 1 / 48
DURATION = 20.0
N_STEPS  = int(DURATION / CTRL_DT)
 
TRAJECTORIES = {
    "Hover":    traj_hover,
    "Circle":   traj_circle,
    "Figure-8": traj_figure8,
}
 
def _ref_path(traj_fn):
    return np.array([traj_fn(i * CTRL_DT) for i in range(N_STEPS)])
 
def load_and_plot_real_data(traj_name, condition_name):
    runs = []
    for ctrl in ["PID", "MPC", "PID+MPC"]:
        # Standardizing naming convention to match saved files
        safe = f"{ctrl}_{traj_name}_{condition_name}".replace(" ","_").replace("+","plus")
        af = f"traj_{safe}_actual.npy"
        tf = f"traj_{safe}_target.npy"
        
        if os.path.exists(af) and os.path.exists(tf):
            actual = np.load(af)
            ref = np.load(tf)
            runs.append((ctrl, ref, actual, traj_name, condition_name))
    
    if runs:
        plot_runs(runs)
    else:
        print(f"No real data files found for {traj_name} {condition_name}")

# ── main plot function ────────────────────────────────────────────
def plot_runs(runs, title=None):
    if not runs:
        print("No runs to plot.")
        return
 
    n = len(runs)
    traj_name = runs[0][3]
    cond_name = runs[0][4]
    sup = title or f"{traj_name}  |  {cond_name}"
 
    fig = plt.figure(figsize=(5 * n + 2, 14))
    fig.suptitle(sup, fontsize=13, fontweight="bold", y=0.98)
 
    gs = fig.add_gridspec(4, n,
                          height_ratios=[2.2, 1, 1, 1],
                          hspace=0.45, wspace=0.35)
 
    for col, (ctrl, ref, actual, _, _) in enumerate(runs):
        ref    = np.asarray(ref)
        actual = np.asarray(actual)
        color  = COLORS.get(ctrl, "#aaaaaa")
        errors = np.linalg.norm(actual - ref, axis=1)
        
        # Recalculate time based on loaded data length
        times = np.arange(len(actual)) * CTRL_DT
 
        # ── row 0: 3D ──
        ax3d = fig.add_subplot(gs[0, col], projection="3d")
        ax3d.set_facecolor("#13161d")
        ax3d.plot(ref[:,0], ref[:,1], ref[:,2], color=REF_COLOR, lw=1.2, ls="--", alpha=0.5, label="Reference")
        ax3d.plot(actual[:,0], actual[:,1], actual[:,2], color=color, lw=1.8, alpha=0.9, label="Actual")
        ax3d.plot(actual[:,0], actual[:,1], np.zeros(len(actual)), color=color, lw=0.6, alpha=0.18, ls=":")
        ax3d.scatter(*actual[0],  color="white", s=20, zorder=5)
        ax3d.scatter(*actual[-1], color=color,   s=35, zorder=5, marker="*")
        ax3d.set_xlabel("X (m)", fontsize=7, labelpad=1)
        ax3d.set_ylabel("Y (m)", fontsize=7, labelpad=1)
        ax3d.set_zlabel("Z (m)", fontsize=7, labelpad=1)
        ax3d.set_title(ctrl, fontsize=11, fontweight="bold", pad=6)
        ax3d.legend(fontsize=7, loc="upper left")
        ax3d.view_init(elev=28, azim=-55)
        for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#1e2333")
 
        # ── row 1: Top view XY ──
        ax_top = fig.add_subplot(gs[1, col])
        ax_top.plot(ref[:,0], ref[:,1], color=REF_COLOR, lw=1.0, ls="--", alpha=0.5)
        ax_top.plot(actual[:,0], actual[:,1], color=color, lw=1.6, alpha=0.9)
        ax_top.set_xlabel("X (m)", fontsize=7)
        ax_top.set_ylabel("Y (m)", fontsize=7)
        ax_top.set_title("Top  (XY)", fontsize=8)
        ax_top.set_aspect("equal", adjustable="box")
 
        # ── row 2: Front view XZ ──
        ax_front = fig.add_subplot(gs[2, col])
        ax_front.plot(ref[:,0], ref[:,2], color=REF_COLOR, lw=1.0, ls="--", alpha=0.5)
        ax_front.plot(actual[:,0], actual[:,2], color=color, lw=1.6, alpha=0.9)
        ax_front.set_xlabel("X (m)", fontsize=7)
        ax_front.set_ylabel("Z (m)", fontsize=7)
        ax_front.set_title("Front (XZ)", fontsize=8)
        ax_front.set_aspect("equal", adjustable="box")
 
        # ── row 3: Error over time ──
        ax_err = fig.add_subplot(gs[3, col])
        ax_err.plot(times, errors, color=color, lw=1.6)
        ax_err.fill_between(times, errors, alpha=0.15, color=color)
        ax_err.set_xlabel("Time (s)", fontsize=7)
        ax_err.set_ylabel("Error (m)", fontsize=7)
        ax_err.set_title("Position error", fontsize=8)
 
        # metrics text box
        n_e    = len(errors)
        rmse   = np.sqrt(np.mean(errors**2))
        steady = np.mean(errors[n_e//2:])
        ax_err.text(0.97, 0.95, f"RMSE  {rmse:.4f}\nSteady {steady:.4f}",
                    transform=ax_err.transAxes, fontsize=7, va="top", ha="right",
                    color="#c8cfe8", bbox=dict(boxstyle="round,pad=0.3",
                    facecolor="#1a1e28", edgecolor="#2a2f40"))
 
    plt.show()

# ── Entry Point ──────────────────────────────────────────────────
if __name__ == "__main__":
    t_name = input("Enter Trajectory (Hover, Circle, Figure-8): ").strip()
    c_name = input("Enter Condition (Nominal, Wind Only, etc.): ").strip()
    load_and_plot_real_data(t_name, c_name)