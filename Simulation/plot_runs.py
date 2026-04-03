import numpy as np
import matplotlib.pyplot as plt
import glob, os

COLORS   = {"PID": "#ff6b4a", "MPC": "#4c9fff", "PIDplus MPC": "#3ddba6"}
CTRL_DT  = 1/48
DURATION = 20.0
N_STEPS  = int(DURATION / CTRL_DT)
times    = np.arange(N_STEPS) * CTRL_DT

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


def load_run(controller, traj, condition):
    safe = f"{controller}_{traj}_{condition}".replace(" ","_").replace("+","plus")
    af = f"traj_{safe}_actual.npy"
    tf = f"traj_{safe}_target.npy"
    if not os.path.exists(af):
        print(f"  Missing: {af}")
        return None, None
    return np.load(af), np.load(tf)


def plot_comparison(traj_name, condition, controllers=["PID","MPC","PID+MPC"]):
    runs = []
    for ctrl in controllers:
        actual, target = load_run(ctrl, traj_name, condition)
        if actual is not None:
            runs.append((ctrl, actual, target))

    if not runs:
        print(f"No data found for {traj_name} / {condition}")
        return

    n   = len(runs)
    fig = plt.figure(figsize=(5*n+2, 14))
    fig.suptitle(f"{traj_name}  |  {condition}", fontsize=13, fontweight="bold", y=0.98)
    gs  = fig.add_gridspec(4, n, height_ratios=[2.2,1,1,1], hspace=0.45, wspace=0.35)
    t   = np.arange(N_STEPS) * CTRL_DT

    for col, (ctrl, actual, target) in enumerate(runs):
        color  = COLORS.get(ctrl.replace("+","plus"), "#aaaaaa")
        n_pts  = min(len(actual), len(target), N_STEPS)
        actual = actual[:n_pts]
        target = target[:n_pts]
        t_plot = t[:n_pts]
        errors = np.linalg.norm(actual - target, axis=1)

        # 3D
        ax3d = fig.add_subplot(gs[0, col], projection="3d")
        ax3d.set_facecolor("#13161d")
        ax3d.plot(target[:,0], target[:,1], target[:,2],
                  color="#ffffff", lw=1.2, ls="--", alpha=0.5, label="Reference")
        ax3d.plot(actual[:,0], actual[:,1], actual[:,2],
                  color=color, lw=1.8, alpha=0.9, label="Actual")
        ax3d.plot(actual[:,0], actual[:,1], np.zeros(n_pts),
                  color=color, lw=0.6, alpha=0.18, ls=":")
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

        # XY top view
        ax_top = fig.add_subplot(gs[1, col])
        ax_top.plot(target[:,0], target[:,1], color="#ffffff", lw=1.0, ls="--", alpha=0.5)
        ax_top.plot(actual[:,0], actual[:,1], color=color, lw=1.6, alpha=0.9)
        ax_top.scatter(actual[0,0],  actual[0,1],  color="white", s=15, zorder=5)
        ax_top.scatter(actual[-1,0], actual[-1,1], color=color,   s=25, zorder=5, marker="*")
        ax_top.set_xlabel("X (m)", fontsize=7)
        ax_top.set_ylabel("Y (m)", fontsize=7)
        ax_top.set_title("Top (XY)", fontsize=8)
        ax_top.set_aspect("equal", adjustable="box")

        # XZ front view
        ax_front = fig.add_subplot(gs[2, col])
        ax_front.plot(target[:,0], target[:,2], color="#ffffff", lw=1.0, ls="--", alpha=0.5)
        ax_front.plot(actual[:,0], actual[:,2], color=color, lw=1.6, alpha=0.9)
        ax_front.scatter(actual[0,0],  actual[0,2],  color="white", s=15, zorder=5)
        ax_front.scatter(actual[-1,0], actual[-1,2], color=color,   s=25, zorder=5, marker="*")
        ax_front.set_xlabel("X (m)", fontsize=7)
        ax_front.set_ylabel("Z (m)", fontsize=7)
        ax_front.set_title("Front (XZ)", fontsize=8)
        ax_front.set_aspect("equal", adjustable="box")

        # Error over time
        ax_err = fig.add_subplot(gs[3, col])
        ax_err.plot(t_plot, errors, color=color, lw=1.6)
        ax_err.fill_between(t_plot, errors, alpha=0.15, color=color)
        ax_err.axvline(DURATION/2, color="#555", lw=0.8, ls=":")
        ax_err.set_xlabel("Time (s)", fontsize=7)
        ax_err.set_ylabel("Error (m)", fontsize=7)
        ax_err.set_title("Position error", fontsize=8)
        ax_err.set_xlim(0, DURATION)
        rmse   = np.sqrt(np.mean(errors**2))
        steady = np.mean(errors[len(errors)//2:])
        ax_err.text(0.97, 0.95,
                    f"RMSE  {rmse:.4f}\nSteady {steady:.4f}",
                    transform=ax_err.transAxes,
                    fontsize=7, va="top", ha="right", color="#c8cfe8",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="#1a1e28", edgecolor="#2a2f40"))

    plt.savefig(f"plot_{traj_name}_{condition}.png".replace(" ","_"), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: plot_{traj_name}_{condition}.png".replace(" ","_"))


if __name__ == "__main__":
    # Plot whichever combinations you have data for
    for traj in ["Circle", "Figure-8", "Hover"]:
        for cond in ["Nominal", "Wind_Only", "Payload_Only", "Wind_+_Payload"]:
            plot_comparison(traj, cond)