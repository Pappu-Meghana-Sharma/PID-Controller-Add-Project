import os
import numpy as np
import matplotlib.pyplot as plt
from .config import CTRL_DT

COLORS = {
    "PID": "#ff6b4a", 
    "MPC": "#4c9fff", 
    "PID+MPC": "#3ddba6",
    "Adaptive MPC": "#3ddba6",
    "PID (Lin)": "#ff6b4a", 
    "MPC (Lin)": "#4c9fff", 
    "PID+MPC (Lin)": "#1e996e",
    "PID (NonLin)": "#ff9b85",
    "MPC (NonLin)": "#8ecaff",
    "Adaptive MPC (NonLin)": "#3ddba6"
}
REF_COLOR = "#ffffff"
 
def setup_dark_theme():
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

def _get_padded_limits(runs, min_span=1.0):
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    min_z, max_z = float('inf'), float('-inf')
    
    for _, ref, actual, _, _ in runs:
        all_pts = np.vstack((ref, actual))
        min_x = min(min_x, all_pts[:,0].min())
        max_x = max(max_x, all_pts[:,0].max())
        min_y = min(min_y, all_pts[:,1].min())
        max_y = max(max_y, all_pts[:,1].max())
        min_z = min(min_z, all_pts[:,2].min())
        max_z = max(max_z, all_pts[:,2].max())
        
    def pad(cmin, cmax):
        span = cmax - cmin
        if span < min_span:
            p = (min_span - span) / 2
            return cmin - p, cmax + p
        p = span * 0.1
        return cmin - p, cmax + p
        
    return pad(min_x, max_x), pad(min_y, max_y), pad(min_z, max_z)

def plot_runs(runs, title=None, auto_save=True, output_dir="experiment_results"):
    if not runs:
        print("No runs to plot.")
        return
 
    setup_dark_theme()
    n = len(runs)
    traj_name = runs[0][3]
    cond_name = runs[0][4]
    sup = title or f"{traj_name}  |  {cond_name}"
    
    xlim, ylim, zlim = _get_padded_limits(runs, min_span=1.0)
 
    fig = plt.figure(figsize=(5 * n + 2, 14))
    fig.suptitle(sup, fontsize=13, fontweight="bold", y=0.98)
 
    gs = fig.add_gridspec(4, n, height_ratios=[2.2, 1, 1, 1], hspace=0.45, wspace=0.35)
 
    for col, (ctrl, ref, actual, _, _) in enumerate(runs):
        ref    = np.asarray(ref)
        actual = np.asarray(actual)
        
        min_len = min(len(ref), len(actual))
        ref = ref[:min_len]
        actual = actual[:min_len]
        
        color  = COLORS.get(ctrl, "#aaaaaa")
        errors = np.linalg.norm(actual - ref, axis=1)
        times = np.arange(len(actual)) * CTRL_DT
 
        # Row 0: 3D
        ax3d = fig.add_subplot(gs[0, col], projection="3d")
        ax3d.set_facecolor("#13161d")
        ax3d.plot(ref[:,0], ref[:,1], ref[:,2], color=REF_COLOR, lw=1.2, ls="--", alpha=0.5, label="Reference")
        ax3d.plot(actual[:,0], actual[:,1], actual[:,2], color=color, lw=1.8, alpha=0.9, label="Actual")
        ax3d.plot(actual[:,0], actual[:,1], np.zeros(len(actual)), color=color, lw=0.6, alpha=0.18, ls=":")
        ax3d.scatter(*actual[0],  color="white", s=20, zorder=5)
        ax3d.scatter(*actual[-1], color=color,   s=35, zorder=5, marker="*")
        ax3d.set_xlim(xlim)
        ax3d.set_ylim(ylim)
        ax3d.set_zlim(zlim)
        ax3d.set_xlabel("X (m)", fontsize=7, labelpad=1)
        ax3d.set_ylabel("Y (m)", fontsize=7, labelpad=1)
        ax3d.set_zlabel("Z (m)", fontsize=7, labelpad=1)
        ax3d.set_title(ctrl, fontsize=11, fontweight="bold", pad=6)
        ax3d.legend(fontsize=7, loc="upper left")
        ax3d.view_init(elev=28, azim=-55)
        for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#1e2333")
 
        # Row 1: Top view XY
        ax_top = fig.add_subplot(gs[1, col])
        ax_top.plot(ref[:,0], ref[:,1], color=REF_COLOR, lw=1.0, ls="--", alpha=0.5)
        ax_top.plot(actual[:,0], actual[:,1], color=color, lw=1.6, alpha=0.9)
        ax_top.scatter(ref[-1,0], ref[-1,1], color=REF_COLOR, marker='x', s=20, alpha=0.5)
        ax_top.scatter(actual[-1,0], actual[-1,1], color=color, marker='o', s=20, alpha=0.9)
        ax_top.set_xlim(xlim)
        ax_top.set_ylim(ylim)
        ax_top.set_xlabel("X (m)", fontsize=7)
        ax_top.set_ylabel("Y (m)", fontsize=7)
        ax_top.set_title("Top  (XY)", fontsize=8)
        ax_top.set_aspect("equal", adjustable="box")
 
        # Row 2: Front view XZ
        ax_front = fig.add_subplot(gs[2, col])
        ax_front.plot(ref[:,0], ref[:,2], color=REF_COLOR, lw=1.0, ls="--", alpha=0.5)
        ax_front.plot(actual[:,0], actual[:,2], color=color, lw=1.6, alpha=0.9)
        ax_front.scatter(ref[-1,0], ref[-1,2], color=REF_COLOR, marker='x', s=20, alpha=0.5)
        ax_front.scatter(actual[-1,0], actual[-1,2], color=color, marker='o', s=20, alpha=0.9)
        ax_front.set_xlim(xlim)
        ax_front.set_ylim(zlim)
        ax_front.set_xlabel("X (m)", fontsize=7)
        ax_front.set_ylabel("Z (m)", fontsize=7)
        ax_front.set_title("Front (XZ)", fontsize=8)
        ax_front.set_aspect("equal", adjustable="box")
 
        # Row 3: Error over time
        ax_err = fig.add_subplot(gs[3, col])
        ax_err.plot(times, errors, color=color, lw=1.6)
        ax_err.fill_between(times, errors, alpha=0.15, color=color)
        ax_err.set_xlabel("Time (s)", fontsize=7)
        ax_err.set_ylabel("Error (m)", fontsize=7)
        ax_err.set_title("Position error", fontsize=8)
 
        n_e    = len(errors)
        rmse   = np.sqrt(np.mean(errors**2))
        steady = np.mean(errors[n_e//2:])
        ax_err.text(0.97, 0.95, f"RMSE  {rmse:.4f}\nSteady {steady:.4f}",
                    transform=ax_err.transAxes, fontsize=7, va="top", ha="right",
                    color="#c8cfe8", bbox=dict(boxstyle="round,pad=0.3",
                    facecolor="#1a1e28", edgecolor="#2a2f40"))

    os.makedirs(output_dir, exist_ok=True)
    safe_cond = cond_name.replace(" ", "_").replace("+", "plus")
    filename = f"{output_dir}/plot_{traj_name}_{safe_cond}.png"
 
    if not auto_save:
        plt.show(block=False)
        plt.pause(0.1)
        confirm = input(f"\nVisualization ready. Save PNG to {filename}? (y/n): ").strip().lower()
        if confirm == 'y':
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"  [Saved Plot] {filename}")
        else:
            print("  [Plot saving skipped by user]")
        print("Close the plot window to continue...")
        plt.show()
    else:
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"  [Saved Plot] {filename}")
        plt.close(fig)

def plot_runs_paper(runs, title=None, auto_save=True, output_dir="experiment_results"):
    if not runs:
        return
 
    with plt.rc_context({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "black",
        "axes.labelcolor":  "black",
        "axes.titlecolor":  "black",
        "axes.grid":        True,
        "grid.color":       "#cccccc",
        "grid.linewidth":   0.5,
        "xtick.color":      "black",
        "ytick.color":      "black",
        "text.color":       "black",
        "legend.facecolor": "white",
        "legend.edgecolor": "black",
        "lines.linewidth":  1.5,
        "font.family":      "serif",
        "font.size":        10,
    }):
        n = len(runs)
        traj_name = runs[0][3]
        cond_name = runs[0][4]
        sup = title or f"{traj_name} tracking under {cond_name}"
        
        xlim, ylim, zlim = _get_padded_limits(runs, min_span=1.0)
     
        fig = plt.figure(figsize=(4 * n + 2, 10))
        fig.suptitle(sup, fontsize=14, fontweight="bold", y=0.98)
     
        gs = fig.add_gridspec(3, n, height_ratios=[1.8, 1, 1], hspace=0.35, wspace=0.3)
     
        for col, (ctrl, ref, actual, _, _) in enumerate(runs):
            ref    = np.asarray(ref)
            actual = np.asarray(actual)
            min_len = min(len(ref), len(actual))
            ref = ref[:min_len]
            actual = actual[:min_len]
            
            color = "#D95319" if "PID" in ctrl else "#0072BD"
            if "Adaptive" in ctrl or "PID+MPC" in ctrl:
                color = "#77AC30"
                
            errors = np.linalg.norm(actual - ref, axis=1)
            times = np.arange(len(actual)) * CTRL_DT
     
            # Row 0: 3D
            ax3d = fig.add_subplot(gs[0, col], projection="3d")
            ax3d.plot(ref[:,0], ref[:,1], ref[:,2], 'k--', lw=1.2, alpha=0.7, label="Reference")
            ax3d.plot(actual[:,0], actual[:,1], actual[:,2], color=color, lw=1.8, label="Actual")
            ax3d.set_xlim(xlim)
            ax3d.set_ylim(ylim)
            ax3d.set_zlim(zlim)
            ax3d.set_xlabel("X (m)", labelpad=1)
            ax3d.set_ylabel("Y (m)", labelpad=1)
            ax3d.set_zlabel("Z (m)", labelpad=1)
            ax3d.set_title(ctrl, fontweight="bold")
            if col == 0:
                ax3d.legend(loc="upper left", fontsize=8)
            ax3d.view_init(elev=28, azim=-55)
     
            # Row 1: Top view XY
            ax_top = fig.add_subplot(gs[1, col])
            ax_top.plot(ref[:,0], ref[:,1], 'k--', lw=1.0, alpha=0.7)
            ax_top.plot(actual[:,0], actual[:,1], color=color, lw=1.5)
            ax_top.scatter(ref[-1,0], ref[-1,1], color='black', marker='x', s=30, alpha=0.7)
            ax_top.scatter(actual[-1,0], actual[-1,1], color=color, marker='o', s=30)
            ax_top.set_xlim(xlim)
            ax_top.set_ylim(ylim)
            ax_top.set_xlabel("X (m)")
            ax_top.set_ylabel("Y (m)")
            ax_top.set_aspect("equal", adjustable="box")
     
            # Row 2: Error over time
            ax_err = fig.add_subplot(gs[2, col])
            ax_err.plot(times, errors, color=color, lw=1.5)
            ax_err.set_xlabel("Time (s)")
            ax_err.set_ylabel("Error (m)")
     
            n_e    = len(errors)
            rmse   = np.sqrt(np.mean(errors**2))
            steady = np.mean(errors[n_e//2:])
            ax_err.text(0.95, 0.90, f"RMSE: {rmse:.3f}m\nSteady: {steady:.3f}m",
                        transform=ax_err.transAxes, fontsize=9, va="top", ha="right",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8))
                        
        os.makedirs(output_dir, exist_ok=True)
        safe_cond = cond_name.replace(" ", "_").replace("+", "plus")
        pdf_file = f"{output_dir}/paper_plot_{traj_name}_{safe_cond}.pdf"
     
        if not auto_save:
            plt.show(block=False)
            plt.pause(0.1)
            confirm = input(f"\nSave Paper PDF to {pdf_file}? (y/n): ").strip().lower()
            if confirm == 'y':
                fig.savefig(pdf_file, dpi=300, bbox_inches="tight")
                print(f"  [Saved PDF] {pdf_file}")
            plt.show()
        else:
            fig.savefig(pdf_file, dpi=300, bbox_inches="tight")
            fig.savefig(pdf_file.replace('.pdf', '.png'), dpi=300, bbox_inches="tight")
            print(f"  [Saved Paper Plots] {pdf_file} (and .png)")
            plt.close(fig)
