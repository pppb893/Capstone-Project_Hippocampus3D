import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()

def main():
    parser = argparse.ArgumentParser(description="Plot ICP convergence time-series")
    parser.add_argument("--output_dir", type=str, help="Output directory containing icp_convergence_history.json")
    parser.add_argument("--show", action="store_true", help="Display the plot window")
    args, unknown = parser.parse_known_args()

    if args.output_dir:
        output_root = os.path.abspath(args.output_dir)
    else:
        output_root = os.path.join(SCRIPT_DIR, "output")

    json_path = os.path.join(output_root, "icp_convergence_history.json")
    output_plot = os.path.join(output_root, "icp_convergence.png")

    print(f"--- Plotting ICP Convergence Time-Series for {os.path.basename(output_root)} ---")

    if not os.path.exists(json_path):
        print(f"[ERROR] Convergence log not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    rounds = data.get("rounds", [])
    mean_dists = data.get("mean_distances", [])
    dist_changes = data.get("dist_changes", [])
    subject_dists = data.get("subject_distances", {})
    elapsed_times = data.get("elapsed_times_sec", [])

    pw_iters = data.get("pairwise_iterations", [])
    pw_dists = data.get("mean_pairwise_distances", [])
    subj_pw_dists = data.get("subject_pairwise_distances", {})

    if not rounds or not mean_dists:
        print("[ERROR] Convergence log is empty.")
        return

    x_vals_gw = rounds
    x_label_gw = "Groupwise Round"

    has_pairwise = bool(pw_iters) and bool(pw_dists)
    n_cols = 3 if has_pairwise else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(18 if n_cols == 3 else 15, 5.8))

    ax1 = axes[0]
    ax2 = axes[1] if has_pairwise else axes[1]
    ax3 = axes[2] if has_pairwise else None

    # -------------------------------------------------------------
    # Panel 1: Groupwise Global Convergence Across Rounds
    # -------------------------------------------------------------
    ax1.set_title('Groupwise Global Convergence', fontsize=13, fontweight='bold', pad=15)
    line1 = ax1.plot(x_vals_gw, mean_dists, color='forestgreen', marker='o', linewidth=2.2, markersize=7, label='Mean ICP Distance')
    ax1.set_xlabel(x_label_gw, fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean ICP Distance (mm / norm. unit)', fontsize=12, fontweight='bold', color='forestgreen')
    ax1.tick_params(axis='y', labelcolor='forestgreen', labelsize=11)
    ax1.tick_params(axis='x', labelsize=11)
    ax1.set_xticks(rounds)
    ax1.grid(True, linestyle='--', alpha=0.5)

    for r, x, d in zip(rounds, x_vals_gw, mean_dists):
        t_sec = elapsed_times[r - 1] if (elapsed_times and len(elapsed_times) >= r) else None
        lbl = f'Round {r}\n({t_sec:.1f}s)' if t_sec is not None else f'Round {r}'
        ax1.annotate(lbl, (x, d), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9.5, fontweight='bold', color='forestgreen')

    if len(rounds) > 1:
        ax1_right = ax1.twinx()
        line2 = ax1_right.plot(x_vals_gw[1:], dist_changes[1:], color='navy', marker='s', linestyle='-', linewidth=2.0, markersize=6, label='Round-to-Round Change (|Δ|)')
        line3 = ax1_right.axhline(y=0.00005, color='darkorange', linestyle=':', linewidth=1.8, label='Tolerance (0.00005)')
        ax1_right.set_ylabel('Convergence Change (|Δ Distance|)', fontsize=12, fontweight='bold', color='navy')
        ax1_right.tick_params(axis='y', labelcolor='navy', labelsize=11)

        lines = line1 + line2 + [line3]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right', frameon=True, facecolor='white', edgecolor='gray', fontsize=9)
    else:
        ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='gray', fontsize=10)

    # -------------------------------------------------------------
    # Panel 2: Pairwise VTK Alignment Sub-iterations (Mean)
    # -------------------------------------------------------------
    if has_pairwise:
        ax2.set_title('Pairwise VTK Alignment (Mean)', fontsize=13, fontweight='bold', pad=15)
        ax2.plot(pw_iters, pw_dists, color='royalblue', marker='^', linewidth=2.2, markersize=7, label='Mean Pairwise Distance')
        ax2.set_xlabel('VTK Pairwise Iteration', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Pairwise Mean Distance', fontsize=12, fontweight='bold', color='royalblue')
        ax2.tick_params(axis='both', labelsize=11)
        ax2.set_xticks(pw_iters)
        ax2.grid(True, linestyle='--', alpha=0.5)

        ax2.annotate(f'Start: {pw_dists[0]:.4f}', (pw_iters[0], pw_dists[0]), textcoords="offset points", xytext=(10, 5), fontweight='bold', color='royalblue')
        ax2.annotate(f'Final: {pw_dists[-1]:.4f}', (pw_iters[-1], pw_dists[-1]), textcoords="offset points", xytext=(-40, 10), fontweight='bold', color='royalblue')
        ax2.legend(loc='upper right', fontsize=10)

    # -------------------------------------------------------------
    # Panel 3: Subject Pairwise Trajectories Across VTK Sub-iterations (Matching Graph 2 X-axis)
    # -------------------------------------------------------------
    subj_ax = ax3 if has_pairwise else ax2

    # Check if we have individual pairwise trajectory data matching pw_iters
    target_dict = subj_pw_dists if (has_pairwise and subj_pw_dists) else subject_dists
    target_x = pw_iters if (has_pairwise and subj_pw_dists) else x_vals_gw
    target_xlabel = "VTK Pairwise Iteration" if (has_pairwise and subj_pw_dists) else "Groupwise Round"

    subjs = [item for item in target_dict.items() if len(item[1]) == len(target_x)]
    num_subjs = len(subjs)

    cmap = plt.colormaps['turbo']
    colors = [cmap(i / max(1, num_subjs - 1)) for i in range(num_subjs)]

    for idx, (subj_name, dists) in enumerate(subjs):
        color = colors[idx]
        subj_ax.plot(target_x, dists, color=color, alpha=0.55, linewidth=1.0)

    # Group mean overlay (matching Graph 2 X-axis)
    mean_overlay = pw_dists if (has_pairwise and subj_pw_dists) else mean_dists
    subj_ax.plot(target_x, mean_overlay, color='black', linestyle='--', marker='o', linewidth=3.2, markersize=6, label='Group Mean Distance')
    subj_ax.set_xlabel(target_xlabel, fontsize=12, fontweight='bold')
    subj_ax.set_ylabel('Pairwise ICP Distance to Template', fontsize=12, fontweight='bold')
    subj_ax.set_title(f'Subject Trajectories Across VTK Iterations ({num_subjs} Meshes)', fontsize=13, fontweight='bold', pad=15)
    subj_ax.set_xticks(target_x)
    subj_ax.tick_params(axis='both', labelsize=11)
    subj_ax.grid(True, linestyle='--', alpha=0.5)
    subj_ax.legend(loc='upper right', fontsize=9.5)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"[OK] Time-series convergence plot saved to: {output_plot}")

    if args.show:
        try:
            print("[INFO] Displaying convergence plot window...")
            plt.show()
        except Exception as e:
            print(f"[NOTE] Could not open matplotlib window: {e}")
            try:
                if sys.platform == "win32":
                    os.startfile(output_plot)
            except Exception:
                pass

if __name__ == "__main__":
    main()
