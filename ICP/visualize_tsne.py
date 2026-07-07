import os
import sys
import glob
import csv
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from datetime import datetime

# =============================================================================
# Helper: Folder Picker
# =============================================================================
def prompt_folder(title):
    try:
        import qt
        folder = qt.QFileDialog.getExistingDirectory(None, title)
        return folder if folder else None
    except ImportError:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder = filedialog.askdirectory(title=title, initialdir=os.getcwd())
        root.destroy()
        return folder if folder else None

# =============================================================================
# Parse SPHARM-PDM .coef format
# =============================================================================
def parse_coef(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    # Match all triplets {x, y, z}
    pattern = re.compile(r"\{([-+]?[\d\.eE+-]+),\s*([-+]?[\d\.eE+-]+),\s*([-+]?[\d\.eE+-]+)\}")
    matches = pattern.findall(content)
    
    coeffs = []
    for m in matches:
        coeffs.append([float(x) for x in m])
    
    num_match = re.search(r"\{\s*(\d+)", content)
    if num_match:
        num_coeffs = int(num_match.group(1))
        return coeffs[:num_coeffs]
    
    return coeffs

# =============================================================================
# Group Classification
# =============================================================================
def classify_subject(subject_name):
    is_left_side = subject_name.startswith("left_")
    
    # 1. Healthy Control (royalblue)
    if "_Healthy" in subject_name or "HFH_" in subject_name:
        return "Healthy Control", "royalblue", 0
    
    # 2. Ipsilateral TLE (Diseased) (crimson)
    elif (is_left_side and "_Left-TLE" in subject_name) or (not is_left_side and "_Right-TLE" in subject_name):
        return "Ipsilateral TLE (Diseased)", "crimson", 1
        
    # 3. Contralateral TLE (Healthy-side) (royalblue)
    elif (is_left_side and "_Right-TLE" in subject_name) or (not is_left_side and "_Left-TLE" in subject_name):
        return "Contralateral TLE (Healthy-side)", "royalblue", 2
        
    return "Unknown", "gray", -1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory containing spharm_results/")
    parser.add_argument("--perplexity", type=float, default=30.0,
                        help="t-SNE perplexity parameter (default: 30.0)")
    args, unknown = parser.parse_known_args()

    if args.output_dir:
        output_root = os.path.abspath(args.output_dir)
    else:
        print("No --output_dir given. Opening folder picker...")
        chosen = prompt_folder("Select output folder (containing spharm_results/)")
        if not chosen:
            print("ERROR: No folder selected. Exiting.")
            return
        if os.path.basename(chosen.rstrip("\\/")) == "spharm_results":
            output_root = os.path.abspath(os.path.dirname(chosen))
        else:
            output_root = os.path.abspath(chosen)

    spharm_results_dir = os.path.join(output_root, "spharm_results")
    tsne_dir = os.path.join(output_root, "tsne_results")
    if not os.path.exists(tsne_dir):
        os.makedirs(tsne_dir)

    print("="*60)
    print("--- SPHARM COEFFICIENT t-SNE ANALYSIS ---")
    print("="*60)

    # 1. Find .coef files
    all_coef_files = sorted(glob.glob(os.path.join(spharm_results_dir, "*_SPHARM.coef")))
    coef_files = [f for f in all_coef_files
                  if not any(s in os.path.basename(f)
                             for s in ("_ellalign", "_grid", "_realigned", "_procalign", "_pca_ready"))]

    if not coef_files:
        print("ERROR: No *_SPHARM.coef files found in spharm_results/")
        return
    
    print(f"Found {len(coef_files)} SPHARM coefficient files for processing.")

    # 2. Load and parse SPHARM coefficients
    subject_names = []
    coef_vectors = []
    groups = []
    colors = []
    classes = []
    L = None

    for fpath in coef_files:
        basename = os.path.basename(fpath)
        subject_name = basename.replace("_SPHARM.coef", "")
        
        coeffs = parse_coef(fpath)
        if not coeffs:
            continue
            
        if L is None:
            L = int(np.sqrt(len(coeffs))) - 1
            print(f"Detected SPHARM degree L = {L} (number of coefficients = {len(coeffs)})")
            
        flat_coeffs = np.array(coeffs).ravel()
        coef_vectors.append(flat_coeffs)
        
        subject_names.append(subject_name)
        g_name, col, cls = classify_subject(subject_name)
        groups.append(g_name)
        colors.append(col)
        classes.append(cls)

    coef_vectors = np.array(coef_vectors)
    N, D = coef_vectors.shape
    print(f"Data matrix shape: {N} subjects x {D} coefficient features")

    # 3. Perform t-SNE (3 components to support both 2D and 3D visualization)
    print(f"Running t-SNE (perplexity={args.perplexity}, components=3)...")
    tsne = TSNE(n_components=3, perplexity=args.perplexity, random_state=42)
    tsne_results = tsne.fit_transform(coef_vectors)

    # 4. Save Scores to CSV
    scores_csv = os.path.join(tsne_dir, "tsne_scores.csv")
    with open(scores_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Subject", "t-SNE 1", "t-SNE 2", "t-SNE 3", "Group", "Class"])
        for i in range(N):
            writer.writerow([subject_names[i], f"{tsne_results[i,0]:.8f}", f"{tsne_results[i,1]:.8f}", f"{tsne_results[i,2]:.8f}", groups[i], classes[i]])
    print(f"Saved t-SNE scores to: {scores_csv}")

    # 5. Plot t-SNE
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_groups = sorted(list(set(groups)))
    # Plot each group separately to get clear labels in the legend
    for g_name in unique_groups:
        idx = [i for i, g in enumerate(groups) if g == g_name]
        col = colors[idx[0]]
        ax.scatter(tsne_results[idx, 0], tsne_results[idx, 1], c=col, alpha=0.7, edgecolors='w', s=100, label=g_name)
        
    ax.legend(loc='best', fontsize=10)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title(f't-SNE Distribution: {os.path.basename(output_root)}', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)

    output_plot = os.path.join(tsne_dir, "tsne_visualization.png")
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE plot to: {output_plot}")
    print("="*60)

if __name__ == '__main__':
    main()
