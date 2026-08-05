import os
import sys
import glob
import csv
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
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
    is_left_side = subject_name.startswith("left_") or subject_name.startswith("lh_") or "_lh" in subject_name.lower()
    name_upper = subject_name.upper()
    
    # 1. Healthy Control / Normal (royalblue, 0)
    if "_HEALTHY" in name_upper or "HEALTHY" in name_upper or "HFH_" in name_upper or "NORMAL" in name_upper:
        return "Healthy Control", "royalblue", 0
    
    # 2. Ipsilateral TLE (Diseased) (crimson, 1)
    elif (is_left_side and "LEFT-TLE" in name_upper) or (not is_left_side and "RIGHT-TLE" in name_upper):
        return "Ipsilateral TLE (Diseased)", "crimson", 1
        
    # 3. Contralateral TLE (Healthy-side) (royalblue, 2)
    elif (is_left_side and "RIGHT-TLE" in name_upper) or (not is_left_side and "LEFT-TLE" in name_upper):
        return "Contralateral TLE (Healthy-side)", "royalblue", 2
        
    # 4. General TLE (crimson, 1)
    elif "TLE" in name_upper:
        return "Ipsilateral TLE (Diseased)", "crimson", 1
        
    return "Unknown", "gray", -1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory containing spharm_results/")
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

    if os.path.isdir(os.path.join(output_root, "spharm_results")):
        spharm_results_dir = os.path.join(output_root, "spharm_results")
    else:
        spharm_results_dir = output_root
    plsda_dir = os.path.join(output_root, "plsda_results")
    if not os.path.exists(plsda_dir):
        os.makedirs(plsda_dir)

    print("="*60)
    print("--- SPHARM COEFFICIENT PLS-DA ANALYSIS ---")
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
    expected_len = None

    # First pass: find the first file with a valid number of coefficients (len >= 9) to determine L and expected_len
    for fpath in coef_files:
        coeffs = parse_coef(fpath)
        if coeffs and len(coeffs) >= 9:
            expected_len = len(coeffs)
            L = int(np.sqrt(expected_len)) - 1
            print(f"Detected SPHARM degree L = {L} (number of coefficients = {expected_len})")
            break

    if expected_len is None:
        print("ERROR: No valid coefficient files found (all are empty or too small).")
        return

    # Second pass: read and parse all files, skipping those with mismatched coefficient lengths
    for fpath in coef_files:
        basename = os.path.basename(fpath)
        subject_name = basename.replace("_SPHARM.coef", "")
        
        coeffs = parse_coef(fpath)
        if not coeffs:
            continue
            
        if len(coeffs) != expected_len:
            print(f"WARNING: Skipping {basename} - got {len(coeffs)} coefficients, expected {expected_len}")
            continue
            
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

    # 3. Prepare target labels (One-Hot Encoded Y matrix for binary PLS-DA: Normal vs Diseased)
    # Class 0: Normal (Healthy Control + Contralateral TLE)
    # Class 1: Diseased (Ipsilateral TLE)
    num_classes = 2
    Y = np.zeros((N, num_classes))
    for i, cls in enumerate(classes):
        if cls == 1:  # Diseased (Ipsilateral TLE)
            Y[i, 1] = 1.0
        else:         # Normal (Healthy Control or Contralateral TLE)
            Y[i, 0] = 1.0

    # 4. Perform PLS-DA (PLSRegression with 3 components to support 3D visualization)
    print("Running PLS-DA (PLSRegression with 10 components)...")
    pls = PLSRegression(n_components=10)
    X_scores, _ = pls.fit_transform(coef_vectors, Y)

    # 5. Save Scores to CSV
    scores_csv = os.path.join(plsda_dir, "plsda_scores.csv")
    with open(scores_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["Subject"] + [f"PLS-DA {k}" for k in range(1, 11)] + ["Group", "Class"]
        writer.writerow(header)
        for i in range(N):
            scores_row = [subject_names[i]] + [f"{X_scores[i,j]:.8f}" for j in range(10)] + [groups[i], classes[i]]
            writer.writerow(scores_row)
    print(f"Saved PLS-DA scores to: {scores_csv}")

    # 6. Plot PLS-DA
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_groups = sorted(list(set(groups)))
    # Plot each group separately to get clear labels in the legend
    for g_name in unique_groups:
        idx = [i for i, g in enumerate(groups) if g == g_name]
        col = colors[idx[0]]
        ax.scatter(X_scores[idx, 0], X_scores[idx, 1], c=col, alpha=0.7, edgecolors='w', s=100, label=g_name)
        
    ax.legend(loc='best', fontsize=10)
    ax.set_xlabel('PLS-DA Component 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('PLS-DA Component 2', fontsize=12, fontweight='bold')
    ax.set_title(f'PLS-DA Distribution: {os.path.basename(output_root)}', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)

    output_plot = os.path.join(plsda_dir, "plsda_visualization.png")
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PLS-DA plot to: {output_plot}")
    print("="*60)

if __name__ == '__main__':
    main()
