import os
import sys
import glob
import csv
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression

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
    
    # 1. Healthy Control / Normal (0, royalblue)
    if "_HEALTHY" in name_upper or "HEALTHY" in name_upper or "HFH_" in name_upper or "NORMAL" in name_upper:
        return "Healthy Control", 0, "royalblue"
    
    # 2. Ipsilateral TLE (Diseased) (1, crimson)
    elif (is_left_side and "LEFT-TLE" in name_upper) or (not is_left_side and "RIGHT-TLE" in name_upper):
        return "Ipsilateral TLE (Diseased)", 1, "crimson"
        
    # 3. Contralateral TLE (Healthy-side) (2, royalblue)
    elif (is_left_side and "RIGHT-TLE" in name_upper) or (not is_left_side and "LEFT-TLE" in name_upper):
        return "Contralateral TLE (Healthy-side)", 2, "royalblue"
        
    # 4. General TLE (1, crimson)
    elif "TLE" in name_upper:
        return "Ipsilateral TLE (Diseased)", 1, "crimson"
        
    return "Unknown", -1, "gray"

def main():
    print("="*60)
    print("--- PLS-DA Augmented Scores Visualization ---")
    print("="*60)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory containing spharm_results/")
    parser.add_argument("--n_components", type=int, default=10,
                        help="Number of PLS-DA components to fit (default=10)")
    args, unknown = parser.parse_known_args()

    if args.output_dir:
        output_root = os.path.abspath(args.output_dir)
    else:
        print("No --output_dir given. Opening folder picker...")
        chosen = prompt_folder("Select output folder (containing spharm_results/)")
        if not chosen:
            print("ERROR: No folder selected. Exiting.")
            return
        chosen_abs = os.path.abspath(chosen)
        base = os.path.basename(chosen_abs.rstrip("\\/")).lower()
        if base in ("spharm_results", "plsda_interpolated_surfaces", "plsda_results", "plsda_split_results"):
            output_root = os.path.dirname(chosen_abs.rstrip("\\/"))
        elif base in ("healthy", "diseased"):
            parent = os.path.dirname(chosen_abs.rstrip("\\/"))
            if os.path.basename(parent.rstrip("\\/")).lower() == "plsda_interpolated_surfaces":
                output_root = os.path.dirname(parent.rstrip("\\/"))
            else:
                output_root = chosen_abs
        else:
            output_root = chosen_abs

    if os.path.isdir(os.path.join(output_root, "spharm_results")):
        spharm_results_dir = os.path.join(output_root, "spharm_results")
    else:
        spharm_results_dir = output_root

    # Find .coef files
    all_coef_files = sorted(glob.glob(os.path.join(spharm_results_dir, "*_SPHARM.coef")))
    coef_files = [f for f in all_coef_files
                  if not any(s in os.path.basename(f)
                             for s in ("_ellalign", "_grid", "_realigned", "_procalign", "_pca_ready"))]

    if not coef_files:
        print("ERROR: No SPHARM coefficient files (*_SPHARM.coef) found.")
        return

    # Parse and check dimensions
    subject_names = []
    coef_vectors = []
    groups = []
    colors = []
    classes = []
    expected_len = None

    # First pass: find the first file with a valid number of coefficients (len >= 9) to determine expected_len
    for fpath in coef_files:
        coeffs = parse_coef(fpath)
        if coeffs and len(coeffs) >= 9:
            expected_len = len(coeffs)
            break

    if expected_len is None:
        print("ERROR: No valid SPHARM coefficient files found.")
        return

    # Second pass: read and parse all files, skipping mismatched ones
    for fpath in coef_files:
        basename = os.path.basename(fpath)
        
        coeffs = parse_coef(fpath)
        if not coeffs:
            continue
            
        if len(coeffs) != expected_len:
            continue
            
        flat_coeffs = np.array(coeffs).ravel()
        coef_vectors.append(flat_coeffs)
        
        sub_name = basename.replace("_SPHARM.coef", "")
        subject_names.append(sub_name)
        g_name, cls, col = classify_subject(basename)
        groups.append(g_name)
        colors.append(col)
        classes.append(cls)

    coef_vectors = np.array(coef_vectors)
    N, D = coef_vectors.shape
    print(f"Loaded {N} original subjects.")

    # Prepare Y labels (0 = Normal, 1 = Diseased)
    binary_labels = np.array([1 if cls == 1 else 0 for cls in classes])
    
    Y = np.zeros((N, 2))
    for i, label in enumerate(binary_labels):
        Y[i, label] = 1.0

    # Fit PLS-DA
    n_comp = min(args.n_components, N - 1)
    print(f"Fitting PLS-DA model with {n_comp} components on original subjects...")
    pls = PLSRegression(n_components=n_comp, scale=True)
    X_scores, _ = pls.fit_transform(coef_vectors, Y)

    # 3. Load Augmented Metadata
    aug_dir = os.path.join(output_root, "plsda_interpolated_surfaces")
    metadata_path = os.path.join(aug_dir, "interpolated_metadata.csv")
    
    if not os.path.exists(metadata_path):
        print(f"ERROR: metadata file not found at: {metadata_path}")
        print("Please run augment_plsda_interpolation.py first to generate augmented data.")
        return

    augmented_scores = []
    augmented_classes = [] # 'Healthy' or 'Diseased'

    subject_to_idx = {name: idx for idx, name in enumerate(subject_names)}

    print("Loading augmented data and computing interpolated scores...")
    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parent_A = row["Parent_A_Subject"]
            parent_B = row["Parent_B_Subject"]
            alpha = float(row["Alpha"])
            dir_class = row["DirectoryClass"]

            if parent_A in subject_to_idx and parent_B in subject_to_idx:
                idx_A = subject_to_idx[parent_A]
                idx_B = subject_to_idx[parent_B]

                score_A = X_scores[idx_A]
                score_B = X_scores[idx_B]

                # Linear interpolation in PLS-DA score space
                score_new = (1.0 - alpha) * score_A + alpha * score_B
                augmented_scores.append(score_new)
                augmented_classes.append(dir_class)
            else:
                print(f"  Warning: Parent subject not found in original dataset ({parent_A} or {parent_B})")

    augmented_scores = np.array(augmented_scores)
    M = len(augmented_scores)
    print(f"Loaded {M} augmented subjects.")

    # 4. Plot Scatter Plot
    fig, ax = plt.subplots(figsize=(11, 9))

    # Plot augmented points (semi-transparent cloud)
    if M > 0:
        aug_healthy_idx = [i for i, c in enumerate(augmented_classes) if c == "Healthy"]
        aug_diseased_idx = [i for i, c in enumerate(augmented_classes) if c == "Diseased"]

        if aug_healthy_idx:
            ax.scatter(
                augmented_scores[aug_healthy_idx, 0],
                augmented_scores[aug_healthy_idx, 1],
                color="royalblue",
                marker="o",
                s=90,
                alpha=0.9,
                edgecolors="black",
                linewidths=0.8,
                label="Augmented Healthy"
            )
        if aug_diseased_idx:
            ax.scatter(
                augmented_scores[aug_diseased_idx, 0],
                augmented_scores[aug_diseased_idx, 1],
                color="crimson",
                marker="o",
                s=90,
                alpha=0.9,
                edgecolors="black",
                linewidths=0.8,
                label="Augmented Diseased"
            )

    # Plot original points (solid/distinct)
    unique_groups = sorted(list(set(groups)))
    for g_name in unique_groups:
        idx = [i for i, g in enumerate(groups) if g == g_name]
        col = colors[idx[0]]
        ax.scatter(
            X_scores[idx, 0],
            X_scores[idx, 1],
            c=col,
            edgecolors="black",
            linewidths=0.8,
            s=90,
            alpha=0.9,
            label=f"Original: {g_name}"
        )

    ax.legend(loc="best", fontsize=10)
    ax.set_xlabel("PLS-DA Component 1", fontsize=12, fontweight="bold")
    ax.set_ylabel("PLS-DA Component 2", fontsize=12, fontweight="bold")
    ax.set_title(f"PLS-DA Score Space: Original vs Augmented ({os.path.basename(output_root)})", 
                 fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Save visualization plot
    out_plot = os.path.join(aug_dir, "plsda_augmented_visualization.png")
    plt.savefig(out_plot, dpi=300, bbox_inches="tight")
    print(f"Saved visualization plot to: {out_plot}")
    
    # Display plot window
    plt.show()

if __name__ == "__main__":
    main()
