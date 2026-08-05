import os
import sys
import glob
import csv
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
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

    # Cleanup old results from previous runs
    results_dir = os.path.join(output_root, "plsda_split_results")
    if os.path.exists(results_dir):
        print(f"Cleaning up old results directory: {results_dir}")
        try:
            import shutil
            shutil.rmtree(results_dir)
        except Exception as e:
            print(f"Warning: could not remove old results directory: {e}")

    for old_file in ["train_data.csv", "test_data.csv", "all_data.csv"]:
        old_path = os.path.join(output_root, old_file)
        if os.path.exists(old_path):
            print(f"Removing old file: {old_path}")
            try:
                os.remove(old_path)
            except Exception as e:
                print(f"Warning: could not remove old file {old_path}: {e}")

    print("="*60)
    print("--- SPHARM COEFFICIENT PLS-DA TRAINING & CSV SPLIT PIPELINE ---")
    print("="*60)
    print(f"Selected Output Root: {output_root}")
    print(f"Spharm Results Dir:  {spharm_results_dir}")

    # 1. Find .coef files
    all_coef_files = sorted(glob.glob(os.path.join(spharm_results_dir, "*_SPHARM.coef")))
    coef_files = [f for f in all_coef_files
                  if not any(s in os.path.basename(f)
                             for s in ("_ellalign", "_grid", "_realigned", "_procalign", "_pca_ready"))]

    if not coef_files:
        print("ERROR: No *_SPHARM.coef files found in spharm_results/")
        return
    
    print(f"Found {len(coef_files)} SPHARM coefficient files.")

    # 2. Parse and classify subjects
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
    print(f"Total dataset: {N} subjects x {D} coefficient features")

    # Target labels for split stratification:
    # 0 = Normal (Healthy Control + Contralateral TLE)
    # 1 = Diseased (Ipsilateral TLE)
    binary_labels = np.array([1 if cls == 1 else 0 for cls in classes])

    # Print current dataset count
    num_diseased = np.sum(binary_labels == 1)
    num_normal = np.sum(binary_labels == 0)
    print(f"Dataset summary: Normal = {num_normal}, Diseased (Patient) = {num_diseased}")

    # 3. Perform 80:20 train/test stratified split
    print("\nSplitting dataset (80% Train / 20% Test) with stratification...")
    try:
        train_idx, test_idx = train_test_split(
            np.arange(N),
            test_size=0.20,
            random_state=42,
            stratify=binary_labels
        )
    except Exception as e:
        print(f"[WARNING] Stratified split failed: {e}. Falling back to standard split.")
        train_idx, test_idx = train_test_split(
            np.arange(N),
            test_size=0.20,
            random_state=42
        )

    # Output directory for results
    results_dir = os.path.join(output_root, "plsda_split_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 4. Save input datasets directly as CSV files (Subject, Group, Class, BinaryClass, Coef_1...Coef_D)
    print("Writing input datasets to CSV files...")
    header = ["Subject", "Group", "Class", "BinaryClass"] + [f"Coef_{i+1}" for i in range(D)]

    def save_split_csv(indices, filename):
        filepath = os.path.join(results_dir, filename)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for idx in indices:
                row = [
                    subject_names[idx], 
                    groups[idx], 
                    classes[idx], 
                    binary_labels[idx]
                ] + ["{:.8f}".format(v) for v in coef_vectors[idx]]
                writer.writerow(row)
        print(f"  Saved: {filename} ({len(indices)} rows) to {results_dir}")

    save_split_csv(train_idx, "train_data.csv")
    save_split_csv(test_idx, "test_data.csv")
    save_split_csv(np.arange(N), "all_data.csv")

    # Print stratification statistics in train and test sets
    train_labels = binary_labels[train_idx]
    test_labels = binary_labels[test_idx]
    print(f"  Train Set: Normal = {np.sum(train_labels == 0)}, Diseased = {np.sum(train_labels == 1)}")
    print(f"  Test Set:  Normal = {np.sum(test_labels == 0)}, Diseased = {np.sum(test_labels == 1)}")

    # =============================================================================
    # PLS-DA on Train Data only
    # =============================================================================
    print("\n" + "="*50)
    print("--- RUNNING PLS-DA ON TRAIN DATASET ONLY ---")
    print("="*50)

    train_coef_vectors = coef_vectors[train_idx]
    N_train = len(train_idx)

    # Prepare Y target matrix (One-Hot Encoded Y matrix for binary PLS-DA: Normal vs Diseased)
    Y_train = np.zeros((N_train, 2))
    for i, idx in enumerate(train_idx):
        cls = classes[idx]
        if cls == 1:  # Diseased (Ipsilateral TLE)
            Y_train[i, 1] = 1.0
        else:         # Normal (Healthy Control or Contralateral TLE)
            Y_train[i, 0] = 1.0

    print("Fitting PLS-DA (PLSRegression with 10 components)...")
    pls = PLSRegression(n_components=10)
    X_scores_train, _ = pls.fit_transform(train_coef_vectors, Y_train)

    # Save PLS-DA Scores to CSV
    scores_csv = os.path.join(results_dir, "plsda_scores_train.csv")
    with open(scores_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["Subject"] + [f"PLS-DA {k}" for k in range(1, 11)] + ["Group", "Class"]
        writer.writerow(header)
        for i, idx in enumerate(train_idx):
            scores_row = [subject_names[idx]] + [f"{X_scores_train[i,j]:.8f}" for j in range(10)] + [groups[idx], classes[idx]]
            writer.writerow(scores_row)
    print(f"Saved train PLS-DA scores to: {scores_csv}")

    # Plot PLS-DA for train set
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Track which groups are present in training set
    train_groups = [groups[idx] for idx in train_idx]
    train_colors = [colors[idx] for idx in train_idx]
    unique_groups = sorted(list(set(train_groups)))
    
    # Plot each group separately to get clear labels in the legend
    for g_name in unique_groups:
        idx_in_train = [i for i, g in enumerate(train_groups) if g == g_name]
        col = train_colors[idx_in_train[0]]
        ax.scatter(
            X_scores_train[idx_in_train, 0], 
            X_scores_train[idx_in_train, 1], 
            c=col, 
            alpha=0.7, 
            edgecolors='w', 
            s=100, 
            label=g_name
        )
        
    ax.legend(loc='best', fontsize=10)
    ax.set_xlabel('PLS-DA Component 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('PLS-DA Component 2', fontsize=12, fontweight='bold')
    ax.set_title(f'PLS-DA Distribution (Train Set Only): {os.path.basename(output_root)}', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)

    output_plot = os.path.join(results_dir, "plsda_visualization_train.png")
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved train PLS-DA plot to: {output_plot}")
    print("="*60)

if __name__ == '__main__':
    main()
