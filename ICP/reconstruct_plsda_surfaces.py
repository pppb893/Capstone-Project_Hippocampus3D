import os
import sys
import glob
import csv
import re
import argparse
import numpy as np
from sklearn.cross_decomposition import PLSRegression

# Import resampling functions from existing workspace file
try:
    from resample_spharm_grid import evaluate_spharm, save_grid_vtk
except ImportError:
    print("ERROR: Could not import resample_spharm_grid. Please ensure you are running this script in the ICP directory.")
    sys.exit(1)

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
    
    # 1. Healthy Control / Normal (0)
    if "_HEALTHY" in name_upper or "HEALTHY" in name_upper or "HFH_" in name_upper or "NORMAL" in name_upper:
        return "Healthy Control", 0
    
    # 2. Ipsilateral TLE (Diseased) (1)
    elif (is_left_side and "LEFT-TLE" in name_upper) or (not is_left_side and "RIGHT-TLE" in name_upper):
        return "Ipsilateral TLE (Diseased)", 1
        
    # 3. Contralateral TLE (Healthy-side) (2)
    elif (is_left_side and "RIGHT-TLE" in name_upper) or (not is_left_side and "LEFT-TLE" in name_upper):
        return "Contralateral TLE (Healthy-side)", 2
        
    # 4. General TLE (1)
    elif "TLE" in name_upper:
        return "Ipsilateral TLE (Diseased)", 1
        
    return "Unknown", -1

def main():
    print("="*60)
    print("--- PLS-DA Shape Reconstruction Pipeline ---")
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
        if os.path.basename(chosen.rstrip("\\/")) == "spharm_results":
            output_root = os.path.abspath(os.path.dirname(chosen))
        else:
            output_root = os.path.abspath(chosen)

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
    classes = []
    L = None
    expected_len = None

    # First pass: find the first file with a valid number of coefficients (len >= 9) to determine expected_len
    for fpath in coef_files:
        coeffs = parse_coef(fpath)
        if coeffs and len(coeffs) >= 9:
            expected_len = len(coeffs)
            L = int(np.sqrt(expected_len)) - 1
            print(f"Detected SPHARM degree L = {L} (number of coefficients = {expected_len})")
            break

    if expected_len is None:
        print("ERROR: No valid SPHARM coefficient files found.")
        return

    # Second pass: read and parse all files, skipping mismatched ones
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
        g_name, cls = classify_subject(subject_name)
        groups.append(g_name)
        classes.append(cls)

    coef_vectors = np.array(coef_vectors)
    N, D = coef_vectors.shape
    print(f"Loaded {N} subjects. Coef vector size: {D}")

    # Prepare Y labels (0 = Normal, 1 = Diseased)
    binary_labels = np.array([1 if cls == 1 else 0 for cls in classes])
    
    Y = np.zeros((N, 2))
    for i, label in enumerate(binary_labels):
        Y[i, label] = 1.0

    # Fit PLS-DA
    n_comp = min(args.n_components, N - 1)
    print(f"Fitting PLS-DA model with {n_comp} components...")
    pls = PLSRegression(n_components=n_comp, scale=True)
    X_scores, _ = pls.fit_transform(coef_vectors, Y)

    # Compute group averages on each component to understand the direction (sign)
    std_devs = np.std(X_scores, axis=0)
    print("\n--- Component Direction Interpretation ---")
    for j in range(min(3, n_comp)):
        normal_idx = np.where(binary_labels == 0)[0]
        diseased_idx = np.where(binary_labels == 1)[0]
        mean_normal = np.mean(X_scores[normal_idx, j])
        mean_diseased = np.mean(X_scores[diseased_idx, j])
        print(f"PLS-DA Component {j+1}:")
        print(f"  Mean Score - Normal/Control Group: {mean_normal:+.4f}")
        print(f"  Mean Score - Diseased (Patient):    {mean_diseased:+.4f}")
        direction = "Positive (+)" if mean_diseased > mean_normal else "Negative (-)"
        print(f"  Interpretation: {direction} direction correlates with TLE Disease shape changes.")

    # Reconstruction output directory
    recon_dir = os.path.join(output_root, "plsda_reconstructed_surfaces")
    if not os.path.exists(recon_dir):
        os.makedirs(recon_dir)

    # Grid parameters (same as resample_spharm_grid.py defaults)
    theta_step = 9.0
    phi_step = 9.0
    theta_deg = np.linspace(0, 180, int(180/theta_step) + 1)
    phi_deg = np.linspace(0, 360, int(360/phi_step) + 1)[:-1]

    weights = [-3, -2, -1, 0, 1, 2, 3]
    num_comp_reconstruct = min(3, n_comp)

    print(f"\nGenerating reconstructed surfaces in: {recon_dir}")
    
    for comp_idx in range(num_comp_reconstruct):
        print(f"--- Reconstructing PLS-DA Component {comp_idx + 1} ---")
        comp_dir = os.path.join(recon_dir, f"PLS{comp_idx + 1}")
        if not os.path.exists(comp_dir):
            os.makedirs(comp_dir)

        for w in weights:
            # Create target score vector (size = n_components)
            score = np.zeros(n_comp)
            score[comp_idx] = w * std_devs[comp_idx]

            # Reconstruct coefficient vector (1 x 507) using inverse_transform
            flat_recon = pls.inverse_transform(score.reshape(1, -1))[0]

            # Reshape to (169, 3) SPHARM coefficients
            coeffs_recon = flat_recon.reshape(expected_len, 3)

            # Evaluate SPHARM on grid
            X_grid, Y_grid, Z_grid = evaluate_spharm(
                coeffs_recon, L, np.radians(theta_deg), np.radians(phi_deg)
            )

            # Set output filename
            if w > 0:
                w_str = f"plus{w}SD"
            elif w < 0:
                w_str = f"minus{abs(w)}SD"
            else:
                w_str = "Mean"

            out_filename = os.path.join(comp_dir, f"PLS{comp_idx + 1}_{w_str}.vtk")

            # Save as VTK Structured Grid / PolyData
            save_grid_vtk(X_grid, Y_grid, Z_grid, theta_deg, phi_deg, out_filename)
            print(f"  Saved: {os.path.basename(out_filename)}")

    print("\n" + "="*60)
    print("Reconstruction Complete!")
    print(f"All VTK meshes saved to: {os.path.abspath(recon_dir)}")
    print("="*60)

if __name__ == "__main__":
    main()
