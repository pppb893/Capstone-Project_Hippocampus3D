import os
import sys
import glob
import csv
import re
import argparse
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import NearestNeighbors

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
    is_left_side = subject_name.startswith("left_")
    
    # 1. Healthy Control
    if "_Healthy" in subject_name or "HFH_" in subject_name:
        return "Healthy Control", 0
    
    # 2. Ipsilateral TLE (Diseased)
    elif (is_left_side and "_Left-TLE" in subject_name) or (not is_left_side and "_Right-TLE" in subject_name):
        return "Ipsilateral TLE (Diseased)", 1
        
    # 3. Contralateral TLE (Healthy-side)
    elif (is_left_side and "_Right-TLE" in subject_name) or (not is_left_side and "_Left-TLE" in subject_name):
        return "Contralateral TLE (Healthy-side)", 2
        
    return "Unknown", -1

def main():
    print("="*60)
    print("--- PLS-DA Local Interpolation Augmentation ---")
    print("="*60)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory containing spharm_results/")
    parser.add_argument("--n_components", type=int, default=10,
                        help="Number of PLS-DA components to fit (default=10)")
    parser.add_argument("--num_augmented", type=int, default=None,
                        help="Number of interpolated meshes to generate (prompts if None)")
    parser.add_argument("--k_neighbors", type=int, default=5,
                        help="Number of nearest neighbors to define 'close' pairs (default=5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args, unknown = parser.parse_known_args()

    np.random.seed(args.seed)

    # Interactive prompt if num_augmented is not provided via CLI
    num_augmented = args.num_augmented
    if num_augmented is None:
        while True:
            try:
                user_input = input("Enter the number of interpolated meshes to generate: ")
                num_augmented = int(user_input)
                if num_augmented <= 0:
                    print("Please enter a positive integer.")
                    continue
                break
            except ValueError:
                print("Invalid input. Please enter a valid integer.")

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
    if not os.path.exists(spharm_results_dir):
        print(f"ERROR: spharm_results directory not found under output root: {output_root}")
        return

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
        
        subject_names.append(basename.replace("_SPHARM.coef", ""))
        g_name, cls = classify_subject(basename)
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

    # Fit K-Nearest Neighbors on the PLS-DA scores
    print(f"Finding {args.k_neighbors} nearest neighbors for each subject...")
    # n_neighbors = k_neighbors + 1 (since the point itself is neighbor #1)
    nn = NearestNeighbors(n_neighbors=args.k_neighbors + 1, metric='euclidean')
    nn.fit(X_scores)
    _, nn_indices = nn.kneighbors(X_scores)

    # Setup output directories
    aug_dir = os.path.join(output_root, "plsda_interpolated_surfaces")
    healthy_out = os.path.join(aug_dir, "Healthy")
    diseased_out = os.path.join(aug_dir, "Diseased")
    
    for path in [healthy_out, diseased_out]:
        if not os.path.exists(path):
            os.makedirs(path)

    # Grid parameters (defaults)
    theta_step = 9.0
    phi_step = 9.0
    theta_deg = np.linspace(0, 180, int(180/theta_step) + 1)
    phi_deg = np.linspace(0, 360, int(360/phi_step) + 1)[:-1]

    # CSV metadata file to document the parents and interpolation factor
    metadata_csv_path = os.path.join(aug_dir, "interpolated_metadata.csv")
    metadata_rows = []

    # To prevent duplicate meshes (same pair and close alpha)
    generated_shapes = {} # maps (min_idx, max_idx) -> list of alphas

    print(f"\nGenerating {num_augmented} interpolated meshes...")
    for i in range(num_augmented):
        while True:
            # 1. Randomly select subject A
            idx_A = np.random.randint(0, N)
            
            # 2. Randomly select subject B from A's nearest neighbors (excluding index A itself)
            neighbors_of_A = nn_indices[idx_A, 1:] # Exclude index 0 which is A itself
            idx_B = np.random.choice(neighbors_of_A)

            # 3. Sample random weight alpha from Uniform(0, 1)
            alpha = np.random.rand()

            # Check uniqueness of shape
            pair_key = (min(idx_A, idx_B), max(idx_A, idx_B))
            if pair_key in generated_shapes:
                # If this pair was used, make sure the new alpha is at least 0.01 away from all previous alphas
                too_close = False
                for prev_alpha in generated_shapes[pair_key]:
                    if abs(alpha - prev_alpha) < 0.01:
                        too_close = True
                        break
                if too_close:
                    continue  # Re-sample pair and alpha
                else:
                    generated_shapes[pair_key].append(alpha)
                    break
            else:
                generated_shapes[pair_key] = [alpha]
                break

        # 4. Interpolate scores
        score_A = X_scores[idx_A]
        score_B = X_scores[idx_B]
        score_new = (1 - alpha) * score_A + alpha * score_B

        # 5. Class assignment based on proximity (closer to A vs closer to B)
        if alpha < 0.5:
            # Closer to A
            assigned_class = binary_labels[idx_A]
            assigned_group = groups[idx_A]
            assigned_parent = subject_names[idx_A]
        else:
            # Closer to B
            assigned_class = binary_labels[idx_B]
            assigned_group = groups[idx_B]
            assigned_parent = subject_names[idx_B]

        # Map assigned group clean directory label
        dir_name = "Diseased" if assigned_class == 1 else "Healthy"
        out_path = diseased_out if assigned_class == 1 else healthy_out

        # 6. Reconstruct coefficients (1 x 507)
        flat_recon = pls.inverse_transform(score_new.reshape(1, -1))[0]
        coeffs_recon = flat_recon.reshape(expected_len, 3)

        # Evaluate SPHARM grid
        X_grid, Y_grid, Z_grid = evaluate_spharm(
            coeffs_recon, L, np.radians(theta_deg), np.radians(phi_deg)
        )

        # Save mesh
        filename = f"interp_{dir_name}_{i+1:03d}.vtk"
        filepath = os.path.join(out_path, filename)
        save_grid_vtk(X_grid, Y_grid, Z_grid, theta_deg, phi_deg, filepath)

        # Log metadata row
        metadata_rows.append([
            filename,
            dir_name,
            subject_names[idx_A],
            groups[idx_A],
            subject_names[idx_B],
            groups[idx_B],
            f"{alpha:.4f}",
            assigned_parent,
            assigned_group
        ])

        if (i + 1) % 10 == 0 or (i + 1) == num_augmented:
            print(f"  Progress: {i+1}/{num_augmented} meshes generated")

    # Write metadata CSV
    with open(metadata_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Filename", "DirectoryClass", "Parent_A_Subject", "Parent_A_Group",
            "Parent_B_Subject", "Parent_B_Group", "Alpha", "Closest_Parent", "Assigned_Group"
        ])
        writer.writerows(metadata_rows)

    print("\n" + "="*60)
    print("Interpolation Augmentation Complete!")
    print(f"Metadata file saved to:     {metadata_csv_path}")
    print(f"Healthy meshes saved to:    {os.path.abspath(healthy_out)}")
    print(f"Diseased meshes saved to:   {os.path.abspath(diseased_out)}")
    print("="*60)

if __name__ == "__main__":
    main()
