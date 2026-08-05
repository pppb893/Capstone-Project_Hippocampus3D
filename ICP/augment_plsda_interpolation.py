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
# Write SPHARM-PDM .coef format
# =============================================================================
def save_coef(coeffs, filepath):
    num_coeffs = len(coeffs)
    with open(filepath, 'w') as f:
        # Write first line
        f.write(f"{{ {num_coeffs},{{{coeffs[0][0]:.6f}, {coeffs[0][1]:.6f}, {coeffs[0][2]:.6f}}},\n")
        # Write middle lines
        for i in range(1, num_coeffs - 1):
            f.write(f"{{{coeffs[i][0]:.6f}, {coeffs[i][1]:.6f}, {coeffs[i][2]:.6f}}},\n")
        # Write last line
        f.write(f"{{{coeffs[-1][0]:.6f}, {coeffs[-1][1]:.6f}, {coeffs[-1][2]:.6f}}}}}")

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
    print("--- PLS-DA Local Interpolation Augmentation ---")
    print("="*60)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory containing spharm_results/")
    parser.add_argument("--n_components", type=int, default=10,
                        help="Number of PLS-DA components to fit (default=10)")
    parser.add_argument("--num_per_pair", type=int, default=None,
                        help="Number of children (interpolations) to generate per parent pair (prompts if None)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args, unknown = parser.parse_known_args()

    np.random.seed(args.seed)

    # Interactive prompt if num_per_pair is not provided via CLI
    num_per_pair = args.num_per_pair
    if num_per_pair is None:
        while True:
            try:
                user_input = input("Enter the number of children to generate per parent pair (e.g. 8): ")
                num_per_pair = int(user_input)
                if num_per_pair <= 0:
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

    # Check if selected folder is directly the directory containing coef files, or contains 'spharm_results'
    if os.path.isdir(os.path.join(output_root, "spharm_results")):
        coef_dir = os.path.join(output_root, "spharm_results")
    else:
        coef_dir = output_root

    # Find .coef files
    all_coef_files = sorted(glob.glob(os.path.join(coef_dir, "*_SPHARM.coef")))
    coef_files = [f for f in all_coef_files
                  if not any(s in os.path.basename(f)
                             for s in ("_ellalign", "_grid", "_realigned", "_procalign", "_pca_ready"))]

    if not coef_files:
        print(f"ERROR: No SPHARM coefficient files (*_SPHARM.coef) found in: {coef_dir}")
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

    # NearestNeighbors fitting is not needed since we calculate absolute closest candidate dynamically within the remaining pool

    # Setup output directory
    aug_dir = os.path.join(output_root, "plsda_interpolated_surfaces")
    os.makedirs(aug_dir, exist_ok=True)

    # Grid parameters (defaults)
    theta_step = 9.0
    phi_step = 9.0
    theta_deg = np.linspace(0, 180, int(180/theta_step) + 1)
    phi_deg = np.linspace(0, 360, int(360/phi_step) + 1)[:-1]

    # CSV metadata file to document the parents and interpolation factor
    metadata_csv_path = os.path.join(aug_dir, "interpolated_metadata.csv")
    metadata_rows = []

    # Prepare a single pool containing all subjects (allowing parents from different classes)
    pool = list(range(N))

    # Function to form pairs from a pool using closest distance
    def form_pairs(pool):
        formed = []
        temp_pool = list(pool)
        while len(temp_pool) >= 2:
            # 1. Randomly select subject A from the remaining pool
            idx_A = np.random.choice(temp_pool)
            temp_pool.remove(idx_A)
            
            # 2. Find the closest neighbor B to A among the remaining pool in score space
            distances = [np.linalg.norm(X_scores[idx_A] - X_scores[cand]) for cand in temp_pool]
            min_idx = np.argmin(distances)
            idx_B = temp_pool[min_idx]
            temp_pool.remove(idx_B)
            
            formed.append((idx_A, idx_B))
        return formed

    all_pairs = form_pairs(pool)
    
    total_pairs = len(all_pairs)
    print(f"\nFormed {total_pairs} parent pairs from the entire dataset pool.")
    num_augmented = total_pairs * num_per_pair
    print(f"Generating {num_per_pair} children per pair. Total output: {num_augmented} meshes.")

    global_idx = 1
    for pair_idx, (idx_A, idx_B) in enumerate(all_pairs):
        # Generate evenly spaced alphas with small random jitter
        alphas = np.linspace(0.1, 0.9, num_per_pair)
        if num_per_pair > 1:
            jitter = np.random.uniform(-0.02, 0.02, num_per_pair)
            alphas = np.clip(alphas + jitter, 0.05, 0.95)

        for child_idx, alpha in enumerate(alphas):
            # 4. Interpolate scores
            score_A = X_scores[idx_A]
            score_B = X_scores[idx_B]
            score_new = (1 - alpha) * score_A + alpha * score_B

            # 5. Class and parent assignment based on proximity (closer to A vs closer to B)
            if alpha < 0.5:
                assigned_class = binary_labels[idx_A]
                assigned_group = groups[idx_A]
                assigned_parent = subject_names[idx_A]
            else:
                assigned_class = binary_labels[idx_B]
                assigned_group = groups[idx_B]
                assigned_parent = subject_names[idx_B]

            dir_name = "Diseased" if assigned_class == 1 else "Healthy"
            
            # Determine side (left/right) from parent A's name
            parent_name = subject_names[idx_A]
            parent_name_upper = parent_name.upper()
            is_left = parent_name.startswith("left_") or parent_name.startswith("lh_") or "_lh" in parent_name.lower() or "left" in parent_name.lower()
            
            if parent_name.startswith("lh_") or "_lh_" in parent_name.lower():
                side_str = "lh_"
            elif parent_name.startswith("rh_") or "_rh_" in parent_name.lower():
                side_str = "rh_"
            else:
                side_str = "left_" if is_left else "right_"
            
            # Determine group prefix similar to parents
            if assigned_class == 0:
                class_prefix = "Normal" if "NORMAL" in parent_name_upper else "Healthy"
            else:
                if "LEFT-TLE" in parent_name_upper or "RIGHT-TLE" in parent_name_upper:
                    class_prefix = "Left-TLE" if is_left else "Right-TLE"
                else:
                    class_prefix = "TLE"

            # 6. Reconstruct coefficients (1 x D)
            flat_recon = pls.inverse_transform(score_new.reshape(1, -1))[0]
            coeffs_recon = flat_recon.reshape(expected_len, 3)

            # Evaluate SPHARM grid
            X_grid, Y_grid, Z_grid = evaluate_spharm(
                coeffs_recon, L, np.radians(theta_deg), np.radians(phi_deg)
            )

            # Save mesh (flat layout with prefixed names)
            filename = f"{side_str}{class_prefix}_interp_{global_idx:04d}.vtk"
            filepath = os.path.join(aug_dir, filename)
            save_grid_vtk(X_grid, Y_grid, Z_grid, theta_deg, phi_deg, filepath)

            # Save reconstructed coefficients in SPHARM .coef format
            coef_filename = f"{side_str}{class_prefix}_interp_{global_idx:04d}_SPHARM.coef"
            coef_filepath = os.path.join(aug_dir, coef_filename)
            save_coef(coeffs_recon, coef_filepath)

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

            if global_idx % 20 == 0 or global_idx == num_augmented:
                print(f"  Progress: {global_idx}/{num_augmented} meshes generated")
            global_idx += 1

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
    print(f"Augmented meshes saved to:  {os.path.abspath(aug_dir)}")
    print("="*60)

if __name__ == "__main__":
    main()
