import json
import csv
import os
import glob

# 1. Setup Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
pca_json_path = os.path.join(SCRIPT_DIR, "output", "pca_results", "pca_model.json")
vtk_results_dir = os.path.join(SCRIPT_DIR, "output", "spharm_results")
output_csv_path = os.path.join(SCRIPT_DIR, "output", "pca_results", "pca_scores.csv")

def extract():
    print("="*60)
    print("--- PCA SCORE EXTRACTION (Final Fix) ---")
    print("="*60)
    
    # Get subject names from VTK files to ensure correctness
    vtk_files = sorted(glob.glob(os.path.join(vtk_results_dir, "*_SPHARM_ellalign.vtk")))
    subject_names = [os.path.basename(f).replace("_SPHARM_ellalign.vtk", "") for f in vtk_files]

    if not os.path.exists(pca_json_path):
        print(f"ERROR: JSON results not found at {pca_json_path}")
        print("Please run start_pca_batch.bat first.")
        return

    print(f"Loading results from {pca_json_path} (67MB, please wait)...")
    with open(pca_json_path, 'r') as f:
        root_data = json.load(f)

    # Access Group '0' or 'All' or root
    data = root_data.get('0', root_data.get('All', root_data))
    
    # Try all known keys for scores
    scores = data.get('data_projection', 
             data.get('projected_points', 
             data.get('scores', [])))

    if not scores:
        print("ERROR: Could not find any score data inside the JSON.")
        # Debugging: show keys in first level
        print(f"Available keys in data group: {list(data.keys())}")
        return

    # 4. Calculate Cumulative Variance to determine X components
    TARGET_VARIANCE = 0.95 # 95% threshold, edit this value if needed (e.g. 0.67)
    evr = data.get('explained_variance_ratio', [])
    num_pcs = len(scores[0])
    
    if evr:
        cumulative_variance = 0.0
        for x, var in enumerate(evr):
            cumulative_variance += var
            if cumulative_variance >= TARGET_VARIANCE:
                num_pcs = x + 1
                break
        print(f"Target Cumulative Variance: >= {TARGET_VARIANCE*100}%")
        print(f"Determined X = {num_pcs} components to cover {cumulative_variance*100:.2f}% of variation.")
    else:
        print("Warning: 'explained_variance_ratio' not found in JSON. Exporting all components.")

    # 5. Save to CSV
    print(f"Found {len(scores)} subjects. SAVING to CSV (PC1 to PC{num_pcs})...")
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Create Header: Subject, PC1, PC2, ...
        header = ['Subject'] + [f'PC{i+1}' for i in range(num_pcs)]
        writer.writerow(header)
        
        for i in range(min(len(subject_names), len(scores))):
            # Keep only the first `num_pcs` scores
            subject_scores = scores[i][:num_pcs]
            # Using 8 decimal places for EXCEL compatibility
            formatted_scores = ["{:.8f}".format(s) for s in subject_scores]
            writer.writerow([subject_names[i]] + formatted_scores)

    print("-" * 60)
    print(f"SUCCESS! Output saved to: {output_csv_path}")
    print(f"Extracted {len(scores)} samples and top {num_pcs} Principal Components.")
    print("-" * 60)
    print("You can now open 'pca_scores.csv' or run 'visualize_pca.py'.")

if __name__ == "__main__":
    extract()
