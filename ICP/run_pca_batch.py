import os
import glob
import csv
import slicer
import time
import json
import sys
import argparse
from datetime import datetime

# --- Utility Functions ---
def sprint(msg, log_file):
    print(msg)
    sys.stdout.flush()
    timestamp = datetime.now().strftime("%H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] {msg}\n")

def run_pca_analysis():
    # --- Configuration via Argparse ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="Directory to save all results")
    args, unknown = parser.parse_known_args()

    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SCRIPT_DIR = os.getcwd()

    if args.output_dir:
        output_root = os.path.abspath(args.output_dir)
    else:
        output_root = os.path.join(SCRIPT_DIR, "output")

    pca_output_dir = os.path.join(output_root, "pca_results")
    if not os.path.exists(pca_output_dir):
        os.makedirs(pca_output_dir)

    log_file = os.path.join(pca_output_dir, "progress.log")
    
    # Clear old log
    with open(log_file, 'w') as f:
        f.write(f"--- PCA Log Started: {datetime.now()} ---\n")
        f.write(f"Output: {output_root}\n\n")

    # --- Setup Paths ---
    results_dir = os.path.join(output_root, "spharm_results")
    output_csv = os.path.join(pca_output_dir, "pca_scores.csv")
    
    sprint("="*60, log_file)
    sprint("--- SlicerSALT PCA ANALYSIS ---", log_file)
    sprint("="*60, log_file)

    # 1. CLEANUP OLD FILES
    if os.path.exists(output_csv):
        try: os.remove(output_csv)
        except: pass

    # 2. FIND VTK FILES
    vtk_files = sorted(glob.glob(os.path.join(results_dir, "*_SPHARM_ellalign.vtk")))
    if not vtk_files:
        sprint(f"ERROR: No aligned VTK files found in {results_dir}", log_file)
        return

    sprint(f"Verified {len(vtk_files)} subjects.", log_file)

    # 3. CREATE INPUT CSV
    csv_input_path = os.path.join(pca_output_dir, "pca_input.csv")
    with open(csv_input_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Path", "Group"])
        for vtk_file in vtk_files:
            writer.writerow([vtk_file.replace("\\", "/"), 0])

    # 4. RUN PCA MODULE
    save_json = os.path.join(pca_output_dir, "pca_model.json")
    parameters = {
        'inputCsv': csv_input_path,
        'outputJson': save_json,
        'evaluation': 0,
        'shapeNum': 50
    }

    sprint("Executing ShapePCA module... Please wait (Check this log for %)", log_file)

    try:
        cli_node = slicer.cli.run(slicer.modules.shapepca, None, parameters)
    except AttributeError:
        sprint("Searching for PCA module alternatives...", log_file)
        available = [m for m in dir(slicer.modules) if 'pca' in m.lower()]
        if available:
            cli_node = slicer.cli.run(getattr(slicer.modules, available[0]), None, parameters)
        else:
            sprint("CRITICAL ERROR: No PCA module found in this SlicerSALT.", log_file)
            return

    last_progress = -1
    while cli_node.IsBusy():
        progress = int(cli_node.GetProgress() * 100)
        if progress != last_progress:
            sprint(f"  Calculation Progress: {progress}%", log_file)
            last_progress = progress
        time.sleep(1)

    # 5. EXTRACT SCORES
    if cli_node.GetStatusString() == 'Completed' and os.path.exists(save_json):
        sprint("--- Computation SUCCESS! Extracting scores to CSV ---", log_file)
        try:
            with open(save_json, 'r') as f:
                data = json.load(f)
            scores_data = data.get('All', data.get('0', data))
            scores = scores_data.get('data_projection', 
                     scores_data.get('projected_points', 
                     scores_data.get('scores', [])))
            if scores:
                with open(output_csv, 'w', newline='') as f:
                    writer = csv.writer(f)
                    header = ["Subject"] + [f"PC{i+1}" for i in range(len(scores[0]))]
                    writer.writerow(header)
                    for i in range(min(len(scores), len(vtk_files))):
                        subject_id = os.path.basename(vtk_files[i]).replace("_SPHARM_ellalign.vtk", "")
                        row = [subject_id] + ["{:.8f}".format(s) for s in scores[i]]
                        writer.writerow(row)
                sprint(f"DONE: {len(scores)} subjects exported to {output_csv}", log_file)
                sprint("="*60, log_file)
                sprint("PCA ANALYSIS COMPLETED!", log_file)
                sprint("="*60, log_file)
            else:
                sprint("Warning: No scores found in JSON.", log_file)
        except Exception as e:
            sprint(f"Error during extraction: {e}", log_file)
    else:
        sprint(f"FAILED: {cli_node.GetStatusString()}", log_file)
        sprint(f"Error Detail: {cli_node.GetErrorText()}", log_file)

if __name__ == "__main__":
    run_pca_analysis()
    try:
        import qt
        print("Closing SlicerSALT...")
        qt.QTimer.singleShot(500, slicer.util.exit)
    except:
        import sys
        sys.exit(0)
