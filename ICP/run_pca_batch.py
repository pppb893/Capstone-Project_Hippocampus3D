import os
import sys

# =============================================================================
# BOOTSTRAP: ถ้ารันด้วย Python ปกติ -> re-launch ผ่าน SlicerSALT
# (slicer module อยู่ใน SlicerSALT เท่านั้น)
# =============================================================================

def _bootstrap_slicer():
    try:
        import slicer  # noqa: F401
        return
    except ImportError:
        pass

    slicer_exe = os.environ.get(
        "SLICER_EXE",
        r"C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe",
    )
    if not os.path.exists(slicer_exe):
        print(f"[ERROR] SlicerSALT not found at: {slicer_exe}")
        print("        Set env var SLICER_EXE or edit the default path.")
        sys.exit(1)

    import subprocess
    script_path = os.path.abspath(__file__)
    cmd = [slicer_exe, "--no-main-window", "--no-splash",
           "--python-script", script_path] + sys.argv[1:]
    print(f"[INFO] No 'slicer' module here. Re-launching via SlicerSALT:")
    print(f"       {slicer_exe}")
    sys.exit(subprocess.call(cmd))


_bootstrap_slicer()


# =============================================================================
# ใน SlicerSALT แล้ว -> import เต็มได้
# =============================================================================

import glob
import csv
import slicer
import time
import json
import argparse
from datetime import datetime


def prompt_folder(title):
    """เปิด QFileDialog ให้ user เลือก folder (Qt ที่ฝังใน SlicerSALT)"""
    import qt
    folder = qt.QFileDialog.getExistingDirectory(None, title)
    return folder if folder else None

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
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory containing spharm_results/ (output ของ main2.py). "
                             "ถ้าไม่ระบุ จะเด้ง folder picker ให้เลือก")
    args, unknown = parser.parse_known_args()

    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SCRIPT_DIR = os.getcwd()

    if args.output_dir:
        output_root = os.path.abspath(args.output_dir)
    else:
        print("No --output_dir given. Opening folder picker...")
        chosen = prompt_folder(
            "Select output folder (มี spharm_results/ อยู่ข้างใน — เช่น output_left_hippocampus)"
        )
        if not chosen:
            print("ERROR: No folder selected. Exiting.")
            return
        # รองรับกรณี user เลือก spharm_results โดยตรง
        if os.path.basename(chosen.rstrip("\\/")) == "spharm_results":
            output_root = os.path.abspath(os.path.dirname(chosen))
        else:
            output_root = os.path.abspath(chosen)
        print(f"Output root: {output_root}")

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
    vtk_files = sorted(glob.glob(os.path.join(results_dir, "*_SPHARM_pca_ready.vtk")))
    if not vtk_files:
        vtk_files = sorted(glob.glob(os.path.join(results_dir, "*_SPHARM_realigned.vtk")))
    if not vtk_files:
        vtk_files = sorted(glob.glob(os.path.join(results_dir, "*_SPHARM_procalign.vtk")))
    if not vtk_files:
        vtk_files = sorted(glob.glob(os.path.join(results_dir, "*_SPHARM_ellalign.vtk")))
    
    if vtk_files:
        sprint(f"Using {len(vtk_files)} meshes: {os.path.basename(vtk_files[0])}", log_file)
    else:
        sprint(f"ERROR: No aligned VTK files found in {results_dir}", log_file)
        return

    # 3. CREATE INPUT CSV
    csv_input_path = os.path.join(pca_output_dir, "pca_input.csv")
    with open(csv_input_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Path", "Group"])
        for vtk_file in vtk_files:
            writer.writerow([vtk_file.replace("\\", "/"), 0])

    # 4. RUN PCA MODULE
    # shapeNum ต้องไม่เกิน (N - 1) — PCA produce ได้สูงสุด min(N-1, features) modes
    # ถ้า hardcode 50 แต่มี subject < 50 จะ overspec และอาจ error / warning จาก SlicerSALT
    save_json = os.path.join(pca_output_dir, "pca_model.json")
    shape_num = max(1, min(50, len(vtk_files) - 1))
    sprint(f"Using shapeNum = {shape_num} (from {len(vtk_files)} subjects)", log_file)
    parameters = {
        'inputCsv': csv_input_path,
        'outputJson': save_json,
        'evaluation': 0,
        'shapeNum': shape_num
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
                        subject_id = os.path.basename(vtk_files[i])
                        for suf in ("_SPHARM_realigned.vtk", "_SPHARM_ellalign.vtk"):
                            subject_id = subject_id.replace(suf, "")
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
    # ทำไม os._exit forced: --no-main-window mode บางครั้ง Qt event loop ไม่ start
    #   -> slicer.util.exit() ที่ใช้ QTimer ไม่ trigger -> Slicer process ค้าง
    #   -> batch script รอ exit ไม่จบ (ดูเหมือน Slicer "เด้ง")
    #   forced os._exit() = kill process ทันที, รับประกัน batch ไปต่อได้
    import sys, os
    exit_code = 0
    try:
        run_pca_analysis()
    except Exception:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        try:
            slicer.util.exit(exit_code)
        except Exception:
            pass
        try:
            import time, threading
            def _force_kill():
                time.sleep(1.5)
                os._exit(exit_code)
            threading.Thread(target=_force_kill, daemon=True).start()
        except Exception:
            os._exit(exit_code)
