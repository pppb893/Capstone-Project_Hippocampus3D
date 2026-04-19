import os
import glob
import slicer
import time
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

def run_batch_spharm():
    # --- Configuration via Argparse ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Directory containing .nii.gz labels")
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

    if args.input_dir:
        input_dir = os.path.abspath(args.input_dir)
    else:
        # Route to the newly aligned volumes instead of the raw data!
        input_dir = os.path.join(output_root, "aligned_nifti")

    output_base_dir = os.path.join(output_root, "spharm_results")
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    log_file = os.path.join(output_base_dir, "spharm_progress.log")
    
    # Clear log at start
    with open(log_file, 'w') as f:
        f.write(f"--- SPHARM Batch Mode Start: {datetime.now()} ---\n")
        f.write(f"Input: {input_dir}\n")
        f.write(f"Output: {output_root}\n\n")

    # Template check
    mean_vtk = os.path.join(output_root, "spharm_mean", "mean_result_SPHARM.vtk").replace("\\", "/")
    # Get all medical volume files (.nii.gz, .nii, .hdr) anywhere inside input_dir
    extensions = ["*.nii.gz", "*.nii", "*.hdr"]
    file_list = []
    for ext in extensions:
        file_list.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))
    
    file_list = sorted(list(set(file_list))) # Unique and sorted
    
    # Smart Filter: Prioritize files with 'label' in name (Shape Analysis is done on segmentations)
    label_files = [f for f in file_list if "label" in os.path.basename(f).lower()]
    if label_files:
        sprint(f"Detected {len(label_files)} label-specific files (filtering out original images).", log_file)
        file_list = label_files

    sprint(f"Verified {len(file_list)} files for processing.", log_file)
    
    for i, file_path in enumerate(file_list):
        file_path = os.path.normpath(file_path).replace("\\", "/")
        basename = os.path.basename(file_path).split('.')[0]
        sprint(f"\n>>> [{i+1}/{len(file_list)}] STARTING: {basename}", log_file)
        
        # Define output paths
        pp_mask_path = os.path.join(output_base_dir, f"{basename}_pp.nrrd").replace("\\", "/")
        para_mesh_path = os.path.join(output_base_dir, f"{basename}_para.vtk").replace("\\", "/")
        surf_mesh_path = os.path.join(output_base_dir, f"{basename}_surf.vtk").replace("\\", "/")
        spharm_base = os.path.join(output_base_dir, basename).replace("\\", "/")
        
        input_node = None
        pp_node = None
        para_node = None
        surf_node = None
        
        try:
            # 1. Load Input
            sprint(f"  - Loading volume...", log_file)
            input_node = slicer.util.loadLabelVolume(file_path)
            if not input_node:
                sprint(f"  - FAILED to load: {file_path}", log_file)
                continue

            # 2. SegPostProcess
            sprint(f"  - STEP 1: SegPostProcess (Cleaning labels)...", log_file)
            pp_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"{basename}_pp")
            pp_params = {
                'fileName': input_node.GetID(),
                'outfileName': pp_node.GetID(),
                'label': 1
            }
            slicer.cli.run(slicer.modules.segpostprocessclp, None, pp_params, wait_for_completion=True)
            slicer.util.saveNode(pp_node, pp_mask_path)

            # 3. GenParaMesh
            sprint(f"  - STEP 2: GenParaMesh (Creating spherical mesh)...", log_file)
            para_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"{basename}_para")
            surf_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"{basename}_surf")
            para_params = {
                'infile': pp_node.GetID(),
                'outParaName': para_node.GetID(),
                'outSurfName': surf_node.GetID(),
                'numIterations': 500,
                'label': 1
            }
            slicer.cli.run(slicer.modules.genparameshclp, None, para_params, wait_for_completion=True)
            slicer.util.saveNode(para_node, para_mesh_path)
            slicer.util.saveNode(surf_node, surf_mesh_path)

            # 4. ParaToSPHARMMesh
            sprint(f"  - STEP 3: ParaToSPHARMMesh (Computing PDM)...", log_file)
            spharm_params = {
                'inParaFile': para_node.GetID(),
                'inSurfFile': surf_node.GetID(),
                'outbase': spharm_base,
                'subdivLevel': 10,
                'spharmDegree': 12
            }
            # If template exists, add it for alignment
            if os.path.exists(mean_vtk):
                spharm_params['regTemplateFile'] = mean_vtk

            slicer.cli.run(slicer.modules.paratospharmmeshclp, None, spharm_params, wait_for_completion=True)
            
            final_vtk = f"{spharm_base}_SPHARM.vtk"
            if os.path.exists(final_vtk):
                sprint(f"  - SUCCESS: {basename} completed.", log_file)
            else:
                sprint(f"  - ERROR: Result VTK not generated for {basename}.", log_file)

        except Exception as e:
            sprint(f"  - CRITICAL ERROR for {basename}: {str(e)}", log_file)
        
        finally:
            # Cleanup nodes
            for node in [input_node, pp_node, para_node, surf_node]:
                if node:
                    try: slicer.mrmlScene.RemoveNode(node)
                    except: pass
            sprint(f"  - Cleanup done.", log_file)

    sprint("\n" + "="*60, log_file)
    sprint("!!! ALL BATCH PROCESSING COMPLETED !!!", log_file)
    sprint("="*60, log_file)

if __name__ == "__main__":
    run_batch_spharm()
    try:
        import qt
        print("Closing SlicerSALT...")
        qt.QTimer.singleShot(500, slicer.util.exit)
    except:
        import sys
        sys.exit(0)
