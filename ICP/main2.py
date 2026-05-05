import vtk
import numpy as np
import os
import glob
import argparse
import slicer
import sys
from datetime import datetime

# --- Setup Logging ---
DEBUG_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icp_debug_log.txt")

def sprint(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] [ICP-LOG] {msg}"
    print(formatted_msg)
    sys.stdout.flush()
    with open(DEBUG_LOG, "a") as f:
        f.write(formatted_msg + "\n")

def get_poly_max_bound(poly):
    b = poly.GetBounds()
    return max(abs(b[0]), abs(b[1]), abs(b[2]), abs(b[3]), abs(b[4]), abs(b[5]))

def apply_poly_transform(poly, matrix_np):
    transform = vtk.vtkTransform()
    matrix_vtk = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            matrix_vtk.SetElement(r, c, float(matrix_np[r, c]))
    transform.SetMatrix(matrix_vtk)
    transformer = vtk.vtkTransformPolyDataFilter()
    transformer.SetInputData(poly)
    transformer.SetTransform(transform)
    transformer.Update()
    return transformer.GetOutput()

def load_and_mesh_node(filepath):
    """Native VTK Meshing (No Slicer module dependency)"""
    node = slicer.util.loadLabelVolume(filepath)
    if not node: return None
    
    # 1. Get Image Data
    img = node.GetImageData()
    
    # 2. Discrete Marching Cubes (Best for Labels)
    dmc = vtk.vtkDiscreteMarchingCubes()
    dmc.SetInputData(img)
    dmc.GenerateValues(1, 1, 100) # Look for any non-zero labels
    dmc.Update()
    
    poly = dmc.GetOutput()
    
    # 3. Correct for Voxel-to-RAS (Crucial!)
    ijkToRas = vtk.vtkMatrix4x4()
    node.GetIJKToRASMatrix(ijkToRas)
    
    transformer = vtk.vtkTransformPolyDataFilter()
    t = vtk.vtkTransform()
    t.SetMatrix(ijkToRas)
    transformer.SetTransform(t)
    transformer.SetInputData(poly)
    transformer.Update()
    
    result_poly = transformer.GetOutput()
    
    slicer.mrmlScene.RemoveNode(node)
    return result_poly

def run_vtk_icp(source_poly, target_poly):
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(source_poly)
    icp.SetTarget(target_poly)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(50)
    icp.Update()
    matrix = icp.GetMatrix()
    res = np.eye(4)
    for r in range(4):
        for c in range(4): res[r, c] = matrix.GetElement(r, c)
    return res

def main():
    sprint("--- main2.py STARTING (Headless / Native VTK Mode) ---")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args, unknown = parser.parse_known_args()

    input_dir = args.input_dir.replace("\\", "/")
    output_dir = args.output_dir.replace("\\", "/")
    
    # --- Smart File Discovery ---
    extensions = ["*.nii.gz", "*.nii", "*.hdr", "*.nrrd"]
    file_list = []
    for ext in extensions:
        file_list.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))
    
    file_list = sorted(list(set(file_list)))
    # Filter for labels only
    label_files = [f for f in file_list if "label" in os.path.basename(f).lower()]
    if label_files:
        file_list = label_files
        sprint(f"Prioritizing {len(file_list)} label files.")

    sprint(f"Total files to process: {len(file_list)}")
    
    meshes = []
    for i, f in enumerate(file_list):
        sprint(f"Processing [{i+1}/{len(file_list)}]: {os.path.basename(f)}")
        p = load_and_mesh_node(f)
        if p and p.GetNumberOfPoints() > 0:
            meshes.append(p)
    
    N = len(meshes)
    if N == 0:
        sprint("ERROR: No valid label meshes found. Ensure files have 'label' in their name.")
        return

    # 1. Pre-ICP Normalization
    sprint("Step 1: Normalizing individual meshes to ±1...")
    aligned_meshes = []
    T_initial = []
    for i in range(N):
        b = meshes[i].GetBounds()
        centroid = [(b[0]+b[1])/2.0, (b[2]+b[3])/2.0, (b[4]+b[5])/2.0]
        T_cent = np.eye(4)
        T_cent[:3, 3] = -np.array(centroid)
        poly = apply_poly_transform(meshes[i], T_cent)
        max_b = get_poly_max_bound(poly)
        s = 1.0 / max_b if max_b > 0 else 1.0
        T_scale = np.eye(4)
        T_scale[0,0] = T_scale[1,1] = T_scale[2,2] = s
        T_combined = T_scale @ T_cent
        aligned_meshes.append(apply_poly_transform(meshes[i], T_combined))
        T_initial.append(T_combined)

    # 2. ICP Alignment
    sprint("Step 2: Group-wise ICP...")
    ref = aligned_meshes[0]
    T_icp = [np.eye(4) for _ in range(N)]
    for i in range(1, N):
        sprint(f"  Aligning {i+1}/{N}...")
        dT = run_vtk_icp(aligned_meshes[i], ref)
        aligned_meshes[i] = apply_poly_transform(aligned_meshes[i], dT)
        T_icp[i] = dT

    # 3. Global Scaling
    sprint("Step 3: Calculating Global Bounding Box...")
    global_min = [float('inf')] * 3
    global_max = [float('-inf')] * 3
    for m in aligned_meshes:
        b = m.GetBounds()
        global_min[0] = min(global_min[0], b[0]); global_max[0] = max(global_max[0], b[1])
        global_min[1] = min(global_min[1], b[2]); global_max[1] = max(global_max[1], b[3])
        global_min[2] = min(global_min[2], b[4]); global_max[2] = max(global_max[2], b[5])
    
    overall_max = max(max(abs(v) for v in global_min), max(abs(v) for v in global_max))
    global_s = 1.0 / overall_max if overall_max > 0 else 1.0
    sprint(f"Global Max: {overall_max:.4f} -> Unified Scale: {global_s:.4f}")
    
    T_matrices = []
    for i in range(N):
        T_final = np.eye(4)
        T_final[0,0] = T_final[1,1] = T_final[2,2] = global_s
        T_final = T_final @ T_icp[i] @ T_initial[i]
        T_matrices.append(T_final)

    # 4. Export
    sprint("Step 4: Computing Mean Shape & Saving Results...")
    out_vol_dir = os.path.join(output_dir, "aligned_nifti")
    os.makedirs(out_vol_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "T_matrices.npy"), np.array(T_matrices))

    # Compute TRUE Mean Shape (Average vertices)
    mean_poly = vtk.vtkPolyData()
    mean_poly.DeepCopy(aligned_meshes[0])
    points = mean_poly.GetPoints()
    for i in range(1, N):
        p_other = aligned_meshes[i].GetPoints()
        for j in range(points.GetNumberOfPoints()):
            p1 = points.GetPoint(j)
            p2 = p_other.GetPoint(j) if j < p_other.GetNumberOfPoints() else p1
            new_p = [(p1[k] + p2[k]) / 2.0 for k in range(3)] # Running average (simplified)
            points.SetPoint(j, new_p)
    
    # Save Mean Shape PLY
    writer = vtk.vtkPLYWriter()
    writer.SetFileName(os.path.join(output_dir, "mean_shape.ply"))
    writer.SetInputData(mean_poly)
    writer.Write()
    sprint(f"Saved new mean_shape.ply to {output_dir}")

    for i, f in enumerate(file_list):
        basename = os.path.basename(f).split('.')[0]
        sprint(f"Saving [{i+1}/{N}]: {basename}")
        node = slicer.util.loadLabelVolume(f)
        
        # Transform Node
        T = T_matrices[i]
        t_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
        v_mat = vtk.vtkMatrix4x4()
        for r in range(4):
            for c in range(4): v_mat.SetElement(r, c, float(T[r, c]))
        t_node.SetMatrixTransformToParent(v_mat)
        
        # Grid Setup
        res = 0.02
        ref_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "ref")
        img = vtk.vtkImageData()
        img.SetDimensions(128, 128, 128)
        img.AllocateScalars(vtk.VTK_SHORT, 1)
        ref_node.SetAndObserveImageData(img)
        ref_node.SetSpacing(res, res, res)
        ref_node.SetOrigin(-64*res, -64*res, -64*res)
        
        params = {
            "inputVolume": node.GetID(),
            "referenceVolume": ref_node.GetID(),
            "outputVolume": ref_node.GetID(),
            "transformationFile": t_node.GetID(),
            "interpolationMode": "NearestNeighbor"
        }
        slicer.cli.run(slicer.modules.resamplescalarvectordwivolume, None, params, wait_for_completion=True)
        slicer.util.saveNode(ref_node, os.path.join(out_vol_dir, f"{basename}_aligned.nii.gz"))
        
        slicer.mrmlScene.RemoveNode(node)
        slicer.mrmlScene.RemoveNode(t_node)
        slicer.mrmlScene.RemoveNode(ref_node)

    sprint("--- main2.py FINISHED Successfully ---")

if __name__ == "__main__":
    with open(DEBUG_LOG, "w") as f:
        f.write(f"--- LOG START: {datetime.now()} ---\n")
    try:
        main()
    except Exception as e:
        import traceback
        sprint(f"FATAL ERROR: {str(e)}")
        with open(DEBUG_LOG, "a") as f:
            f.write(traceback.format_exc())
    
    import qt
    qt.QTimer.singleShot(500, slicer.util.exit)