import os
import glob
import json
import sys
import argparse
import vtk
import slicer
import numpy as np

# ============================================================
#  Utility Functions (Slicer/VTK Based)
# ============================================================

def load_and_mesh_node(filepath):
    """Load a label volume and convert it to a mesh using VTK in Slicer."""
    print(f"Loading {os.path.basename(filepath)}...")
    
    # 1. Load Label Volume
    node = slicer.util.loadLabelVolume(filepath)
    if not node:
        print(f"Failed to load: {filepath}")
        return None, None

    # 2. Extract Mesh using Discrete Marching Cubes (Best for labels)
    marching_cubes = vtk.vtkDiscreteMarchingCubes()
    marching_cubes.SetInputData(node.GetImageData())
    marching_cubes.SetValue(0, 1) # Extract label 1
    marching_cubes.Update()
    
    polydata = marching_cubes.GetOutput()
    
    # Optional: Transform to correctly match volume coordinates (IJK to RAS)
    mask_to_ras = vtk.vtkMatrix4x4()
    node.GetIJKToRASMatrix(mask_to_ras)
    
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform = vtk.vtkTransform()
    transform.SetMatrix(mask_to_ras)
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(polydata)
    transform_filter.Update()
    
    result_poly = transform_filter.GetOutput()
    
    # Cleanup node
    slicer.mrmlScene.RemoveNode(node)
    
    return result_poly

def get_poly_centroid(polydata):
    """Calculate the geometric centroid of a polydata."""
    points = polydata.GetPoints()
    n = points.GetNumberOfPoints()
    if n == 0: return np.zeros(3)
    
    center = np.zeros(3)
    for i in range(n):
        pt = points.GetPoint(i)
        center += np.array(pt)
    return center / n

def get_poly_max_bound(polydata):
    """Calculate the maximum absolute coordinate (bounding box distance from origin)."""
    bounds = polydata.GetBounds() # (xmin, xmax, ymin, ymax, zmin, zmax)
    max_bound = max(abs(b) for b in bounds)
    return max_bound

def apply_poly_transform(polydata, matrix_np):
    """Apply a 4x4 numpy matrix transform to polydata."""
    vtk_matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtk_matrix.SetElement(i, j, float(matrix_np[i, j]))
            
    transform = vtk.vtkTransform()
    transform.SetMatrix(vtk_matrix)
    
    filter = vtk.vtkTransformPolyDataFilter()
    filter.SetTransform(transform)
    filter.SetInputData(polydata)
    filter.Update()
    return filter.GetOutput()

def run_vtk_icp(source_poly, target_poly, max_iters=20):
    """Run ICP using VTK's native iterative closest point transform."""
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(source_poly)
    icp.SetTarget(target_poly)
    icp.GetLandmarkTransform().SetModeToRigidBody() # Rotation + Translation
    icp.SetMaximumNumberOfIterations(max_iters)
    icp.Update()
    
    matrix = icp.GetMatrix()
    res = np.eye(4)
    for i in range(4):
        for j in range(4):
            res[i, j] = matrix.GetElement(i, j)
    return res

def compute_mean_shape_vtk(polys):
    """Simplified Mean Shape: Just average the first one with the others after ICP."""
    # Note: Complex mean shape computation in VTK is hard. 
    # For a template, we just need a reliable representative shape.
    return polys[0]

# ============================================================
#  Main Loop
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--headless", action="store_true")
    args, unknown = parser.parse_known_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    input_dir = os.path.abspath(args.input_dir) if args.input_dir else SCRIPT_DIR
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(SCRIPT_DIR, "output")

    os.makedirs(output_dir, exist_ok=True)

    # 1. FIND FILES (Recursive Smart Search)
    extensions = ["*.nii.gz", "*.nii", "*.hdr"]
    file_list = []
    for ext in extensions:
        file_list.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))
    
    file_list = sorted(list(set(file_list)))
    label_files = [f for f in file_list if "label" in os.path.basename(f).lower()]
    if label_files:
        print(f"Detected {len(label_files)} label-specific files.")
        file_list = label_files

    if not file_list:
        print("No files found. Exiting.")
        return

    # 2. LOAD & MESH
    meshes = []
    for file_path in file_list:
        try:
            poly = load_and_mesh_node(file_path)
            if poly and poly.GetNumberOfPoints() > 0:
                meshes.append(poly)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    N = len(meshes)
    if N == 0:
        print("No valid meshes loaded.")
        return

    print(f"Successfully loaded {N} meshes.")

    # 3. CENTROID ALIGNMENT & PRE-ICP NORMALIZATION
    aligned_meshes = []
    T_matrices = []
    for i in range(N):
        c = get_poly_centroid(meshes[i])
        T_i = np.eye(4)
        T_i[:3, 3] = -c
        
        # Translate to origin first to calculate accurate symmetrical bounds
        centered_poly = apply_poly_transform(meshes[i], T_i)
        
        # Pre-ICP Normalization: Scale down ONLY if it exceeds [-1, 1] bounds
        max_bound = get_poly_max_bound(centered_poly)
        T_scale = np.eye(4)
        if max_bound > 1.0:
            s_factor = 1.0 / max_bound
            T_scale[0, 0] = s_factor
            T_scale[1, 1] = s_factor
            T_scale[2, 2] = s_factor
            
        T_combined = T_scale @ T_i
        
        aligned_meshes.append(apply_poly_transform(centered_poly, T_scale))
        T_matrices.append(T_combined)

    # 4. GROUP-WISE ICP (Simplified: Align to first mesh for template)
    print("Running Group-wise ICP...")
    ref_mesh = aligned_meshes[0]
    for i in range(1, N):
        if i % 10 == 0: print(f"  Aligning mesh {i}/{N}...")
        delta_T = run_vtk_icp(aligned_meshes[i], ref_mesh)
        aligned_meshes[i] = apply_poly_transform(aligned_meshes[i], delta_T)
        T_matrices[i] = delta_T @ T_matrices[i]

    # 4.5. POST-ICP NORMALIZATION
    print("Applying Post-ICP Scale Normalization (if exceeding bounds)...")
    for i in range(N):
        max_bound_post = get_poly_max_bound(aligned_meshes[i])
        # If the bound exceeded 1.0 (e.g., due to rotation during ICP), scale it back down
        if max_bound_post > 1.0:
            T_scale_post = np.eye(4)
            s_factor_post = 1.0 / max_bound_post
            T_scale_post[0, 0] = s_factor_post
            T_scale_post[1, 1] = s_factor_post
            T_scale_post[2, 2] = s_factor_post
            
            aligned_meshes[i] = apply_poly_transform(aligned_meshes[i], T_scale_post)
            T_matrices[i] = T_scale_post @ T_matrices[i]

    # 5. EXPORT
    print("Saving results...")
    # Save Mean Shape (Representative)
    writer = vtk.vtkPLYWriter()
    writer.SetFileName(os.path.join(output_dir, "mean_shape.ply"))
    writer.SetInputData(ref_mesh)
    writer.Write()

    # Save Matrices
    np.save(os.path.join(output_dir, "T_matrices.npy"), np.array(T_matrices))
    
    # 6. APPLY TRANSFORM TO ORIGINAL VOLUMES AND SAVE
    print("\nApplying alignments to original .nii.gz volumes...")
    aligned_volumes_dir = os.path.join(output_dir, "aligned_nifti")
    os.makedirs(aligned_volumes_dir, exist_ok=True)
    
    for i, file_path in enumerate(file_list):
        basename = os.path.basename(file_path).split('.')[0]
        if i % 10 == 0: print(f"  Saving aligned volume {i}/{N}...")
        
        # Load volume
        node = slicer.util.loadLabelVolume(file_path)
        if not node: continue
        
        # Create transform node from T_matrices[i]
        T_i = T_matrices[i]
        transform_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
        vtk_matrix = vtk.vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                vtk_matrix.SetElement(r, c, float(T_i[r, c]))
                
        transform_node.SetMatrixTransformToParent(vtk_matrix)
        
        # Apply and harden transform (modifies header only, NO voxel degradation)
        node.SetAndObserveTransformNodeID(transform_node.GetID())
        slicer.vtkSlicerTransformLogic().hardenTransform(node)
        
        # Save as new aligned NIFTI
        out_name = f"{basename}_aligned.nii.gz"
        out_path = os.path.join(aligned_volumes_dir, out_name)
        slicer.util.saveNode(node, out_path)
        
        # Cleanup
        slicer.mrmlScene.RemoveNode(node)
        slicer.mrmlScene.RemoveNode(transform_node)
    
    print(f"\nDONE! Results saved to {output_dir}")
    print(f"Aligned volumes (to be used by SPHARM) are in: {aligned_volumes_dir}")

if __name__ == "__main__":
    main()
    try:
        import qt
        print("Closing SlicerSALT...")
        qt.QTimer.singleShot(500, slicer.util.exit)
    except:
        import sys
        sys.exit(0)