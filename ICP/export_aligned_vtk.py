import os
import glob
import numpy as np
import nibabel as nib
from skimage import measure
import vtk

def load_and_mesh_nifti(filepath):
    """Loads NIfTI and returns vertices and faces using Marching Cubes."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    nii = nib.load(filepath)
    data = nii.get_fdata()
    # Assuming the same logic as main2.py
    mask = (data > 0).astype(np.uint8)
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
    return np.asarray(verts), np.asarray(faces)

def save_vtk(verts, faces, output_path):
    """Saves mesh as a VTK polydata file."""
    points = vtk.vtkPoints()
    for v in verts:
        points.InsertNextPoint(v[0], v[1], v[2])
    
    triangles = vtk.vtkCellArray()
    for f in faces:
        tri = vtk.vtkTriangle()
        tri.GetPointIds().SetId(0, int(f[0]))
        tri.GetPointIds().SetId(1, int(f[1]))
        tri.GetPointIds().SetId(2, int(f[2]))
        triangles.InsertNextCell(tri)
    
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(triangles)
    
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(polydata)
    writer.Write()

def main():
    # Paths from main2.py
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(SCRIPT_DIR, "msd-hippocampus", "labelsTr")
    matrix_path = os.path.join(SCRIPT_DIR, "output", "T_matrices.npy")
    output_dir = os.path.join(SCRIPT_DIR, "output", "aligned_vtk")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load matrices
    if not os.path.exists(matrix_path):
        print(f"Error: {matrix_path} not found. Run main2.py first.")
        return
    T_matrices = np.load(matrix_path)
    
    # Load file list (must match main2.py sorting)
    file_list = sorted(glob.glob(os.path.join(folder_path, "*.nii.gz")))
    
    print(f"Loaded {len(T_matrices)} matrices for {len(file_list)} files.")
    
    if len(T_matrices) != len(file_list):
        print("Warning: Number of matrices and files do not match!")
        # We will process up to the minimum of both
        count = min(len(T_matrices), len(file_list))
    else:
        count = len(file_list)

    for i in range(count):
        file_path = file_list[i]
        basename = os.path.basename(file_path).replace(".nii.gz", "")
        output_name = f"{basename}_aligned.vtk"
        output_path = os.path.join(output_dir, output_name)
        
        print(f"[{i+1}/{count}] Processing {basename}...")
        
        try:
            verts, faces = load_and_mesh_nifti(file_path)
            
            # Apply transformation T[i]
            T = T_matrices[i]
            # Convert to homogeneous coords for multiplication
            # (N, 3) -> (N, 4)
            ones = np.ones((len(verts), 1))
            verts_h = np.hstack([verts, ones])
            # (N, 4) @ (4, 4).T -> (N, 4)
            aligned_verts_h = verts_h @ T.T
            aligned_verts = aligned_verts_h[:, :3]
            
            # Save as VTK
            save_vtk(aligned_verts, faces, output_path)
            
        except Exception as e:
            print(f"Error processing {basename}: {e}")

    print(f"\nDone! Aligned VTK files saved to: {output_dir}")

if __name__ == "__main__":
    main()
