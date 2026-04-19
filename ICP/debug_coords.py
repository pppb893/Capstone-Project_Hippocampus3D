import vtk
import os
import nibabel as nib
import numpy as np

# Path Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
PLY_PATH = os.path.join(SCRIPT_DIR, "output", "mean_shape.ply")
NII_PATH = os.path.join(SCRIPT_DIR, "msd-hippocampus", "labelsTr", "hippocampus_001.nii.gz")

def inspect():
    results = []
    
    # Check PLY Bounds
    if os.path.exists(PLY_PATH):
        reader = vtk.vtkPLYReader()
        reader.SetFileName(PLY_PATH)
        reader.Update()
        poly = reader.GetOutput()
        bounds = poly.GetBounds() # (xmin, xmax, ymin, ymax, zmin, zmax)
        results.append(f"PLY Bounds: {bounds}")
    else:
        results.append("PLY file not found.")

    # Check NIfTI header and bounds
    if os.path.exists(NII_PATH):
        img = nib.load(NII_PATH)
        shape = img.shape
        affine = img.affine
        results.append(f"NII Shape: {shape}")
        results.append(f"NII Affine:\n{affine}")
        
        # Calculate NII Physical Bounds
        v_min = np.array([0, 0, 0, 1])
        v_max = np.array(shape) - 1
        v_max = np.append(v_max, 1)
        
        p_min = affine @ v_min
        p_max = affine @ v_max
        results.append(f"NII Approx Physical Range: {p_min[:3]} to {p_max[:3]}")
    else:
        results.append("NII file not found.")

    print("\n".join(results))
    with open(os.path.join(SCRIPT_DIR, "coordinate_check.txt"), "w") as f:
        f.write("\n".join(results))

if __name__ == "__main__":
    inspect()
