import vtk
import os

# Path Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
PLY_PATH = os.path.join(SCRIPT_DIR, "output", "mean_shape.ply")
NII_PATH = os.path.join(SCRIPT_DIR, "msd-hippocampus", "labelsTr", "hippocampus_001.nii.gz")

def check():
    print("--- Spatial Alignment Check ---")
    
    # 1. PLY Bounds
    if os.path.exists(PLY_PATH):
        reader = vtk.vtkPLYReader()
        reader.SetFileName(PLY_PATH)
        reader.Update()
        poly = reader.GetOutput()
        bounds = poly.GetBounds()
        print(f"PLY Mesh Bounds (Physical):")
        print(f"  X: {bounds[0]:.2f} to {bounds[1]:.2f} (mid: {(bounds[0]+bounds[1])/2:.2f})")
        print(f"  Y: {bounds[2]:.2f} to {bounds[3]:.2f} (mid: {(bounds[2]+bounds[3])/2:.2f})")
        print(f"  Z: {bounds[4]:.2f} to {bounds[5]:.2f} (mid: {(bounds[4]+bounds[5])/2:.2f})")
    else:
        print("PLY not found.")

    # 2. NIfTI Bounds
    if os.path.exists(NII_PATH):
        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(NII_PATH)
        reader.Update()
        data = reader.GetOutput()
        
        # Physical bounds in VTK
        bounds = data.GetBounds()
        print(f"\nNIfTI Volume Bounds (Physical):")
        print(f"  X: {bounds[0]:.2f} to {bounds[1]:.2f} (mid: {(bounds[0]+bounds[1])/2:.2f})")
        print(f"  Y: {bounds[2]:.2f} to {bounds[3]:.2f} (mid: {(bounds[2]+bounds[3])/2:.2f})")
        print(f"  Z: {bounds[4]:.2f} to {bounds[5]:.2f} (mid: {(bounds[4]+bounds[5])/2:.2f})")
        
        # Check Spacing/Origin
        origin = data.GetOrigin()
        spacing = data.GetSpacing()
        dim = data.GetDimensions()
        print(f"\nNIfTI Props:")
        print(f"  Dimensions: {dim}")
        print(f"  Spacing: {spacing}")
        print(f"  Origin: {origin}")
    else:
        print("NIfTI not found.")

if __name__ == "__main__":
    check()
