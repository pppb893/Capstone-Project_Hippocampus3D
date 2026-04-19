import os
import slicer
import vtk
import numpy as np

# Path Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
MEAN_PLY = os.path.join(SCRIPT_DIR, "output", "mean_shape.ply")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output", "spharm_mean")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

LOG_FILE = os.path.join(OUTPUT_DIR, "mean_process_log.txt")

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def process():
    log("--- Starting Mean Shape Fixing (VTK Stencil Method) ---")
    
    try:
        # 1. Load Mean PLY
        log("Loading PLY...")
        reader = vtk.vtkPLYReader()
        reader.SetFileName(MEAN_PLY)
        reader.Update()
        poly = reader.GetOutput()
        
        if poly.GetNumberOfPoints() == 0:
            raise Exception("PLY file has no points.")
            
        # 2. Calculate Bounds and Setup Image
        bounds = poly.GetBounds()
        log(f"Mesh Bounds: {bounds}")
        
        padding = 10.0
        spacing = 0.5
        
        origin = [bounds[0] - padding, bounds[2] - padding, bounds[4] - padding]
        dim_x = int((bounds[1] - bounds[0] + 2*padding) / spacing) + 1
        dim_y = int((bounds[3] - bounds[2] + 2*padding) / spacing) + 1
        dim_z = int((bounds[5] - bounds[4] + 2*padding) / spacing) + 1
        
        log(f"Image Dims: {dim_x}x{dim_y}x{dim_z}, Origin: {origin}")
        
        # Create blank image
        white_image = vtk.vtkImageData()
        white_image.SetDimensions(dim_x, dim_y, dim_z)
        white_image.SetSpacing(spacing, spacing, spacing)
        white_image.SetOrigin(origin)
        white_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        white_image.GetPointData().GetScalars().Fill(1) # Fill all with 1
        
        # 3. Create Stencil from PolyData
        pol2stenc = vtk.vtkPolyDataToImageStencil()
        pol2stenc.SetInputData(poly)
        pol2stenc.SetOutputOrigin(origin)
        pol2stenc.SetOutputSpacing(spacing, spacing, spacing)
        pol2stenc.SetOutputWholeExtent(0, dim_x-1, 0, dim_y-1, 0, dim_z-1)
        pol2stenc.Update()
        
        # 4. Apply Stencil to create the label map
        imgstenc = vtk.vtkImageStencil()
        imgstenc.SetInputData(white_image)
        imgstenc.SetStencilData(pol2stenc.GetOutput())
        imgstenc.ReverseStencilOff()
        imgstenc.SetBackgroundValue(0) # Keep background 0, stencil 1
        imgstenc.Update()
        
        # 5. Save to MRML and File
        mean_label_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "mean_label")
        mean_label_node.SetAndObserveImageData(imgstenc.GetOutput())
        
        # Count non-zero voxels for verification
        label_data = imgstenc.GetOutput()
        stats = vtk.vtkImageAccumulate()
        stats.SetInputData(label_data)
        stats.Update()
        non_zero = stats.GetVoxelCount() - stats.GetOutput().GetScalarComponentAsDouble(0,0,0,0)
        log(f"Non-zero voxels: {non_zero:.0f}")
        
        if non_zero < 100:
             raise Exception(f"Voxelization too small ({non_zero}). Is the mesh closed and solid?")

        slicer.util.saveNode(mean_label_node, os.path.join(OUTPUT_DIR, "mean_label.nrrd"))
        log("LabelMap saved.")

        # 6. SPHARM Pipeline
        log("Running SPHARM steps...")
        
        # 6.1 SegPostProcess
        pp_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "mean_pp")
        pp_params = {'fileName': mean_label_node.GetID(), 'outfileName': pp_node.GetID(), 'label': 1}
        slicer.cli.run(slicer.modules.segpostprocessclp, None, pp_params, wait_for_completion=True)
        log("  SegPostProcess done.")

        # 6.2 GenParaMesh
        para_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "mean_para")
        surf_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "mean_surf")
        para_params = {
            'infile': pp_node.GetID(),
            'outParaName': para_node.GetID(),
            'outSurfName': surf_node.GetID(),
            'numIterations': 500,
            'label': 1
        }
        slicer.cli.run(slicer.modules.genparameshclp, None, para_params, wait_for_completion=True)
        log("  GenParaMesh done.")

        # 6.3 ParaToSPHARMMesh
        spharm_base = os.path.join(OUTPUT_DIR, "mean_result").replace("\\", "/")
        spharm_params = {
            'inParaFile': para_node.GetID(),
            'inSurfFile': surf_node.GetID(),
            'outbase': spharm_base,
            'subdivLevel': 10,
            'spharmDegree': 12
        }
        slicer.cli.run(slicer.modules.paratospharmmeshclp, None, spharm_params, wait_for_completion=True)
        log(f"  ParaToSPHARMMesh done. Results in: {spharm_base}")

        log("\nPROCESS COMPLETE!")

    except Exception as e:
        log(f"CRITICAL ERROR: {str(e)}")
    
    finally:
        log("Exiting Slicer.")
        slicer.util.exit()

if __name__ == "__main__":
    with open(LOG_FILE, "w") as f: f.write("Log started (VTK Method).\n")
    process()
