import os
import glob
import slicer

# Path Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
LOG_FILE = os.path.join(SCRIPT_DIR, "slicer_batch_log.txt")

def log(msg):
    print(msg)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(msg + "\n")
    except:
        pass
try:
    with open(LOG_FILE, "w") as f:
        f.write("--- SPHARM Batch Start (SlicerSALT 2.2.1) ---\n")
except:
    pass
def visualize_results(vtk_files):
    """Visualizes the first few VTK results in a pop-up window."""
    import vtk
    log("\nLaunching 3D Visualization...")   
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.1, 0.1, 0.2)
    max_to_show = min(len(vtk_files), 3)   
    found_any = False
    for i in range(max_to_show):
        if not os.path.exists(vtk_files[i]):
            continue           
        found_any = True
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(vtk_files[i])
        reader.Update()       
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.SetPosition(i * 30, 0, 0) 
        actor.GetProperty().SetOpacity(0.7)
        renderer.AddActor(actor)      
    if not found_any:
        log("No VTK files found to visualize.")
        return
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)
    render_window.SetWindowName("SPHARM-PDM Results Preview")
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    log(f"Showing first {max_to_show} models as a sample.")
    render_window.Render()
    interactor.Start()
def run_batch_spharm():
    input_dir = os.path.join(SCRIPT_DIR, "msd-hippocampus", "labelsTr")
    output_base_dir = os.path.join(SCRIPT_DIR, "output", "spharm_results")
    TEST_MODE = False  # Set to False to run all 260 files
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir) 
    file_list = sorted(glob.glob(os.path.join(input_dir, "*.nii.gz")))
    if TEST_MODE:
        file_list = file_list[:1]
        log("TEST MODE ENABLED: Processing only the first file.")
    log(f"Found {len(file_list)} files to process.") 
    output_vtk_list = []
    for i, file_path in enumerate(file_list):
        file_path = os.path.normpath(file_path).replace("\\", "/")
        basename = os.path.basename(file_path).split('.')[0]
        log(f"\n--- [{i+1}/{len(file_list)}] Processing {basename} ---")
        pp_mask_path = os.path.join(output_base_dir, f"{basename}_pp.nrrd").replace("\\", "/")
        para_mesh_path = os.path.join(output_base_dir, f"{basename}_para.vtk").replace("\\", "/")
        surf_mesh_path = os.path.join(output_base_dir, f"{basename}_surf.vtk").replace("\\", "/")
        spharm_base = os.path.join(output_base_dir, basename).replace("\\", "/")
        input_node = None
        pp_node = None
        para_node = None
        surf_node = None       
        try:
            log("  Loading input volume...")
            input_node = slicer.util.loadLabelVolume(file_path)
            if not input_node:
                raise Exception(f"Failed to load {file_path}")
            log("  Running SegPostProcess...")
            pp_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"{basename}_pp")
            pp_params = {
                'fileName': input_node.GetID(),
                'outfileName': pp_node.GetID(),
                'label': 1
            }
            slicer.cli.run(slicer.modules.segpostprocessclp, None, pp_params, wait_for_completion=True)
            slicer.util.saveNode(pp_node, pp_mask_path)
            if not os.path.exists(pp_mask_path):
                raise Exception("SegPostProcess failed to generate output.")
            log("  Running GenParaMesh...")
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
            if not os.path.exists(para_mesh_path):
                raise Exception("GenParaMesh failed to generate output.")
            log("  Running ParaToSPHARMMesh...")
            spharm_params = {
                'inParaFile': para_node.GetID(),
                'inSurfFile': surf_node.GetID(),
                'outbase': spharm_base,
                'subdivLevel': 10,
                'spharmDegree': 12
            }
            slicer.cli.run(slicer.modules.paratospharmmeshclp, None, spharm_params, wait_for_completion=True)
            final_vtk = f"{spharm_base}_SPHARM.vtk"
            if os.path.exists(final_vtk):
                output_vtk_list.append(final_vtk)
                log(f"  Success: {basename} completed.")
            else:
                log(f"  FAILED: {basename} - SPHARM result not found.")
        except Exception as e:
            log(f"  FAILED: {basename} error: {str(e)}")      
        finally:
            for node in [input_node, pp_node, para_node, surf_node]:
                if node:
                    slicer.mrmlScene.RemoveNode(node)
    log("\nBATCH PROCESSING COMPLETE!")
    if output_vtk_list:
        visualize_results(output_vtk_list)
    else:
        log("No successful reconstructions to visualize.")
run_batch_spharm()
