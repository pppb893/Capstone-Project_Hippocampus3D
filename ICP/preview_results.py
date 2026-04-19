import os
import glob
import vtk
import sys

# Path Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(SCRIPT_DIR, "output", "spharm_results")

# Fetch Aligned VTK files
# We use _ellalign.vtk because they are the standard result of SPHARM-PDM
vtk_files = sorted(glob.glob(os.path.join(results_dir, "*_SPHARM_ellalign.vtk")))

if not vtk_files:
    print(f"Error: No VTK files found in {results_dir}")
    print("Please make sure you have run the SPHARM-PDM analysis first.")
    sys.exit(1)

# Limit to 5 files to avoid overlapping/clutter
max_to_show = min(len(vtk_files), 5)
print(f"--- SPHARM-PDM Results Preview ---")
print(f"Found {len(vtk_files)} files total.")
print(f"Showing the first {max_to_show} files as a preview.")

def main():
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.05, 0.1, 0.15) # Modern dark slate background
    
    # Add a light source
    light = vtk.vtkLight()
    light.SetPosition(1, 1, 1)
    renderer.AddLight(light)

    for i in range(max_to_show):
        file_path = vtk_files[i]
        basename = os.path.basename(file_path).replace("_SPHARM_ellalign.vtk", "")
        print(f"  Adding model: {basename}")

        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(file_path)
        reader.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Shift models horizontally so they don't overlap
        actor.SetPosition(i * 35, 0, 0)
        
        # Set a slightly transparent blue-ish material
        actor.GetProperty().SetColor(0.2, 0.6, 0.9)
        actor.GetProperty().SetOpacity(0.8)
        actor.GetProperty().SetSpecular(0.3)
        
        renderer.AddActor(actor)
        
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1280, 720)
    render_window.SetWindowName("SPHARM-PDM Results Preview (First 5 Models)")
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    # Modern camera orientation
    renderer.ResetCamera()
    renderer.GetActiveCamera().Azimuth(30)
    renderer.GetActiveCamera().Elevation(20)
    
    print("\n--- Interaction Controls ---")
    print("Left Click: Rotate")
    print("Middle Click (Scroll): Zoom")
    print("Right Click: Pan")
    print("Press 'q' or click X to close this window.\n")
    
    render_window.Render()
    interactor.Start()

if __name__ == "__main__":
    main()
