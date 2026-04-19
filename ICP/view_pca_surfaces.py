import os
import tkinter as tk
from tkinter import filedialog
import vtk

def popup_select_directory(title):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askdirectory(title=title, initialdir=os.getcwd())
    root.destroy()
    return path

def visualize_series(vtk_files, title="PCA Variations"):
    """Pops up a VTK 3D window to view the files side-by-side."""
    print("\nLaunching 3D VTK Viewer...")
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.2, 0.2, 0.2)
    
    # Text Property
    text_prop = vtk.vtkTextProperty()
    text_prop.SetFontSize(14)
    text_prop.SetColor(1.0, 1.0, 1.0)
    text_prop.SetJustificationToCentered()
    
    offset_step = 40.0 # Distance between shapes
    
    for i, fpath in enumerate(vtk_files):
        if not os.path.exists(fpath): continue
        
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(fpath)
        reader.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Color coding: Mean is white, - is blue tint, + is red tint
        weight_str = os.path.basename(fpath).split('_')[-1].replace('.vtk','')
        if 'minus' in weight_str:
            actor.GetProperty().SetColor(0.2, 0.6, 1.0) # Light blue
        elif 'plus' in weight_str:
            actor.GetProperty().SetColor(1.0, 0.4, 0.4) # Light red
        else:
            actor.GetProperty().SetColor(0.9, 0.9, 0.9) # White
            
        actor.GetProperty().SetSpecular(0.3)
        actor.GetProperty().SetSpecularPower(20)
        
        # Position them in a line
        pos_x = (i - len(vtk_files)/2.0) * offset_step
        actor.SetPosition(pos_x, 0, 0)
        renderer.AddActor(actor)
        
        # Add label
        text_mapper = vtk.vtkTextMapper()
        text_mapper.SetInput(weight_str)
        text_mapper.SetTextProperty(text_prop)
        text_actor = vtk.vtkActor2D()
        text_actor.SetMapper(text_mapper)
        text_actor.GetPositionCoordinate().SetCoordinateSystemToWorld()
        text_actor.GetPositionCoordinate().SetValue(pos_x, -15, 0)
        renderer.AddActor(text_actor)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1200, 400)
    render_window.SetWindowName(title)
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    # Auto-position camera
    renderer.ResetCamera()
    cam = renderer.GetActiveCamera()
    cam.Zoom(1.2)
    
    render_window.Render()
    print("Close the 3D window to finish!")
    interactor.Start()

def main():
    print("="*60)
    print("--- View PCA Surface Variations ---")
    print("="*60)
    
    print("\nPlease select a PC folder (e.g., 'PC1', 'PC2', 'PC3') to view its variations...")
    pc_dir = popup_select_directory("Select PC Folder")
    if not pc_dir:
        print("Canceled. No folder selected.")
        return
        
    pc_name = os.path.basename(pc_dir) # e.g. "PC1"
    
    files_to_show = [
        f"{pc_name}_minus3SD.vtk", f"{pc_name}_minus2SD.vtk", f"{pc_name}_minus1SD.vtk", 
        f"{pc_name}_Mean.vtk", 
        f"{pc_name}_plus1SD.vtk", f"{pc_name}_plus2SD.vtk", f"{pc_name}_plus3SD.vtk"
    ]
    
    full_paths = [os.path.join(pc_dir, f) for f in files_to_show]
    
    # Check if files exist
    existing_paths = [p for p in full_paths if os.path.exists(p)]
    if not existing_paths:
        print(f"Error: Could not find VTK files in {pc_dir}.")
        print(f"Make sure you selected a folder containing files like {pc_name}_minus3SD.vtk")
        return
        
    visualize_series(existing_paths, title=f"Visualizing {pc_name} Variations (-3SD to +3SD)")

if __name__ == "__main__":
    main()
