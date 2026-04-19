import os
import json
import tkinter as tk
from tkinter import filedialog
import vtk
import numpy as np

def popup_select_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(title=title, filetypes=filetypes, initialdir=os.getcwd())
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
    print("--- PCA Surface Reconstruction (Mean +/- 3SD) ---")
    print("="*60)
    
    # 1. Select PCA Model JSON
    print("Waiting for you to select 'pca_model.json'...")
    json_path = popup_select_file("Select pca_model.json", [('JSON Files', '*.json'), ('All Files', '*.*')])
    if not json_path:
        print("Canceled. No JSON selected.")
        return
        
    out_dir = os.path.join(os.path.dirname(json_path), "reconstructed_surfaces")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    # 2. Select Template VTK
    print("\nPlease select a Template VTK file (e.g., any of your '_SPHARM_ellalign.vtk' meshes)")
    vtk_path = popup_select_file("Select Template VTK", [('VTK Files', '*.vtk'), ('All Files', '*.*')])
    if not vtk_path:
        print("Canceled. No Template VTK selected.")
        return

    # 3. Load JSON Data
    print(f"\nLoading JSON model: {os.path.basename(json_path)}")
    print("This might take a moment due to file size (e.g., 60MB+)...")
    try:
        with open(json_path, 'r') as f:
            full_data = json.load(f)
    except Exception as e:
        print(f"Failed to load JSON: {e}")
        return
        
    # SlicerSALT PCA data often nested under '0' or 'All'
    data = full_data.get('0', full_data.get('All', full_data))
    
    if 'data_mean' not in data or 'eigenvalues' not in data or 'components' not in data:
        print("Error: Required PCA keys missing in JSON. Ensure this is a valid SlicerSALT PCA output.")
        return
        
    mean_vec = np.array(data['data_mean'][0])
    components = np.array(data['components'])
    
    # Safely compute standard deviation of PC scores from the data itself
    # SlicerSALT saves scores in 'data_projection', 'projected_points', or 'scores'
    scores_key = 'data_projection'
    if scores_key not in data:
        scores_key = 'projected_points' if 'projected_points' in data else 'scores'
        
    if scores_key in data and len(data[scores_key]) > 0:
        scores_matrix = np.array(data[scores_key])
        std_devs = np.std(scores_matrix, axis=0) # Actual standard deviation across subjects
    else:
        # Fallback to eigenvalues if scores are somehow missing
        print("[Warning] Could not find empirical scores. Falling back to eigenvalues.")
        std_devs = np.sqrt(np.array(data['eigenvalues']))
    
    print(f"Extracted properties for {len(components)} Principal Components.")
    
    # 4. Load Template VTK
    print(f"Loading template VTK: {os.path.basename(vtk_path)}")
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_path)
    reader.Update()
    template_poly = reader.GetOutput()
    
    num_points = template_poly.GetNumberOfPoints()
    if len(mean_vec) != num_points * 3:
        print(f"\n[WARNING] JSON mean length ({len(mean_vec)}) != VTK points * 3 ({num_points * 3}).")
        print("The VTK topology might get broken. Continuing anyway, but please ensure")
        print("you selected a VTK from the exact SAME dataset run as the PCA.\n")
    
    # 5. Reconstruct meshes for top 3 PCs
    num_pcs_to_reconstruct = min(3, len(components))
    weights = [-3, -2, -1, 0, 1, 2, 3]
    
    print("\nStarting generation over PC1, PC2, PC3...")
    for pc_idx in range(num_pcs_to_reconstruct):
        print(f"--- Reconstructing PC{pc_idx + 1} ---")
        pc_dir = os.path.join(out_dir, f"PC{pc_idx + 1}")
        if not os.path.exists(pc_dir):
            os.makedirs(pc_dir)
            
        for w in weights:
            # Core Formula: X = Mean + (Weight * SD * Eigenvector)
            # Both mean_vec and components[pc_idx] are flattened 1D arrays of size (N points * 3)
            new_points_flat = mean_vec + (w * std_devs[pc_idx] * components[pc_idx])
            
            # Reshape to (N, 3) to iterate row by row
            new_points_3d = new_points_flat.reshape(-1, 3)
            
            # Create a new vtkPoints array
            vtk_pts = vtk.vtkPoints()
            for pt in new_points_3d:
                vtk_pts.InsertNextPoint(pt[0], pt[1], pt[2])
                
            # Create new PolyData copied from template to keep polygons
            new_poly = vtk.vtkPolyData()
            new_poly.DeepCopy(template_poly)
            new_poly.SetPoints(vtk_pts) # Overwrite points
            
            # Formatted weight label for naming
            if w > 0:
                w_str = f"plus{w}SD"
            elif w < 0:
                w_str = f"minus{abs(w)}SD"
            else:
                w_str = "Mean"
                
            out_filename = os.path.join(pc_dir, f"PC{pc_idx + 1}_{w_str}.vtk")
            
            # Save to disk
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(out_filename)
            writer.SetInputData(new_poly)
            writer.Write()
            
            print(f"  Saved: {os.path.basename(out_filename)}")
            
    print("\n" + "="*60)
    print(f"Reconstruction Complete! ALL files securely saved to:")
    print(f"{os.path.abspath(out_dir)}")
    print("="*60)

if __name__ == "__main__":
    main()
