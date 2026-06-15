import os
import json
import argparse
import glob
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


def auto_pick_template(json_path):
    """หา template VTK โดยอัตโนมัติจาก spharm_results ที่อยู่ใน output folder ของ JSON"""
    pca_dir = os.path.dirname(json_path)
    output_dir = os.path.dirname(pca_dir)
    spharm_dir = os.path.join(output_dir, "spharm_results")
    if not os.path.isdir(spharm_dir):
        return None
    candidates = sorted(glob.glob(os.path.join(spharm_dir, "*_SPHARM_realigned.vtk")))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(spharm_dir, "*_SPHARM_ellalign.vtk")))
    return candidates[0] if candidates else None

def fix_pca_sign_convention(components, scores):
    """
    [FIX] แก้ Sign Ambiguity ของ PCA Eigenvectors
    
    PCA/SVD ให้ eigenvector ที่ทิศทางสุ่ม (+ หรือ − ก็ได้)
    ถ้าไม่แก้: บาง subject จะมี score −4 แทนที่จะเป็น +2 ทั้งที่ shape เหมือนกัน
    ผลคือ ±3SD reconstruct shape คนละขั้ว (โค้งคนละทาง)
    
    วิธีแก้: บังคับให้ค่า max absolute ใน eigenvector แต่ละ PC เป็นบวกเสมอ
    ถ้า flip eigenvector → flip scores ด้วย (เพื่อให้ consistent)
    
    Args:
        components: np.array (n_components, n_features) — eigenvectors
        scores:     np.array (n_samples, n_components) — projected scores
    
    Returns:
        components_fixed, scores_fixed (ค่าเดิมแต่ sign consistent)
    """
    components = components.copy()
    scores = scores.copy()
    
    for i in range(components.shape[0]):
        # หา index ที่มี |value| มากที่สุดใน eigenvector นี้
        max_abs_idx = np.argmax(np.abs(components[i]))
        # ถ้าค่านั้นเป็นลบ → flip ทั้ง eigenvector และ scores column นั้น
        if components[i, max_abs_idx] < 0:
            components[i] *= -1
            scores[:, i] *= -1
    
    return components, scores

def visualize_series(vtk_files, title="PCA Variations"):
    """Pops up a VTK 3D window to view the files side-by-side."""
    print("\nLaunching 3D VTK Viewer...")
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.2, 0.2, 0.2)
    
    text_prop = vtk.vtkTextProperty()
    text_prop.SetFontSize(14)
    text_prop.SetColor(1.0, 1.0, 1.0)
    text_prop.SetJustificationToCentered()
    
    offset_step = 40.0
    
    for i, fpath in enumerate(vtk_files):
        if not os.path.exists(fpath): continue
        
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(fpath)
        reader.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        weight_str = os.path.basename(fpath).split('_')[-1].replace('.vtk','')
        if 'minus' in weight_str:
            actor.GetProperty().SetColor(0.2, 0.6, 1.0)
        elif 'plus' in weight_str:
            actor.GetProperty().SetColor(1.0, 0.4, 0.4)
        else:
            actor.GetProperty().SetColor(0.9, 0.9, 0.9)
            
        actor.GetProperty().SetSpecular(0.3)
        actor.GetProperty().SetSpecularPower(20)
        
        pos_x = (i - len(vtk_files)/2.0) * offset_step
        actor.SetPosition(pos_x, 0, 0)
        renderer.AddActor(actor)
        
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", default=None,
                        help="Path to pca_model.json (ถ้าไม่ระบุจะเด้ง dialog)")
    parser.add_argument("--template", default=None,
                        help="Path to template VTK (ถ้าไม่ระบุจะหาให้อัตโนมัติ)")
    parser.add_argument("--output_dir", default=None,
                        help="Output folder ที่มี pca_results/ — shortcut แทน --json_path")
    args, _ = parser.parse_known_args()

    json_path = args.json_path
    if not json_path and args.output_dir:
        json_path = os.path.join(args.output_dir, "pca_results", "pca_model.json")
    if not json_path:
        print("Waiting for you to select 'pca_model.json'...")
        json_path = popup_select_file("Select pca_model.json",
                                      [('JSON Files', '*.json'), ('All Files', '*.*')])
    if not json_path or not os.path.exists(json_path):
        print(f"ERROR: pca_model.json not found at: {json_path}")
        return

    out_dir = os.path.join(os.path.dirname(json_path), "reconstructed_surfaces")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    vtk_path = args.template
    if not vtk_path:
        vtk_path = auto_pick_template(json_path)
        if vtk_path:
            print(f"\nAuto-picked template: {vtk_path}")
    if not vtk_path:
        print("\nPlease select a Template VTK file "
              "(e.g., any of your '_SPHARM_realigned.vtk' meshes)")
        vtk_path = popup_select_file("Select Template VTK",
                                     [('VTK Files', '*.vtk'), ('All Files', '*.*')])
    if not vtk_path or not os.path.exists(vtk_path):
        print(f"ERROR: Template VTK not found at: {vtk_path}")
        return

    print(f"\nLoading JSON model: {os.path.basename(json_path)}")
    print("This might take a moment due to file size (e.g., 60MB+)...")
    try:
        with open(json_path, 'r') as f:
            full_data = json.load(f)
    except Exception as e:
        print(f"Failed to load JSON: {e}")
        return
        
    data = full_data.get('0', full_data.get('All', full_data))
    
    if 'data_mean' not in data or 'eigenvalues' not in data or 'components' not in data:
        print("Error: Required PCA keys missing in JSON.")
        return
        
    mean_vec = np.array(data['data_mean'][0])
    components = np.array(data['components'])
    
    # โหลด scores
    scores_key = 'data_projection'
    if scores_key not in data:
        scores_key = 'projected_points' if 'projected_points' in data else 'scores'
        
    if scores_key in data and len(data[scores_key]) > 0:
        scores_matrix = np.array(data[scores_key])
    else:
        print("[Warning] Could not find empirical scores. Falling back to eigenvalues.")
        scores_matrix = None
    
    # ========================================================
    # [FIX] แก้ Sign Convention ของ Eigenvectors ก่อน reconstruct
    # นี่คือสาเหตุหลักที่ ±3SD โค้งคนละด้าน
    # ========================================================
    if scores_matrix is not None:
        print("\n[FIX] Applying PCA sign convention correction...")
        components, scores_matrix = fix_pca_sign_convention(components, scores_matrix)
        std_devs = np.std(scores_matrix, axis=0)
        print(f"  PC1 std after fix: {std_devs[0]:.4f}")
        print(f"  PC1 score range: [{scores_matrix[:,0].min():.3f}, {scores_matrix[:,0].max():.3f}]")
        print("  Sign convention applied successfully.")
    else:
        std_devs = np.sqrt(np.array(data['eigenvalues']))
    
    print(f"\nExtracted properties for {len(components)} Principal Components.")
    
    print(f"Loading template VTK: {os.path.basename(vtk_path)}")
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_path)
    reader.Update()
    template_poly = reader.GetOutput()
    
    num_points = template_poly.GetNumberOfPoints()
    if len(mean_vec) != num_points * 3:
        print(f"\n[WARNING] JSON mean length ({len(mean_vec)}) != VTK points * 3 ({num_points * 3}).")
        print("Ensure you selected a VTK from the exact SAME dataset run as the PCA.\n")
    
    num_pcs_to_reconstruct = min(3, len(components))
    weights = [-3, -2, -1, 0, 1, 2, 3]
    
    print("\nStarting generation over PC1, PC2, PC3...")
    for pc_idx in range(num_pcs_to_reconstruct):
        print(f"--- Reconstructing PC{pc_idx + 1} ---")
        print(f"    std_dev = {std_devs[pc_idx]:.4f}")
        pc_dir = os.path.join(out_dir, f"PC{pc_idx + 1}")
        if not os.path.exists(pc_dir):
            os.makedirs(pc_dir)
            
        for w in weights:
            # Core Formula: X = Mean + (Weight * SD * Eigenvector)
            new_points_flat = mean_vec + (w * std_devs[pc_idx] * components[pc_idx])
            new_points_3d = new_points_flat.reshape(-1, 3)
            
            vtk_pts = vtk.vtkPoints()
            for pt in new_points_3d:
                vtk_pts.InsertNextPoint(pt[0], pt[1], pt[2])
                
            new_poly = vtk.vtkPolyData()
            new_poly.DeepCopy(template_poly)
            new_poly.SetPoints(vtk_pts)
            
            if w > 0:
                w_str = f"plus{w}SD"
            elif w < 0:
                w_str = f"minus{abs(w)}SD"
            else:
                w_str = "Mean"
                
            out_filename = os.path.join(pc_dir, f"PC{pc_idx + 1}_{w_str}.vtk")
            
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(out_filename)
            writer.SetInputData(new_poly)
            writer.Write()
            
            print(f"  Saved: {os.path.basename(out_filename)}")
            
    print("\n" + "="*60)
    print(f"Reconstruction Complete! ALL files saved to:")
    print(f"{os.path.abspath(out_dir)}")
    print("="*60)

if __name__ == "__main__":
    main()