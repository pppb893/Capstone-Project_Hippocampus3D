import vtk
import nibabel as nib
import numpy as np
import cv2
from skimage import measure
import trimesh
import os

def load_and_mesh_nifti(filepath):
    print(f"Loading {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    nii = nib.load(filepath)
    data = nii.get_fdata()
    print(f"shape: {data.shape}")

    mask = (data > 0).astype(np.uint8)

    print("Smoothing with OpenCV...")
    for i in range(mask.shape[2]):
        slice_img = mask[:, :, i]
        slice_img = cv2.GaussianBlur(slice_img, (5,5), 0)
        _, slice_img = cv2.threshold(slice_img, 0.1, 1, cv2.THRESH_BINARY)
        mask[:, :, i] = slice_img

    print("Generating mesh...")
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
    return np.asarray(verts), np.asarray(faces)

def update_actor_vertices(actor, new_verts):
    polydata = actor.GetMapper().GetInput()
    points = polydata.GetPoints()

    for i in range(len(new_verts)):
        points.SetPoint(i, float(new_verts[i][0]), float(new_verts[i][1]), float(new_verts[i][2]))
        
    points.Modified() 
    polydata.Modified()
    

def create_vtk_actor(verts, faces, color, opacity=0.8):
    points = vtk.vtkPoints()
    for v in verts:
        points.InsertNextPoint(float(v[0]), float(v[1]), float(v[2]))

    triangles = vtk.vtkCellArray()
    for f in faces:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, int(f[0]))
        triangle.GetPointIds().SetId(1, int(f[1]))
        triangle.GetPointIds().SetId(2, int(f[2]))
        triangles.InsertNextCell(triangle)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(triangles)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetInterpolationToPhong()
    actor.GetProperty().SetOpacity(opacity)
    return actor

def main():
    # Path Configuration
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    file1 = os.path.join(SCRIPT_DIR, "msd-hippocampus", "labelsTr", "hippocampus_001.nii.gz")
    file2 = os.path.join(SCRIPT_DIR, "msd-hippocampus", "labelsTr", "hippocampus_003.nii.gz")
    
    try:
        verts1, faces1 = load_and_mesh_nifti(file1)
        verts2, faces2 = load_and_mesh_nifti(file2)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("Setting up visual ICP animation...")
    
    centroid_src = np.mean(verts1, axis=0)
    centroid_tgt = np.mean(verts2, axis=0)
    initial_transform = np.eye(4)
    initial_transform[:3, 3] = centroid_tgt - centroid_src
    
    current_verts = trimesh.transformations.transform_points(verts1, initial_transform)

    print("Setting up VTK visualization...")
    actor1 = create_vtk_actor(current_verts, faces1, color=(0.9, 0.1, 0.1), opacity=0.8)
    actor2 = create_vtk_actor(verts2, faces2, color=(0.1, 0.1, 0.9), opacity=0.8)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor1)
    renderer.AddActor(actor2)
    renderer.SetBackground(0.05, 0.05, 0.1)
    
    text_actor = vtk.vtkTextActor()
    text_actor.SetInput("Initializing...")
    text_actor.GetTextProperty().SetFontSize(24)
    text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
    text_actor.SetDisplayPosition(20, 20)
    renderer.AddActor(text_actor)

    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetSize(900, 700)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    
    class ICPTimerCallback:
        def __init__(self, moving_verts, static_verts, faces_for_avg, renderer, dynamic_actor, static_actor, text_display, max_iter=50):
            self.current_verts = moving_verts
            self.static_verts = static_verts
            self.faces_for_avg = faces_for_avg 
            self.dynamic_actor = dynamic_actor
            self.static_actor = static_actor
            self.renderer = renderer
            self.text_display = text_display
            
            self.iteration = 0
            self.max_iter = max_iter
            self.done = False

        def execute(self, obj, event):
            if self.done:
                return

            if self.iteration < self.max_iter:
                print(f"Starting ICP iteration {self.iteration + 1}...")
                self.iteration += 1
                
        
                result_transform, cost, _ = trimesh.registration.icp(
                    self.current_verts, 
                    self.static_verts, 
                    initial=np.eye(4),
                    max_iterations=1 
                )
                print(f"Finished ICP iteration {self.iteration}, cost: {cost}")
                
            
                self.current_verts = trimesh.transformations.transform_points(self.current_verts, result_transform)
                
                update_actor_vertices(self.dynamic_actor, self.current_verts)
       
                self.text_display.SetInput(f"ICP Iteration: {self.iteration}/{self.max_iter}\nCost: {cost}")
                
                if np.allclose(result_transform, np.eye(4), atol=1e-5):
                    self.text_display.SetInput(f"ICP Converged!\nCost: {cost}\nGenerating Average Mesh...")
                    self.done = True
                    self.generate_average_mesh(obj)
                
                obj.GetRenderWindow().Render()
            else:
                self.done = True
                self.text_display.SetInput("ICP Reached Max Iterations.\nGenerating Average Mesh...")
                self.generate_average_mesh(obj)
                obj.GetRenderWindow().Render()
                
        def generate_average_mesh(self, obj):
            print("Generating average mesh (using fast cKDTree)...")
            from scipy.spatial import cKDTree
            
            tree = cKDTree(self.static_verts)
            distances, indices = tree.query(self.current_verts, k=1, workers=-1)
            
            matched_static_verts = self.static_verts[indices]
            
            average_verts = (self.current_verts + matched_static_verts) / 2.0
            
            avg_actor = create_vtk_actor(average_verts, self.faces_for_avg, color=(0.5, 0.5, 0.5), opacity=1.0)
            
            self.renderer.AddActor(avg_actor)

            self.dynamic_actor.SetVisibility(False)
            self.static_actor.SetVisibility(False)
            
            self.text_display.SetInput("ICP Converged. Showing Average Mesh (Gray).")

            obj.GetRenderWindow().Render()
                
    icp_callback = ICPTimerCallback(current_verts, verts2, faces1, renderer, actor1, actor2, text_actor, max_iter=100)
    interactor.AddObserver('TimerEvent', icp_callback.execute)
    
    interactor.Initialize()
    timer_id = interactor.CreateRepeatingTimer(100)

    print("READY — Animation starting...")
    window.Render()
    interactor.Start()

if __name__ == "__main__":
    main()