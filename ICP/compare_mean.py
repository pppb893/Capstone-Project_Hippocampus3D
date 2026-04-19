import vtk
import os

# Paths
MEAN_ORIGINAL = r"c:\Users\jckky\Desktop\ICP\output\spharm_mean\mean_shape.vtk"
MEAN_SPHARM = r"c:\Users\jckky\Desktop\ICP\output\spharm_mean\mean_result_SPHARM.vtk"

def visualize():
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.1, 0.1, 0.2)
    
    # 1. Original ICP Mean Shape
    if os.path.exists(MEAN_ORIGINAL):
        reader1 = vtk.vtkPolyDataReader()
        reader1.SetFileName(MEAN_ORIGINAL)
        reader1.Update()
        
        mapper1 = vtk.vtkPolyDataMapper()
        mapper1.SetInputConnection(reader1.GetOutputPort())
        
        actor1 = vtk.vtkActor()
        actor1.SetMapper(mapper1)
        actor1.SetPosition(-20, 0, 0)
        actor1.GetProperty().SetColor(0.7, 0.7, 0.7) # Grey
        renderer.AddActor(actor1)
        
        # Label/Caption would be nice but let's keep it simple
        print("Blue/Grey (Left): Original ICP Mean Shape")

    # 2. SPHARM Smoothed Mean Shape
    if os.path.exists(MEAN_SPHARM):
        reader2 = vtk.vtkPolyDataReader()
        reader2.SetFileName(MEAN_SPHARM)
        reader2.Update()
        
        mapper2 = vtk.vtkPolyDataMapper()
        mapper2.SetInputConnection(reader2.GetOutputPort())
        
        actor2 = vtk.vtkActor()
        actor2.SetMapper(mapper2)
        actor2.SetPosition(20, 0, 0)
        actor2.GetProperty().SetColor(0.2, 0.8, 0.2) # Greenish
        renderer.AddActor(actor2)
        print("Green (Right): SPHARM Smoothed Mean Shape")

    render_win = vtk.vtkRenderWindow()
    render_win.AddRenderer(renderer)
    render_win.SetSize(1000, 600)
    render_win.SetWindowName("Mean Shape Comparison: Original vs SPHARM")
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_win)
    
    print("\nVisualizing comparison...")
    render_win.Render()
    interactor.Start()

if __name__ == "__main__":
    visualize()
