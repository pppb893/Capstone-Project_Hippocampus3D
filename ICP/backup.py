import vtk
import nibabel as nib
import numpy as np
from skimage import measure
import trimesh
import os
import glob
import json
from scipy.spatial import cKDTree


def load_and_mesh_nifti(filepath):
    print(f"Loading {os.path.basename(filepath)}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    nii = nib.load(filepath)
    data = nii.get_fdata()
    mask = (data > 0).astype(np.uint8)
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
    return np.asarray(verts), np.asarray(faces)


def update_actor_vertices(actor, new_verts):
    polydata = actor.GetMapper().GetInput()
    points = polydata.GetPoints()
    for i in range(len(new_verts)):
        points.SetPoint(i, float(new_verts[i][0]), float(new_verts[i][1]), float(new_verts[i][2]))
    points.Modified()
    polydata.Modified()


def update_actor_transform(actor, matrix):
    vtk_matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtk_matrix.SetElement(i, j, float(matrix[i, j]))
    actor.SetUserMatrix(vtk_matrix)


def create_vtk_actor(verts, faces, color, opacity=0.8, wireframe=False):
    points = vtk.vtkPoints()
    for v in verts:
        points.InsertNextPoint(float(v[0]), float(v[1]), float(v[2]))
    triangles = vtk.vtkCellArray()
    for f in faces:
        tri = vtk.vtkTriangle()
        tri.GetPointIds().SetId(0, int(f[0]))
        tri.GetPointIds().SetId(1, int(f[1]))
        tri.GetPointIds().SetId(2, int(f[2]))
        triangles.InsertNextCell(tri)
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
    if wireframe:
        actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetLineWidth(2.0)
    return actor


def get_random_color():
    return (np.random.uniform(0.2, 0.8),
            np.random.uniform(0.2, 0.8),
            np.random.uniform(0.2, 0.8))


class TransformHistory:
    """Stores the labeled chain of transformations for each mesh."""

    def __init__(self, N):
        self.N = N
        self.chains = [[] for _ in range(N)]

    def append(self, i, label, matrix):
        self.chains[i].append((label, matrix.copy()))

    def append_all(self, label, matrices):
        for i, M in enumerate(matrices):
            self.append(i, label, M)

    def get_composite(self, i):
        result = np.eye(4)
        for _, M in self.chains[i]:
            result = M @ result
        return result

    def to_dict(self):
        out = {}
        for i in range(self.N):
            out[f"mesh_{i}"] = [
                {"label": label, "matrix": M.tolist()}
                for label, M in self.chains[i]
            ]
        return out


def export_results(T_matrices, ref_shape_verts, ref_faces, history, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    T_array = np.array(T_matrices)
    np.save(os.path.join(output_dir, "T_matrices.npy"), T_array)
    print(f"Saved T_matrices.npy  shape={T_array.shape}")

    with open(os.path.join(output_dir, "transform_history.json"), "w") as f:
        json.dump(history.to_dict(), f, indent=2)
    print("Saved transform_history.json")

    mesh = trimesh.Trimesh(vertices=ref_shape_verts, faces=ref_faces)
    ply_path = os.path.join(output_dir, "mean_shape.ply")
    mesh.export(ply_path)
    print(f"Saved mean_shape.ply  ({len(ref_shape_verts)} verts, {len(ref_faces)} faces)")


def compute_mean_shape(meshes_verts, T_matrices, ref_verts):
    """Compute mean shape by averaging NN-matched vertices from all aligned meshes."""
    N = len(meshes_verts)
    mean_shape = np.zeros_like(ref_verts)
    for i in range(N):
        aligned = trimesh.transformations.transform_points(meshes_verts[i], T_matrices[i])
        tree = cKDTree(aligned)
        _, indices = tree.query(ref_verts, k=1, workers=-1)
        mean_shape += aligned[indices]
    mean_shape /= N
    return mean_shape


def main():
    folder_path = "msd-hippocampus/labelsTr/"
    file_list = sorted(glob.glob(os.path.join(folder_path, "*.nii.gz")))

    if not file_list:
        print(f"No .nii.gz files found in {folder_path}")
        return

    print(f"Found {len(file_list)} files. Loading all...")

    meshes_verts = []
    meshes_faces = []

    for file_path in file_list:
        try:
            v, f = load_and_mesh_nifti(file_path)
            if len(v) == 0:
                continue
            meshes_verts.append(v)
            meshes_faces.append(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    N = len(meshes_verts)
    if N == 0:
        print("No valid meshes loaded.")
        return

    print(f"Successfully loaded {N} meshes.")

    # --- Initialize history ---
    history = TransformHistory(N)

    # --- Initial centroid alignment ---
    T_matrices = []
    for i in range(N):
        centroid = np.mean(meshes_verts[i], axis=0)
        T_i = np.eye(4)
        T_i[:3, 3] = -centroid
        T_matrices.append(T_i)
        history.append(i, "init_centroid", T_i)

    # --- Initial reference: mesh[0] (will be updated to mean shape) ---
    ref_template_faces = meshes_faces[0]
    ref_shape_verts = trimesh.transformations.transform_points(meshes_verts[0], T_matrices[0])

    # --- VTK Setup ---
    print("Setting up VTK visualization...")
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.05, 0.05, 0.1)

    actors = []
    for i in range(N):
        color = get_random_color()
        actor = create_vtk_actor(meshes_verts[i], meshes_faces[i], color=color, opacity=0.15)
        update_actor_transform(actor, T_matrices[i])
        actors.append(actor)
        renderer.AddActor(actor)

    ref_actor = create_vtk_actor(ref_shape_verts, ref_template_faces,
                                  color=(1.0, 1.0, 1.0), opacity=1.0, wireframe=True)
    renderer.AddActor(ref_actor)

    text_actor = vtk.vtkTextActor()
    text_actor.SetInput(f"Group-wise ICP for {N} meshes...")
    text_actor.GetTextProperty().SetFontSize(24)
    text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
    text_actor.SetDisplayPosition(20, 20)
    renderer.AddActor(text_actor)

    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetSize(1000, 800)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)

    class GroupwiseICPTimerCallback:
        def __init__(self, meshes_verts, meshes_faces, T_matrices, actors,
                     ref_shape_verts, ref_template_faces, ref_actor,
                     renderer, text_display, history,
                     max_outer=20, max_inner=20):
            self.meshes_verts = meshes_verts
            self.meshes_faces = meshes_faces
            self.T = T_matrices
            self.actors = actors
            self.ref_shape_verts = ref_shape_verts
            self.ref_template_faces = ref_template_faces
            self.ref_actor = ref_actor
            self.renderer = renderer
            self.text_display = text_display
            self.history = history

            self.N = len(meshes_verts)
            self.max_outer = max_outer
            self.max_inner = max_inner
            self.outer_iter = 0
            self.inner_iter = 0
            self.done = False
            self.last_avg_cost: float | None = None
            self.last_outer_cost: float | None = None
            self.inner_conv_threshold = 1e-4
            self.outer_conv_threshold = 0.01

        def execute(self, obj, event):
            if self.done:
                return

            # --- Start new outer iteration if inner loop finished/not started ---
            if self.inner_iter == 0:
                self.outer_iter += 1
                if self.outer_iter > self.max_outer:
                    self.finish(obj)
                    return
                print(f"\n=== OUTER ITERATION {self.outer_iter}/{self.max_outer} ===")
                self.last_avg_cost = None  # reset for new inner loop

            # --- Inner loop: one ICP pass per timer tick ---
            self.inner_iter += 1
            total_cost = 0.0

            for i in range(self.N):
                current_verts = trimesh.transformations.transform_points(
                    self.meshes_verts[i], self.T[i])

                delta_T, _, cost = trimesh.registration.icp(
                    current_verts,
                    self.ref_shape_verts,
                    initial=np.eye(4),
                    max_iterations=5
                )
                if cost is None:
                    cost = 0.0

                self.history.append(i, f"outer{self.outer_iter}_icp{self.inner_iter}", delta_T)
                self.T[i] = delta_T @ self.T[i]
                total_cost += cost
                update_actor_transform(self.actors[i], self.T[i])

            avg_cost = total_cost / self.N

            # Check inner convergence
            last = self.last_avg_cost
            if last is not None:
                abs_change = abs(last - avg_cost)
                print(f"  Inner {self.inner_iter}: Avg Cost = {avg_cost:.6f}, Change = {abs_change:.8f}")
                inner_converged = abs_change < self.inner_conv_threshold or self.inner_iter >= self.max_inner
            else:
                print(f"  Inner {self.inner_iter}: Avg Cost = {avg_cost:.6f}")
                inner_converged = False

            self.last_avg_cost = avg_cost

            self.text_display.SetInput(
                f"Outer: {self.outer_iter}/{self.max_outer}  "
                f"Inner: {self.inner_iter}\n"
                f"Avg Cost: {avg_cost:.6f}")
            obj.GetRenderWindow().Render()

            # --- If inner loop converged: update mean shape, start next outer ---
            if inner_converged:
                print(f"  Inner loop converged at step {self.inner_iter}. Cost = {avg_cost:.6f}")

                # Check outer convergence
                outer_converged = False
                last_outer = self.last_outer_cost
                if last_outer is not None:
                    outer_change = abs(last_outer - avg_cost)
                    print(f"  Outer cost change = {outer_change:.6f}")
                    outer_converged = outer_change < self.outer_conv_threshold
                self.last_outer_cost = avg_cost

                if outer_converged or self.outer_iter >= self.max_outer:
                    self.finish(obj)
                else:
                    # Compute new mean shape from all aligned meshes
                    print("  Updating mean shape...")
                    new_mean = compute_mean_shape(
                        self.meshes_verts, self.T, self.ref_shape_verts)
                    self.ref_shape_verts = new_mean

                    # Update VTK reference actor
                    update_actor_vertices(self.ref_actor, self.ref_shape_verts)
                    obj.GetRenderWindow().Render()

                    # Record mean shape update
                    for i in range(self.N):
                        self.history.append(i, f"mean_update_{self.outer_iter}", np.eye(4))

                    # Reset inner loop for next outer iteration
                    self.inner_iter = 0

        def finish(self, obj):
            if not self.done:
                self.done = True
                print(f"\nGroup-wise ICP complete after {self.outer_iter} outer iterations.")
                print("Computing final mean shape...")

                # Final mean shape
                mean_shape = compute_mean_shape(
                    self.meshes_verts, self.T, self.ref_shape_verts)
                update_actor_vertices(self.ref_actor, mean_shape)

                for a in self.actors:
                    a.SetVisibility(False)

                self.ref_actor.GetProperty().SetRepresentationToSurface()
                self.ref_actor.GetProperty().SetColor(0.8, 0.8, 0.8)
                self.ref_actor.GetProperty().SetOpacity(1.0)

                self.text_display.SetInput(
                    f"Converged! Outer: {self.outer_iter}, "
                    f"Final Cost: {self.last_avg_cost:.6f}")
                obj.GetRenderWindow().Render()

                export_results(self.T, mean_shape,
                               self.ref_template_faces, self.history)

    icp_cb = GroupwiseICPTimerCallback(
        meshes_verts, meshes_faces, T_matrices, actors,
        ref_shape_verts, ref_template_faces, ref_actor,
        renderer, text_actor, history,
        max_outer=5, max_inner=20)

    interactor.AddObserver('TimerEvent', icp_cb.execute)
    interactor.Initialize()
    interactor.CreateRepeatingTimer(100)

    print("READY — True Group-wise ICP starting...")
    window.Render()
    interactor.Start()


if __name__ == "__main__":
    main()