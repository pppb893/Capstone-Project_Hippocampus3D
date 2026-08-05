import os
import glob
import csv
import argparse
import tkinter as tk
from tkinter import filedialog
import vtk
import numpy as np

vtk.vtkObject.GlobalWarningDisplayOff()

# =============================================================================
# Helper: Directory Picker
# =============================================================================
def popup_select_directory(title):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askdirectory(title=title, initialdir=os.getcwd())
    root.destroy()
    return path

# =============================================================================
# VTK IO helpers
# =============================================================================
def load_polydata_smoothed(filepath):
    """อ่านไฟล์ .vtk และคำนวณ Normals สำหรับ Smooth Shading"""
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filepath)
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetNumberOfPoints() == 0:
        return None
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.Update()
    return normals.GetOutput()

# =============================================================================
# Viewer Class
# =============================================================================
class PLSDAAugmentedViewer:
    MODE_OVERLAY = "OVERLAY"
    MODE_SLIDESHOW = "SLIDESHOW"
    REF_BOX_HALF = 20.0  # ขอบเขตคร่าวๆ สำหรับกล่องอ้างอิง

    def __init__(self, aug_dir):
        self.aug_dir = aug_dir.replace("\\", "/")

        # 1. ค้นหาไฟล์ VTK ในโฟลเดอร์โดยตรง (หรือในโฟลเดอร์ย่อยย้อนหลังเพื่อความเข้ากันได้)
        self.files = []
        self.class_labels = []

        direct_files = sorted(glob.glob(os.path.join(self.aug_dir, "*.vtk")))
        
        healthy_count = 0
        diseased_count = 0
        
        if direct_files:
            for f in direct_files:
                self.files.append(f)
                basename = os.path.basename(f)
                if "healthy" in basename.lower() or "normal" in basename.lower():
                    self.class_labels.append("Healthy")
                    healthy_count += 1
                else:
                    self.class_labels.append("Diseased")
                    diseased_count += 1
        else:
            # ย้อนกลับไปค้นหาในโฟลเดอร์ย่อย (Healthy/Diseased) เผื่อรันแบบเก่า
            healthy_files = sorted(glob.glob(os.path.join(self.aug_dir, "Healthy", "*.vtk")))
            self.files.extend(healthy_files)
            self.class_labels.extend(["Healthy"] * len(healthy_files))
            healthy_count = len(healthy_files)

            diseased_files = sorted(glob.glob(os.path.join(self.aug_dir, "Diseased", "*.vtk")))
            self.files.extend(diseased_files)
            self.class_labels.extend(["Diseased"] * len(diseased_files))
            diseased_count = len(diseased_files)

        if not self.files:
            print(f"[ERROR] No augmented VTK files found in: {self.aug_dir}")
            self.meshes = []
            return

        print(f"Found {len(self.files)} augmented meshes ({healthy_count} Healthy, {diseased_count} Diseased).")

        # 2. โหลดไฟล์ Metadata CSV
        self.metadata = {}
        metadata_path = os.path.join(self.aug_dir, "interpolated_metadata.csv")
        if os.path.exists(metadata_path):
            print(f"Loading metadata: {metadata_path}")
            try:
                with open(metadata_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self.metadata[row["Filename"]] = row
            except Exception as e:
                print(f"Warning: Could not read metadata CSV: {e}")
        else:
            print("Warning: interpolated_metadata.csv not found in target folder.")

        # 3. โหลดและสร้าง Mesh
        self.meshes = []
        self.filenames = []
        for i, fpath in enumerate(self.files):
            basename = os.path.basename(fpath)
            poly = load_polydata_smoothed(fpath)
            if poly is None:
                continue
            self.meshes.append(poly)
            self.filenames.append(basename)

        # คำนวณขอบเขต (Bounds) สำหรับสร้างกล่องและตั้งมุมกล้อง
        if self.meshes:
            all_bounds = []
            for m in self.meshes:
                all_bounds.append(m.GetBounds())
            all_bounds = np.array(all_bounds)
            min_x, max_x = all_bounds[:, 0].min(), all_bounds[:, 1].max()
            min_y, max_y = all_bounds[:, 2].min(), all_bounds[:, 3].max()
            min_z, max_z = all_bounds[:, 4].min(), all_bounds[:, 5].max()
            self.bounds_center = [(min_x + max_x)/2.0, (min_y + max_y)/2.0, (min_z + max_z)/2.0]
            self.REF_BOX_HALF = max(max_x - min_x, max_y - min_y, max_z - min_z) * 0.6
        else:
            self.bounds_center = [0.0, 0.0, 0.0]

        # State
        self.current_idx = 0
        self.mode = self.MODE_OVERLAY
        self.wireframe = False
        self.opacity_overlay = 0.25

        self.setup_vtk()
        self.build_actors()
        self.apply_mode()

    def setup_vtk(self):
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.07, 0.07, 0.1)
        self.renderer.SetUseDepthPeeling(True)
        self.renderer.SetMaximumNumberOfPeels(8)
        self.renderer.SetOcclusionRatio(0.0)

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetAlphaBitPlanes(1)
        self.render_window.SetMultiSamples(4)
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1300, 950)
        self.render_window.SetWindowName(f"PLS-DA Augmented Mesh Viewer ({len(self.meshes)} generated)")

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        # info overlay (top-left)
        self.text_actor = vtk.vtkTextActor()
        tp = self.text_actor.GetTextProperty()
        tp.SetFontSize(15)
        tp.SetColor(1.0, 1.0, 1.0)
        tp.BoldOn()
        tp.SetShadow(True)
        self.text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.text_actor.GetPositionCoordinate().SetValue(0.02, 0.88)
        self.renderer.AddActor2D(self.text_actor)

        # help overlay (bottom-left)
        help_actor = vtk.vtkTextActor()
        hp = help_actor.GetTextProperty()
        hp.SetFontSize(12)
        hp.SetColor(0.65, 0.85, 1.0)
        hp.SetShadow(True)
        help_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        help_actor.GetPositionCoordinate().SetValue(0.02, 0.02)
        help_actor.SetInput(
            "[1] Overlay   [2] Slideshow   [W] Wireframe\n"
            "[N / P / Space / scroll / Right / Left]  Next / Prev (slideshow)\n"
            "[+] / [-]  Overlay opacity     [R] Reset camera     [Q / Esc] Quit\n"
            "Colors:   BLUE = Healthy   RED = Diseased"
        )
        self.renderer.AddActor2D(help_actor)

        # camera-follower orientation axes
        axes_marker = vtk.vtkAxesActor()
        axes_marker.SetXAxisLabelText("X")
        axes_marker.SetYAxisLabelText("Y")
        axes_marker.SetZAxisLabelText("Z")
        self.axes_widget = vtk.vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(axes_marker)
        self.axes_widget.SetInteractor(self.interactor)
        self.axes_widget.SetViewport(0.82, 0.0, 1.0, 0.22)
        self.axes_widget.SetEnabled(1)
        self.axes_widget.InteractiveOff()

        # reference box
        outline = vtk.vtkOutlineSource()
        c = self.bounds_center
        h = self.REF_BOX_HALF
        outline.SetBounds(c[0]-h, c[0]+h, c[1]-h, c[1]+h, c[2]-h, c[2]+h)
        out_mapper = vtk.vtkPolyDataMapper()
        out_mapper.SetInputConnection(outline.GetOutputPort())
        out_actor = vtk.vtkActor()
        out_actor.SetMapper(out_mapper)
        out_actor.GetProperty().SetColor(0.35, 0.35, 0.45)
        out_actor.GetProperty().SetOpacity(0.4)
        out_actor.GetProperty().SetLineWidth(1)
        self.renderer.AddActor(out_actor)

        # observers
        self.interactor.AddObserver("KeyPressEvent", self.on_key_press)
        self.interactor.AddObserver("MouseWheelForwardEvent", self.on_wheel_forward)
        self.interactor.AddObserver("MouseWheelBackwardEvent", self.on_wheel_backward)

    def build_actors(self):
        self.actors = []
        for i, poly in enumerate(self.meshes):
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly)
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            # Color: Blue for Healthy, Red for Diseased
            lbl = self.class_labels[i]
            if lbl == "Healthy":
                r, g, b = (0.2549, 0.4118, 0.8824)  # royalblue
            else:
                r, g, b = (0.8627, 0.0784, 0.2353)  # crimson

            prop = actor.GetProperty()
            prop.SetColor(r, g, b)
            prop.SetInterpolationToGouraud()
            prop.SetAmbient(0.25)
            prop.SetDiffuse(0.75)
            prop.SetSpecular(0.15)
            self.renderer.AddActor(actor)
            self.actors.append(actor)

    def apply_mode(self):
        for i, a in enumerate(self.actors):
            prop = a.GetProperty()
            if self.wireframe:
                prop.SetRepresentationToWireframe()
            else:
                prop.SetRepresentationToSurface()

            if self.mode == self.MODE_OVERLAY:
                a.SetVisibility(True)
                prop.SetOpacity(self.opacity_overlay)
            else:  # slideshow
                a.SetVisibility(i == self.current_idx)
                prop.SetOpacity(1.0)

        self._update_info_text()
        self.render_window.Render()

    def _update_info_text(self):
        if self.mode == self.MODE_OVERLAY:
            txt = (
                f"MODE: OVERLAY   |   {len(self.actors)} synthetic meshes   |   "
                f"opacity = {self.opacity_overlay:.2f}\n"
                f"Healthy (Blue): {self.class_labels.count('Healthy')} meshes\n"
                f"Diseased (Red): {self.class_labels.count('Diseased')} meshes"
            )
        else:
            fname = self.filenames[self.current_idx]
            lbl = self.class_labels[self.current_idx]
            
            # ดึงประวัติผสมจาก metadata
            meta = self.metadata.get(fname, None)
            if meta:
                parent_info = (
                    f"Parent A: {meta['Parent_A_Subject']} ({meta['Parent_A_Group']})\n"
                    f"Parent B: {meta['Parent_B_Subject']} ({meta['Parent_B_Group']})\n"
                    f"Interpolation Weight (Alpha): {meta['Alpha']} (closer to {meta['Closest_Parent']})"
                )
            else:
                parent_info = "No parent metadata found."

            txt = (
                f"MODE: SLIDESHOW   [{self.current_idx+1}/{len(self.actors)}]\n"
                f"Filename: {fname}  ({lbl})\n"
                f"-----------------------------------------\n"
                f"{parent_info}"
            )
        self.text_actor.SetInput(txt)

    def reset_camera(self):
        c = self.bounds_center
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(c[0], c[1] - 120, c[2] + 20)
        cam.SetFocalPoint(c[0], c[1], c[2])
        cam.SetViewUp(0, 0, 1)
        self.renderer.ResetCamera()
        self.render_window.Render()

    # ---------------- events ----------------
    def on_wheel_forward(self, obj, event):
        if self.mode == self.MODE_SLIDESHOW:
            self.current_idx = (self.current_idx + 1) % len(self.actors)
            self.apply_mode()

    def on_wheel_backward(self, obj, event):
        if self.mode == self.MODE_SLIDESHOW:
            self.current_idx = (self.current_idx - 1) % len(self.actors)
            self.apply_mode()

    def on_key_press(self, obj, event):
        key = (obj.GetKeySym() or "").lower()
        if key == "1":
            self.mode = self.MODE_OVERLAY
            self.apply_mode()
        elif key == "2":
            self.mode = self.MODE_SLIDESHOW
            self.apply_mode()
        elif key in ("n", "right", "space"):
            if self.mode != self.MODE_SLIDESHOW:
                self.mode = self.MODE_SLIDESHOW
            self.current_idx = (self.current_idx + 1) % len(self.actors)
            self.apply_mode()
        elif key in ("p", "left"):
            if self.mode != self.MODE_SLIDESHOW:
                self.mode = self.MODE_SLIDESHOW
            self.current_idx = (self.current_idx - 1) % len(self.actors)
            self.apply_mode()
        elif key == "w":
            self.wireframe = not self.wireframe
            self.apply_mode()
        elif key in ("plus", "equal", "kp_add"):
            self.opacity_overlay = min(1.0, self.opacity_overlay + 0.05)
            self.apply_mode()
        elif key in ("minus", "underscore", "kp_subtract"):
            self.opacity_overlay = max(0.05, self.opacity_overlay - 0.05)
            self.apply_mode()
        elif key == "r":
            self.reset_camera()
        elif key in ("q", "escape"):
            self.interactor.TerminateApp()

    def start(self):
        if not self.meshes:
            return
        print("\n" + "=" * 56)
        print("PLS-DA AUGMENTED MESH VIEWER")
        print("  Modes:        [1] Overlay   [2] Slideshow")
        print("  Slideshow:    [N/P/space/scroll/Right/Left]")
        print("  Wireframe:    [W] toggle")
        print("  Opacity:      [+] / [-]  (overlay mode only)")
        print("  Camera:       [R] reset, drag mouse to rotate/pan/zoom")
        print("  Quit:         [Q] or [Esc]")
        print("=" * 56 + "\n")

        self.reset_camera()
        self.interactor.Initialize()
        self.interactor.Start()

# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("--- View PLS-DA Augmented Meshes ---")
    print("=" * 60)

    print("\nSelect the 'plsda_interpolated_surfaces' folder...")
    aug_dir = popup_select_directory("Select plsda_interpolated_surfaces Folder")
    if not aug_dir:
        print("Canceled.")
        return

    # 1. If user picks "Healthy" or "Diseased" subfolder, go one level up to parent
    base = os.path.basename(aug_dir.rstrip("\\/")).lower()
    if base in ("healthy", "diseased"):
        aug_dir = os.path.dirname(aug_dir.rstrip("\\/"))

    # 2. If user picks "spharm_results" folder, go to parent first
    base = os.path.basename(aug_dir.rstrip("\\/")).lower()
    if base == "spharm_results":
        aug_dir = os.path.dirname(aug_dir.rstrip("\\/"))

    # 3. If user picks parent output directory, append "plsda_interpolated_surfaces" if it exists
    base = os.path.basename(aug_dir.rstrip("\\/")).lower()
    if base != "plsda_interpolated_surfaces":
        candidate = os.path.join(aug_dir, "plsda_interpolated_surfaces")
        if os.path.isdir(candidate):
            aug_dir = candidate

    viewer = PLSDAAugmentedViewer(aug_dir)
    viewer.start()

if __name__ == "__main__":
    main()
