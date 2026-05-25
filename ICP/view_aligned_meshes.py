import vtk
vtk.vtkObject.GlobalWarningDisplayOff()
import os
import glob
import argparse
import colorsys
import tkinter as tk
from tkinter import filedialog


# =============================================================================
# Mesh loading helpers
# =============================================================================

def load_mesh_from_nifti(filepath, smooth_iter=20):
    """อ่าน NIfTI label -> surface mesh (smoothed + normals)"""
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(filepath)
    reader.Update()

    dmc = vtk.vtkDiscreteMarchingCubes()
    dmc.SetInputData(reader.GetOutput())
    dmc.GenerateValues(1, 1, 100)
    dmc.Update()
    if dmc.GetOutput().GetNumberOfPoints() == 0:
        return None

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(dmc.GetOutput())
    smoother.SetNumberOfIterations(smooth_iter)
    smoother.BoundarySmoothingOn()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetPassBand(0.05)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(smoother.GetOutput())
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.Update()
    return normals.GetOutput()


def load_mesh_from_ply(filepath):
    reader = vtk.vtkPLYReader()
    reader.SetFileName(filepath)
    reader.Update()
    poly = reader.GetOutput()
    if poly.GetNumberOfPoints() == 0:
        return None
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.Update()
    return normals.GetOutput()


# =============================================================================
# Viewer
# =============================================================================

class AlignmentViewer:
    """
    Viewer สำหรับตรวจดูว่า mesh หลังทำ ICP align ตรงกันหรือไม่
      - OVERLAY  : ซ้อนทุก mesh แบบกึ่งโปร่งใส, สีต่างกัน -> เห็นทันทีว่า
                   ตัวไหน orientation/ตำแหน่งไม่ตรงกับกลุ่ม
      - SLIDESHOW: เปิดทีละตัว พร้อม mean shape เป็น wireframe อ้างอิง
    """

    MODE_OVERLAY = "OVERLAY"
    MODE_SLIDESHOW = "SLIDESHOW"

    # ขนาด reference box (ตรงกับ OUTPUT_VOXELS * OUTPUT_SPACING / 2 ของ main2.py)
    # หลัง global normalization mesh อยู่ใน [-1, 1] -> box 128*0.02/2 = 1.28
    REF_BOX_HALF_MM = 1.28

    def __init__(self, output_dir):
        self.output_dir = output_dir.replace("\\", "/")
        if os.path.basename(self.output_dir).lower() == "aligned_nifti":
            self.aligned_dir = self.output_dir
            self.parent_dir = os.path.dirname(self.aligned_dir)
        else:
            self.aligned_dir = os.path.join(self.output_dir, "aligned_nifti")
            self.parent_dir = self.output_dir

        self.files = sorted(glob.glob(os.path.join(self.aligned_dir, "*.nii.gz")))
        if not self.files:
            print(f"[ERROR] No *.nii.gz files in {self.aligned_dir}")
            self.meshes = []
            return
        print(f"Found {len(self.files)} aligned mesh files in {self.aligned_dir}")

        # state
        self.current_idx = 0
        self.mode = self.MODE_OVERLAY
        self.wireframe = False
        self.opacity_overlay = 0.25
        self.show_mean = True

        # load all meshes (cache)
        self.meshes = []
        self.basenames = []
        for i, f in enumerate(self.files):
            print(f"  [{i+1}/{len(self.files)}] {os.path.basename(f)}")
            poly = load_mesh_from_nifti(f)
            if poly is None:
                print(f"      skipped (empty mesh)")
                continue
            self.meshes.append(poly)
            self.basenames.append(os.path.basename(f).replace("_aligned.nii.gz", ""))

        if not self.meshes:
            print("[ERROR] No valid meshes loaded.")
            return

        # mean_shape.ply เป็น optional (ถ้าใช้ main2.py จะมีอยู่)
        mean_path = os.path.join(self.parent_dir, "mean_shape.ply")
        if os.path.exists(mean_path):
            print(f"Loading mean shape: {mean_path}")
            self.mean_poly = load_mesh_from_ply(mean_path)
        else:
            self.mean_poly = None
            print("(mean_shape.ply not found - reference outline disabled)")

        self.setup_vtk()
        self.build_actors()
        self.apply_mode()

    # ---------------- VTK scaffolding ----------------

    def setup_vtk(self):
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.07, 0.07, 0.1)
        # depth peeling -> transparency หลายชั้นแสดงผลถูกต้อง
        self.renderer.SetUseDepthPeeling(True)
        self.renderer.SetMaximumNumberOfPeels(8)
        self.renderer.SetOcclusionRatio(0.0)

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetAlphaBitPlanes(1)
        self.render_window.SetMultiSamples(0)
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1300, 950)
        self.render_window.SetWindowName("Aligned Mesh Viewer")

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        # ---------- info overlay (top-left) ----------
        self.text_actor = vtk.vtkTextActor()
        tp = self.text_actor.GetTextProperty()
        tp.SetFontSize(16)
        tp.SetColor(1, 1, 1)
        tp.BoldOn()
        tp.SetShadow(True)
        self.text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.text_actor.GetPositionCoordinate().SetValue(0.02, 0.90)
        self.renderer.AddActor2D(self.text_actor)

        # ---------- help overlay (bottom-left) ----------
        help_actor = vtk.vtkTextActor()
        hp = help_actor.GetTextProperty()
        hp.SetFontSize(12)
        hp.SetColor(0.65, 0.85, 1.0)
        hp.SetShadow(True)
        help_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        help_actor.GetPositionCoordinate().SetValue(0.02, 0.02)
        help_actor.SetInput(
            "[1] Overlay   [2] Slideshow   [M] Toggle mean shape   [W] Wireframe\n"
            "[N / P / Space / scroll / Right / Left]  Next / Prev (slideshow)\n"
            "[+] / [-]  Overlay opacity     [R] Reset camera     [Q / Esc] Quit"
        )
        self.renderer.AddActor2D(help_actor)

        # ---------- camera-follower axes (bottom-right) ----------
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

        # ---------- world-space axes at origin ----------
        world_axes = vtk.vtkAxesActor()
        world_axes.SetTotalLength(20, 20, 20)
        world_axes.AxisLabelsOff()
        world_axes.SetShaftTypeToLine()
        for ax_prop in (world_axes.GetXAxisShaftProperty(),
                        world_axes.GetYAxisShaftProperty(),
                        world_axes.GetZAxisShaftProperty()):
            ax_prop.SetLineWidth(2)
        self.renderer.AddActor(world_axes)

        # ---------- reference bounding box ----------
        outline = vtk.vtkOutlineSource()
        h = self.REF_BOX_HALF_MM
        outline.SetBounds(-h, h, -h, h, -h, h)
        out_mapper = vtk.vtkPolyDataMapper()
        out_mapper.SetInputConnection(outline.GetOutputPort())
        out_actor = vtk.vtkActor()
        out_actor.SetMapper(out_mapper)
        out_actor.GetProperty().SetColor(0.35, 0.35, 0.45)
        out_actor.GetProperty().SetOpacity(0.6)
        out_actor.GetProperty().SetLineWidth(1)
        self.renderer.AddActor(out_actor)

        # ---------- observers ----------
        self.interactor.AddObserver("KeyPressEvent", self.on_key_press)
        self.interactor.AddObserver("MouseWheelForwardEvent", self.on_wheel_forward)
        self.interactor.AddObserver("MouseWheelBackwardEvent", self.on_wheel_backward)

    def build_actors(self):
        """สร้าง actor 1 ตัวต่อ mesh + actor mean shape (ถ้ามี)"""
        N = len(self.meshes)
        self.actors = []
        for i, poly in enumerate(self.meshes):
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly)
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            # ใช้ golden-ratio hue spacing เพื่อให้สีกระจายแม้ N ใหญ่
            hue = (i * 0.61803398875) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
            prop = actor.GetProperty()
            prop.SetColor(r, g, b)
            prop.SetInterpolationToGouraud()
            prop.SetAmbient(0.25)
            prop.SetDiffuse(0.75)
            prop.SetSpecular(0.15)
            self.renderer.AddActor(actor)
            self.actors.append(actor)

        self.mean_actor = None
        if self.mean_poly is not None:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self.mean_poly)
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            p = actor.GetProperty()
            p.SetColor(1.0, 1.0, 1.0)
            p.SetOpacity(0.5)
            p.SetRepresentationToWireframe()
            p.SetLineWidth(1)
            self.renderer.AddActor(actor)
            self.mean_actor = actor

    # ---------------- mode / state ----------------

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
        if self.mean_actor is not None:
            self.mean_actor.SetVisibility(self.show_mean)
        self._update_info_text()
        self.render_window.Render()

    def _update_info_text(self):
        if self.mean_actor is None:
            mean_state = "N/A"
        else:
            mean_state = "ON" if self.show_mean else "OFF"

        if self.mode == self.MODE_OVERLAY:
            txt = (
                f"MODE: OVERLAY   |   {len(self.actors)} meshes   |   "
                f"opacity = {self.opacity_overlay:.2f}\n"
                f"Mean shape (white wireframe): {mean_state}\n"
                f"Reference box: +/- {self.REF_BOX_HALF_MM:.0f} mm"
            )
        else:
            name = self.basenames[self.current_idx]
            b = self.meshes[self.current_idx].GetBounds()
            txt = (
                f"MODE: SLIDESHOW   [{self.current_idx+1}/{len(self.actors)}]\n"
                f"Subject: {name}\n"
                f"Bounds  X[{b[0]:6.1f},{b[1]:6.1f}]  "
                f"Y[{b[2]:6.1f},{b[3]:6.1f}]  Z[{b[4]:6.1f},{b[5]:6.1f}] mm\n"
                f"Mean shape: {mean_state}"
            )
        self.text_actor.SetInput(txt)

    def reset_camera(self):
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(0, -150, 30)
        cam.SetFocalPoint(0, 0, 0)
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
        key = obj.GetKeySym() or ""
        k = key.lower()

        if k == "1":
            self.mode = self.MODE_OVERLAY
            self.apply_mode()
        elif k == "2":
            self.mode = self.MODE_SLIDESHOW
            self.apply_mode()
        elif k in ("n", "right", "space"):
            if self.mode != self.MODE_SLIDESHOW:
                self.mode = self.MODE_SLIDESHOW
            self.current_idx = (self.current_idx + 1) % len(self.actors)
            self.apply_mode()
        elif k in ("p", "left"):
            if self.mode != self.MODE_SLIDESHOW:
                self.mode = self.MODE_SLIDESHOW
            self.current_idx = (self.current_idx - 1) % len(self.actors)
            self.apply_mode()
        elif k == "m":
            self.show_mean = not self.show_mean
            self.apply_mode()
        elif k == "w":
            self.wireframe = not self.wireframe
            self.apply_mode()
        elif k in ("plus", "equal", "kp_add"):
            self.opacity_overlay = min(1.0, self.opacity_overlay + 0.05)
            self.apply_mode()
        elif k in ("minus", "underscore", "kp_subtract"):
            self.opacity_overlay = max(0.05, self.opacity_overlay - 0.05)
            self.apply_mode()
        elif k == "r":
            self.reset_camera()
        elif k in ("q", "escape"):
            self.interactor.TerminateApp()

    # ---------------- entrypoint ----------------

    def start(self):
        if not self.meshes:
            return
        print("\n" + "=" * 56)
        print("ALIGNED MESH VIEWER")
        print("  Modes:        [1] Overlay   [2] Slideshow")
        print("  Slideshow:    [N/P/space/scroll/Right/Left]")
        print("  Mean shape:   [M] toggle")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    out_dir = args.output_dir
    if not out_dir:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        out_dir = filedialog.askdirectory(
            title="Select output folder (e.g. output_labelsTr or aligned_nifti)"
        )
        root.destroy()

    if not out_dir:
        print("No folder selected.")
        return

    viewer = AlignmentViewer(out_dir)
    viewer.start()


if __name__ == "__main__":
    main()
