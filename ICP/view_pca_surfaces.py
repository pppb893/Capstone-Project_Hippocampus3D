import os
import tkinter as tk
from tkinter import filedialog
import vtk


# =============================================================================
# IO helpers
# =============================================================================

def popup_select_directory(title):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askdirectory(title=title, initialdir=os.getcwd())
    root.destroy()
    return path


def load_polydata_smoothed(filepath):
    """อ่าน .vtk + compute normals (smooth shading) -> vtkPolyData"""
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


def mesh_extents(poly):
    b = poly.GetBounds()
    return (b[1] - b[0], b[3] - b[2], b[5] - b[4])


def color_for_weight(weight_str):
    """สีตาม SD weight: -SD ฟ้า, +SD แดง (เข้มขึ้นตาม |SD|), Mean ขาว"""
    import re
    wl = weight_str.lower()
    m = re.search(r"(\d+)", wl)
    sd = float(m.group(1)) if m else 1.0
    t = min(sd / 3.0, 1.0)
    if 'minus' in wl:
        return (0.35 - 0.15*t, 0.65 - 0.15*t, 1.0 - 0.05*t)
    if 'plus' in wl:
        return (1.0 - 0.05*t, 0.55 - 0.2*t, 0.55 - 0.2*t)
    return (0.92, 0.92, 0.92)


# =============================================================================
# Viewer (sub-viewport-per-mesh layout)
# =============================================================================

class PCAVariationsViewer:
    """
    แต่ละ mesh แสดงใน **sub-viewport ของตัวเอง** -> camera แยกกัน
    ขนาด mesh ที่ต่างกันมาก (เช่น +2SD ที่ degenerate) จะไม่กระทบ mesh ตัวอื่น
    ทุก panel auto-fit mesh ของตัวเองให้เห็นชัด
    """

    def __init__(self, vtk_files, title):
        # ---------- load all meshes (+ print bounds ใช้ debug) ----------
        self.polys = []
        self.weights = []
        print("\nLoaded meshes (bounds in original units):")
        print(f"  {'Name':<20} {'#pts':>7}   {'X size':>9} {'Y size':>9} {'Z size':>9}")
        print("  " + "-" * 60)
        for f in vtk_files:
            poly = load_polydata_smoothed(f)
            if poly is None:
                print(f"  WARN: skipping empty {os.path.basename(f)}")
                continue
            base = os.path.basename(f).replace('.vtk', '')
            # อ่าน weight label: minus3SD / plus2SD / Mean
            # ทำงานกับชื่อแบบ "PC1_plus2SD.vtk" และ "pca_model_All_mean.vtk"
            low = base.lower()
            if 'minus' in low or 'plus' in low:
                w = base.split('_')[-1]
            else:
                w = "Mean"
            ext = mesh_extents(poly)
            print(f"  {w:<20} {poly.GetNumberOfPoints():>7}   "
                  f"{ext[0]:>9.2f} {ext[1]:>9.2f} {ext[2]:>9.2f}")
            self.polys.append(poly)
            self.weights.append(w)

        if not self.polys:
            print("[ERROR] No valid meshes to display.")
            return

        # หา reference (Mean) สำหรับ warn เรื่อง outlier
        try:
            ref_idx = self.weights.index("Mean")
        except ValueError:
            ref_idx = len(self.polys) // 2
        ref_ext = mesh_extents(self.polys[ref_idx])
        print(f"\n  Reference ('{self.weights[ref_idx]}'): "
              f"{ref_ext[0]:.2f} x {ref_ext[1]:.2f} x {ref_ext[2]:.2f}")
        for w, p in zip(self.weights, self.polys):
            e = mesh_extents(p)
            denom = max(ref_ext[0], 1e-9)
            ratio = e[0] / denom
            if ratio > 2.0 or ratio < 0.4:
                print(f"  [WARN] '{w}' X-size is {ratio:.2f}x of reference "
                      f"-> likely degenerate PCA reconstruction")

        # state
        self.wireframe = False
        self.view_axis = "front"  # front / top / side

        self.setup_window(title)
        self.build_panels()
        self.apply_view()
        self.start()

    # ---------------- VTK scaffolding ----------------

    def setup_window(self, title):
        N = len(self.polys)
        win_w = max(900, min(1900, N * 240))
        win_h = 640
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetAlphaBitPlanes(1)
        self.render_window.SetMultiSamples(8)
        self.render_window.SetSize(win_w, win_h)
        self.render_window.SetWindowName(title)

        # background renderer (สำหรับ overlay text ของ help)
        self.bg_renderer = vtk.vtkRenderer()
        self.bg_renderer.SetViewport(0, 0, 1, 1)
        self.bg_renderer.SetBackground(0.07, 0.07, 0.1)
        self.bg_renderer.InteractiveOff()
        self.bg_renderer.SetLayer(0)
        self.render_window.SetNumberOfLayers(2)
        self.render_window.AddRenderer(self.bg_renderer)

        # help text
        help_actor = vtk.vtkTextActor()
        hp = help_actor.GetTextProperty()
        hp.SetFontSize(12)
        hp.SetColor(0.65, 0.85, 1.0)
        hp.SetShadow(True)
        help_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        help_actor.GetPositionCoordinate().SetValue(0.005, 0.005)
        help_actor.SetInput(
            "[F] Front  [T] Top  [S] Side   [W] Wireframe   [R] Reset   [Q/Esc] Quit"
        )
        self.bg_renderer.AddActor2D(help_actor)

        # interactor
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.interactor.AddObserver("KeyPressEvent", self.on_key_press)

    def build_panels(self):
        """สร้าง renderer 1 ตัวต่อ mesh, viewport ติดกันเป็นแถว"""
        N = len(self.polys)
        self.renderers = []
        self.actors = []
        top_margin = 0.04
        bottom_margin = 0.08
        for i, (poly, w) in enumerate(zip(self.polys, self.weights)):
            x0 = i / N
            x1 = (i + 1) / N

            ren = vtk.vtkRenderer()
            ren.SetViewport(x0, bottom_margin, x1, 1.0 - top_margin)
            ren.SetBackground(0.07, 0.07, 0.1)
            ren.SetLayer(1)
            ren.SetUseDepthPeeling(True)
            ren.SetMaximumNumberOfPeels(4)
            # parallel projection -> auto-fit ทำงานสม่ำเสมอ
            ren.GetActiveCamera().SetParallelProjection(True)

            # camera lights only (เห็น mesh ชัดทุก panel)
            ren.RemoveAllLights()
            key_light = vtk.vtkLight()
            key_light.SetLightTypeToCameraLight()
            key_light.SetPosition(0.3, 0.5, 1.0)
            key_light.SetFocalPoint(0, 0, 0)
            key_light.SetIntensity(0.9)
            ren.AddLight(key_light)
            fill_light = vtk.vtkLight()
            fill_light.SetLightTypeToCameraLight()
            fill_light.SetPosition(-0.4, -0.3, 0.7)
            fill_light.SetFocalPoint(0, 0, 0)
            fill_light.SetIntensity(0.35)
            ren.AddLight(fill_light)

            # actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly)
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetColor(*color_for_weight(w))
            prop.SetInterpolationToGouraud()
            prop.SetAmbient(0.22)
            prop.SetDiffuse(0.78)
            prop.SetSpecular(0.25)
            prop.SetSpecularPower(28)
            ren.AddActor(actor)

            # label (panel top)
            label = vtk.vtkTextActor()
            tp = label.GetTextProperty()
            tp.SetFontSize(18)
            tp.SetColor(1, 1, 1)
            tp.SetJustificationToCentered()
            tp.SetVerticalJustificationToTop()
            tp.BoldOn()
            tp.SetShadow(True)
            label.SetInput(w)
            label.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            label.GetPositionCoordinate().SetValue(0.5, 0.97)
            ren.AddActor2D(label)

            # bounds info (panel bottom)
            ext = mesh_extents(poly)
            info = vtk.vtkTextActor()
            ip = info.GetTextProperty()
            ip.SetFontSize(11)
            ip.SetColor(0.6, 0.8, 1.0)
            ip.SetJustificationToCentered()
            ip.SetVerticalJustificationToBottom()
            info.SetInput(f"{ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f}")
            info.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            info.GetPositionCoordinate().SetValue(0.5, 0.02)
            ren.AddActor2D(info)

            self.render_window.AddRenderer(ren)
            self.renderers.append(ren)
            self.actors.append(actor)

    # ---------------- camera ----------------

    def apply_view(self):
        """ตั้ง camera ของแต่ละ panel ให้ auto-fit mesh ตัวเอง"""
        for ren, poly in zip(self.renderers, self.polys):
            b = poly.GetBounds()
            cx = (b[0] + b[1]) / 2.0
            cy = (b[2] + b[3]) / 2.0
            cz = (b[4] + b[5]) / 2.0
            sx = b[1] - b[0]
            sy = b[3] - b[2]
            sz = b[5] - b[4]
            diag = max(sx, sy, sz)
            far_dist = max(diag * 4.0, 1.0)

            cam = ren.GetActiveCamera()
            cam.SetParallelProjection(True)
            cam.SetFocalPoint(cx, cy, cz)

            if self.view_axis == "front":
                cam.SetPosition(cx, cy - far_dist, cz)
                cam.SetViewUp(0, 0, 1)
                fit_h = max(sz, sx)
            elif self.view_axis == "top":
                cam.SetPosition(cx, cy, cz + far_dist)
                cam.SetViewUp(0, 1, 0)
                fit_h = max(sy, sx)
            else:  # side
                cam.SetPosition(cx + far_dist, cy, cz)
                cam.SetViewUp(0, 0, 1)
                fit_h = max(sz, sy)

            # parallel scale = half of viewport height in world units
            # เพิ่ม margin 15%
            cam.SetParallelScale(max(fit_h * 0.6, 0.01))
            ren.ResetCameraClippingRange()

        self.render_window.Render()

    # ---------------- key handling ----------------

    def on_key_press(self, obj, event):
        key = (obj.GetKeySym() or "").lower()
        if key == "f":
            self.view_axis = "front"
            self.apply_view()
        elif key == "t":
            self.view_axis = "top"
            self.apply_view()
        elif key == "s":
            self.view_axis = "side"
            self.apply_view()
        elif key == "w":
            self.wireframe = not self.wireframe
            for a in self.actors:
                if self.wireframe:
                    a.GetProperty().SetRepresentationToWireframe()
                else:
                    a.GetProperty().SetRepresentationToSurface()
            self.render_window.Render()
        elif key == "r":
            self.apply_view()
        elif key in ("q", "escape"):
            self.interactor.TerminateApp()

    def start(self):
        print("\n" + "=" * 56)
        print("PCA SURFACE VARIATIONS VIEWER (panel-per-mesh)")
        print("  View:        [F] Front   [T] Top   [S] Side   [R] Reset")
        print("  Display:     [W] toggle Wireframe")
        print("  Quit:        [Q] or [Esc]")
        print("=" * 56 + "\n")
        self.interactor.Initialize()
        self.interactor.Start()


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("--- View PCA Surface Variations ---")
    print("=" * 60)

    print("\nSelect a PC folder (e.g., 'PC1', 'PC2', 'PC3')...")
    pc_dir = popup_select_directory("Select PC Folder")
    if not pc_dir:
        print("Canceled.")
        return

    pc_name = os.path.basename(pc_dir)

    # IMPORTANT: ใช้ pca_model_All_mean.vtk เป็น "Mean" — เพราะ {PC}_Mean.vtk เป็น
    # mean PROJECTED ONTO ONLY {PC} axis (degenerate, แบน), ไม่ใช่ global mean shape
    # global mean อยู่ใน pca_results/pca_model_All_mean.vtk
    parent_dir = os.path.dirname(pc_dir.rstrip("\\/"))
    parent_parent = os.path.dirname(parent_dir)
    global_mean_candidates = [
        os.path.join(parent_dir, "pca_model_All_mean.vtk"),
        os.path.join(parent_parent, "pca_results", "pca_model_All_mean.vtk"),
        os.path.join(pc_dir, "..", "..", "pca_model_All_mean.vtk"),
    ]
    global_mean_path = None
    for c in global_mean_candidates:
        if os.path.exists(c):
            global_mean_path = c
            break

    ordered = [
        (f"{pc_name}_minus3SD.vtk", os.path.join(pc_dir, f"{pc_name}_minus3SD.vtk")),
        (f"{pc_name}_minus2SD.vtk", os.path.join(pc_dir, f"{pc_name}_minus2SD.vtk")),
        (f"{pc_name}_minus1SD.vtk", os.path.join(pc_dir, f"{pc_name}_minus1SD.vtk")),
        ("Mean (global)", global_mean_path or os.path.join(pc_dir, f"{pc_name}_Mean.vtk")),
        (f"{pc_name}_plus1SD.vtk", os.path.join(pc_dir, f"{pc_name}_plus1SD.vtk")),
        (f"{pc_name}_plus2SD.vtk", os.path.join(pc_dir, f"{pc_name}_plus2SD.vtk")),
        (f"{pc_name}_plus3SD.vtk", os.path.join(pc_dir, f"{pc_name}_plus3SD.vtk")),
    ]
    existing = [path for _, path in ordered if path and os.path.exists(path)]

    if global_mean_path:
        print(f"\nUsing global mean from: {global_mean_path}")
    else:
        print(f"\nWARN: pca_model_All_mean.vtk not found — fallback to {pc_name}_Mean.vtk "
              f"(degenerate per-PC mean)")

    if not existing:
        print(f"\n[ERROR] No VTK files in {pc_dir}")
        print(f"        Expected files like {pc_name}_minus3SD.vtk ... {pc_name}_plus3SD.vtk")
        return

    PCAVariationsViewer(existing, title=f"PCA {pc_name} variations (-3SD to +3SD)")


if __name__ == "__main__":
    main()
