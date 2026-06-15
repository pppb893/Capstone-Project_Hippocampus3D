import vtk
vtk.vtkObject.GlobalWarningDisplayOff()
import os
import glob
import argparse
import colorsys
import tkinter as tk
from tkinter import filedialog
import numpy as np


# =============================================================================
# Anatomical landmark detection (copied from realign_spharm.py)
# =============================================================================

def find_landmarks_by_position(pts):
    """
    หา 4 landmark indices ตาม canonical position (สำหรับ realigned mesh).

    หลัง realign — canonical คือ:
      HEAD = +Z   TAIL = -Z   LAT = +X   MED = -X

    -> ใช้ argmax/argmin ในแต่ละแกน → label คงที่ ไม่ flip อีก
    """
    head_idx = int(np.argmax(pts[:, 2]))
    tail_idx = int(np.argmin(pts[:, 2]))
    lateral_idx = int(np.argmax(pts[:, 0]))
    medial_idx = int(np.argmin(pts[:, 0]))
    return head_idx, tail_idx, lateral_idx, medial_idx


def find_anatomical_landmarks(pts):
    """หา 4 anatomical landmarks: head, tail, lateral, medial (สำหรับ non-realigned)"""
    centroid = pts.mean(axis=0)
    pts_c = pts - centroid
    _, _, Vt = np.linalg.svd(pts_c, full_matrices=False)
    long_axis = Vt[0] / np.linalg.norm(Vt[0])

    proj = pts_c @ long_axis
    p90, p10 = np.percentile(proj, 90), np.percentile(proj, 10)
    top_pts = pts_c[proj > p90]
    bot_pts = pts_c[proj < p10]

    def spread_perp(subset, axis):
        perp = subset - np.outer(subset @ axis, axis)
        return float(np.std(perp, axis=0).sum())

    top_spread = spread_perp(top_pts, long_axis) if len(top_pts) > 0 else 0.0
    bot_spread = spread_perp(bot_pts, long_axis) if len(bot_pts) > 0 else 0.0

    if top_spread >= bot_spread:
        head_idx, tail_idx = int(np.argmax(proj)), int(np.argmin(proj))
    else:
        head_idx, tail_idx = int(np.argmin(proj)), int(np.argmax(proj))

    middle_mask = (proj > np.percentile(proj, 25)) & (proj < np.percentile(proj, 75))
    middle_pts = pts_c[middle_mask] if middle_mask.any() else pts_c
    middle_perp = middle_pts - np.outer(middle_pts @ long_axis, long_axis)
    curl_axis = middle_perp.mean(axis=0)
    norm = np.linalg.norm(curl_axis)
    if norm > 1e-9:
        curl_axis /= norm
    else:
        pc2 = Vt[1]
        curl_axis = pc2 - (pc2 @ long_axis) * long_axis
        curl_axis /= np.linalg.norm(curl_axis)

    proj_curl = pts_c @ curl_axis
    lateral_idx = int(np.argmax(proj_curl))
    medial_idx = int(np.argmin(proj_curl))
    return head_idx, tail_idx, lateral_idx, medial_idx


def poly_points_numpy(poly):
    n = poly.GetNumberOfPoints()
    return np.array([poly.GetPoint(i) for i in range(n)])


# =============================================================================
# IO helpers
# =============================================================================

def load_polydata_smoothed(filepath):
    """อ่าน .vtk + compute normals (smooth shading)"""
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
# Viewer
# =============================================================================

class SpharmMeshViewer:
    """
    Viewer สำหรับตรวจ SPHARM mesh (output ของ SPHARM-PDM)
      - OVERLAY  : ซ้อนทุก mesh, สีต่างกัน, semi-transparent
                   -> เห็นทันทีว่า orientation ตรงกันมั้ย
      - SLIDESHOW: เปิดทีละตัว + mean shape เป็น wireframe อ้างอิง
    """

    MODE_OVERLAY = "OVERLAY"
    MODE_SLIDESHOW = "SLIDESHOW"

    # SPHARM meshes อยู่ใน normalized scale (~ +/- 1) -> box +/- 1.0
    REF_BOX_HALF = 1.0

    def __init__(self, spharm_dir):
        self.spharm_dir = spharm_dir.replace("\\", "/")

        # prefer pca_ready, then realigned, fall back to ellalign
        files = sorted(glob.glob(os.path.join(self.spharm_dir,
                                              "*_SPHARM_pca_ready.vtk")))
        source = "pca_ready"
        if not files:
            files = sorted(glob.glob(os.path.join(self.spharm_dir,
                                                  "*_SPHARM_realigned.vtk")))
            source = "realigned"
        if not files:
            files = sorted(glob.glob(os.path.join(self.spharm_dir,
                                                  "*_SPHARM_ellalign.vtk")))
            source = "ellalign"
        if not files:
            files = sorted(glob.glob(os.path.join(self.spharm_dir,
                                                  "*_SPHARM.vtk")))
            # filter out ellalign/grid/realigned variants
            files = [f for f in files
                     if not any(s in os.path.basename(f)
                                for s in ("_ellalign", "_grid", "_realigned",
                                          "_procalign", "_pca_ready"))]
            source = "SPHARM (non-aligned)"

        if not files:
            print(f"[ERROR] No SPHARM .vtk found in {self.spharm_dir}")
            self.meshes = []
            return

        self.files = files
        self.source = source
        print(f"Found {len(files)} '{source}' meshes in {self.spharm_dir}")

        # state
        self.current_idx = 0
        self.mode = self.MODE_OVERLAY
        self.wireframe = False
        self.opacity_overlay = 0.25
        self.show_mean = True
        self.show_landmarks = True

        # load
        self.meshes = []
        self.basenames = []
        for i, f in enumerate(files):
            print(f"  [{i+1}/{len(files)}] {os.path.basename(f)}")
            poly = load_polydata_smoothed(f)
            if poly is None:
                print(f"      skipped (empty)")
                continue
            self.meshes.append(poly)
            name = os.path.basename(f)
            for suf in ("_SPHARM_realigned.vtk", "_SPHARM_ellalign.vtk", "_SPHARM.vtk"):
                name = name.replace(suf, "")
            self.basenames.append(name)

        if not self.meshes:
            print("[ERROR] No valid meshes loaded.")
            return

        # mean shape candidates (จาก PCA output ถ้ามี)
        self.mean_poly = None
        candidates = [
            os.path.join(os.path.dirname(self.spharm_dir), "pca_results",
                         "pca_model_0_mean.vtk"),
            os.path.join(os.path.dirname(self.spharm_dir), "pca_results",
                         "pca_model_All_mean.vtk"),
        ]
        for c in candidates:
            if os.path.exists(c):
                print(f"Loading mean shape: {c}")
                self.mean_poly = load_polydata_smoothed(c)
                if self.mean_poly is not None:
                    break

        if self.mean_poly is None:
            print("(no pca_model mean shape found - reference overlay disabled)")

        self.setup_vtk()
        self.build_actors()
        self.apply_mode()

    # ---------------- VTK scaffolding ----------------

    def setup_vtk(self):
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.07, 0.07, 0.1)
        self.renderer.SetUseDepthPeeling(True)
        self.renderer.SetMaximumNumberOfPeels(8)
        self.renderer.SetOcclusionRatio(0.0)

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetAlphaBitPlanes(1)
        self.render_window.SetMultiSamples(0)
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1300, 950)
        self.render_window.SetWindowName(
            f"SPHARM Mesh Viewer ({self.source}, {len(self.meshes)} subjects)"
        )

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        # info overlay (top-left)
        self.text_actor = vtk.vtkTextActor()
        tp = self.text_actor.GetTextProperty()
        tp.SetFontSize(16)
        tp.SetColor(1, 1, 1)
        tp.BoldOn()
        tp.SetShadow(True)
        self.text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.text_actor.GetPositionCoordinate().SetValue(0.02, 0.90)
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
            "[1] Overlay   [2] Slideshow   [M] Mean   [L] Landmark dots   [W] Wireframe\n"
            "[N / P / Space / scroll / Right / Left]  Next / Prev (slideshow)\n"
            "[+] / [-]  Overlay opacity     [R] Reset camera     [Q / Esc] Quit\n"
            "Landmark colors:  RED=head  BLUE=tail  GREEN=lateral  YELLOW=medial"
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

        # world axes at origin
        world_axes = vtk.vtkAxesActor()
        world_axes.SetTotalLength(0.6, 0.6, 0.6)
        world_axes.AxisLabelsOff()
        world_axes.SetShaftTypeToLine()
        for ax_prop in (world_axes.GetXAxisShaftProperty(),
                        world_axes.GetYAxisShaftProperty(),
                        world_axes.GetZAxisShaftProperty()):
            ax_prop.SetLineWidth(2)
        self.renderer.AddActor(world_axes)

        # reference box
        outline = vtk.vtkOutlineSource()
        h = self.REF_BOX_HALF
        outline.SetBounds(-h, h, -h, h, -h, h)
        out_mapper = vtk.vtkPolyDataMapper()
        out_mapper.SetInputConnection(outline.GetOutputPort())
        out_actor = vtk.vtkActor()
        out_actor.SetMapper(out_mapper)
        out_actor.GetProperty().SetColor(0.35, 0.35, 0.45)
        out_actor.GetProperty().SetOpacity(0.5)
        out_actor.GetProperty().SetLineWidth(1)
        self.renderer.AddActor(out_actor)

        # observers
        self.interactor.AddObserver("KeyPressEvent", self.on_key_press)
        self.interactor.AddObserver("MouseWheelForwardEvent", self.on_wheel_forward)
        self.interactor.AddObserver("MouseWheelBackwardEvent", self.on_wheel_backward)

    def build_actors(self):
        N = len(self.meshes)
        self.actors = []
        for i, poly in enumerate(self.meshes):
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly)
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
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

        # ------------------------------------------------------------
        # Landmark dots: 4 จุด (HEAD/TAIL/LAT/MED) ต่อ subject
        # เก็บ list-of-list เพื่อ filter visibility ตาม mode ได้
        #   OVERLAY  : ทุก subject's dots = visible -> เห็น 4 cluster
        #   SLIDESHOW: เฉพาะ subject ปัจจุบัน -> ไม่ค้าง
        #
        # สำหรับ realigned mesh: ใช้ positional rule (canonical fixed)
        #   -> dot สีจะตรง cluster เสมอ
        # สำหรับ non-realigned (ellalign/SPHARM): ใช้ anatomical detection
        #   -> อาจ flip ได้เพราะ orientation ยังไม่ลง canonical
        # ------------------------------------------------------------
        self.landmark_actors_per_subject = []
        sphere_radius = self._estimate_landmark_dot_radius()
        landmark_colors = [
            (1.0, 0.25, 0.25),   # head  - red
            (0.25, 0.45, 1.0),   # tail  - blue
            (0.3,  1.0,  0.3),   # lat   - green
            (1.0,  1.0,  0.3),   # med   - yellow
        ]
        
        # Determine fixed vertex indices from the template shape to ensure correspondence visualization
        use_fixed_indices = (self.source in ("pca_ready", "realigned", "procalign") or "procalign" in self.source)
        fixed_indices = None
        if use_fixed_indices and len(self.meshes) > 0:
            ref_pts = poly_points_numpy(self.meshes[0])
            try:
                # First try positional detection since it's aligned
                fixed_indices = find_landmarks_by_position(ref_pts)
            except Exception:
                try:
                    fixed_indices = find_anatomical_landmarks(ref_pts)
                except Exception:
                    use_fixed_indices = False

        use_positional = self.source == "realigned"
        for poly in self.meshes:
            pts = poly_points_numpy(poly)
            try:
                if use_fixed_indices and fixed_indices is not None:
                    h, t, l, m = fixed_indices
                elif use_positional:
                    h, t, l, m = find_landmarks_by_position(pts)
                else:
                    h, t, l, m = find_anatomical_landmarks(pts)
            except Exception:
                self.landmark_actors_per_subject.append([])
                continue
            positions = [pts[h], pts[t], pts[l], pts[m]]
            subject_actors = []
            for color, pos in zip(landmark_colors, positions):
                sphere = vtk.vtkSphereSource()
                sphere.SetCenter(float(pos[0]), float(pos[1]), float(pos[2]))
                sphere.SetRadius(sphere_radius)
                sphere.SetThetaResolution(10)
                sphere.SetPhiResolution(10)
                sphere.Update()
                s_mapper = vtk.vtkPolyDataMapper()
                s_mapper.SetInputConnection(sphere.GetOutputPort())
                s_actor = vtk.vtkActor()
                s_actor.SetMapper(s_mapper)
                sp = s_actor.GetProperty()
                sp.SetColor(*color)
                sp.SetAmbient(0.6)
                sp.SetDiffuse(0.4)
                self.renderer.AddActor(s_actor)
                subject_actors.append(s_actor)
            self.landmark_actors_per_subject.append(subject_actors)

    def _estimate_landmark_dot_radius(self):
        """ขนาด sphere ของ landmark dot = 2% ของ mesh diagonal เฉลี่ย"""
        diag_total = 0.0
        for m in self.meshes:
            b = m.GetBounds()
            diag_total += np.sqrt((b[1]-b[0])**2 + (b[3]-b[2])**2 + (b[5]-b[4])**2)
        avg_diag = diag_total / max(1, len(self.meshes))
        return max(avg_diag * 0.02, 0.01)

    # ---------------- modes ----------------

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
        # landmark dots: filter per-subject ตาม mode
        #   OVERLAY  -> ทุก subject's dots
        #   SLIDESHOW -> เฉพาะ current_idx
        for i, subj_actors in enumerate(
                getattr(self, "landmark_actors_per_subject", [])):
            if self.mode == self.MODE_OVERLAY:
                show_subj = self.show_landmarks
            else:
                show_subj = self.show_landmarks and (i == self.current_idx)
            for la in subj_actors:
                la.SetVisibility(show_subj)
        self._update_info_text()
        self.render_window.Render()

    def _update_info_text(self):
        if self.mean_actor is None:
            mean_state = "N/A"
        else:
            mean_state = "ON" if self.show_mean else "OFF"

        if self.mode == self.MODE_OVERLAY:
            txt = (
                f"MODE: OVERLAY   |   {len(self.actors)} meshes ({self.source})   |   "
                f"opacity = {self.opacity_overlay:.2f}\n"
                f"Mean shape (white wireframe): {mean_state}\n"
                f"Reference box: +/- {self.REF_BOX_HALF:.1f} units"
            )
        else:
            name = self.basenames[self.current_idx]
            b = self.meshes[self.current_idx].GetBounds()
            n_pts = self.meshes[self.current_idx].GetNumberOfPoints()
            txt = (
                f"MODE: SLIDESHOW   [{self.current_idx+1}/{len(self.actors)}]   "
                f"({self.source})\n"
                f"Subject: {name}   ({n_pts} pts)\n"
                f"Bounds  X[{b[0]:+.3f},{b[1]:+.3f}]  "
                f"Y[{b[2]:+.3f},{b[3]:+.3f}]  Z[{b[4]:+.3f},{b[5]:+.3f}]\n"
                f"Mean shape: {mean_state}"
            )
        self.text_actor.SetInput(txt)

    def reset_camera(self):
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(0, -4, 0.8)
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
        elif key == "m":
            self.show_mean = not self.show_mean
            self.apply_mode()
        elif key == "l":
            self.show_landmarks = not self.show_landmarks
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
        print("SPHARM MESH VIEWER")
        print(f"  Source:       {self.source}")
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
    parser.add_argument("--spharm_dir", default=None,
                        help="Path to spharm_results folder. ถ้าไม่ระบุจะเด้ง dialog")
    args = parser.parse_args()

    spharm_dir = args.spharm_dir
    if not spharm_dir:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        spharm_dir = filedialog.askdirectory(
            title="Select 'spharm_results' folder (or output_xxx folder)"
        )
        root.destroy()

    if not spharm_dir:
        print("No folder selected.")
        return

    # ถ้า user เลือก output_xxx (parent) -> เข้าไปใน spharm_results subfolder
    if os.path.basename(spharm_dir.rstrip("\\/")).lower() != "spharm_results":
        candidate = os.path.join(spharm_dir, "spharm_results")
        if os.path.isdir(candidate):
            spharm_dir = candidate

    viewer = SpharmMeshViewer(spharm_dir)
    viewer.start()


if __name__ == "__main__":
    main()
