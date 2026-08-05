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
    """
    head_idx = int(np.argmax(pts[:, 2]))
    tail_idx = int(np.argmin(pts[:, 2]))
    lateral_idx = int(np.argmax(pts[:, 0]))
    medial_idx = int(np.argmin(pts[:, 0]))
    return head_idx, tail_idx, lateral_idx, medial_idx


def find_anatomical_landmarks(pts):
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


def create_mean_polydata(template_poly, mean_coords):
    if mean_coords is None or template_poly is None:
        return None
    
    new_poly = vtk.vtkPolyData()
    new_poly.DeepCopy(template_poly)
    
    vtk_pts = vtk.vtkPoints()
    for pt in mean_coords:
        vtk_pts.InsertNextPoint(pt[0], pt[1], pt[2])
    new_poly.SetPoints(vtk_pts)
    
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(new_poly)
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

class SpharmMeanViewer:
    """
    Viewer สำหรับเปรียบเทียบ Mean Shape ของกลุ่มปกติ (น้ำเงิน) และกลุ่มโรค (แดง) เคียงข้างกัน
    พร้อมแสดงเส้น Displacement Vectors ของแต่ละฝั่งเพื่อเปรียบเทียบ
    """

    MODE_OVERLAY = "OVERLAY"
    MODE_SLIDESHOW = "SLIDESHOW"

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
        self.opacity_overlay = 0.8
        self.show_vectors = True
        self.show_landmarks = False

        # load and classify coordinates
        self.basenames = []
        blue_points = []
        red_points = []
        template_poly = None

        for i, f in enumerate(files):
            poly = load_polydata_smoothed(f)
            if poly is None:
                continue
            if template_poly is None:
                template_poly = poly
            
            name = os.path.basename(f)
            for suf in ("_SPHARM_realigned.vtk", "_SPHARM_ellalign.vtk", "_SPHARM.vtk"):
                name = name.replace(suf, "")
            self.basenames.append(name)
            
            is_left_side = name.startswith("left_")
            # Classification
            if "_Healthy" in name or "HFH_" in name:
                is_red = False
            elif (is_left_side and "_Left-TLE" in name) or (not is_left_side and "_Right-TLE" in name):
                is_red = True
            elif (is_left_side and "_Right-TLE" in name) or (not is_left_side and "_Left-TLE" in name):
                is_red = False
            else:
                is_red = False

            pts = poly_points_numpy(poly)
            if is_red:
                red_points.append(pts)
            else:
                blue_points.append(pts)

        if template_poly is None:
            print("[ERROR] No valid meshes loaded.")
            return

        # 1. คำนวณพิกัดเฉลี่ยของแต่ละกลุ่ม
        print(f"Computing Mean Shape for Normal group (N={len(blue_points)})...")
        self.mean_blue_coords = np.mean(blue_points, axis=0) if blue_points else None
        print(f"Computing Mean Shape for Diseased group (N={len(red_points)})...")
        self.mean_red_coords = np.mean(red_points, axis=0) if red_points else None

        if self.mean_blue_coords is None or self.mean_red_coords is None:
            print("[ERROR] Need both Normal and Diseased groups to compute means.")
            return

        # 2. สร้าง PolyData ของ Mean ทั้งสอง
        self.mean_blue_poly = create_mean_polydata(template_poly, self.mean_blue_coords)
        self.mean_red_poly = create_mean_polydata(template_poly, self.mean_red_coords)

        # 3. คำนวณ Displacement Vector และ Magnitude ของจุดแต่ละจุด (ก่อนการ shift ตน.)
        displacement_vectors_blue = []
        displacement_vectors_red = []
        distances = []
        for i in range(len(self.mean_blue_coords)):
            p_blue = self.mean_blue_coords[i]
            p_red = self.mean_red_coords[i]
            vec = p_red - p_blue
            dist = np.linalg.norm(vec)
            displacement_vectors_blue.append(vec)     # Normal -> Diseased (ทิศชี้ไปหาสีแดง)
            displacement_vectors_red.append(-vec)     # Diseased -> Normal (ทิศชี้กลับมาสีน้ำเงิน)
            distances.append(dist)

        self.max_dist = max(distances)
        self.mean_dist = np.mean(distances)
        print(f"  Max displacement:  {self.max_dist:.4f}")
        print(f"  Mean displacement: {self.mean_dist:.4f}")

        # แนบ Scalars และ Vectors ของทั้งสองฝั่ง
        scalars_blue = vtk.vtkDoubleArray()
        scalars_blue.SetName("DisplacementMagnitude")
        for val in distances:
            scalars_blue.InsertNextValue(val)
        self.mean_blue_poly.GetPointData().SetScalars(scalars_blue)

        vectors_blue = vtk.vtkDoubleArray()
        vectors_blue.SetNumberOfComponents(3)
        vectors_blue.SetName("DisplacementVectors")
        for vec in displacement_vectors_blue:
            vectors_blue.InsertNextTuple3(vec[0], vec[1], vec[2])
        self.mean_blue_poly.GetPointData().SetVectors(vectors_blue)

        scalars_red = vtk.vtkDoubleArray()
        scalars_red.SetName("DisplacementMagnitude")
        for val in distances:
            scalars_red.InsertNextValue(val)
        self.mean_red_poly.GetPointData().SetScalars(scalars_red)

        vectors_red = vtk.vtkDoubleArray()
        vectors_red.SetNumberOfComponents(3)
        vectors_red.SetName("DisplacementVectors")
        for vec in displacement_vectors_red:
            vectors_red.InsertNextTuple3(vec[0], vec[1], vec[2])
        self.mean_red_poly.GetPointData().SetVectors(vectors_red)

        self.meshes = []
        self.mesh_names = []
        self.mesh_colors = []
        self.mesh_is_red = []

        if self.mean_blue_poly:
            self.meshes.append(self.mean_blue_poly)
            self.mesh_names.append("Mean Normal (Blue)")
            self.mesh_colors.append((0.2549, 0.4118, 0.8824))
            self.mesh_is_red.append(False)
        if self.mean_red_poly:
            self.meshes.append(self.mean_red_poly)
            self.mesh_names.append("Mean Diseased (Red)")
            self.mesh_colors.append((0.8627, 0.0784, 0.2353))
            self.mesh_is_red.append(True)

        # คำนวณระยะห่าง (shift_amount) จากขนาดเฉลี่ยของ Mesh
        b = self.meshes[0].GetBounds()
        self.shift_amount = 1.1 * (b[1] - b[0])

        # คำนวณ Scale Factor เริ่มต้นสำหรับหัวลูกศร (10% ของ mesh diagonal)
        mesh_diag = np.sqrt((b[1]-b[0])**2 + (b[3]-b[2])**2 + (b[5]-b[4])**2)
        self.arrow_scale = (mesh_diag * 0.1) / self.max_dist if self.max_dist > 1e-6 else 1.0

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
        self.render_window.SetMultiSamples(8)
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1300, 950)
        self.render_window.SetWindowName(
            f"SPHARM Mean Shapes Split Viewer (Left: Normal, Right: Diseased)"
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
            "[1] Overlay   [2] Slideshow   [V] Toggle vector arrows   [W] Wireframe\n"
            "[ [ ] Decrease arrow size   [ ] ] Increase arrow size   [L] Toggle landmarks\n"
            "[N / P / Space / scroll / Right / Left]  Next / Prev (slideshow)\n"
            "[+] / [-]  Overlay opacity     [R] Reset camera     [Q / Esc] Quit"
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

        # reference box (Left and Right)
        for offset_x in (-self.shift_amount, self.shift_amount):
            outline = vtk.vtkOutlineSource()
            h = self.REF_BOX_HALF
            outline.SetBounds(-h, h, -h, h, -h, h)
            out_mapper = vtk.vtkPolyDataMapper()
            out_mapper.SetInputConnection(outline.GetOutputPort())
            out_actor = vtk.vtkActor()
            out_actor.SetMapper(out_mapper)
            out_actor.SetPosition(offset_x, 0, 0)
            out_actor.GetProperty().SetColor(0.35, 0.35, 0.45)
            out_actor.GetProperty().SetOpacity(0.5)
            out_actor.GetProperty().SetLineWidth(1)
            self.renderer.AddActor(out_actor)

        # observers
        self.interactor.AddObserver("KeyPressEvent", self.on_key_press)
        self.interactor.AddObserver("MouseWheelForwardEvent", self.on_wheel_forward)
        self.interactor.AddObserver("MouseWheelBackwardEvent", self.on_wheel_backward)

    def build_actors(self):
        # 1. Lookup Table (Color Map): Blue (min) -> Cyan -> Green -> Yellow -> Red (max)
        self.lut = vtk.vtkLookupTable()
        self.lut.SetNumberOfTableValues(256)
        self.lut.SetHueRange(0.6667, 0.0) # blue to red
        self.lut.Build()

        # 2. Color Bar Legend (ขวา)
        self.scalar_bar = vtk.vtkScalarBarActor()
        self.scalar_bar.SetLookupTable(self.lut)
        self.scalar_bar.SetTitle("Deformation (mm)" if "realigned" in self.source or "pca_ready" in self.source else "Deformation (units)")
        self.scalar_bar.SetNumberOfLabels(5)
        self.scalar_bar.GetTitleTextProperty().SetColor(1, 1, 1)
        self.scalar_bar.GetTitleTextProperty().BoldOn()
        self.scalar_bar.GetTitleTextProperty().SetFontSize(14)
        self.scalar_bar.GetLabelTextProperty().SetColor(1, 1, 1)
        self.scalar_bar.SetWidth(0.1)
        self.scalar_bar.SetHeight(0.7)
        self.scalar_bar.GetPositionCoordinate().SetValue(0.88, 0.15)
        self.renderer.AddActor2D(self.scalar_bar)

        # 3. สร้าง Actor ของตัวเฉลี่ยซ้ายขวา
        self.actors = []
        for i, poly in enumerate(self.meshes):
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly)
            mapper.SetScalarRange(0.0, self.max_dist)
            mapper.SetLookupTable(self.lut)
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            
            is_red = self.mesh_is_red[i]
            offset_x = self.shift_amount if is_red else -self.shift_amount
            actor.SetPosition(offset_x, 0, 0)
            
            prop = actor.GetProperty()
            prop.SetInterpolationToGouraud()
            prop.SetAmbient(0.15)
            prop.SetDiffuse(0.85)
            prop.SetSpecular(0.2)
            prop.SetSpecularPower(20)
            
            self.renderer.AddActor(actor)
            self.actors.append(actor)

        # 4. Glyph สำหรับหัวลูกศร
        self.arrow_source = vtk.vtkArrowSource()
        self.arrow_source.SetTipResolution(16)
        self.arrow_source.SetTipLength(0.25)
        self.arrow_source.SetTipRadius(0.08)
        self.arrow_source.SetShaftRadius(0.025)

        self.glyphs = []
        self.glyph_actors = []
        for i, poly in enumerate(self.meshes):
            is_red = self.mesh_is_red[i]
            offset_x = self.shift_amount if is_red else -self.shift_amount
            
            glyph = vtk.vtkGlyph3D()
            glyph.SetSourceConnection(self.arrow_source.GetOutputPort())
            glyph.SetInputData(poly)
            glyph.SetVectorModeToUseVector()
            glyph.SetScaleModeToScaleByVector()
            glyph.SetScaleFactor(self.arrow_scale)
            glyph.OrientOn()
            glyph.Update()
            self.glyphs.append(glyph)
            
            glyph_mapper = vtk.vtkPolyDataMapper()
            glyph_mapper.SetInputConnection(glyph.GetOutputPort())
            glyph_mapper.SetScalarRange(0.0, self.max_dist)
            glyph_mapper.SetLookupTable(self.lut)
            
            glyph_actor = vtk.vtkActor()
            glyph_actor.SetMapper(glyph_mapper)
            glyph_actor.GetProperty().SetAmbient(0.3)
            glyph_actor.GetProperty().SetDiffuse(0.7)
            glyph_actor.SetPosition(offset_x, 0, 0) # เลื่อนหัวลูกศรตามฝั่งของโมเดล
            
            self.renderer.AddActor(glyph_actor)
            self.glyph_actors.append(glyph_actor)

        # 5. Landmark dots
        self.landmark_actors_per_subject = []
        sphere_radius = self._estimate_landmark_dot_radius()
        landmark_colors = [
            (1.0, 0.25, 0.25),   # head 470 - red (+Z)
            (0.25, 0.45, 1.0),   # tail 276 - blue (-Z)
            (1.0,  1.0,  0.3),   # index 0  - yellow (-X)
            (0.3,  1.0,  0.3),   # opp 0    - green (+X)
        ]
        
        fixed_indices = (470, 276, 0, 272)
        for i, poly in enumerate(self.meshes):
            is_red = self.mesh_is_red[i]
            offset_x = self.shift_amount if is_red else -self.shift_amount
            
            pts = poly_points_numpy(poly)
            try:
                r_idx, b_idx, y_idx, g_idx = fixed_indices
                positions = [pts[r_idx], pts[b_idx], pts[y_idx], pts[g_idx]]
            except Exception:
                self.landmark_actors_per_subject.append([])
                continue
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
                
                # Apply position shift
                s_actor.SetPosition(offset_x, 0, 0)
                
                self.renderer.AddActor(s_actor)
                subject_actors.append(s_actor)
            self.landmark_actors_per_subject.append(subject_actors)

    def _estimate_landmark_dot_radius(self):
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

        # ควบคุมการแสดงผลลูกศร
        for i, ga in enumerate(self.glyph_actors):
            if self.mode == self.MODE_OVERLAY:
                ga.SetVisibility(self.show_vectors)
            else:
                ga.SetVisibility(self.show_vectors and (i == self.current_idx))

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
        v_state = "ON" if self.show_vectors else "OFF"
        
        if self.mode == self.MODE_OVERLAY:
            txt = (
                f"MODE: OVERLAY (Mean Shapes Split Compare)   |   ({self.source})\n"
                f"Left side: MEAN Normal (Blue)   |   Right side: MEAN Diseased (Red)\n"
                f"Vector Arrows: {v_state} (Scale: {self.arrow_scale:.2f}x)   |   Opacity = {self.opacity_overlay:.2f}"
            )
        else:
            name = self.mesh_names[self.current_idx]
            b = self.meshes[self.current_idx].GetBounds()
            n_pts = self.meshes[self.current_idx].GetNumberOfPoints()
            side_str = "Right (Diseased)" if self.mesh_is_red[self.current_idx] else "Left (Normal)"
            txt = (
                f"MODE: SLIDESHOW   [{self.current_idx+1}/{len(self.actors)}]   "
                f"({self.source})   |   Side: {side_str}\n"
                f"Group Mean: {name}   ({n_pts} pts)\n"
                f"Vector Arrows: {v_state} (Scale: {self.arrow_scale:.2f}x)\n"
                f"Bounds  X[{b[0]:+.3f},{b[1]:+.3f}]  "
                f"Y[{b[2]:+.3f},{b[3]:+.3f}]  Z[{b[4]:+.3f},{b[5]:+.3f}]"
            )
        self.text_actor.SetInput(txt)

    def reset_camera(self):
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(0, -5, 1.2)
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
        elif key == "v":
            self.show_vectors = not self.show_vectors
            self.apply_mode()
        elif key == "l":
            self.show_landmarks = not self.show_landmarks
            self.apply_mode()
        elif key == "w":
            self.wireframe = not self.wireframe
            self.apply_mode()
        elif key in ("bracketright", "equal", "kp_add"):
            self.arrow_scale *= 1.2
            for g in self.glyphs:
                g.SetScaleFactor(self.arrow_scale)
            self.apply_mode()
        elif key in ("bracketleft", "minus", "kp_subtract"):
            self.arrow_scale /= 1.2
            for g in self.glyphs:
                g.SetScaleFactor(self.arrow_scale)
            self.apply_mode()
        elif key in ("plus", "equal", "kp_add") and not obj.GetShiftKey(): # Opacity if no bracket
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
        print("SPHARM MEAN SHAPES SPLIT VIEWER (Vector Compare)")
        print(f"  Source:       {self.source}")
        print("  Left side:    MEAN Normal shape (Blue, vectors point to diseased)")
        print("  Right side:   MEAN Diseased shape (Red, vectors point to normal)")
        print("  Modes:        [1] Overlay   [2] Slideshow")
        print("  Slideshow:    [N/P/space/scroll/Right/Left]")
        print("  Vectors:      [V] toggle vector arrows")
        print("  Scale vector: [ ] ] increase size, [ [ ] decrease size")
        print("  Landmarks:    [L] toggle")
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

    viewer = SpharmMeanViewer(spharm_dir)
    viewer.start()


if __name__ == "__main__":
    main()
