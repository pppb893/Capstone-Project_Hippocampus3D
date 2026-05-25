import os
import sys

# =============================================================================
# BOOTSTRAP: ถ้ารันด้วย Python ปกติ (ไม่ใช่ SlicerSALT) ให้ re-launch
# ตัวเองผ่าน SlicerSALT.exe เพราะ slicer / vtk module อยู่ในตัว SlicerSALT
# เท่านั้น ใช้ env var SLICER_EXE override ได้ถ้า path ไม่ตรง
# =============================================================================

def _bootstrap_slicer():
    try:
        import slicer  # noqa: F401
        return
    except ImportError:
        pass

    slicer_exe = os.environ.get(
        "SLICER_EXE",
        r"C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe",
    )
    if not os.path.exists(slicer_exe):
        print(f"[ERROR] SlicerSALT not found at: {slicer_exe}")
        print("        Set env var SLICER_EXE or edit the default path in main2.py")
        sys.exit(1)

    import subprocess
    script_path = os.path.abspath(__file__)
    cmd = [slicer_exe, "--no-main-window", "--no-splash",
           "--python-script", script_path] + sys.argv[1:]
    print(f"[INFO] No 'slicer' module in this Python. Re-launching via SlicerSALT:")
    print(f"       {slicer_exe}")
    sys.exit(subprocess.call(cmd))


_bootstrap_slicer()


# =============================================================================
# จากตรงนี้ลงไป: เรารันอยู่ใน SlicerSALT แล้ว -> import เต็มได้
# =============================================================================

import vtk
import numpy as np
import glob
import argparse
import slicer
from datetime import datetime

# =============================================================================
# SCRIPT_DIR: หา path ของสคริปต์นี้ให้ถูกต้องใน SlicerSALT
# =============================================================================
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
    for arg in sys.argv:
        if arg.endswith(".py") and os.path.isfile(arg):
            SCRIPT_DIR = os.path.dirname(os.path.abspath(arg))
            break

DEBUG_LOG = os.path.join(SCRIPT_DIR, "icp_debug_log.txt")


# =============================================================================
# UTILITIES
# =============================================================================

def sprint(msg):
    """Print พร้อม timestamp และบันทึกลง log file"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] [ICP-LOG] {msg}"
    print(formatted_msg)
    sys.stdout.flush()
    with open(DEBUG_LOG, "a") as f:
        f.write(formatted_msg + "\n")


def get_points_numpy(poly):
    """แปลง vtkPolyData points -> numpy array (N, 3)"""
    n = poly.GetNumberOfPoints()
    pts = poly.GetPoints()
    arr = np.zeros((n, 3))
    for i in range(n):
        arr[i] = pts.GetPoint(i)
    return arr


def prompt_folder(title):
    """
    เปิด QFileDialog ให้ user เลือก folder (ใช้ Qt ที่ฝังใน SlicerSALT)
    คืน path string หรือ None ถ้า user กด Cancel
    """
    import qt
    folder = qt.QFileDialog.getExistingDirectory(None, title)
    if not folder:
        return None
    return folder


def apply_poly_transform(poly, matrix_np):
    """Apply 4x4 numpy transform matrix กับ vtkPolyData"""
    transform = vtk.vtkTransform()
    matrix_vtk = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            matrix_vtk.SetElement(r, c, float(matrix_np[r, c]))
    transform.SetMatrix(matrix_vtk)
    transformer = vtk.vtkTransformPolyDataFilter()
    transformer.SetInputData(poly)
    transformer.SetTransform(transform)
    transformer.Update()
    return transformer.GetOutput()


def icp_distance(source_poly, target_poly, n_sample=500):
    """
    Mean nearest-neighbor distance จาก source -> target
    (ใช้ sample 500 จุดเพื่อความเร็ว: ตรวจ convergence + เปรียบเทียบ orientation)
    """
    locator = vtk.vtkKdTreePointLocator()
    locator.SetDataSet(target_poly)
    locator.BuildLocator()
    src_pts = get_points_numpy(source_poly)
    n_sample = min(len(src_pts), n_sample)
    idx = np.linspace(0, len(src_pts) - 1, n_sample, dtype=int)
    total_dist = 0.0
    for i in idx:
        pt = src_pts[i].tolist()
        closest_id = locator.FindClosestPoint(pt)
        cp = target_poly.GetPoint(closest_id)
        total_dist += np.sqrt(sum((src_pts[i][k] - cp[k]) ** 2 for k in range(3)))
    return total_dist / n_sample


# =============================================================================
# STEP A: LOAD + MESH
# โหลด NIfTI label -> surface mesh (vtkPolyData) ใน RAS space
# =============================================================================

def load_and_mesh_node(filepath, max_retries=2):
    """
    โหลด label volume ผ่าน Slicer -> vtkDiscreteMarchingCubes -> apply IJK->RAS
    เพื่อให้ mesh อยู่ใน physical mm space เดียวกับ NIfTI ต้นฉบับ

    Robust loading:
      - slicer.util.loadLabelVolume() raise RuntimeError ตอน failed
        (ไม่ใช่ return None) -> ต้อง try/except กัน script crash
      - ลอง normalize path (\\ -> /) ก่อน retry (Slicer ไวกับ mixed-slash บน Windows)
      - retry สูงสุด max_retries ครั้งกัน transient failure (file lock, GC pressure)
    return None ถ้า load ไม่สำเร็จ -> caller จะ skip subject ตัวนั้น (ไม่ stop pipeline)
    """
    normalized = os.path.abspath(filepath).replace("\\", "/")
    last_err = None
    for attempt in range(max_retries + 1):
        node = None
        try:
            node = slicer.util.loadLabelVolume(normalized)
            if not node:
                last_err = "loadLabelVolume returned None"
                continue
            img = node.GetImageData()
            if img is None or img.GetNumberOfPoints() == 0:
                last_err = "empty image data"
                slicer.mrmlScene.RemoveNode(node)
                continue
            dmc = vtk.vtkDiscreteMarchingCubes()
            dmc.SetInputData(img)
            dmc.GenerateValues(1, 1, 100)
            dmc.Update()
            poly = dmc.GetOutput()
            ijkToRas = vtk.vtkMatrix4x4()
            node.GetIJKToRASMatrix(ijkToRas)
            t = vtk.vtkTransform()
            t.SetMatrix(ijkToRas)
            transformer = vtk.vtkTransformPolyDataFilter()
            transformer.SetTransform(t)
            transformer.SetInputData(poly)
            transformer.Update()
            result_poly = transformer.GetOutput()
            slicer.mrmlScene.RemoveNode(node)
            return result_poly
        except Exception as e:
            last_err = str(e).split("\n")[0][:200]
            if node is not None:
                try: slicer.mrmlScene.RemoveNode(node)
                except Exception: pass
            if attempt < max_retries:
                # transient failure -> ลอง garbage collect แล้ว retry
                try:
                    import gc; gc.collect()
                    import time; time.sleep(0.5)
                except Exception:
                    pass
    sprint(f"  !!! load_and_mesh_node failed after {max_retries+1} attempts: {last_err}")
    return None


# =============================================================================
# STEP B: PRINCIPAL AXIS ALIGNMENT (PCA)
# หมุน mesh ให้ long axis = Z, second axis = X (right-handed)
# =============================================================================

def principal_axis_align(poly):
    """
    ใช้ SVD หา principal axes แล้ว rotate ให้:
      PC1 (variance สูงสุด) -> Z, PC2 -> X, PC3 -> Y
    บังคับให้เป็น right-handed coordinate (det = +1) เพื่อ proper rotation
    """
    pts = get_points_numpy(poly)
    centroid = pts.mean(axis=0)
    pts_c = pts - centroid
    _, _, Vt = np.linalg.svd(pts_c, full_matrices=False)
    z_ax = Vt[0]
    x_ax = Vt[1]
    y_ax = np.cross(z_ax, x_ax)
    y_ax /= np.linalg.norm(y_ax)
    x_ax = np.cross(y_ax, z_ax)
    x_ax /= np.linalg.norm(x_ax)
    R = np.eye(3)
    R[:, 0] = x_ax
    R[:, 1] = y_ax
    R[:, 2] = z_ax
    T_rot = np.eye(4)
    T_rot[:3, :3] = R.T
    T_cent = np.eye(4)
    T_cent[:3, 3] = -centroid
    T_combined = T_rot @ T_cent
    return apply_poly_transform(poly, T_combined), T_combined


# =============================================================================
# STEP C: ORIENTATION DISAMBIGUATION (PROPER ROTATIONS ONLY)
# หลัง PCA แกน Z อาจชี้กลับด้านได้ (sign ambiguity ของ SVD)
# ทดสอบ 4 candidate ที่เป็น proper rotation 180 องศา รอบแกน X / Y / Z (det = +1)
# *** หลีกเลี่ยง reflection (det = -1) ที่ทำให้ chirality เปลี่ยน ***
# =============================================================================

_FLIP_CANDIDATES = [
    np.eye(4),                                # identity
    np.diag([1.0, -1.0, -1.0, 1.0]),          # 180 deg about X (flip Y, Z)
    np.diag([-1.0, 1.0, -1.0, 1.0]),          # 180 deg about Y (flip X, Z)
    np.diag([-1.0, -1.0, 1.0, 1.0]),          # 180 deg about Z (flip X, Y)
]
_FLIP_NAMES = ["identity", "rotX180", "rotY180", "rotZ180"]


def pole_flip_correction(poly, reference_poly=None):
    """
    เลือก orientation ที่ดีที่สุดจาก 4 proper rotations (ไม่มี mirror)
    - ไม่มี reference: heuristic 'หัวอ้วน' ตามแกน Z (rotate รอบ Y ถ้าหัวอยู่ Z-)
    - มี reference  : ทดลองทั้ง 4 candidate -> เลือก ICP distance ต่ำสุด
    """
    if reference_poly is None:
        pts = get_points_numpy(poly)
        z_mid = (pts[:, 2].min() + pts[:, 2].max()) / 2.0
        pts_pos = pts[pts[:, 2] > z_mid]
        pts_neg = pts[pts[:, 2] <= z_mid]
        fat_pos = (np.std(pts_pos[:, 0]) + np.std(pts_pos[:, 1])) if len(pts_pos) > 0 else 0.0
        fat_neg = (np.std(pts_neg[:, 0]) + np.std(pts_neg[:, 1])) if len(pts_neg) > 0 else 0.0
        if fat_neg > fat_pos:
            idx = 2  # rotY180 -> flip Z (proper rotation)
        else:
            idx = 0
        sprint(f"    Pole check (no ref): fat_neg={fat_neg:.4f}, fat_pos={fat_pos:.4f} -> {_FLIP_NAMES[idx]}")
        T = _FLIP_CANDIDATES[idx].copy()
        return apply_poly_transform(poly, T), T, (idx != 0)

    best_idx = 0
    best_dist = float("inf")
    dists = []
    for idx, T in enumerate(_FLIP_CANDIDATES):
        poly_test = apply_poly_transform(poly, T)
        d = icp_distance(poly_test, reference_poly)
        dists.append(d)
        if d < best_dist:
            best_dist = d
            best_idx = idx
    sprint(f"    Pole check: dists=[{', '.join(f'{d:.5f}' for d in dists)}] -> {_FLIP_NAMES[best_idx]}")
    T = _FLIP_CANDIDATES[best_idx].copy()
    return apply_poly_transform(poly, T), T, (best_idx != 0)


# =============================================================================
# STEP D: MEAN SHAPE (NN-matching)
# ใช้ topology ของ meshes[0] เป็น anchor; สำหรับแต่ละจุดใน reference
# หา nearest neighbor ใน mesh อื่น แล้ว average ตำแหน่ง
# =============================================================================

def compute_mean_poly(meshes):
    counts = [m.GetNumberOfPoints() for m in meshes]
    sprint(f"    compute_mean_poly: NN-matching, point counts {min(counts)}-{max(counts)}, N={len(meshes)}")

    ref_pts = get_points_numpy(meshes[0])
    mean_points = np.zeros_like(ref_pts)

    for m in meshes:
        locator = vtk.vtkKdTreePointLocator()
        locator.SetDataSet(m)
        locator.BuildLocator()
        pts = get_points_numpy(m)
        for j, pt in enumerate(ref_pts):
            closest_id = locator.FindClosestPoint(pt.tolist())
            mean_points[j] += pts[closest_id]

    mean_points /= len(meshes)

    mean_poly = vtk.vtkPolyData()
    mean_poly.DeepCopy(meshes[0])
    vtk_pts = mean_poly.GetPoints()
    for j in range(len(mean_points)):
        vtk_pts.SetPoint(j, mean_points[j].tolist())

    return mean_poly


# =============================================================================
# STEP E: VTK ICP (Rigid)
# หา rotation + translation ที่ทำให้ source เข้าใกล้ target ที่สุด
# convergence threshold = 0.01 mm (เหมาะกับ physical scale ของ hippocampus)
# =============================================================================

def run_vtk_icp(source_poly, target_poly):
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(source_poly)
    icp.SetTarget(target_poly)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(100)
    icp.SetMaximumMeanDistance(0.0001)  # ใน normalized units [-1,1]
    icp.CheckMeanDistanceOn()
    icp.Update()
    matrix = icp.GetMatrix()
    res = np.eye(4)
    for r in range(4):
        for c in range(4):
            res[r, c] = matrix.GetElement(r, c)
    return res


# =============================================================================
# STEP F: EXPORT aligned NIfTI
# Resample volume ต้นฉบับด้วย T_matrices ที่คำนวณได้ ลงบน reference grid
# ขนาด n_voxels^3 ที่ spacing_mm/voxel โดยจัด origin ให้ box อยู่กึ่งกลาง 0
# =============================================================================

def export_aligned_nifti(file_list, T_matrices, output_dir,
                         spacing_mm=0.5, n_voxels=128):
    out_vol_dir = os.path.join(output_dir, "aligned_nifti")
    os.makedirs(out_vol_dir, exist_ok=True)
    N = len(file_list)
    half = (n_voxels / 2.0) * spacing_mm

    for i, f in enumerate(file_list):
        basename = os.path.basename(f).split('.')[0]
        sprint(f"Saving [{i+1}/{N}]: {basename} (box={2*half:.1f}mm, spacing={spacing_mm}mm)")
        node = slicer.util.loadLabelVolume(f)
        T = T_matrices[i]

        t_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
        v_mat = vtk.vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                v_mat.SetElement(r, c, float(T[r, c]))
        t_node.SetMatrixTransformToParent(v_mat)

        ref_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "ref")
        img = vtk.vtkImageData()
        img.SetDimensions(n_voxels, n_voxels, n_voxels)
        img.AllocateScalars(vtk.VTK_SHORT, 1)
        ref_node.SetAndObserveImageData(img)
        ref_node.SetSpacing(spacing_mm, spacing_mm, spacing_mm)
        ref_node.SetOrigin(-half, -half, -half)

        params = {
            "inputVolume": node.GetID(),
            "referenceVolume": ref_node.GetID(),
            "outputVolume": ref_node.GetID(),
            "transformationFile": t_node.GetID(),
            "interpolationMode": "NearestNeighbor"
        }
        slicer.cli.run(
            slicer.modules.resamplescalarvectordwivolume,
            None, params, wait_for_completion=True
        )
        out_path = os.path.join(out_vol_dir, f"{basename}_aligned.nii.gz")
        slicer.util.saveNode(ref_node, out_path)

        slicer.mrmlScene.RemoveNode(node)
        slicer.mrmlScene.RemoveNode(t_node)
        slicer.mrmlScene.RemoveNode(ref_node)


# =============================================================================
# MAIN PIPELINE
# Two-stage normalization + rigid groupwise ICP:
#   1. Pre-ICP per-mesh normalize: center + scale ให้ |coord| <= 1
#   2. PCA orientation alignment
#   3. Orientation disambiguation (proper rotations, no reflection)
#   4. Groupwise ICP (rigid: rotation + translation only)
#   5. Post-ICP GLOBAL normalize: หา union bounding box ของทุก mesh
#                                  แล้วใช้ค่าเดียวกัน normalize ทุก mesh
#                                  -> เก็บ relative size ในกลุ่ม + fit ใน [-1, 1]
#   6. Export aligned NIfTI
# =============================================================================

# หลัง pre-ICP per-mesh normalize + post-ICP global normalize
# coords ของ mesh จะอยู่ใน [-1, 1] (normalized units, ไม่ใช่ mm แล้ว)
# output volume = 128 voxels x 0.02 unit -> box [-1.28, 1.28] (margin 28%)
OUTPUT_SPACING = 0.02
OUTPUT_VOXELS = 128


def main():
    sprint("--- main2.py STARTING (rigid groupwise ICP) ---")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    args, unknown = parser.parse_known_args()

    # ----------------------------------------------------------------
    # ถ้าไม่มี --input_dir : เปิด folder picker dialog
    # ถ้าไม่มี --output_dir : ใช้ <SCRIPT_DIR>/output_<basename ของ input>
    # (พฤติกรรมเดียวกับที่ run_everything.bat เคยจัดให้)
    # ----------------------------------------------------------------
    input_dir = args.input_dir
    if not input_dir:
        sprint("No --input_dir given. Opening folder picker...")
        input_dir = prompt_folder("Select input folder containing NIfTI labels")
        if not input_dir:
            sprint("ERROR: No folder selected. Exiting.")
            return

    output_dir = args.output_dir
    if not output_dir:
        basename = os.path.basename(os.path.normpath(input_dir))
        output_dir = os.path.join(SCRIPT_DIR, f"output_{basename}")
        sprint(f"No --output_dir given. Using default: {output_dir}")

    input_dir = input_dir.replace("\\", "/")
    output_dir = output_dir.replace("\\", "/")
    os.makedirs(output_dir, exist_ok=True)
    sprint(f"Input  dir: {input_dir}")
    sprint(f"Output dir: {output_dir}")

    extensions = ["*.nii.gz", "*.nii", "*.hdr", "*.nrrd"]
    file_list = []
    for ext in extensions:
        file_list.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))
    file_list = sorted(list(set(file_list)))

    label_files = [f for f in file_list if "label" in os.path.basename(f).lower()]
    if label_files:
        file_list = label_files
        sprint(f"Prioritizing {len(file_list)} label files.")
    sprint(f"Total files to process: {len(file_list)}")

    if not file_list:
        sprint("ERROR: No files found in input_dir.")
        return

    # ----------------------------------------------------------------
    # STEP A: load + mesh
    # ----------------------------------------------------------------
    sprint("Step A: Loading and meshing label volumes...")
    meshes = []
    valid_files = []
    for i, f in enumerate(file_list):
        sprint(f"  Loading [{i+1}/{len(file_list)}]: {os.path.basename(f)}")
        p = load_and_mesh_node(f)
        if p and p.GetNumberOfPoints() > 0:
            meshes.append(p)
            valid_files.append(f)
        else:
            sprint(f"  WARNING: Skipping {os.path.basename(f)} (no points)")

    N = len(meshes)
    if N == 0:
        sprint("ERROR: No valid label meshes found.")
        return
    file_list = valid_files
    sprint(f"  Loaded {N} meshes successfully.")

    # ----------------------------------------------------------------
    # STEP 1: Per-mesh normalization (center + scale to fit |coord| <= 1)
    # ทำเพื่อให้ ICP convergence เสถียร + ทุก mesh เริ่มจาก scale เดียวกัน
    # (per-mesh scaling นี้ -- post-ICP จะ global-normalize อีกชั้น Step 5)
    # ----------------------------------------------------------------
    sprint("Step 1: Per-mesh normalization (center + scale to |coord| <= 1)...")
    aligned_meshes = []
    T_initial = []
    for i in range(N):
        pts = get_points_numpy(meshes[i])
        centroid = pts.mean(axis=0)
        T_cent = np.eye(4)
        T_cent[:3, 3] = -centroid
        poly_centered = apply_poly_transform(meshes[i], T_cent)

        b = poly_centered.GetBounds()
        max_abs = max(abs(b[0]), abs(b[1]), abs(b[2]),
                      abs(b[3]), abs(b[4]), abs(b[5]))
        s = 1.0 / max_abs if max_abs > 0 else 1.0

        T_scale = np.eye(4)
        T_scale[0, 0] = T_scale[1, 1] = T_scale[2, 2] = s

        T_combined = T_scale @ T_cent
        aligned_meshes.append(apply_poly_transform(meshes[i], T_combined))
        T_initial.append(T_combined)
        if i < 3 or i == N - 1:
            sprint(f"  Mesh {i+1}: centroid={centroid}, scale={s:.6f}")

    # ----------------------------------------------------------------
    # STEP 2: Principal Axis Alignment (PCA)
    # ----------------------------------------------------------------
    sprint("Step 2: Principal Axis Alignment (PCA)...")
    T_pca_list = []
    for i in range(N):
        aligned_meshes[i], T_p = principal_axis_align(aligned_meshes[i])
        T_pca_list.append(T_p)

    # ----------------------------------------------------------------
    # STEP 3: Orientation disambiguation (proper rotations only)
    # mesh[0] ใช้ heuristic; mesh[1..N] เปรียบกับ mesh[0]
    # ----------------------------------------------------------------
    sprint("Step 3: Orientation disambiguation (proper rotations vs reference)...")
    T_flip = []
    aligned_meshes[0], tf0, flipped0 = pole_flip_correction(aligned_meshes[0], reference_poly=None)
    T_flip.append(tf0)
    sprint(f"  Mesh 1 (reference): {'REORIENTED' if flipped0 else 'OK'}")

    for i in range(1, N):
        aligned_meshes[i], tfi, flippedi = pole_flip_correction(
            aligned_meshes[i], reference_poly=aligned_meshes[0]
        )
        T_flip.append(tfi)
        sprint(f"  Mesh {i+1}: {'REORIENTED' if flippedi else 'OK'}")

    # รวม transform chain: flip @ pca @ cent
    for i in range(N):
        T_initial[i] = T_flip[i] @ T_pca_list[i] @ T_initial[i]

    # ----------------------------------------------------------------
    # STEP 4: Groupwise ICP (rigid)
    # แต่ละรอบ: re-check orientation -> mean shape -> ICP to mean
    # ICP rigid (rotation+translation) ไม่สามารถ flip mesh ได้
    # -> ตรวจ orientation เฉพาะก่อนคำนวณ mean (pre-mean) ก็พอ
    # ----------------------------------------------------------------
    sprint("Step 4: Groupwise ICP (rigid, 5 rounds)...")
    T_icp = [np.eye(4) for _ in range(N)]
    NUM_GW_ITERATIONS = 5

    for gw_iter in range(NUM_GW_ITERATIONS):
        sprint(f"  [Groupwise Round {gw_iter+1}/{NUM_GW_ITERATIONS}]")

        sprint(f"  [Round {gw_iter+1}] Pre-mean orientation re-check...")
        for i in range(1, N):
            aligned_meshes[i], tfi, flippedi = pole_flip_correction(
                aligned_meshes[i], reference_poly=aligned_meshes[0]
            )
            if flippedi:
                sprint(f"    NOTE: Mesh {i+1} reoriented before mean update")
                T_icp[i] = tfi @ T_icp[i]

        ref_mean = compute_mean_poly(aligned_meshes)

        for i in range(N):
            dT = run_vtk_icp(aligned_meshes[i], ref_mean)
            aligned_meshes[i] = apply_poly_transform(aligned_meshes[i], dT)
            T_icp[i] = dT @ T_icp[i]

        total_d = sum(icp_distance(aligned_meshes[i], ref_mean) for i in range(N))
        sprint(f"  [Round {gw_iter+1}] Mean ICP dist to template: {total_d/N:.6f}")

    # ----------------------------------------------------------------
    # STEP 5: GLOBAL bounding-box normalization
    # หา union bounding box ของทุก mesh หลัง ICP
    # แล้วใช้ค่า min/max "เดียวกัน" normalize ทุก mesh
    # -> รักษา relative size ภายในกลุ่ม (subject ใหญ่ก็ยังใหญ่กว่า subject เล็ก)
    #    แต่ทั้งกลุ่มถูกย่อ/ขยายเข้าหา [-1, 1] พร้อมกัน
    # -> ทำให้ PC1 ของ shape PCA จับ "shape variation" ไม่ใช่ "size variation"
    # ----------------------------------------------------------------
    sprint("Step 5: Global bounding-box normalization (union of all aligned meshes)...")
    g_min = np.array([float('inf')] * 3)
    g_max = np.array([float('-inf')] * 3)
    for m in aligned_meshes:
        b = m.GetBounds()
        g_min[0] = min(g_min[0], b[0]); g_max[0] = max(g_max[0], b[1])
        g_min[1] = min(g_min[1], b[2]); g_max[1] = max(g_max[1], b[3])
        g_min[2] = min(g_min[2], b[4]); g_max[2] = max(g_max[2], b[5])
    sprint(f"  Union bounds (before): "
           f"X[{g_min[0]:+.4f},{g_max[0]:+.4f}]  "
           f"Y[{g_min[1]:+.4f},{g_max[1]:+.4f}]  "
           f"Z[{g_min[2]:+.4f},{g_max[2]:+.4f}]")

    # center on (g_min + g_max)/2, scale uniformly ด้วย max half-extent
    # uniform scale = รักษา aspect ratio
    g_center = (g_min + g_max) / 2.0
    g_half = (g_max - g_min) / 2.0
    max_half = float(g_half.max())
    global_scale = 1.0 / max_half if max_half > 0 else 1.0

    T_g_cent = np.eye(4)
    T_g_cent[:3, 3] = -g_center
    T_g_scale = np.eye(4)
    T_g_scale[0, 0] = T_g_scale[1, 1] = T_g_scale[2, 2] = global_scale
    T_global = T_g_scale @ T_g_cent
    sprint(f"  Global center = ({g_center[0]:+.4f}, {g_center[1]:+.4f}, {g_center[2]:+.4f})")
    sprint(f"  Global scale  = {global_scale:.6f}  "
           f"(max half-extent {max_half:.4f} -> 1.0)")

    for i in range(N):
        aligned_meshes[i] = apply_poly_transform(aligned_meshes[i], T_global)

    # verify post-normalize
    v_min = np.array([float('inf')] * 3)
    v_max = np.array([float('-inf')] * 3)
    for m in aligned_meshes:
        b = m.GetBounds()
        v_min[0] = min(v_min[0], b[0]); v_max[0] = max(v_max[0], b[1])
        v_min[1] = min(v_min[1], b[2]); v_max[1] = max(v_max[1], b[3])
        v_min[2] = min(v_min[2], b[4]); v_max[2] = max(v_max[2], b[5])
    sprint(f"  Union bounds (after) : "
           f"X[{v_min[0]:+.4f},{v_max[0]:+.4f}]  "
           f"Y[{v_min[1]:+.4f},{v_max[1]:+.4f}]  "
           f"Z[{v_min[2]:+.4f},{v_max[2]:+.4f}]")

    # final transform chain: global @ icp @ initial
    T_matrices = [T_global @ T_icp[i] @ T_initial[i] for i in range(N)]

    # ----------------------------------------------------------------
    # STEP 6: Save results
    # ----------------------------------------------------------------
    sprint("Step 6: Saving results...")
    np.save(os.path.join(output_dir, "T_matrices.npy"), np.array(T_matrices))
    sprint(f"  Saved T_matrices.npy ({N} matrices)")

    mean_poly = compute_mean_poly(aligned_meshes)
    writer = vtk.vtkPLYWriter()
    writer.SetFileName(os.path.join(output_dir, "mean_shape.ply"))
    writer.SetInputData(mean_poly)
    writer.Write()
    sprint(f"  Saved mean_shape.ply")

    export_aligned_nifti(file_list, T_matrices, output_dir,
                         spacing_mm=OUTPUT_SPACING,
                         n_voxels=OUTPUT_VOXELS)

    sprint(f"  All {N} aligned NIfTI saved to: {os.path.join(output_dir, 'aligned_nifti')}")
    sprint("--- main2.py FINISHED ---")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # ทำไม os._exit forced: --no-main-window ของ SlicerSALT บางครั้ง Qt event loop
    # ไม่ run -> slicer.util.exit ผ่าน QTimer ไม่ trigger -> Slicer process ค้าง
    # -> batch script รอ exit ไม่ได้ (เด้ง/hang) แก้โดย force-kill หลัง 1.5s
    import sys, os
    exit_code = 0
    with open(DEBUG_LOG, "w") as f:
        f.write(f"--- LOG START: {datetime.now()} ---\n")
    try:
        main()
    except Exception as e:
        import traceback
        sprint(f"FATAL ERROR: {str(e)}")
        with open(DEBUG_LOG, "a") as f:
            f.write(traceback.format_exc())
        exit_code = 1
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        try:
            slicer.util.exit(exit_code)
        except Exception:
            pass
        try:
            import time, threading
            def _force_kill():
                time.sleep(1.5)
                os._exit(exit_code)
            threading.Thread(target=_force_kill, daemon=True).start()
        except Exception:
            os._exit(exit_code)
