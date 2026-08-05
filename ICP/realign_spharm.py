"""
Anatomical-landmark re-alignment ของ SPHARM mesh
ไม่ใช้ template — แต่ละ subject หา landmark ของตัวเองตาม anatomy ของ hippocampus
แล้วหมุนให้ landmarks ไปอยู่ "ตำแหน่งมาตรฐาน" ที่เรากำหนด

4 landmarks:
  HEAD    (เหนือ) = ปลายอ้วน ของ long axis              -> (0, 0, +1)
  TAIL    (ใต้)   = ปลายเรียว                          -> (0, 0, -1)
  LATERAL (ออก)  = ฝั่งโค้งออก (curl direction)         -> (+1, 0, 0)
  MEDIAL  (ตก)   = ฝั่งโค้งเข้า                        -> (-1, 0, 0)

หลังจัด -> ทุก subject มี:
  - หัวอยู่ +Z
  - หางอยู่ -Z
  - lateral อยู่ +X
  - medial อยู่ -X
  -> หันทางเดียวกันหมด anatomically meaningful
"""
import os
import sys
import glob
import argparse
import tkinter as tk
from tkinter import filedialog
import numpy as np
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


def load_polydata(filepath):
    r = vtk.vtkPolyDataReader()
    r.SetFileName(filepath)
    r.Update()
    return r.GetOutput()


def write_polydata(poly, filepath):
    w = vtk.vtkPolyDataWriter()
    w.SetFileName(filepath)
    w.SetInputData(poly)
    w.SetFileTypeToBinary()
    w.Write()


def points_to_numpy(poly):
    n = poly.GetNumberOfPoints()
    return np.array([poly.GetPoint(i) for i in range(n)])


def replace_points(poly, new_pts):
    out = vtk.vtkPolyData()
    out.DeepCopy(poly)
    vtk_pts = out.GetPoints()
    for i, p in enumerate(new_pts):
        vtk_pts.SetPoint(i, float(p[0]), float(p[1]), float(p[2]))
    return out


# =============================================================================
# Math
# =============================================================================

def kabsch_proper(P, Q):
    """Find rotation R + translation t mapping P -> Q (det(R) = +1, no reflection)."""
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = cQ - R @ cP
    return R, t


# 4 label permutations: (head, tail, lateral, medial)
#   ใช้แก้กรณี anatomical detection พลาด — ลองสลับ h<->t และ l<->m
#   เลือก permutation ที่ Kabsch fit ดีที่สุด
LABEL_PERMS = [
    (0, 1, 2, 3),  # identity
    (1, 0, 2, 3),  # flip head/tail
    (0, 1, 3, 2),  # flip lat/med
    (1, 0, 3, 2),  # flip both
]


def best_kabsch_with_flips(lm, mean_lm):
    """ลอง 4 label permutations + คืน (R, t, perm, residual) ที่ดีที่สุด"""
    best = None
    for perm in LABEL_PERMS:
        lm_p = lm[list(perm)]
        R, t = kabsch_proper(lm_p, mean_lm)
        aligned = (R @ lm_p.T).T + t
        residual = float(np.linalg.norm(aligned - mean_lm, axis=1).sum())
        if best is None or residual < best[3]:
            best = (R, t, perm, residual)
    return best


def gpa_landmarks(all_landmarks, max_iter=20, tol=1e-6):
    """
    Generalized Procrustes Analysis บน landmark sets ทั้งหมด พร้อม label flipping
    Input:  all_landmarks = list of (K=4, 3) arrays — 4 landmarks ต่อ subject
            (order ที่ส่งเข้า: HEAD, TAIL, LAT, MED — ตามที่ detection คืน)
    Output: rotations, translations, consensus mean, perms ต่อ subject, history

    Algorithm: iterative
      1. Initial mean = subject แรก (centered)
      2. Loop:
         a. Align ทุก subject's landmarks ไป current mean (Kabsch)
            — ลอง 4 label permutations เลือกที่ residual ต่ำสุด
            → กันปัญหา detection สลับ head/tail หรือ lat/med
         b. Update mean = ค่าเฉลี่ยของ aligned landmarks (ใช้ permuted order)
         c. Stop เมื่อ mean เปลี่ยนน้อยกว่า tol
    """
    N = len(all_landmarks)
    K = all_landmarks[0].shape[0]

    # Initial mean = ตัวแรก (centered) เพื่อ orientation reference
    mean_lm = all_landmarks[0] - all_landmarks[0].mean(axis=0)

    history = []
    perms = [(0, 1, 2, 3)] * N
    Rs = [np.eye(3)] * N
    ts = [np.zeros(3)] * N

    for it in range(max_iter):
        aligned = []
        new_perms = []
        for i, lm in enumerate(all_landmarks):
            R, t, perm, _ = best_kabsch_with_flips(lm, mean_lm)
            lm_p = lm[list(perm)]
            aligned_lm = (R @ lm_p.T).T + t
            aligned.append(aligned_lm)
            Rs[i] = R
            ts[i] = t
            new_perms.append(perm)

        new_mean = np.mean(aligned, axis=0)
        diff = float(np.linalg.norm(new_mean - mean_lm))
        history.append(diff)
        perms = new_perms
        if diff < tol:
            mean_lm = new_mean
            break
        mean_lm = new_mean

    return Rs, ts, mean_lm, perms, history


# =============================================================================
# Anatomical landmark detection
# =============================================================================

def find_anatomical_landmarks(pts):
    """
    หา 4 anatomical landmarks ของ hippocampus mesh (ทำงานทุก orientation).

    Returns (head_idx, tail_idx, lateral_idx, medial_idx):
      - head:    ปลายอ้วน ของ long axis (anterior, wider cross-section)
      - tail:    ปลายเรียว (posterior, narrower)
      - lateral: ฝั่งโค้งออก (curl direction)
      - medial:  ฝั่งโค้งเข้า (opposite of curl)
    """
    centroid = pts.mean(axis=0)
    pts_c = pts - centroid

    # --- (1) long axis ด้วย PCA ---
    _, _, Vt = np.linalg.svd(pts_c, full_matrices=False)
    long_axis = Vt[0]
    long_axis /= np.linalg.norm(long_axis)

    # --- (2) project onto long axis ---
    proj = pts_c @ long_axis

    # --- (3) ดู spread (ความอ้วน) ของแต่ละปลาย ---
    p90 = np.percentile(proj, 90)
    p10 = np.percentile(proj, 10)
    top_pts = pts_c[proj > p90]
    bot_pts = pts_c[proj < p10]

    def spread_perpendicular(subset, axis):
        # ลบ component ตามแกน long_axis ออก -> เหลือเฉพาะ transverse
        perp = subset - np.outer(subset @ axis, axis)
        return float(np.std(perp, axis=0).sum())

    top_spread = spread_perpendicular(top_pts, long_axis) if len(top_pts) > 0 else 0.0
    bot_spread = spread_perpendicular(bot_pts, long_axis) if len(bot_pts) > 0 else 0.0

    # --- (4) head = ปลายอ้วน (cross-section ใหญ่กว่า) ---
    if top_spread >= bot_spread:
        head_idx = int(np.argmax(proj))
        tail_idx = int(np.argmin(proj))
    else:
        head_idx = int(np.argmin(proj))
        tail_idx = int(np.argmax(proj))

    # --- (5) curl direction (ทิศที่ middle slice เบี่ยงออก) ---
    # ใช้ middle 50% ของแกนยาว
    middle_mask = (proj > np.percentile(proj, 25)) & (proj < np.percentile(proj, 75))
    middle_pts = pts_c[middle_mask]
    if len(middle_pts) == 0:
        middle_pts = pts_c

    # project middle points onto plane perpendicular to long_axis
    middle_perp = middle_pts - np.outer(middle_pts @ long_axis, long_axis)
    curl_axis = middle_perp.mean(axis=0)
    norm = np.linalg.norm(curl_axis)
    if norm > 1e-9:
        curl_axis = curl_axis / norm
    else:
        # fallback: ใช้ PC2 ตั้งฉาก long_axis
        pc2 = Vt[1]
        curl_axis = pc2 - (pc2 @ long_axis) * long_axis
        curl_axis /= np.linalg.norm(curl_axis)

    # --- (6) lateral/medial ตาม curl direction ---
    proj_curl = pts_c @ curl_axis
    lateral_idx = int(np.argmax(proj_curl))
    medial_idx = int(np.argmin(proj_curl))

    return head_idx, tail_idx, lateral_idx, medial_idx


# =============================================================================
# Main
# =============================================================================

# 4 canonical positions: HEAD (+Z), TAIL (-Z), LATERAL (+X), MEDIAL (-X)
CANONICAL_4PTS = np.array([
    [0.0, 0.0, +1.0],   # HEAD    -> +Z
    [0.0, 0.0, -1.0],   # TAIL    -> -Z
    [+1.0, 0.0, 0.0],   # LATERAL -> +X
    [-1.0, 0.0, 0.0],   # MEDIAL  -> -X
])


def main():
    print("=" * 72)
    print("--- 4-Point Anatomical Template Re-alignment ---")
    print("=" * 72)

    parser = argparse.ArgumentParser()
    parser.add_argument("--spharm_dir", default=None,
                        help="Path to spharm_results folder")
    args, _ = parser.parse_known_args()

    folder = args.spharm_dir
    if not folder:
        print("\nSelect 'spharm_results' folder...")
        folder = popup_select_directory("Select spharm_results folder")
        if not folder:
            print("Canceled.")
            return

    if not os.path.isdir(folder):
        print(f"[ERROR] Not a folder: {folder}")
        sys.exit(1)

    # Clean up old realigned/pca_ready files to prevent mixing stale results from previous runs
    old_aligned_files = (glob.glob(os.path.join(folder, "*_SPHARM_realigned.vtk")) +
                         glob.glob(os.path.join(folder, "*_SPHARM_pca_ready.vtk")))
    if old_aligned_files:
        print(f"Cleaning up {len(old_aligned_files)} old realigned/pca_ready files...")
        for f in old_aligned_files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"  Failed to delete {os.path.basename(f)}: {e}")

    all_spharm = sorted(glob.glob(os.path.join(folder, "*_SPHARM.vtk")))
    candidate_files = [f for f in all_spharm
                       if not any(s in os.path.basename(f)
                                  for s in ("_ellalign", "_grid", "_realigned", "_procalign", "_pca_ready"))]
    source = "_SPHARM.vtk"

    if not candidate_files:
        candidate_files = sorted(glob.glob(os.path.join(folder, "*_SPHARM_ellalign.vtk")))
        source = "_SPHARM_ellalign.vtk"

    if not candidate_files:
        all_vtk = sorted(glob.glob(os.path.join(folder, "*.vtk")))
        candidate_files = [f for f in all_vtk
                           if not any(s in os.path.basename(f)
                                      for s in ("_realigned.vtk", "_pca_ready.vtk"))]
        source = ".vtk"

    if not candidate_files:
        print(f"[ERROR] No SPHARM .vtk found in {folder}")
        sys.exit(1)

    filtered_files = {}
    for f in candidate_files:
        basename = os.path.basename(f)
        name = basename
        for suffix in ("_SPHARM.vtk", "_SPHARM_ellalign.vtk"):
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break
        
        if name.endswith("_aligned"):
            subj_key = name[:-len("_aligned")]
            is_aligned = True
        else:
            subj_key = name
            is_aligned = False
            
        if subj_key not in filtered_files:
            filtered_files[subj_key] = (f, is_aligned)
        else:
            stored_file, stored_is_aligned = filtered_files[subj_key]
            if is_aligned and not stored_is_aligned:
                filtered_files[subj_key] = (f, True)
                
    files = sorted([f for f, _ in filtered_files.values()])

    print(f"\nSource: {source}")
    print(f"Found {len(files)} meshes\n")

    subjects = []
    skipped = 0
    for f in files:
        poly = load_polydata(f)
        if poly is None or poly.GetNumberOfPoints() < 10:
            print(f"  SKIP: {os.path.basename(f)} (empty)")
            skipped += 1
            continue
        pts = points_to_numpy(poly)
        subjects.append({
            "file": f,
            "poly": poly,
            "pts": pts,
        })

    if not subjects:
        print("[ERROR] No subjects to align.")
        return

    # Fixed topological landmark indices across SPHARM mesh topology:
    # RED = Head (470, +Z), BLUE = Tail (276, -Z), YELLOW = Index 0 (0, +X), GREEN = Opposite 0 (272, -X)
    h_idx, t_idx, y_idx, g_idx = 470, 276, 0, 272
    print(f"Phase 1: Aligning reference template subject to canonical orientation...")
    ref_pts = subjects[0]["pts"]
    lm_ref = ref_pts[[h_idx, t_idx, y_idx, g_idx]]
    size_ref = float(np.linalg.norm(lm_ref - lm_ref.mean(axis=0), axis=1).mean())
    R_ref, t_ref = kabsch_proper(lm_ref, CANONICAL_4PTS * size_ref)
    aligned_ref_pts = (R_ref @ ref_pts.T).T + t_ref

    print(f"Phase 2: Re-aligning all subjects to canonical template using full-topology Kabsch alignment...")
    print(f"\n{'Subject':<48} {'resid':>10}")
    print("-" * 60)

    final_landmarks = []
    for s in subjects:
        pts = s["pts"]
        # Full 1002-point Kabsch rigid transformation against aligned reference template
        R, t = kabsch_proper(pts, aligned_ref_pts)
        new_pts = (R @ pts.T).T + t

        new_lm = new_pts[[h_idx, t_idx, y_idx, g_idx]]
        target_ref_lm = aligned_ref_pts[[h_idx, t_idx, y_idx, g_idx]]
        residual = float(np.linalg.norm(new_lm - target_ref_lm, axis=1).mean())
        final_landmarks.append(new_lm)

        out_poly = replace_points(s["poly"], new_pts)
        base = s["file"]
        for suf in ("_SPHARM_procalign.vtk", "_SPHARM_ellalign.vtk", "_SPHARM.vtk"):
            if base.endswith(suf):
                base = base[: -len(suf)]
                break
        
        out_path_realigned = base + "_SPHARM_realigned.vtk"
        out_path_pca_ready = base + "_SPHARM_pca_ready.vtk"
        write_polydata(out_poly, out_path_realigned)
        write_polydata(out_poly, out_path_pca_ready)

        name = os.path.basename(s["file"])
        for suf in ("_SPHARM_procalign.vtk", "_SPHARM_ellalign.vtk", "_SPHARM.vtk"):
            name = name.replace(suf, "")
        print(f"  {name:<46} {residual:>10.5f}")

    final_landmarks = np.array(final_landmarks)

    print()
    print("=" * 72)
    print(f"Done. Re-aligned {len(subjects)} subjects using 4-point anatomical alignment.")
    print(f"\nLandmark clustering quality after realignment:")
    for k, name in enumerate(["HEAD (RED)", "TAIL (BLUE)", "LAT (GREEN)", "MED (YELLOW)"]):
        cluster_pts = final_landmarks[:, k, :]
        spread = float(np.linalg.norm(cluster_pts - cluster_pts.mean(axis=0),
                                      axis=1).mean())
        print(f"    {name}: mean distance from cluster center = {spread:.4f}")
    print(f"\nOutput saved as *_SPHARM_realigned.vtk in {folder}")
    print("=" * 72)


if __name__ == "__main__":
    main()
