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

# canonical positions ที่ landmarks ควรไปอยู่ (ทุก subject ปลายทางเหมือนกัน)
CANONICAL = np.array([
    [0.0, 0.0, +1.0],   # HEAD    -> +Z
    [0.0, 0.0, -1.0],   # TAIL    -> -Z
    [+1.0, 0.0, 0.0],   # LATERAL -> +X
    [-1.0, 0.0, 0.0],   # MEDIAL  -> -X
])


def main():
    print("=" * 72)
    print("--- Anatomical Landmark Re-alignment ---")
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

    # ลำดับ preference (สูง -> ต่ำ):
    #   1. *_SPHARM_procalign.vtk   (template-aligned; consistent correspondences ระหว่าง
    #                                subjects -> PCA สะอาด)
    #   2. *_SPHARM_ellalign.vtk    (ellipsoid-aligned; correspondences ตรงกันใน "subject เดียว"
    #                                แต่ระหว่าง subjects มี sign-flip ambiguity)
    #   3. *_SPHARM.vtk            (raw; parameterization ของแต่ละ subject ต่างกัน)
    # procalign ต้องการรัน SPHARM-PDM ด้วย --reference_template (deep fix)
    # ถ้ามี procalign สำหรับ "ทุก" subject -> ใช้ procalign (correspondences สะอาดที่สุด)
    files = sorted(glob.glob(os.path.join(folder, "*_SPHARM_procalign.vtk")))
    source = ""
    if files and len(files) > 0:
        ellalign_count = len(glob.glob(os.path.join(folder, "*_SPHARM_ellalign.vtk")))
        if len(files) >= ellalign_count and ellalign_count > 0:
            source = (f"_SPHARM_procalign.vtk (template-aligned, "
                      f"{len(files)} subjects — recommended for PCA)")
        else:
            files = []

    # ใน SlicerSALT ผลลัพธ์สุดท้ายที่ผ่านการทำ Template alignment จะถูกเซฟเป็น _SPHARM.vtk
    # ดังนั้นเราต้องโหลด _SPHARM.vtk แทนที่จะเป็น _ellalign.vtk
    if not files:
        all_spharm = sorted(glob.glob(os.path.join(folder, "*_SPHARM.vtk")))
        files = [f for f in all_spharm
                 if not any(s in os.path.basename(f)
                            for s in ("_ellalign", "_grid", "_realigned", "_procalign"))]
        source = "_SPHARM.vtk (SlicerSALT final output, should have correspondence if templates were used)"

    # Fallback สุดท้ายถ้าไม่มี _SPHARM.vtk (ซึ่งเป็นไปได้ยาก)
    if not files:
        files = sorted(glob.glob(os.path.join(folder, "*_SPHARM_ellalign.vtk")))
        source = "_SPHARM_ellalign.vtk (WARNING: No vertex correspondence)"

    if not files:
        print(f"[ERROR] No SPHARM .vtk found in {folder}")
        sys.exit(1)

    print(f"\nSource: {source}")
    print(f"Found {len(files)} meshes\n")
    print(f"Canonical orientation:")
    print(f"  HEAD    -> (0, 0, +1)  (+Z 'เหนือ')")
    print(f"  TAIL    -> (0, 0, -1)  (-Z 'ใต้')")
    print(f"  LATERAL -> (+1, 0, 0)  (+X 'ออก')")
    print(f"  MEDIAL  -> (-1, 0, 0)  (-X 'ตก')")
    print()

    # ---------- Phase 1: หา landmarks ของทุก subject ----------
    print(f"Phase 1: Detecting anatomical landmarks per subject...")
    subjects = []
    skipped = 0
    for f in files:
        poly = load_polydata(f)
        if poly is None or poly.GetNumberOfPoints() < 10:
            print(f"  SKIP: {os.path.basename(f)} (empty)")
            skipped += 1
            continue
        pts = points_to_numpy(poly)
        h, t, l, m = find_anatomical_landmarks(pts)
        subjects.append({
            "file": f,
            "poly": poly,
            "pts": pts,
            "lm_indices": (h, t, l, m),
            "lm_pts": pts[[h, t, l, m]],
        })

    if not subjects:
        print("[ERROR] No subjects to align.")
        return

    # ---------- Phase 2: GPA + label flipping ----------
    # แทนที่จะใช้ canonical คงที่ ใช้ consensus (mean ของ landmarks หลัง align)
    # iterate จนกว่า mean ไม่เปลี่ยนแล้ว -> landmarks ทั้งกลุ่ม "ทับกัน"
    # ระหว่าง iterate ลอง flip h<->t, l<->m เลือกที่ fit ดีสุด
    # -> กันปัญหา anatomical detection พลาดในบาง subject
    print(f"\nPhase 2: GPA + label flipping on 4 landmarks ({len(subjects)} subjects)...")
    all_landmarks = [s["lm_pts"] for s in subjects]
    Rs_gpa, ts_gpa, mean_lm, perms, history = gpa_landmarks(
        all_landmarks, max_iter=30, tol=1e-7)
    print(f"  Convergence history (||delta-mean||): "
          f"{', '.join(f'{d:.2e}' for d in history[:10])}"
          f"{'...' if len(history) > 10 else ''}")
    print(f"  Converged in {len(history)} iterations.")

    # นับการ flip — ดูว่ามีกี่ subject ที่ detection พลาด
    flipped_ht = sum(1 for p in perms if p[0] != 0)
    flipped_lm = sum(1 for p in perms if p[2] != 2)
    print(f"  Label corrections: head/tail flipped in {flipped_ht}/{len(subjects)} subjects, "
          f"lat/med flipped in {flipped_lm}/{len(subjects)} subjects")

    # apply perms to lm_indices ของแต่ละ subject (เพื่อ Phase 4 print + final spread)
    for i, s in enumerate(subjects):
        old_idx = list(s["lm_indices"])
        s["lm_indices"] = tuple(old_idx[k] for k in perms[i])
        s["lm_pts"] = s["pts"][list(s["lm_indices"])]

    # ---------- Phase 3: หมุนทั้งกลุ่มไป canonical (HEAD=+Z) ----------
    # หลัง GPA - consensus mean_lm อยู่ใน "consensus frame"
    # หมุนทั้งกลุ่มอีกครั้งเพื่อ HEAD=+Z, TAIL=-Z, LAT=+X, MED=-X
    print(f"\nPhase 3: Rotating group to canonical orientation (HEAD=+Z, LAT=+X)...")
    size = float(np.linalg.norm(mean_lm - mean_lm.mean(axis=0), axis=1).mean())
    canonical_scaled = CANONICAL * size
    R_group, t_group = kabsch_proper(mean_lm, canonical_scaled)
    print(f"  Group rotation angle: "
          f"{np.degrees(np.arccos(np.clip((np.trace(R_group)-1)/2, -1, 1))):.2f} deg")

    # ---------- Phase 4: Apply landmark-based rotation ----------
    print(f"\nPhase 4: Applying landmark-based rotation...")
    aligned_pts_list = []
    for i, s in enumerate(subjects):
        R_init = R_group @ Rs_gpa[i]
        t_init = R_group @ ts_gpa[i] + t_group
        aligned_pts_list.append((R_init @ s["pts"].T).T + t_init)

    # ---------- Phase 5: Write output + report ----------
    print(f"\nPhase 5: Writing output...")
    print(f"{'Subject':<48} {'head':>5} {'tail':>5} {'lat':>5} {'med':>5} {'resid':>10}")
    print("-" * 92)

    final_landmarks = []
    for i, s in enumerate(subjects):
        new_pts = aligned_pts_list[i]
        new_lm = new_pts[list(s["lm_indices"])]
        residual = float(np.linalg.norm(new_lm - canonical_scaled, axis=1).mean())
        final_landmarks.append(new_lm)

        out_poly = replace_points(s["poly"], new_pts)
        base = s["file"]
        for suf in ("_SPHARM_procalign.vtk", "_SPHARM_ellalign.vtk", "_SPHARM.vtk"):
            if base.endswith(suf):
                base = base[: -len(suf)]
                break
        out_path = base + "_SPHARM_realigned.vtk"
        write_polydata(out_poly, out_path)

        name = os.path.basename(s["file"])
        for suf in ("_SPHARM_procalign.vtk", "_SPHARM_ellalign.vtk", "_SPHARM.vtk"):
            name = name.replace(suf, "")
        h, t, l, m = s["lm_indices"]
        print(f"  {name:<46} {h:>5} {t:>5} {l:>5} {m:>5} {residual:>10.5f}")

    final_landmarks = np.array(final_landmarks)  # (N, 4, 3) — Phase-1 detected

    # Position-based landmarks (เหมือนที่ view_spharm_meshes.py ใช้):
    #   หลัง realign — canonical คือ head=+Z, tail=-Z, lat=+X, med=-X
    #   ใช้ argmax/argmin บนแต่ละแกน → ตำแหน่งที่เห็นจริงในรูป
    position_landmarks = []
    for i, _ in enumerate(subjects):
        pts = aligned_pts_list[i]
        h_idx = int(np.argmax(pts[:, 2]))
        t_idx = int(np.argmin(pts[:, 2]))
        l_idx = int(np.argmax(pts[:, 0]))
        m_idx = int(np.argmin(pts[:, 0]))
        position_landmarks.append(pts[[h_idx, t_idx, l_idx, m_idx]])
    position_landmarks = np.array(position_landmarks)  # (N, 4, 3)

    print()
    print("=" * 72)
    print(f"Done. Processed {len(subjects)}/{len(files)} subjects.")
    print(f"\nLandmark clustering quality (Phase-1 detected indices):")
    print(f"  (might be noisy เพราะ Phase 1 anatomical detection ไม่ stable เสมอ)")
    for k, name in enumerate(["HEAD", "TAIL", "LAT ", "MED "]):
        cluster_pts = final_landmarks[:, k, :]
        spread = float(np.linalg.norm(cluster_pts - cluster_pts.mean(axis=0),
                                      axis=1).mean())
        print(f"    {name}: mean distance from cluster center = {spread:.4f}")
    print(f"\nLandmark clustering quality (position-based, ตรงกับ viewer):")
    print(f"  (วัดว่า argmax/argmin บนแต่ละแกน ของทุก subject ตกที่ตำแหน่งใกล้กันมั้ย)")
    for k, name in enumerate(["HEAD", "TAIL", "LAT ", "MED "]):
        cluster_pts = position_landmarks[:, k, :]
        spread = float(np.linalg.norm(cluster_pts - cluster_pts.mean(axis=0),
                                      axis=1).mean())
        print(f"    {name}: mean distance from cluster center = {spread:.4f}")
    print(f"\nOutput: *_SPHARM_realigned.vtk in {folder}")

    # ---------- Phase 6: Resolve SPHARM parameterization flips ----------
    # หลัง realign vertex correspondences ระหว่าง subjects ยังอาจมี ambiguity
    # จาก SPHARM parameterization (180° rotations รอบแกน X/Y/Z บน parameter sphere)
    # → ใช้ _para.vtk คำนวณ 4 permutations เลือกที่ทำให้ subject ตรงกับ template
    #   (subject 0) ที่สุด แล้ว reorder vertices ให้ correspondences ถูกต้องสำหรับ PCA
    # หมายเหตุ: cell topology ของ pca_ready จะ "พัง" (ไม่ใช่ render-able surface)
    #           แต่ vertex correspondences ถูกต้อง — เป็น input สำหรับ ShapePCA
    print(f"\nPhase 6: Resolving SPHARM parameterization flips (pca_ready output)...")

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print("  [WARN] scipy not installed — skipping flip resolution.")
        print(f"        Run PCA on *_SPHARM_realigned.vtk instead.")
        print("=" * 72)
        return

    para_files = sorted(glob.glob(os.path.join(folder, "*_para.vtk")))
    if not para_files:
        print(f"  [WARN] No _para.vtk in {folder} — skipping flip resolution.")
        print("=" * 72)
        return

    print(f"  Using parameter sphere: {os.path.basename(para_files[0])}")
    para_poly = load_polydata(para_files[0])
    para_verts = points_to_numpy(para_poly)
    para_verts = para_verts - para_verts.mean(axis=0)

    tree = cKDTree(para_verts)
    perms = {"Identity": np.arange(len(para_verts))}
    for name, axes in [("FlipX", (1, 2)), ("FlipY", (0, 2)), ("FlipZ", (0, 1))]:
        rot = para_verts.copy()
        for ax in axes:
            rot[:, ax] *= -1
        _, nn = tree.query(rot, k=1)
        perms[name] = nn

    template_pts = aligned_pts_list[0]
    flip_counts = {k: 0 for k in perms}

    print(f"  {'Subject':<48} {'flip':<10} {'dist':>10}")
    print("  " + "-" * 72)
    for i, s in enumerate(subjects):
        subj_pts = aligned_pts_list[i]
        best_flip = "Identity"
        best_dist = float("inf")
        for name, P in perms.items():
            d = float(np.mean(np.linalg.norm(subj_pts[P] - template_pts, axis=1)))
            if d < best_dist:
                best_dist = d
                best_flip = name
        flip_counts[best_flip] += 1

        fixed_pts = subj_pts[perms[best_flip]]
        out_poly = replace_points(s["poly"], fixed_pts)

        base = s["file"]
        for suf in ("_SPHARM_procalign.vtk", "_SPHARM_ellalign.vtk", "_SPHARM.vtk"):
            if base.endswith(suf):
                base = base[: -len(suf)]
                break
        out_path = base + "_SPHARM_pca_ready.vtk"
        write_polydata(out_poly, out_path)

        name = os.path.basename(s["file"])
        for suf in ("_SPHARM_procalign.vtk", "_SPHARM_ellalign.vtk", "_SPHARM.vtk"):
            name = name.replace(suf, "")
        print(f"  {name:<48} {best_flip:<10} {best_dist:>10.4f}")

    print(f"\n  Flip distribution:")
    for k, v in flip_counts.items():
        print(f"    {k:<10}: {v} subjects")
    print(f"\nOutput: *_SPHARM_pca_ready.vtk in {folder}")
    print("=" * 72)


if __name__ == "__main__":
    main()
