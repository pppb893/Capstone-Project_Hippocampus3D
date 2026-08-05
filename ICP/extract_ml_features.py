import os
import glob
import csv
import argparse
import tkinter as tk
from tkinter import filedialog
import vtk
from vtk.util import numpy_support
import numpy as np


# =============================================================================
# Helper: Folder Picker
# =============================================================================
def prompt_folder(title):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title=title, initialdir=os.getcwd())
    root.destroy()
    return folder if folder else None


# =============================================================================
# Group Classification
# =============================================================================
def classify_subject(subject_name):
    is_left_side = subject_name.startswith("left_") or subject_name.startswith("lh_") or "_lh" in subject_name.lower()
    name_upper = subject_name.upper()
    
    # 1. Healthy Control / Normal (0)
    if "_HEALTHY" in name_upper or "HEALTHY" in name_upper or "HFH_" in name_upper or "NORMAL" in name_upper:
        return "Healthy Control", 0
    
    # 2. Ipsilateral TLE (Diseased) (1)
    elif (is_left_side and "LEFT-TLE" in name_upper) or (not is_left_side and "RIGHT-TLE" in name_upper):
        return "Ipsilateral TLE (Diseased)", 1
        
    # 3. Contralateral TLE (Healthy-side) (2)
    elif (is_left_side and "RIGHT-TLE" in name_upper) or (not is_left_side and "LEFT-TLE" in name_upper):
        return "Contralateral TLE (Healthy-side)", 2
        
    # 4. General TLE (1)
    elif "TLE" in name_upper:
        return "Ipsilateral TLE (Diseased)", 1
        
    return "Unknown", -1


# =============================================================================
# Main Extraction
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spharm_dir", default=None,
                        help="Path to spharm_results folder. ถ้าไม่ระบุจะเด้ง dialog")
    args = parser.parse_args()

    spharm_dir = args.spharm_dir
    if not spharm_dir:
        print("Opening folder picker...")
        spharm_dir = prompt_folder("Select 'spharm_results' folder (or output_xxx folder)")
    
    if not spharm_dir:
        print("Error: No folder selected. Exiting.")
        return

    # Handle parent directory selection
    if os.path.basename(spharm_dir.rstrip("\\/")).lower() != "spharm_results":
        candidate = os.path.join(spharm_dir, "spharm_results")
        if os.path.isdir(candidate):
            spharm_dir = candidate

    print(f"Target folder: {spharm_dir}")

    # 1. ค้นหาไฟล์ aligned vtk
    vtk_files = sorted(glob.glob(os.path.join(spharm_dir, "*_SPHARM_pca_ready.vtk")))
    source = "pca_ready"
    if not vtk_files:
        vtk_files = sorted(glob.glob(os.path.join(spharm_dir, "*_SPHARM_realigned.vtk")))
        source = "realigned"
    if not vtk_files:
        vtk_files = sorted(glob.glob(os.path.join(spharm_dir, "*_SPHARM_procalign.vtk")))
        source = "procalign"
    if not vtk_files:
        vtk_files = sorted(glob.glob(os.path.join(spharm_dir, "*_SPHARM_ellalign.vtk")))
        source = "ellalign"
    if not vtk_files:
        vtk_files = sorted([f for f in glob.glob(os.path.join(spharm_dir, "*.vtk")) if not f.endswith("_grid.vtk")])
        source = "vtk"

    if not vtk_files:
        print(f"Error: No aligned SPHARM .vtk files found in {spharm_dir}")
        return

    print(f"Found {len(vtk_files)} aligned meshes (source type: '{source}'). Processing...")

    # ตั้งค่าโฟลเดอร์สำหรับผลลัพธ์ ML
    ml_output_dir = os.path.join(os.path.dirname(spharm_dir), "ml_features")
    if not os.path.exists(ml_output_dir):
        os.makedirs(ml_output_dir)

    coords_csv_path = os.path.join(ml_output_dir, "spharm_xyz_coords.csv")
    edges_csv_path = os.path.join(ml_output_dir, "mesh_edges.csv")

    # 2. อ่านไฟล์แรกเพื่อสร้างหัวคอลัมน์และดึง Graph Topology (Edges)
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_files[0])
    reader.Update()
    poly = reader.GetOutput()

    num_points = poly.GetNumberOfPoints()
    print(f"Number of points per mesh: {num_points} vertices")

    # ดึงขอบเชื่อมต่อ (Edges) จากสามเหลี่ยมในโครงสร้าง Mesh
    print("Extracting mesh topology (undirected edges)...")
    cells = poly.GetPolys()
    id_list = vtk.vtkIdList()
    cells.InitTraversal()
    
    unique_edges = set()
    while cells.GetNextCell(id_list):
        n_pts = id_list.GetNumberOfIds()
        for i in range(n_pts):
            p1 = id_list.GetId(i)
            p2 = id_list.GetId((i + 1) % n_pts)
            # บันทึกเป็นคู่อันดับที่ไม่ซ้ำกัน
            edge = tuple(sorted((p1, p2)))
            unique_edges.add(edge)

    # เขียนขอบลงไฟล์ mesh_edges.csv
    with open(edges_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Source", "Target"])
        for edge in sorted(unique_edges):
            writer.writerow(edge)
    print(f"Saved {len(unique_edges)} unique edges to: {edges_csv_path}")

    # 3. วนลูปอ่านพิกัดดิบของทุกไฟล์ และบันทึก
    print("Extracting coordinates and classifications...")
    
    # สร้างหัวตาราง (Header)
    header = ["Subject", "Group_Name", "Group_Label"]
    for idx in range(num_points):
        header.extend([f"x_{idx}", f"y_{idx}", f"z_{idx}"])

    with open(coords_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for filepath in vtk_files:
            filename = os.path.basename(filepath)
            # ลบส่วนขยายออกเพื่อใช้เป็นชื่อ Subject
            subject_name = filename
            for suffix in ("_SPHARM_pca_ready.vtk", "_SPHARM_realigned.vtk", "_SPHARM_procalign.vtk", "_SPHARM_ellalign.vtk"):
                subject_name = subject_name.replace(suffix, "")

            # จัดกลุ่ม
            group_name, group_label = classify_subject(subject_name)

            # โหลดพิกัด
            mesh_reader = vtk.vtkPolyDataReader()
            mesh_reader.SetFileName(filepath)
            mesh_reader.Update()
            mesh_poly = mesh_reader.GetOutput()
            
            if mesh_poly.GetNumberOfPoints() != num_points:
                print(f"Warning: Mesh {filename} has {mesh_poly.GetNumberOfPoints()} points instead of {num_points}. Skipping.")
                continue

            pts = numpy_support.vtk_to_numpy(mesh_poly.GetPoints().GetData())
            flat_pts = pts.flatten() # ขนาด (3N,)

            # ประกอบข้อมูลแถว
            row = [subject_name, group_name, group_label]
            row.extend(["{:.8f}".format(val) for val in flat_pts])
            writer.writerow(row)

    print(f"Successfully saved features for {len(vtk_files)} subjects to: {coords_csv_path}")
    print("=" * 60)
    print("Done! You can use these two CSV files directly in your GNN model.")
    print("=" * 60)


if __name__ == "__main__":
    main()
