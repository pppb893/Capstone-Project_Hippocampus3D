import os
import csv
import argparse
import tkinter as tk
from tkinter import filedialog
import numpy as np


# =============================================================================
# Helper: File Picker
# =============================================================================
def prompt_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes, initialdir=os.getcwd())
    root.destroy()
    return file_path if file_path else None


# =============================================================================
# Main PCA Extraction
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coords_csv", default=None,
                        help="Path to spharm_xyz_coords.csv file. ถ้าไม่ระบุจะเด้ง dialog")
    args = parser.parse_args()

    coords_csv = args.coords_csv
    if not coords_csv:
        print("Opening file picker...")
        coords_csv = prompt_file("Select 'spharm_xyz_coords.csv' file", [("CSV Files", "*.csv")])

    if not coords_csv:
        print("Error: No file selected. Exiting.")
        return

    print(f"Loading coordinates from: {coords_csv}")
    
    subjects = []
    group_names = []
    group_labels = []
    coords_list = []

    with open(coords_csv, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        # ตรวจสอบความถูกต้องของคอลัมน์นำเข้า
        # ตัวแรกเป็น Subject, Group_Name, Group_Label ตามด้วย x_0, y_0, z_0 ...
        for row in reader:
            if not row or len(row) < 4:
                continue
            subjects.append(row[0])
            group_names.append(row[1])
            group_labels.append(int(row[2]))
            coords_list.append([float(x) for x in row[3:]])

    coords_matrix = np.array(coords_list)
    N, D = coords_matrix.shape
    num_points = D // 3
    print(f"Data loaded successfully:")
    print(f"  - Subjects (เคสทั้งหมด): {N}")
    print(f"  - Raw features (แกน XYZ ของทุกจุด): {D} มิติ (มาจาก {num_points} จุดยอด)")

    # 1. ทำการ Centering ข้อมูล (เลื่อนจุดศูนย์กลางเป็น 0)
    mean_vec = np.mean(coords_matrix, axis=0)
    coords_centered = coords_matrix - mean_vec

    # 2. คำนวณ PCA ด้วยวิธี Singular Value Decomposition (SVD)
    # วิธีนี้เหมือนกับการทำ PCA ทุกประการและไม่ต้องลง scikit-learn เพิ่มเติม
    U, S, Vt = np.linalg.svd(coords_centered, full_matrices=False)
    
    # 3. คำนวณความแปรปรวนที่อธิบายได้ (Explained Variance)
    eigenvalues = (S ** 2) / (N - 1)
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cum_variance = np.cumsum(explained_variance_ratio)

    # ค้นหาจำนวน PC ที่ครอบคลุมความแปรปรวน 95% และ 99%
    pcs_95 = np.where(cum_variance >= 0.95)[0][0] + 1
    pcs_99 = np.where(cum_variance >= 0.99)[0][0] + 1 if cum_variance[-1] >= 0.99 else N

    print("\n" + "="*60)
    print("PCA RESULTS SUMMARY:")
    print(f"  - Total possible PCs (มิติสูงสุดที่เป็นไปได้): {len(explained_variance_ratio)} PCs")
    print(f"  - PC1 อธิบายความแปรปรวนรูปร่างได้: {explained_variance_ratio[0]*100:.2f}%")
    if len(explained_variance_ratio) > 1:
        print(f"  - PC2 อธิบายความแปรปรวนรูปร่างได้: {explained_variance_ratio[1]*100:.2f}%")
    if len(explained_variance_ratio) > 2:
        print(f"  - PC3 อธิบายความแปรปรวนรูปร่างได้: {explained_variance_ratio[2]*100:.2f}%")
    print(f"  - จำนวน PC ที่ต้องใช้เพื่อให้ครอบคลุม 95% Variance: {pcs_95} PCs")
    print(f"  - จำนวน PC ที่ต้องใช้เพื่อให้ครอบคลุม 99% Variance: {pcs_99} PCs")
    print("="*60 + "\n")

    # 4. คำนวณ PCA Scores (Projected Coordinates ในพื้นที่มิติต่ำ)
    scores = coords_centered @ Vt.T  # ขนาด N x min(N, D)
    
    # 5. เขียนผลลัพธ์ลง CSV ในโฟลเดอร์เดียวกัน
    output_dir = os.path.dirname(coords_csv)
    pca_csv_path = os.path.join(output_dir, "spharm_pca_coords_features.csv")
    
    num_pcs = scores.shape[1]
    
    out_header = ["Subject", "Group_Name", "Group_Label"]
    for i in range(num_pcs):
        out_header.append(f"PC{i+1}")
        
    with open(pca_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(out_header)
        for i in range(N):
            row = [subjects[i], group_names[i], group_labels[i]]
            row.extend(["{:.8f}".format(val) for val in scores[i]])
            writer.writerow(row)
            
    print(f"Successfully saved PCA features to: {pca_csv_path}")
    print("Done! You can use this CSV file for your ML models.")


if __name__ == "__main__":
    main()
