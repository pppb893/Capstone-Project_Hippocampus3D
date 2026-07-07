import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import tkinter as tk
from tkinter import filedialog
import sys

def select_file():
    """Opens a file dialog to select a PCA scores CSV file."""
    root = tk.Tk()
    root.withdraw() # Hide the main window
    root.attributes('-topmost', True) # Bring to front
    
    file_path = filedialog.askopenfilename(
        title="Select PCA Scores CSV File",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        initialdir=os.getcwd()
    )
    root.destroy()
    return file_path

def main():
    print("--- PCA Visualization Picker ---")
    
    # 1. Select File
    csv_path = select_file()
    
    if not csv_path:
        print("No file selected. Exiting.")
        return

    print(f"Loading: {csv_path}")
    
    # 2. Setup Paths
    output_root = os.path.dirname(csv_path)
    json_path = os.path.join(output_root, "pca_model.json")
    
    # 3. Load Data
    # 3. Load Data
    try:
        df = pd.read_csv(csv_path)
        
        # Detect dimensions
        x_col, y_col = None, None
        method_name = ""
        if 'PC1' in df.columns and 'PC2' in df.columns:
            x_col, y_col = 'PC1', 'PC2'
            method_name = "PCA"
        elif 't-SNE 1' in df.columns and 't-SNE 2' in df.columns:
            x_col, y_col = 't-SNE 1', 't-SNE 2'
            method_name = "t-SNE"
        elif 'PLS-DA 1' in df.columns and 'PLS-DA 2' in df.columns:
            x_col, y_col = 'PLS-DA 1', 'PLS-DA 2'
            method_name = "PLS-DA"
            
        if not x_col or not y_col:
            print("Error: Selected CSV must contain columns for PCA ('PC1', 'PC2'), t-SNE ('t-SNE 1', 't-SNE 2'), or PLS-DA ('PLS-DA 1', 'PLS-DA 2').")
            return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print(f"Loaded {len(df)} subjects.")

    # 4. Load Explained Variance Ratio (EVR) from JSON
    evr = []
    if method_name == "PCA":
        try:
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    # Handle nested structures commonly found in these outputs
                    if '0' in data: data = data['0']
                    elif 'All' in data: data = data['All']
                    evr = data.get('explained_variance_ratio', [])
        except Exception as e:
            print(f"Note: Could not load variance ratios from {json_path}: {e}")

    # 5. Create Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Set window title
    fig.canvas.manager.set_window_title(f"{method_name} Results - {os.path.basename(csv_path)}")

    # Panel A: Scatter Plot
    p_evr = evr if evr and len(evr) >= 2 else [0.0, 0.0]
    pc1_label = f'{x_col} ({p_evr[0]*100:.1f}%)' if (evr and method_name == "PCA") else x_col
    pc2_label = f'{y_col} ({p_evr[1]*100:.1f}%)' if (evr and method_name == "PCA") else y_col

    # Classify subjects into 3 categories
    def classify_subject(subject_name):
        is_left_side = subject_name.startswith("left_")
        
        # 1. Healthy Control
        if "_Healthy" in subject_name or "HFH_" in subject_name:
            return "Healthy Control", "royalblue"
        
        # 2. Ipsilateral TLE (Diseased)
        elif (is_left_side and "_Left-TLE" in subject_name) or (not is_left_side and "_Right-TLE" in subject_name):
            return "Ipsilateral TLE (Diseased)", "crimson"
            
        # 3. Contralateral TLE (Healthy-side)
        elif (is_left_side and "_Right-TLE" in subject_name) or (not is_left_side and "_Left-TLE" in subject_name):
            return "Contralateral TLE (Healthy-side)", "royalblue"
            
        return "Unknown", "gray"

    # Map classifications
    classes_and_colors = df['Subject'].apply(classify_subject)
    df['Group'] = [c[0] for c in classes_and_colors]
    df['Color'] = [c[1] for c in classes_and_colors]

    # Plot each group separately to get clear labels in the legend
    groups = df.groupby(['Group', 'Color'])
    has_groups = False
    for (group_name, color), group_df in groups:
        ax1.scatter(group_df[x_col], group_df[y_col], c=color, alpha=0.7, edgecolors='w', s=100, label=group_name)
        has_groups = True
        
    if has_groups:
        ax1.legend(loc='best', fontsize=10)
    
    ax1.set_xlabel(pc1_label, fontsize=12, fontweight='bold')
    ax1.set_ylabel(pc2_label, fontsize=12, fontweight='bold')
    ax1.set_title(f'{method_name} Distribution\n({os.path.basename(output_root)})', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Panel B: Scree Plot and Cumulative Variance
    if evr and method_name == "PCA":
        import numpy as np
        
        # Dynamically determine how many PCs to show:
        # Show until cumulative variance hits 96% plus a small buffer, up to max length
        cum_full = np.cumsum(evr)
        idx = np.where(cum_full >= 0.96)[0]
        
        if len(idx) > 0:
            limit = min(len(evr), idx[0] + 3) # Target + buffer
        else:
            limit = min(len(evr), 50) # If it struggles to hit 96%, show up to 50
            
        limit = max(10, limit) # Always show at least 10 PCs
        
        pcs = range(1, limit + 1)
        cum_evr = cum_full[:limit]
        
        ax2.bar(pcs, evr[:limit], color='lightseagreen', alpha=0.6, label='Individual')
        
        # Cumulative line on right axis
        ax2_twin = ax2.twinx()
        ax2_twin.plot(pcs, cum_evr, color='crimson', marker='o', linewidth=2, markersize=4, label='Cumulative')
        ax2_twin.set_ylabel('Cumulative Variance', fontsize=12, fontweight='bold', color='crimson')
        ax2_twin.tick_params(axis='y', labelcolor='crimson')
        
        # Threshold lines
        ax2_twin.axhline(y=0.67, color='orange', linestyle='--', alpha=0.8)
        ax2_twin.text(1, 0.67+0.01, '67%', color='orange', fontweight='bold')
        ax2_twin.axhline(y=0.95, color='green', linestyle='--', alpha=0.8)
        ax2_twin.text(1, 0.95+0.01, '95%', color='green', fontweight='bold')

        ax2.set_xlabel('Principal Components', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Explained Variance Ratio', fontsize=12, fontweight='bold')
        ax2.set_title('Scree Plot & Cumulative Variance', fontsize=14)
        ax2.set_xticks(range(1, limit + 1, max(1, limit//10)))
        ax2.grid(True, axis='x', linestyle='--', alpha=0.5)
        
        lines_1, labels_1 = ax2.get_legend_handles_labels()
        lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
    else:
        if method_name != "PCA":
            ax2.text(0.5, 0.5, f"Scree plot is not applicable\nfor supervised/non-linear {method_name}.", 
                    ha='center', va='center', fontsize=12, color='gray')
            ax2.set_title(f"Scree Plot ({method_name})", fontsize=14)
        else:
            ax2.text(0.5, 0.5, "No variance data found (pca_model.json)", 
                    ha='center', va='center', fontsize=12, color='gray')
            ax2.set_title("Scree Plot (No Data)", fontsize=14)

    plt.tight_layout()
    
    print("Opening plot window...")
    plt.show()

if __name__ == "__main__":
    main()
