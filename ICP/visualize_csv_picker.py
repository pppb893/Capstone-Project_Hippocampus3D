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
    try:
        df = pd.read_csv(csv_path)
        if 'PC1' not in df.columns or 'PC2' not in df.columns:
            print("Error: Selected CSV does not have 'PC1' and 'PC2' columns.")
            return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print(f"Loaded {len(df)} subjects.")

    # 4. Load Explained Variance Ratio (EVR) from JSON
    evr = []
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
    fig.canvas.manager.set_window_title(f"PCA Results - {os.path.basename(csv_path)}")

    # Panel A: Scatter Plot (PC1 vs PC2)
    p_evr = evr if evr and len(evr) >= 2 else [0.0, 0.0]
    pc1_label = f'PC1 ({p_evr[0]*100:.1f}%)' if evr else 'PC1'
    pc2_label = f'PC2 ({p_evr[1]*100:.1f}%)' if evr else 'PC2'

    scatter = ax1.scatter(df['PC1'], df['PC2'], c='royalblue', alpha=0.7, edgecolors='w', s=100)
    
    # Add labels to points (optional, but helpful)
    if len(df) < 50: # Only if not too many
        for i, txt in enumerate(df['Subject']):
            ax1.annotate(str(txt), (df['PC1'].iloc[i], df['PC2'].iloc[i]), 
                        xytext=(5,5), textcoords='offset points', fontsize=8, alpha=0.6)

    ax1.set_xlabel(pc1_label, fontsize=12, fontweight='bold')
    ax1.set_ylabel(pc2_label, fontsize=12, fontweight='bold')
    ax1.set_title(f'Subject Distribution\n({os.path.basename(output_root)})', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Panel B: Scree Plot and Cumulative Variance
    if evr:
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
        ax2.text(0.5, 0.5, "No variance data found (pca_model.json)", 
                ha='center', va='center', fontsize=12, color='gray')
        ax2.set_title("Scree Plot (No Data)", fontsize=14)

    plt.tight_layout()
    
    print("Opening plot window...")
    plt.show()

if __name__ == "__main__":
    main()
