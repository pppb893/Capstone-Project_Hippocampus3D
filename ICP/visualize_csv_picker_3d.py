import sys
import subprocess
import os
import json
import tkinter as tk
from tkinter import filedialog

# 1. Ensure required dependencies are installed
try:
    import pandas as pd
except ImportError:
    print("Pandas is not installed. Installing pandas...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    import pandas as pd

try:
    import plotly
    import plotly.express as px
    import plotly.io as pio
except ImportError:
    print("Plotly is not installed. Installing plotly...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
    import plotly
    import plotly.express as px
    import plotly.io as pio

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
    print("--- PCA 3D Interactive Visualizer ---")
    
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
        x_col, y_col, z_col = None, None, None
        method_name = ""
        if 'PC1' in df.columns and 'PC2' in df.columns and 'PC3' in df.columns:
            x_col, y_col, z_col = 'PC1', 'PC2', 'PC3'
            method_name = "PCA"
        elif 't-SNE 1' in df.columns and 't-SNE 2' in df.columns and 't-SNE 3' in df.columns:
            x_col, y_col, z_col = 't-SNE 1', 't-SNE 2', 't-SNE 3'
            method_name = "t-SNE"
        elif 'PLS-DA 1' in df.columns and 'PLS-DA 2' in df.columns and 'PLS-DA 3' in df.columns:
            x_col, y_col, z_col = 'PLS-DA 1', 'PLS-DA 2', 'PLS-DA 3'
            method_name = "PLS-DA"
            
        if not x_col or not y_col or not z_col:
            print("Error: Selected CSV must contain columns for PCA ('PC1', 'PC2', 'PC3'), t-SNE ('t-SNE 1', 't-SNE 2', 't-SNE 3'), or PLS-DA ('PLS-DA 1', 'PLS-DA 2', 'PLS-DA 3').")
            print("Please make sure you have run the analysis script to generate 3 components.")
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

    # Set up labels with EVR percentages
    p_evr = evr if evr and len(evr) >= 3 else [0.0, 0.0, 0.0]
    pc1_label = f'{x_col} ({p_evr[0]*100:.1f}%)' if (evr and method_name == "PCA") else x_col
    pc2_label = f'{y_col} ({p_evr[1]*100:.1f}%)' if (evr and method_name == "PCA") else y_col
    pc3_label = f'{z_col} ({p_evr[2]*100:.1f}%)' if (evr and method_name == "PCA") else z_col

    # Classify subjects into 3 categories
    def classify_subject(subject_name):
        is_left_side = subject_name.startswith("left_")
        
        # 1. Healthy Control
        if "_Healthy" in subject_name or "HFH_" in subject_name:
            return "Healthy Control"
        
        # 2. Ipsilateral TLE (Diseased)
        elif (is_left_side and "_Left-TLE" in subject_name) or (not is_left_side and "_Right-TLE" in subject_name):
            return "Ipsilateral TLE (Diseased)"
            
        # 3. Contralateral TLE (Healthy-side)
        elif (is_left_side and "_Right-TLE" in subject_name) or (not is_left_side and "_Left-TLE" in subject_name):
            return "Contralateral TLE (Healthy-side)"
            
        return "Unknown"

    # Map classifications
    df['Group'] = df['Subject'].apply(classify_subject)

    # 5. Create Plotly 3D Figure
    fig = px.scatter_3d(
        df, 
        x=x_col, 
        y=y_col, 
        z=z_col, 
        color='Group',
        color_discrete_map={
            'Healthy Control': 'royalblue',
            'Ipsilateral TLE (Diseased)': 'crimson',
            'Contralateral TLE (Healthy-side)': 'royalblue',
            'Unknown': 'gray'
        },
        hover_name='Subject',
        hover_data={
            x_col: ':.3f',
            y_col: ':.3f',
            z_col: ':.3f',
            'Group': False
        }
    )

    # Custom styling
    fig.update_traces(
        marker=dict(
            size=7, 
            opacity=0.85, 
            line=dict(width=1, color='white')
        )
    )

    # Layout configurations
    fig.update_layout(
        template='plotly_white',
        title=dict(
            text=f"<b>3D Subject Distribution ({x_col} vs {y_col} vs {z_col})</b><br>File: {os.path.basename(csv_path)}",
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=16, color='#2c3e50')
        ),
        scene=dict(
            xaxis=dict(title=dict(text=pc1_label, font=dict(size=11, color='#2c3e50'))),
            yaxis=dict(title=dict(text=pc2_label, font=dict(size=11, color='#2c3e50'))),
            zaxis=dict(title=dict(text=pc3_label, font=dict(size=11, color='#2c3e50')))
        ),
        legend=dict(
            title_text='<b>Subject Group</b>',
            yanchor="top",
            y=0.9,
            xanchor="left",
            x=0.05,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1
        ),
        margin=dict(l=0, r=0, b=0, t=80)
    )

    # Save as HTML and open in the browser
    html_path = os.path.join(output_root, f"{method_name.lower()}_3d_interactive.html")
    pio.write_html(fig, file=html_path, auto_open=True)
    print(f"Interactive 3D plot successfully saved and opened: {html_path}")

if __name__ == "__main__":
    main()
