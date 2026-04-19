import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import argparse

# Path Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="Directory where pca_results are located")
    args, unknown = parser.parse_known_args()

    if args.output_dir:
        output_root = os.path.abspath(args.output_dir)
    else:
        output_root = os.path.join(SCRIPT_DIR, "output")

    csv_path = os.path.join(output_root, "pca_results", "pca_scores.csv")
    json_path = os.path.join(output_root, "pca_results", "pca_model.json")
    output_plot = os.path.join(output_root, "pca_results", "pca_visualization.png")

    print(f"--- Starting PCA Visualization for {os.path.basename(output_root)} ---")

    # 1. Load Scores
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} subjects.")

    # 2. Get Explained Variance from JSON if possible
    evr = []
    try:
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                # Handle nested structure
                if '0' in data: data = data['0']
                elif 'All' in data: data = data['All']
                evr = data.get('explained_variance_ratio', [])
    except Exception as e:
        print(f"Could not load variance ratios: {e}")

    # 3. Create Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: Scatter Plot (PC1 vs PC2)
    p_evr = evr if evr and len(evr) >= 2 else [0.0, 0.0]
    pc1_label = f'PC1 ({p_evr[0]*100:.1f}%)' if evr else 'PC1'
    pc2_label = f'PC2 ({p_evr[1]*100:.1f}%)' if evr else 'PC2'

    scatter = ax1.scatter(df['PC1'], df['PC2'], c='royalblue', alpha=0.6, edgecolors='w', s=100)
    ax1.set_xlabel(pc1_label, fontsize=12, fontweight='bold')
    ax1.set_ylabel(pc2_label, fontsize=12, fontweight='bold')
    ax1.set_title(f'Distribution: {os.path.basename(output_root)}', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Panel B: Scree Plot and Cumulative Variance
    if evr:
        import numpy as np
        
        # Dynamically determine how many PCs to show:
        cum_full = np.cumsum(evr)
        idx = np.where(cum_full >= 0.96)[0]
        
        if len(idx) > 0:
            limit = min(len(evr), idx[0] + 3) # Target + buffer
        else:
            limit = min(len(evr), 50) # If it struggles to hit 96%, show up to 50
            
        limit = max(10, limit)
        pcs = range(1, limit + 1)
        cum_evr = cum_full[:limit]
        
        # Plot individual variance (bar)
        ax2.bar(pcs, evr[:limit], color='lightseagreen', alpha=0.6, label='Individual')
        
        # Plot cumulative variance (line)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(pcs, cum_evr, color='crimson', marker='o', linewidth=2, markersize=4, label='Cumulative')
        ax2_twin.set_ylabel('Cumulative Variance', fontsize=12, fontweight='bold', color='crimson')
        ax2_twin.tick_params(axis='y', labelcolor='crimson')
        
        # Highlight thresholds (e.g. 67%, 95%)
        ax2_twin.axhline(y=0.67, color='orange', linestyle='--', alpha=0.8)
        ax2_twin.text(1, 0.67+0.01, '67%', color='orange', fontweight='bold')
        ax2_twin.axhline(y=0.95, color='green', linestyle='--', alpha=0.8)
        ax2_twin.text(1, 0.95+0.01, '95%', color='green', fontweight='bold')

        ax2.set_xlabel('Principal Components', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Explained Variance Ratio', fontsize=12, fontweight='bold')
        ax2.set_title('Scree Plot & Cumulative Variance', fontsize=14)
        ax2.set_xticks(range(1, limit + 1, max(1, limit//10)))
        ax2.grid(True, axis='x', linestyle='--', alpha=0.5)
        
        # Combine legends
        lines_1, labels_1 = ax2.get_legend_handles_labels()
        lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"Visualization saved to: {output_plot}")
    
    # Try to show if not in headless environment
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    main()
