import json
import os

json_path = r'c:\Users\jckky\Desktop\ICP\output\pca_results\pca_model.json'

if os.path.exists(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        if 'All' in data:
            data = data['All']
        evr = data.get('explained_variance_ratio', [])
        
        cumulative = 0
        for i, v in enumerate(evr):
            cumulative += v
            print(f"PC{i+1}: {v*100:.2f}% (Cumulative: {cumulative*100:.2f}%)")
            if cumulative >= 0.95:
                print(f"--- 95% threshold reached at PC{i+1} ---")
                break
            if i >= 40: # Limit output
                break
else:
    print("File not found")
