import os
import sys
import argparse
import slicer

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()

def create_template():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="Directory where mean_shape.ply is located")
    args, unknown = parser.parse_known_args()

    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(SCRIPT_DIR, "output")

    input_ply = os.path.join(output_dir, "mean_shape.ply")
    spharm_mean_dir = os.path.join(output_dir, "spharm_mean")
    
    if not os.path.exists(spharm_mean_dir):
        os.makedirs(spharm_mean_dir)

    print(f"Creating SPHARM Template from {input_ply}...")
    
    if not os.path.exists(input_ply):
        print(f"FAILED: Could not find {input_ply}")
        return

    # 1. Load Mean Shape
    mesh_node = slicer.util.loadModel(input_ply)
    if not mesh_node:
        print("Failed to load mean_shape.ply")
        return

    # 2. SPHARM Pipeline (No Registration for template itself)
    # Simple conversion to VTK for standard processing
    temp_vtk = os.path.join(spharm_mean_dir, "mean_result_SPHARM.vtk").replace("\\", "/")
    slicer.util.saveNode(mesh_node, temp_vtk)

    print(f"Template VTK saved to {temp_vtk}")

if __name__ == "__main__":
    create_template()
    slicer.util.exit()

if __name__ == "__main__":
    create_template()
    slicer.util.exit()
