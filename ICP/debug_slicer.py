import os
import slicer

MEAN_PLY = r"c:\Users\jckky\Desktop\ICP\output\mean_shape.ply"
REF_NII = r"c:\Users\jckky\Desktop\ICP\msd-hippocampus\labelsTr\hippocampus_001.nii.gz"
LOG_FILE = r"c:\Users\jckky\Desktop\ICP\debug_slicer_output.txt"

def debug():
    with open(LOG_FILE, "w") as f:
        f.write("--- Slicer Diagnostic ---\n")
        
        # Load PLY
        model = slicer.util.loadModel(MEAN_PLY)
        if model:
            bounds = [0]*6
            model.GetBounds(bounds)
            f.write(f"PLY Mesh Bounds: {bounds}\n")
        else:
            f.write("FAILED to load PLY\n")
            
        # Load NIfTI
        volume = slicer.util.loadLabelVolume(REF_NII)
        if volume:
            bounds = [0]*6
            volume.GetBounds(bounds)
            f.write(f"NII Volume Bounds: {bounds}\n")
            
            # Check Spacing/Origin
            origin = volume.GetOrigin()
            spacing = volume.GetSpacing()
            f.write(f"NII Origin: {origin}\n")
            f.write(f"NII Spacing: {spacing}\n")
        else:
            f.write("FAILED to load NII\n")

    slicer.util.exit()

debug()
