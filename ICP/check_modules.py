import slicer

LOG_FILE = r"c:\Users\jckky\Desktop\ICP\slicer_batch_log.txt"

def log(msg):
    print(msg)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(msg + "\n")
    except:
        pass

log("--- SlicerSALT 2.2.1 Module Check (v2) ---")
try:
    # In older Slicer, slicer.modules attribute contains the modules
    # We can use dir() to see them
    all_attrs = dir(slicer.modules)
    log(f"Searching {len(all_attrs)} attributes in slicer.modules...")
    
    target_keywords = ["segpost", "genpara", "paratospharm"]
    found = []
    for attr in all_attrs:
        if any(kw in attr.lower() for kw in target_keywords):
            found.append(attr)
            log(f"Found Attribute/Module: {attr}")
            
    if not found:
        log("No matching modules found in dir(slicer.modules).")
        # Alternative attempt
        try:
            from slicer.util import moduleNames
            names = moduleNames()
            log(f"Attempting slicer.util.moduleNames(): {len(names)} found")
            for n in names:
                if any(kw in n.lower() for kw in target_keywords):
                    log(f"Found via moduleNames(): {n}")
        except:
            log("slicer.util.moduleNames() failed.")

except Exception as e:
    log(f"Error checking modules: {str(e)}")
