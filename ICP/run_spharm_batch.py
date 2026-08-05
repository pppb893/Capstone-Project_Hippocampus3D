import os
import glob
import slicer
import time
import sys
import argparse
import subprocess
from datetime import datetime

# --- [FIX] หา SCRIPT_DIR ที่ถูกต้องใน SlicerSALT ---
# __file__ มักไม่ถูก define ใน SlicerSALT ใช้ sys.argv แทน
def get_script_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        for arg in sys.argv:
            if arg.endswith(".py") and os.path.isfile(arg):
                return os.path.dirname(os.path.abspath(arg))
        return os.getcwd()

SCRIPT_DIR = get_script_dir()

# --- Utility Functions ---
def sprint(msg, log_file):
    print(msg)
    sys.stdout.flush()
    timestamp = datetime.now().strftime("%H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] {msg}\n")


def cleanup_subject_files(output_base_dir, basename):
    """
    ลบไฟล์ output ทั้งหมดของ subject ก่อนรันใหม่ — กัน stale files จาก run
    ก่อนหน้าทำให้ false-SUCCESS (เช็คไฟล์มีอยู่อย่างเดียวเลยหลอก)
    """
    patterns = [
        f"{basename}_pp.*",
        f"{basename}_para.vtk", f"{basename}_paraMix.txt",
        f"{basename}_paraPhi.txt", f"{basename}_paraPhiHalf.txt",
        f"{basename}_paraTheta.txt",
        f"{basename}_surf.vtk",
        f"{basename}_SPHARM.vtk", f"{basename}_SPHARM.coef",
        f"{basename}_SPHARM_ellalign.vtk", f"{basename}_SPHARM_ellalign.coef",
        f"{basename}_SPHARM_procalign.vtk", f"{basename}_SPHARM_procalign.coef",
        f"{basename}_SPHARM_grid.vtk",
        f"{basename}_SPHARMMedialAxis.vtk",
        f"{basename}_MedialAxisScalars.csv",
    ]
    removed = 0
    for pat in patterns:
        for f in glob.glob(os.path.join(output_base_dir, pat)):
            try:
                os.remove(f)
                removed += 1
            except OSError:
                pass
    return removed


def run_cli_checked(module, params, step_name, log_file):
    """
    รัน CLI module แล้วตรวจ status จริง ๆ
    return True ถ้า status == 'Completed' หรือ 'Completed with errors', False ถ้าไม่ใช่
    """
    cli_node = slicer.cli.run(module, None, params, wait_for_completion=True)
    status = cli_node.GetStatusString()
    if status not in ("Completed", "Completed with errors"):
        sprint(f"    !!! CLI '{step_name}' status = {status}", log_file)
        err = cli_node.GetErrorText() or ""
        out = cli_node.GetOutputText() or ""
        if err.strip():
            sprint(f"    stderr: {err.strip()[:1000]}", log_file)
        if out.strip():
            sprint(f"    stdout: {out.strip()[:1000]}", log_file)
        return False
    return True


def node_has_points(node, min_points=10):
    """ตรวจว่า vtkMRMLModelNode มี polydata + จำนวน point > min_points"""
    if node is None:
        return False
    pd = node.GetPolyData()
    if pd is None:
        return False
    return pd.GetNumberOfPoints() >= min_points


def improve_label_quality(input_node, log_file):
    """
    ปรับปรุงคุณภาพ binary label ก่อนเข้า SPHARM:
      1. binary_fill_holes  -> อุดรูใน (interior cavities)
      2. binary_closing 1x  -> เก็บผิวเรียบ ปิด gap เล็ก ๆ
      3. keep largest connected component -> กัน noise/floating bits
    SPHARM ต้องการ topology ที่เป็น "ลูกบอลทึบ" (genus 0) -> รูเล็ก ๆ
    ทำให้ parameterization fail หรือ mesh ขรุขระ
    """
    try:
        from scipy import ndimage
        import numpy as _np
    except ImportError:
        sprint("    (skipping hole-fill: scipy not available in this Python)", log_file)
        return

    arr = slicer.util.arrayFromVolume(input_node)
    if arr is None or arr.size == 0:
        return
    binary = (arr > 0)
    before = int(binary.sum())
    if before == 0:
        return

    # 1) fill interior holes
    filled = ndimage.binary_fill_holes(binary)

    # 2) morphological closing (dilate then erode) — smooth + close gaps
    struct = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    closed = ndimage.binary_closing(filled, structure=struct, iterations=1)

    # 3) keep only the largest connected component
    labeled, n_comp = ndimage.label(closed, structure=struct)
    if n_comp > 1:
        sizes = ndimage.sum(closed, labeled, range(1, n_comp + 1))
        largest = int(_np.argmax(sizes)) + 1
        cleaned = (labeled == largest)
        removed_comps = n_comp - 1
    else:
        cleaned = closed
        removed_comps = 0

    after = int(cleaned.sum())
    new_arr = cleaned.astype(_np.uint8)
    slicer.util.updateVolumeFromArray(input_node, new_arr)

    delta = after - before
    if delta != 0 or removed_comps > 0:
        sprint(f"    Quality fix: {before} -> {after} voxels (delta={delta:+d})  "
               f"components removed: {removed_comps}", log_file)

def run_batch_spharm():
    # --- Configuration via Argparse ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Directory containing .nii.gz labels")
    parser.add_argument("--output_dir", type=str, help="Directory to save all results")
    parser.add_argument("--fast", action="store_true",
                        help="Fast/test mode: ~4-5x faster but lower mesh resolution")
    # --- [DEEP FIX] template-based alignment options ---
    # ทำไม: SPHARM-PDM ellalign โดยไม่มี template ให้ sign-flip ambiguity ต่อ subject
    #       -> vertex correspondences “แตก” ระหว่าง subjects (PC1 ใน PCA ระเบิด)
    # การแก้: ส่ง _SPHARM.vtk ของ reference subject เป็น regTemplate + flipTemplate
    #         ทำให้ SPHARM-PDM align ทุก subject กับ reference เดียวกัน -> correspondences คงที่
    parser.add_argument("--reference_template", type=str, default=None,
                        help="Path to reference _SPHARM.vtk used as regTemplate/flipTemplate. "
                             "If omitted, auto-picks the first subject's existing _SPHARM.vtk.")
    parser.add_argument("--regenerate_spharm_only", action="store_true",
                        help="Skip SegPostProcess and GenParaMesh — re-run only ParaToSPHARMMesh "
                             "from existing _para.vtk and _surf.vtk files. Much faster (~5x).")
    args, unknown = parser.parse_known_args()

    # SPHARM params: fast vs production
    if args.fast:
        NUM_ITER = 200
        SUBDIV   = 5      # -> 252 points (vs 1002)
        DEGREE   = 6      # SH coefficients
        MODE_TAG = "FAST"
    else:
        NUM_ITER = 1000
        SUBDIV   = 10     # -> 1002 points
        DEGREE   = 12
        MODE_TAG = "PRODUCTION"

    if args.output_dir:
        output_root = os.path.abspath(args.output_dir)
    else:
        output_root = os.path.join(SCRIPT_DIR, "output")

    if args.input_dir:
        input_dir = os.path.abspath(args.input_dir)
    else:
        input_dir = os.path.join(output_root, "aligned_nifti")

    output_base_dir = os.path.join(output_root, "spharm_results")
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    log_file = os.path.join(output_base_dir, "spharm_progress.log")

    with open(log_file, 'w') as f:
        f.write(f"--- SPHARM Batch Mode Start: {datetime.now()} ---\n")
        f.write(f"Mode: {MODE_TAG} (iter={NUM_ITER}, subdiv={SUBDIV}, degree={DEGREE})\n")
        f.write(f"Input: {input_dir}\n")
        f.write(f"Output: {output_root}\n")
        f.write(f"SCRIPT_DIR: {SCRIPT_DIR}\n\n")  # log เพื่อ debug

    sprint(f"SCRIPT_DIR resolved to: {SCRIPT_DIR}", log_file)

    mean_vtk = os.path.join(output_root, "spharm_mean", "mean_result_SPHARM.vtk").replace("\\", "/")

    extensions = ["*.nii.gz", "*.nii", "*.hdr"]
    file_list = []
    for ext in extensions:
        file_list.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))

    file_list = sorted(list(set(file_list)))

    label_files = [f for f in file_list if "label" in os.path.basename(f).lower()]
    if label_files:
        sprint(f"Detected {len(label_files)} label-specific files (filtering out original images).", log_file)
        file_list = label_files

    sprint(f"Verified {len(file_list)} files for processing.", log_file)

    # ---- Determine reference template for sign-flip/registration alignment ----
    # ใช้ existing _SPHARM_ellalign.vtk ของ subject แรกเป็น template
    # (ellalign = principal-axis-aligned orientation -> semantic ของ canonical frame)
    # ทุก subject จะถูก align กับ template เดียวกัน -> vertex correspondences consistent ทั้ง batch
    reference_template = args.reference_template
    template_basename = None
    if not reference_template and file_list:
        first_basename = os.path.basename(file_list[0]).split('.')[0]
        # ลำดับ preference: ellalign (semantic) -> raw SPHARM (fallback)
        for suffix in ("_SPHARM_ellalign.vtk", "_SPHARM.vtk"):
            candidate = os.path.join(output_base_dir,
                                     f"{first_basename}{suffix}").replace("\\", "/")
            if os.path.exists(candidate):
                reference_template = candidate
                template_basename = first_basename
                sprint(f"\nAuto-picked reference template: {os.path.basename(reference_template)}",
                       log_file)
                break
        if not reference_template:
            sprint(f"\nNo existing SPHARM output for first subject — will process it without "
                   f"template first, then use its output as template for rest.", log_file)

    if reference_template:
        reference_template = reference_template.replace("\\", "/")
        # extract basename ของ template เพื่อ skip self-reference
        if template_basename is None:
            tmpl_fname = os.path.basename(reference_template)
            for suf in ("_SPHARM_ellalign.vtk", "_SPHARM.vtk"):
                if tmpl_fname.endswith(suf):
                    template_basename = tmpl_fname[: -len(suf)]
                    break
        sprint(f"Using regTemplate + flipTemplate = {reference_template}", log_file)
        sprint(f"Template subject basename: {template_basename}", log_file)
    if args.regenerate_spharm_only:
        sprint(f"Mode: REGENERATE-ONLY (skip SegPostProcess + GenParaMesh)", log_file)

    for i, file_path in enumerate(file_list):
        file_path = os.path.normpath(file_path).replace("\\", "/")
        basename = os.path.basename(file_path).split('.')[0]
        sprint(f"\n>>> [{i+1}/{len(file_list)}] STARTING: {basename}", log_file)

        pp_mask_path = os.path.join(output_base_dir, f"{basename}_pp.nrrd").replace("\\", "/")
        para_mesh_path = os.path.join(output_base_dir, f"{basename}_para.vtk").replace("\\", "/")
        surf_mesh_path = os.path.join(output_base_dir, f"{basename}_surf.vtk").replace("\\", "/")
        spharm_base = os.path.join(output_base_dir, basename).replace("\\", "/")

        # ใน regenerate-only mode: ใช้ _para.vtk + _surf.vtk เดิม -> skip pp/genpara
        # ถ้าไฟล์ไม่ครบจะ fall back ไป full pipeline สำหรับ subject ตัวนั้น
        regen_mode = (args.regenerate_spharm_only
                      and os.path.exists(para_mesh_path)
                      and os.path.exists(surf_mesh_path))

        # *** ลบ stale files (เฉพาะ SPHARM step) — กัน false-SUCCESS จาก run เก่า ***
        # ใน regen mode ต้องไม่ลบ _para.vtk / _surf.vtk ที่จะเอามาใช้
        if not regen_mode:
            n_removed = cleanup_subject_files(output_base_dir, basename)
            if n_removed > 0:
                sprint(f"  - Cleaned {n_removed} stale files from previous run", log_file)
        else:
            # ลบเฉพาะ SPHARM output ไม่ลบ pp/para/surf
            for pat in [f"{basename}_SPHARM*.*",
                        f"{basename}_SPHARMMedialAxis.vtk",
                        f"{basename}_MedialAxisScalars.csv"]:
                for f in glob.glob(os.path.join(output_base_dir, pat)):
                    try: os.remove(f)
                    except OSError: pass

        input_node = None
        pp_node = None
        para_node = None
        surf_node = None

        try:
            if regen_mode:
                sprint(f"  - REGEN MODE: reusing _para.vtk + _surf.vtk", log_file)
                para_node = slicer.util.loadModel(para_mesh_path)
                surf_node = slicer.util.loadModel(surf_mesh_path)
                if not node_has_points(para_node) or not node_has_points(surf_node):
                    sprint(f"  - !!! Failed to load _para/_surf — skipping", log_file)
                    continue
            else:
                sprint(f"  - Loading volume...", log_file)
                input_node = slicer.util.loadLabelVolume(file_path)
                if not input_node:
                    sprint(f"  - FAILED to load: {file_path}", log_file)
                    continue

                sprint(f"  - STEP 0: Improving label quality (fill holes + close + largest comp)...",
                       log_file)
                improve_label_quality(input_node, log_file)

                sprint(f"  - STEP 1: SegPostProcess (Cleaning labels)...", log_file)
                pp_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode",
                                                              f"{basename}_pp")
                pp_params = {
                    'fileName': input_node.GetID(),
                    'outfileName': pp_node.GetID(),
                    'label': 1
                }
                if not run_cli_checked(slicer.modules.segpostprocessclp, pp_params,
                                       "SegPostProcess", log_file):
                    continue
                slicer.util.saveNode(pp_node, pp_mask_path)

                sprint(f"  - STEP 2: GenParaMesh (Creating spherical mesh)...", log_file)
                para_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode",
                                                                f"{basename}_para")
                surf_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode",
                                                                f"{basename}_surf")
                para_params = {
                    'infile': pp_node.GetID(),
                    'outParaName': para_node.GetID(),
                    'outSurfName': surf_node.GetID(),
                    'numIterations': NUM_ITER,
                    'label': 1
                }
                if not run_cli_checked(slicer.modules.genparameshclp, para_params,
                                       "GenParaMesh", log_file):
                    continue
                if not node_has_points(para_node) or not node_has_points(surf_node):
                    sprint(f"  - !!! GenParaMesh returned EMPTY mesh — skipping", log_file)
                    continue
                slicer.util.saveNode(para_node, para_mesh_path)
                slicer.util.saveNode(surf_node, surf_mesh_path)

            sprint(f"  - STEP 3: ParaToSPHARMMesh (Computing PDM)...", log_file)
            # [DEEP FIX] ใส่ regTemplate + flipTemplate -> SPHARM-PDM align ทุก subject
            # กับ reference เดียวกัน -> vertex correspondences consistent ทั้ง batch
            #   regTemplate  = rigid registration target (rotation alignment)
            #   flipTemplate = sign-flip ambiguity resolution target
            # การไม่มี template ทำให้ ellalign ของแต่ละ subject สุ่มทิศ -> PCA PC1 ระเบิด
            spharm_params = {
                'inParaFile': para_node.GetID(),
                'inSurfFile': surf_node.GetID(),
                'outbase': spharm_base,
                'subdivLevel': SUBDIV,
                'spharmDegree': DEGREE
            }
            # ใช้ template ถ้ามี — แต่ skip กรณี subject นี้คือ template เอง (กัน self-reference)
            own_spharm_path = f"{spharm_base}_SPHARM.vtk"
            own_ellalign_path = f"{spharm_base}_SPHARM_ellalign.vtk"
            is_template_subject = (template_basename is not None
                                    and basename == template_basename)
            if reference_template and not is_template_subject:
                # [DEEP FIX] Slicer's CLI interface expects parameter names with On flags to trigger template alignment
                spharm_params['regTemplateFile'] = reference_template
                spharm_params['regTemplateFileOn'] = True
                spharm_params['flipTemplateFile'] = reference_template.replace(".vtk", ".coef")
                spharm_params['flipTemplateFileOn'] = True
                
                # Also pass short flags / long flags for safety
                spharm_params['regTemplate'] = reference_template
                spharm_params['flipTemplate'] = reference_template.replace(".vtk", ".coef")
                spharm_params['flipTemplateOn'] = True
                sprint(f"    Using template alignment -> will produce _SPHARM_procalign.vtk",
                       log_file)
            elif is_template_subject:
                sprint(f"    (Template subject — no self-reference)", log_file)

            run_cli_checked(slicer.modules.paratospharmmeshclp, spharm_params,
                            "ParaToSPHARMMesh", log_file)
            
            # Check actual output files instead of relying strictly on CLI status
            # (SPHARM CLI often writes warnings to stderr, which Slicer marks as Completed with errors)
            if not (os.path.exists(own_spharm_path) and os.path.exists(own_spharm_path.replace(".vtk", ".coef"))):
                sprint(f"  - ERROR: SPHARM outputs not found for {basename}", log_file)
                cleanup_subject_files(output_base_dir, basename)
                continue

            # Verify VTK validity (check for 0 cells or NaN corrupted files)
            try:
                reader = vtk.vtkPolyDataReader()
                reader.SetFileName(own_spharm_path)
                reader.Update()
                poly = reader.GetOutput()
                if poly is None or poly.GetNumberOfPoints() < 10 or poly.GetNumberOfCells() == 0:
                    sprint(f"  - ERROR: Invalid/Corrupted SPHARM output (0 cells / NaN) for {basename}", log_file)
                    cleanup_subject_files(output_base_dir, basename)
                    continue
            except Exception:
                pass

            # ถ้าเป็น template subject -> copy ellalign เป็น procalign เพื่อให้ pipeline uniform
            # (subjects อื่น จะมี _procalign.vtk สำหรับ subject นี้ก็ต้องมีด้วย ใช้ ellalign ของตัวเอง)
            if is_template_subject and os.path.exists(own_ellalign_path):
                own_procalign_path = f"{spharm_base}_SPHARM_procalign.vtk"
                import shutil
                try:
                    shutil.copy2(own_ellalign_path, own_procalign_path)
                    sprint(f"    Copied template's ellalign -> procalign for pipeline uniformity",
                           log_file)
                except Exception as e:
                    sprint(f"    WARN: failed to copy ellalign->procalign: {e}", log_file)

            # ถ้า reference_template ยังไม่ได้ตั้งตอนเริ่ม (เพราะไม่มีไฟล์เก่า)
            # ใช้ output ของ subject แรกเป็น template สำหรับที่เหลือ
            if not reference_template and os.path.exists(own_ellalign_path):
                reference_template = own_ellalign_path
                template_basename = basename
                sprint(f"    Set as reference template for remaining subjects: "
                       f"{os.path.basename(own_ellalign_path)}", log_file)

            final_vtk = f"{spharm_base}_SPHARM.vtk"
            final_coef = f"{spharm_base}_SPHARM.coef"
            grid_vtk = f"{spharm_base}_SPHARM_grid.vtk"

            if os.path.exists(final_vtk) and os.path.exists(final_coef):
                theta_step = "9"
                phi_step = "9"
                sprint(f"  - SUCCESS: {basename} completed. Resampling to {theta_step}x{phi_step} degree grid...", log_file)

                # [FIX] ใช้ sys.executable ของ Slicer (ถูกต้องสำหรับ script ที่รันใน Slicer env)
                # ถ้า resample_spharm_grid.py ต้องการ system Python ให้เปลี่ยนเป็น "python" หรือ "python3"
                resample_script = os.path.join(SCRIPT_DIR, "resample_spharm_grid.py")

                if not os.path.isfile(resample_script):
                    sprint(f"  - WARNING: resample_spharm_grid.py not found at {resample_script}", log_file)
                else:
                    cmd = [sys.executable, resample_script, final_coef, grid_vtk, theta_step, phi_step]
                    sprint(f"  - Running: {' '.join(cmd)}", log_file)

                    # [FIX] check=True + capture stderr เพื่อไม่ให้ error หายเงียบๆ
                    # [FIX] ใช้ creationflags=0x08000000 (CREATE_NO_WINDOW) ป้องกันหน้าต่างดำเด้งขึ้นมากวนตอนรันบน Windows
                    try:
                        kwargs = {}
                        if os.name == 'nt':
                            kwargs['creationflags'] = 0x08000000
                        result = subprocess.run(
                            cmd,
                            check=True,
                            capture_output=True,
                            text=True,
                            **kwargs
                        )
                        if result.stdout:
                            sprint(f"  - Resample stdout: {result.stdout.strip()}", log_file)
                    except subprocess.CalledProcessError as sub_err:
                        sprint(f"  - ERROR: resample_spharm_grid.py failed (exit {sub_err.returncode})", log_file)
                        sprint(f"  - stderr: {sub_err.stderr.strip()}", log_file)

                    if os.path.exists(grid_vtk):
                        sprint(f"  - SUCCESS: {basename} Grid VTK created.", log_file)
                    else:
                        sprint(f"  - WARNING: Grid resampling failed for {basename}.", log_file)
            else:
                sprint(f"  - ERROR: Result VTK not generated for {basename}.", log_file)
                sprint(f"    Expected: {final_vtk}", log_file)
                sprint(f"    Expected: {final_coef}", log_file)

        except Exception as e:
            import traceback
            sprint(f"  - CRITICAL ERROR for {basename}: {str(e)}", log_file)
            sprint(traceback.format_exc(), log_file)

        finally:
            for node in [input_node, pp_node, para_node, surf_node]:
                if node:
                    try:
                        slicer.mrmlScene.RemoveNode(node)
                    except Exception:
                        pass
            sprint(f"  - Cleanup done.", log_file)

    sprint("\n" + "="*60, log_file)
    sprint("!!! ALL BATCH PROCESSING COMPLETED !!!", log_file)
    sprint("="*60, log_file)

if __name__ == "__main__":
    # ทำไม try/finally + os._exit:
    #   --no-main-window mode ของ SlicerSALT อาจ "ไม่ตาย" หลัง python-script จบ
    #     (Qt event loop ยัง active, GUI thread ยัง alive แต่ไม่มี main window)
    #     -> batch script ค้างรอ Slicer process exit -> user เห็นเป็นเหมือน "เด้ง"
    #   slicer.util.exit() ใช้ Qt event -> ถ้า loop ไม่ run จะไม่ trigger
    #   slicer.app.quit() ก็ขึ้นกับ event loop
    #   os._exit() = bypass ทุก cleanup, kill process ทันที — รับประกัน batch ไปต่อได้
    # ก่อน os._exit ต้อง flush stdout/stderr ให้ log ถูกเขียนจบ
    exit_code = 0
    try:
        run_batch_spharm()
    except Exception as _e:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        # ลอง slicer's exit ก่อน (เผื่อ Qt loop run อยู่)
        try:
            slicer.util.exit(exit_code)
        except Exception:
            pass
        # ถ้ายังไม่ตายภายใน ~1 วินาที -> kill process
        try:
            import time, threading
            def force_kill():
                time.sleep(1.5)
                os._exit(exit_code)
            t = threading.Thread(target=force_kill, daemon=True)
            t.start()
        except Exception:
            os._exit(exit_code)