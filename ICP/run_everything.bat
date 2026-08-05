@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM กัน window ปิดเอง — แทนที่ `exit /b N` ใช้ `goto :end_pipeline`
REM   ทุก fail path ลงท้ายที่ :end_pipeline ซึ่งมี pause ค้างไว้
REM   double-click ก็จะค้าง pause ทุกกรณี ไม่ต้องสร้าง window ใหม่ผ่าน cmd /k
REM ============================================================

set "PIPELINE_DIR=%~dp0"
if "%PIPELINE_DIR:~-1%"=="\" set "PIPELINE_DIR=%PIPELINE_DIR:~0,-1%"

set "SLICER_EXE=C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe"

REM ============================================================
REM Pick a Python interpreter that has vtk + numpy + scipy +
REM pandas + matplotlib. Use goto-based flow to avoid the
REM "nested if + delayed expansion" bugs that bite parens-only forms.
REM ============================================================
set "USER_PYTHON="
set "PYTEST=import vtk, numpy, scipy, pandas, matplotlib"

REM --- (1) try venv ---
if not exist "%PIPELINE_DIR%\venv\Scripts\python.exe" goto try_system
"%PIPELINE_DIR%\venv\Scripts\python.exe" -c "%PYTEST%" >nul 2>&1
if errorlevel 1 goto try_system
set "USER_PYTHON=%PIPELINE_DIR%\venv\Scripts\python.exe"
goto python_found

:try_system
python -c "%PYTEST%" >nul 2>&1
if errorlevel 1 goto try_py
set "USER_PYTHON=python"
goto python_found

:try_py
py -c "%PYTEST%" >nul 2>&1
if errorlevel 1 goto no_python
set "USER_PYTHON=py"
goto python_found

:no_python
echo.
echo [ERROR] No Python with required modules found.
echo         Required: vtk, numpy, scipy, pandas, matplotlib
echo         Install:  pip install vtk numpy scipy pandas matplotlib
echo.
pause
exit /b 1

:python_found
echo Using Python: %USER_PYTHON%
echo.

echo ============================================================
echo   THE ULTIMATE SHAPE ANALYSIS PIPELINE (DYNAMIC DATASET)
echo ============================================================
echo.

echo [0/5] SELECTING DATASET...
set "PS_CMD=Add-Type -AssemblyName System.Windows.Forms; $f = New-Object System.Windows.Forms.FolderBrowserDialog; $f.Description = 'Select folder'; $f.ShowDialog() | Out-Null; $f.SelectedPath"

for /f "delims=" %%I in ('powershell -Command "%PS_CMD%"') do set "INPUT_DIR=%%I"

if not defined INPUT_DIR (
    echo.
    echo [CANCELLED] No folder selected. Exiting...
    pause
    exit /b 1
)
if "%INPUT_DIR%"=="" (
    echo.
    echo [CANCELLED] No folder selected. Exiting...
    pause
    exit /b 1
)

for %%I in ("%INPUT_DIR%") do set "FOLDER_NAME=%%~nxI"
set "OUTPUT_DIR=%PIPELINE_DIR%\output_%FOLDER_NAME%"

echo.
echo SELECTED INPUT:  %INPUT_DIR%
echo TARGET OUTPUT:   %OUTPUT_DIR%
echo SCRIPT DIR:      %PIPELINE_DIR%
echo ============================================================
echo.

if exist "%OUTPUT_DIR%" (
    echo [WARNING] Output folder already exists: %OUTPUT_DIR%
    echo           Files with the same name will be overwritten!
    echo.
    set /p CONFIRM="Continue anyway? (Y/N): "
    if /i "!CONFIRM!" NEQ "Y" (
        echo.
        echo [CANCELLED] User cancelled. Exiting...
        pause
        exit /b 1
    )
    if exist "%OUTPUT_DIR%\spharm_results" (
        echo [INFO] Cleaning up old SPHARM results to prevent duplicate/stale files...
        rmdir /s /q "%OUTPUT_DIR%\spharm_results"
    )
    echo.
)

if not exist "%SLICER_EXE%" (
    echo [ERROR] SlicerSALT not found at: %SLICER_EXE%
    echo         Please edit SLICER_EXE in this bat file.
    pause
    exit /b 1
)

REM ============================================================
REM IMPORTANT: SlicerSALT มักคืน exit code != 0 แม้ทำงานสำเร็จ
REM   -> เช็ค output file/folder แทน ERRORLEVEL หลัง Slicer step
REM Python step เช็ค ERRORLEVEL ปกติ
REM ============================================================

title Pipeline [1/5]: ICP Alignment (%FOLDER_NAME%)
echo [1/5] Running Group-wise ICP Alignment (ICP.py)...
"%SLICER_EXE%" --no-main-window --no-splash --python-script "%PIPELINE_DIR%\ICP.py" --input_dir "%INPUT_DIR%" --output_dir "%OUTPUT_DIR%"
if not exist "%OUTPUT_DIR%\aligned_nifti" (
    echo.
    echo [ERROR] Step 1 failed - 'aligned_nifti' folder not created.
    echo         Check icp_debug_log.txt for details.
    pause
    exit /b 1
)
echo [OK] aligned_nifti created.
echo.

title Pipeline [2/5]: Batch SPHARM (%FOLDER_NAME%)
echo [2/5] Running Batch SPHARM Processing (run_spharm_batch.py)...
echo        - Subject 0 processed without template (generates reference)
echo        - Subjects 1..N processed with subject 0's ellalign as template
echo        - Output: _SPHARM_procalign.vtk for consistent vertex correspondences
"%SLICER_EXE%" --no-main-window --no-splash --python-script "%PIPELINE_DIR%\run_spharm_batch.py" --input_dir "%OUTPUT_DIR%\aligned_nifti" --output_dir "%OUTPUT_DIR%"
if not exist "%OUTPUT_DIR%\spharm_results" (
    echo.
    echo [ERROR] Step 2 failed - 'spharm_results' folder not created.
    echo         Check spharm_progress.log for details.
    pause
    exit /b 1
)
REM ทำไมต้อง wait: Slicer process อาจยังอยู่ระหว่าง cleanup แม้ python script จบแล้ว
REM   ให้เวลา Windows release file handles + flush disk cache ก่อนรัน step ต่อไป
REM   กัน "file in use" errors ใน step ถัดไป
timeout /t 3 /nobreak >nul
echo [OK] spharm_results created.
echo.

REM ---- Pre-check: ต้องมี SPHARM*.vtk อย่างน้อย 2 ตัวก่อน realign ----
REM   ถ้าไม่มี realign_spharm.py จะ sys.exit(1) -> stop pipeline พร้อม message ชัด
set /a VTK_COUNT=0
for %%F in ("%OUTPUT_DIR%\spharm_results\*_SPHARM*.vtk") do set /a VTK_COUNT+=1
if !VTK_COUNT! LSS 2 (
    echo.
    echo [ERROR] Step 3 pre-check failed: only !VTK_COUNT! SPHARM .vtk file in spharm_results/
    echo         [need at least 2 — SPHARM step seems to have failed for most subjects]
    echo         Check %OUTPUT_DIR%\spharm_results\spharm_progress.log
    pause
    exit /b 1
)
echo [OK] Step 3 pre-check: !VTK_COUNT! SPHARM .vtk found.
echo.

title Pipeline [3/5]: SPHARM Re-alignment (%FOLDER_NAME%)
echo [3/5] Re-aligning SPHARM meshes (anatomical landmarks: head/tail/lateral/medial)...
echo        Output also logged to: %OUTPUT_DIR%\realign_log.txt

REM ทำไม redirect: ถ้า realign crash กลางทาง ข้อความ error จะอยู่ในไฟล์
REM   user เปิดดูได้แม้ window ปิด (เผื่อ Window manager kill cmd ระหว่าง pause)
REM ใช้ /b 0 reset errorlevel ก่อนรัน + log ทั้ง stdout & stderr
ver > nul
"%USER_PYTHON%" "%PIPELINE_DIR%\realign_spharm.py" --spharm_dir "%OUTPUT_DIR%\spharm_results" > "%OUTPUT_DIR%\realign_log.txt" 2>&1
set "REALIGN_EXIT=!errorlevel!"
type "%OUTPUT_DIR%\realign_log.txt"
echo.
echo [DIAG] realign_spharm.py exit code: !REALIGN_EXIT!
if not "!REALIGN_EXIT!"=="0" (
    echo.
    echo [ERROR] Step 3 [realign] failed with exit code !REALIGN_EXIT!.
    echo         Full log:  %OUTPUT_DIR%\realign_log.txt
    echo.
    echo Press any key to acknowledge — pipeline will stop here.
    pause
    exit /b 1
)
echo [OK] realign complete.
echo.

title Pipeline [4/5]: PCA Analysis (%FOLDER_NAME%)
echo [4/5] Running PCA Analysis (run_pca_batch.py)...
"%SLICER_EXE%" --no-main-window --no-splash --python-script "%PIPELINE_DIR%\run_pca_batch.py" --output_dir "%OUTPUT_DIR%"
if not exist "%OUTPUT_DIR%\pca_results\pca_scores.csv" (
    echo.
    echo [ERROR] Step 4 failed - pca_scores.csv not created.
    echo         Check pca_results\progress.log for details.
    pause
    exit /b 1
)
timeout /t 2 /nobreak >nul
echo [OK] pca_scores.csv created.
echo.

title Pipeline [5/5]: PCA Visualization (%FOLDER_NAME%)
echo [5/5] Running PCA Visualization (visualize_pca.py)...
"%USER_PYTHON%" "%PIPELINE_DIR%\visualize_pca.py" --output_dir "%OUTPUT_DIR%"
if errorlevel 1 (
    echo.
    echo [ERROR] Step 5 failed.
    pause
    exit /b 1
)
echo [OK] visualization complete.
echo.

title Pipeline Display: ICP Convergence Plot (%FOLDER_NAME%)
echo Displaying ICP Convergence Time-Series Plot...
"%USER_PYTHON%" "%PIPELINE_DIR%\plot_icp_convergence.py" --output_dir "%OUTPUT_DIR%" --show
echo.

echo ============================================================
echo   SUCCESS: PIPELINE COMPLETED FOR DATASET: %FOLDER_NAME%
echo   RESULTS ARE IN: %OUTPUT_DIR%
echo ============================================================
echo.
pause