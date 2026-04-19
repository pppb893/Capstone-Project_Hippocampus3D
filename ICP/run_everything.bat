@echo off
setlocal enabledelayedexpansion

:: --- Setup Executables ---
set "SLICER_EXE=C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe"
set "USER_PYTHON=python"

echo ============================================================
echo   THE ULTIMATE SHAPE ANALYSIS PIPELINE (DYNAMIC DATASET)
echo ============================================================
echo.

:: --- Step 0: Folder Picker ---
echo [0/5] SELECTING DATASET...
set "PS_CMD=Add-Type -AssemblyName System.Windows.Forms; $f = New-Object System.Windows.Forms.FolderBrowserDialog; $f.Description = 'Select the folder containing your .nii.gz datasets'; $f.ShowDialog() | Out-Null; $f.SelectedPath"

for /f "delims=" %%I in ('powershell -Command "%PS_CMD%"') do set "INPUT_DIR=%%I"

if "%INPUT_DIR%"=="" (
    echo.
    echo [CANCELLED] No folder selected. Exiting...
    pause
    exit /b 1
)

:: Extract folder name for output naming
for %%I in ("%INPUT_DIR%") do set "FOLDER_NAME=%%~nxI"
set "OUTPUT_DIR=output_%FOLDER_NAME%"

echo.
echo SELECTED INPUT:  %INPUT_DIR%
echo TARGET OUTPUT:   %OUTPUT_DIR%
echo ============================================================
echo.

:: Step 1: ICP & Mean Shape
title Pipeline [1/5]: ICP Alignment (%FOLDER_NAME%)
echo [1/5] Running Group-wise ICP Alignment (main2.py)...
"%SLICER_EXE%" --no-main-window --python-script main2.py --headless --input_dir "%INPUT_DIR%" --output_dir "%OUTPUT_DIR%"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Step 1 failed.
    pause
    exit /b %ERRORLEVEL%
)
echo.

:: Step 2: Prepare SPHARM Template
title Pipeline [2/5]: Template Creation (%FOLDER_NAME%)
echo [2/5] Creating SPHARM Template Metadata (create_template.py)...
"%SLICER_EXE%" --no-main-window --python-script create_template.py --output_dir "%OUTPUT_DIR%"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Step 2 failed.
    pause
    exit /b %ERRORLEVEL%
)
echo.

:: Step 3: Batch SPHARM
title Pipeline [3/5]: Batch SPHARM (%FOLDER_NAME%)
echo [3/5] Running Batch SPHARM Processing (run_spharm_batch.py)...
"%SLICER_EXE%" --no-main-window --python-script run_spharm_batch.py --input_dir "%OUTPUT_DIR%\aligned_nifti" --output_dir "%OUTPUT_DIR%"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Step 3 failed.
    pause
    exit /b %ERRORLEVEL%
)
echo.

:: Step 4: PCA Statistics
title Pipeline [4/5]: PCA Analysis (%FOLDER_NAME%)
echo [4/5] Running PCA Analysis (run_pca_batch.py)...
"%SLICER_EXE%" --no-main-window --python-script run_pca_batch.py --output_dir "%OUTPUT_DIR%"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Step 4 failed.
    pause
    exit /b %ERRORLEVEL%
)
echo.

:: Step 5: Final Visualization
title Pipeline [5/5]: PCA Visualization (%FOLDER_NAME%)
echo [5/5] Running PCA Visualization (visualize_pca.py)...
"%USER_PYTHON%" visualize_pca.py --output_dir "%OUTPUT_DIR%"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Step 5 failed.
    pause
    exit /b %ERRORLEVEL%
)
echo.

echo ============================================================
echo   SUCCESS: PIPELINE COMPLETED FOR DATASET: %FOLDER_NAME%
echo   RESULTS ARE IN: %OUTPUT_DIR%
echo ============================================================
echo.
pause
