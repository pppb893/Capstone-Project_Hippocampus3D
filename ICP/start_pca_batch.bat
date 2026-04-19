@echo off
set "SLICER_EXE=C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe"
set "SCRIPT_PATH=%~dp0run_pca_batch.py"

echo ============================================================
echo Starting SlicerSALT PCA Analysis (SILENT MODE)
echo ============================================================
echo Runner: %SLICER_EXE%
echo Script: %SCRIPT_PATH%
echo.
echo NOTE: Running in background. NO window will open.
echo       Check 'output/pca_results/progress.log' to see progress!
echo.

if not exist "%SLICER_EXE%" (
    echo Error: SlicerSALT not found at %SLICER_EXE%
    pause
    exit /b 1
)

:: Back to --no-main-window for a clean background run
"%SLICER_EXE%" --no-main-window --python-script "%SCRIPT_PATH%"

echo.
echo Process finished. Results are in 'output/pca_results'.
pause
