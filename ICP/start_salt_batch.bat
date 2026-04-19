@echo off
set "SLICER_EXE=C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe"
set "SCRIPT_PATH=%~dp0run_spharm_batch.py"

echo ============================================================
echo Starting Slicer SALT SPHARM Analysis (Background Mode)
echo ============================================================
echo Runner: %SLICER_EXE%
echo Script: %SCRIPT_PATH%
echo.
echo NOTE: Running in background (Silent). 
echo       Check 'output/spharm_results/spharm_progress.log' to see progress!
echo.

if not exist "%SLICER_EXE%" (
    echo Error: SlicerSALT.exe not found at %SLICER_EXE%
    pause
    exit /b 1
)

:: Using SlicerSALT.exe with --no-main-window (Proven stable for PCA)
"%SLICER_EXE%" --no-main-window --python-script "%SCRIPT_PATH%"

echo.
echo Process finished. Check logs for details.
pause
