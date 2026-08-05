@echo off
setlocal enabledelayedexpansion

set "PIPELINE_DIR=%~dp0"
if "%PIPELINE_DIR:~-1%"=="\" set "PIPELINE_DIR=%PIPELINE_DIR:~0,-1%"

set "SLICER_EXE=C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe"

set "USER_PYTHON="
set "PYTEST=import vtk, numpy, scipy, pandas, matplotlib"

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
set "USER_PYTHON=python"

:python_found

if not exist "%SLICER_EXE%" (
    echo [ERROR] SlicerSALT not found at: %SLICER_EXE%
    echo         Please edit SLICER_EXE in this bat file.
    pause
    exit /b 1
)

echo ============================================================
echo   ICP ALIGNMENT ONLY (ICP.py)
echo ============================================================
echo.
echo Selecting input folder...
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
echo ============================================================
echo.

title ICP Alignment Only (%FOLDER_NAME%)
echo Running Group-wise ICP Alignment (ICP.py)...
"%SLICER_EXE%" --no-main-window --no-splash --python-script "%PIPELINE_DIR%\ICP.py" --input_dir "%INPUT_DIR%" --output_dir "%OUTPUT_DIR%"

if not exist "%OUTPUT_DIR%\aligned_nifti" (
    echo.
    echo [ERROR] ICP failed - 'aligned_nifti' folder not created. Check icp_debug_log.txt for details.
    pause
    exit /b 1
)

echo [OK] ICP complete.
echo.

title ICP Convergence Plot (%FOLDER_NAME%)
echo Displaying ICP Convergence Time-Series Plot...
"%USER_PYTHON%" "%PIPELINE_DIR%\plot_icp_convergence.py" --output_dir "%OUTPUT_DIR%" --show
echo.

echo ============================================================
echo   DONE - aligned NIfTI saved under %OUTPUT_DIR%\aligned_nifti\
echo ============================================================
echo.
pause
