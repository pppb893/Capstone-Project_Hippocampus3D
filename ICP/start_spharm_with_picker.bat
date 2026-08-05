@echo off
setlocal enabledelayedexpansion

set "SLICER_EXE=C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe"
set "SCRIPT_PATH=%~dp0run_spharm_batch.py"
set "PICKER_PATH=%~dp0select_folders_gui.py"

echo ============================================================
echo Starting Slicer SALT SPHARM Analysis (Interactive Mode)
echo ============================================================
echo.
echo Opening folder pickers via standard Python...

:: Find working Python interpreter
set "USER_PYTHON="

:: 1. Try venv first, but verify it actually works
if exist "%~dp0venv\Scripts\python.exe" (
    "%~dp0venv\Scripts\python.exe" -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "USER_PYTHON=%~dp0venv\Scripts\python.exe"
    )
)

:: 2. Try system python
if "%USER_PYTHON%"=="" (
    python -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "USER_PYTHON=python"
    )
)

:: 3. Try py launcher
if "%USER_PYTHON%"=="" (
    py -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "USER_PYTHON=py"
    )
)

if "%USER_PYTHON%"=="" (
    echo [ERROR] No working Python interpreter found on this system.
    echo         Please ensure Python is installed and added to PATH.
    pause
    exit /b 1
)

:: Run the picker script and capture output lines to a temporary file
set "INPUT_DIR="
set "OUTPUT_DIR="
set "TEMP_FILE=%TEMP%\spharm_selected_dirs_%RANDOM%.txt"

if exist "%TEMP_FILE%" del "%TEMP_FILE%"

"%USER_PYTHON%" "%PICKER_PATH%" > "%TEMP_FILE%"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [CANCELLED] Selection was cancelled or failed.
    if exist "%TEMP_FILE%" del "%TEMP_FILE%"
    pause
    exit /b 1
)

:: Read lines from temporary file
set /a LineCount=0
for /f "usebackq delims=" %%x in ("%TEMP_FILE%") do (
    set /a LineCount+=1
    if !LineCount! equ 1 set "INPUT_DIR=%%x"
    if !LineCount! equ 2 set "OUTPUT_DIR=%%x"
)
if exist "%TEMP_FILE%" del "%TEMP_FILE%"

if "%INPUT_DIR%"=="" (
    echo [ERROR] Input directory was not selected.
    pause
    exit /b 1
)
if "%OUTPUT_DIR%"=="" (
    echo [ERROR] Output directory was not selected.
    pause
    exit /b 1
)

echo.
echo Selected Input Folder:  %INPUT_DIR%
echo Selected Output Folder: %OUTPUT_DIR%
echo.
echo Starting SPHARM computation in SlicerSALT background...
echo.

if not exist "%SLICER_EXE%" (
    echo [ERROR] SlicerSALT.exe not found at %SLICER_EXE%
    pause
    exit /b 1
)

:: Run SlicerSALT with the selected arguments
"%SLICER_EXE%" --no-main-window --python-script "%SCRIPT_PATH%" --input_dir "%INPUT_DIR%" --output_dir "%OUTPUT_DIR%"

echo.
echo SPHARM processing completed. Check the progress log inside your output folder/spharm_results/spharm_progress.log!
pause
