@echo off
set "SLICER_EXE=C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe"
set "SCRIPT_PATH=%~dp0preview_results.py"

echo Opening SPHARM Results Preview via SlicerSALT...
echo Slicer: %SLICER_EXE%
echo Script: %SCRIPT_PATH%
echo.

if not exist "%SLICER_EXE%" (
    echo Error: SlicerSALT not found at %SLICER_EXE%
    pause
    exit /b 1
)

"%SLICER_EXE%" --python-script "%SCRIPT_PATH%"

echo.
pause

echo.
pause
