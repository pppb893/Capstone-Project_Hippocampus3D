@echo off
setlocal enabledelayedexpansion

echo Detecting Python...
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.11 from python.org
    pause
    exit /b 1
)

for /f "delims=" %%i in ('where python') do (
    set "PYTHON_EXE=%%i"
    goto :found
)

:found
echo Using Python at: "%PYTHON_EXE%"
"%PYTHON_EXE%" -V

echo.
echo Creating virtual environment (venv)...
if exist venv (
    echo Venv already exists. Refreshing...
    rmdir /s /q venv
)

"%PYTHON_EXE%" -m venv venv
if %errorlevel% neq 0 (
    echo Failed to create venv
    pause
    exit /b %errorlevel%
)

echo Activating venv and installing requirements...
call venv\Scripts\activate.bat

python -m pip install --upgrade pip
if exist requirements.txt (
    pip install -r requirements.txt
) else (
    echo Warning: requirements.txt not found.
)

echo.
echo Setup Completed Successfully!
pause
