@echo off
echo Detecting Conda...
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Conda is not installed or not in PATH.
    echo Please install Miniconda or Anaconda.
    pause
    exit /b 1
)

echo Creating conda environment 'icp_env'...
call conda create -n icp_env python=3.10 -y

echo Activating environment...
call conda activate icp_env

echo Installing dependencies...
call conda install -c conda-forge igl -y
pip install pyshtools scipy trimesh vtk nibabel opencv-python scikit-image

echo.
echo Conda Setup Done!
pause
