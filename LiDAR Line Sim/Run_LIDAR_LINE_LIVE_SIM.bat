@echo off
REM Run the live PyQt LiDAR line-detection simulator.
setlocal
set "ROOT=%~dp0"
set "SIM=%ROOT%simulated_world"
set "PY=%SIM%\.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo [ERROR] Python venv not found at:
    echo   %PY%
    echo Run uv venv inside simulated_world\ and install requirements.txt.
    pause
    exit /b 1
)

cd /d "%SIM%"
"%PY%" "%SIM%\lidar_line_live_gui.py" %*
if errorlevel 1 pause
