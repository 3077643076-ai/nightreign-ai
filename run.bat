@echo off
cd /d "%~dp0"

net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Requesting admin privileges...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

python -c "import dxcam, inputs, pynput, cv2" 2>nul
if %errorlevel% neq 0 (
    echo [!] Installing dependencies...
    pip install -r "%~dp0requirements.txt" -q
)

python "%~dp0record.py"
pause
