@echo off
chcp 65001 >nul
cd /d "%~dp0"

net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Requesting admin...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

python -c "import torch, cv2, pynput, vgamepad, mss" 2>nul
if %errorlevel% neq 0 (
    echo [!] Installing deps...
    pip install -r "%~dp0requirements.txt" -q
)

echo ========================================
echo    AI Inference Starting...
echo    F10 = Toggle AI
echo    ESC = Exit
echo    (Enter game first, then press F10!)
echo ========================================

set PYTHONUNBUFFERED=1
python -m inference.run
pause
