# Game AI Recorder — 一键启动（自动提权到管理员）
param([switch]$NoElevate)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# 自动提权
if (-not $NoElevate -and -not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "[!] 需要管理员权限，正在重新启动..."
    Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -NoElevate"
    exit
}

Write-Host "═══════════════════════════════════════════"
Write-Host "  Game AI Recorder — 一键启动"
Write-Host "═══════════════════════════════════════════"
Write-Host ""

# 检查依赖
$deps = python -c "import dxcam, inputs, pynput, cv2" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] 缺少依赖，正在安装..."
    pip install -r requirements.txt -q
    Write-Host "[√] 依赖安装完成"
}

Write-Host "[√] 环境就绪"
Write-Host ""
Write-Host "  F8 = 开始录制  |  F9 = 停止并保存  |  关闭窗口 = 退出"
Write-Host "═══════════════════════════════════════════"
Write-Host ""

python record.py
pause
