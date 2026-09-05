@echo off
rem LingHe launcher - double click. Auto-starts local AI runtime if present.
setlocal
set "LINGHE_DIR=%~dp0"
set "OLLAMA_EXE_PORTABLE=%LINGHE_DIR%ai_runtime\ollama\ollama.exe"
set "OLLAMA_EXE_OFFICIAL=C:\Users\Administrator\AppData\Local\Programs\Ollama\ollama.exe"
rem 2026-09-05: GPU 诊断结论 —— RTX 2060 驱动/CUDA 正常（Ollama server.log 证实
rem library=CUDA compute=7.5, VRAM 6.0GiB, 张量已 offload 到 GPU）。
rem 此前的 500 报错是模型冷加载耗时 53.5s、客户端提前断开，不是驱动故障。
rem OLLAMA_KEEP_ALIVE=-1 让模型常驻显存，避免每次冷加载。
set "OLLAMA_KEEP_ALIVE=-1"

netstat -ano 2>nul | findstr /c:":11434" | findstr LISTENING >nul 2>&1
if errorlevel 1 (
  if exist "%OLLAMA_EXE_PORTABLE%" start "LingHe-AI" /min "%OLLAMA_EXE_PORTABLE%" serve
  if not exist "%OLLAMA_EXE_PORTABLE%" if exist "%OLLAMA_EXE_OFFICIAL%" start "LingHe-AI" /min "%OLLAMA_EXE_OFFICIAL%" serve
  timeout /t 2 /nobreak >nul
)

start "" "E:\miniconda3\python.exe" "%LINGHE_DIR%linghe.py"
endlocal
