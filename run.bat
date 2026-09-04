@echo off
rem LingHe launcher - double click. Auto-starts local AI runtime if present.
setlocal
set "LINGHE_DIR=%~dp0"
set "OLLAMA_EXE_PORTABLE=%LINGHE_DIR%ai_runtime\ollama\ollama.exe"
set "OLLAMA_EXE_OFFICIAL=C:\Users\Administrator\AppData\Local\Programs\Ollama\ollama.exe"
rem GPU driver stack is broken (nvidia-smi NVML init failed, CUDA compute 500)
rem -> force CPU inference. Delete this line after fixing the NVIDIA driver.
set "CUDA_VISIBLE_DEVICES=-1"
set "OLLAMA_KEEP_ALIVE=-1"

netstat -ano 2>nul | findstr /c:":11434" | findstr LISTENING >nul 2>&1
if errorlevel 1 (
  if exist "%OLLAMA_EXE_PORTABLE%" start "LingHe-AI" /min "%OLLAMA_EXE_PORTABLE%" serve
  if not exist "%OLLAMA_EXE_PORTABLE%" if exist "%OLLAMA_EXE_OFFICIAL%" start "LingHe-AI" /min "%OLLAMA_EXE_OFFICIAL%" serve
  timeout /t 2 /nobreak >nul
)

start "" "E:\miniconda3\python.exe" "%LINGHE_DIR%linghe.py"
endlocal
