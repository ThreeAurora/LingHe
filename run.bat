@echo off
rem LingHe launcher - double click.
rem 2026-09-05: LLM(qwen/Ollama) 兜底退役, 神经重排(RBT3 ONNX)接棒——不再启动 Ollama。
setlocal
set "LINGHE_DIR=%~dp0"

start "" "E:\miniconda3\python.exe" "%LINGHE_DIR%linghe.py"
endlocal
