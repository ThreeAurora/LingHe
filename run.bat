@echo off
rem LingHe launcher - double click.
rem 2026-09-05: LLM(qwen/Ollama) 兜底退役, 神经重排(RBT3 ONNX)接棒——不再启动 Ollama。
rem 2026-09-07: 常驻控制台带标题「灵鹤 LingHe」，实时滚日志；关窗口=退出输入法。
setlocal
set "LINGHE_DIR=%~dp0"

start "灵鹤 LingHe" "E:\miniconda3\python.exe" "%LINGHE_DIR%linghe.py"
endlocal
