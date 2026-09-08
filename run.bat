@echo off
rem LingHe launcher - double click.
rem ASCII-only on purpose: cmd.exe parses .bat in ANSI(GBK) codepage,
rem UTF-8 Chinese comments corrupt the parser (2026-09-08 crash case).
rem 2026-09-08: cmd /k keeps the window open on crash (no more flash-exit).
rem             -X faulthandler prints native crashes too.
rem             Switch back to plain start once stable.
setlocal
set "LINGHE_DIR=%~dp0"

start "LingHe" cmd /k ""E:\miniconda3\python.exe" -X faulthandler "%LINGHE_DIR%linghe.py""
endlocal
