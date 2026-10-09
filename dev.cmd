@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0environment\docker\start-dev.ps1" %*
exit /b %ERRORLEVEL%
