@echo off
setlocal
cd /d "%~dp0"
where docker >nul 2>nul
if errorlevel 1 (
    echo Docker was not found. Install Docker Desktop first.
    exit /b 1
)
docker info >nul 2>nul
if errorlevel 1 (
    echo Start Docker Desktop and select Linux containers, then retry.
    exit /b 1
)
docker compose version >nul 2>nul
if errorlevel 1 (
    echo Docker Compose v2 is required. Update Docker Desktop.
    exit /b 1
)
if "%~1"=="" goto interactive
docker compose run --build --rm -T dev %*
exit /b %ERRORLEVEL%
:interactive
docker compose run --build --rm dev
exit /b %ERRORLEVEL%
