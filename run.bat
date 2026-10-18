@echo off
title AI Driver Safety System
echo ======================================================================
echo    AI DRIVER SAFETY & DROWSINESS PREDICTION SYSTEM
echo    Multi-Model Backend + React Cockpit HUD
echo ======================================================================
echo.
echo Starting Backend API Server (FastAPI on http://localhost:8765)...
cd /d "%~dp0\backend"
start /b "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:8765"
python -m uvicorn api:app --host 127.0.0.1 --port 8765
pause
