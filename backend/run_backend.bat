@echo off
title AI Driver Monitoring - Backend API
echo ======================================================================
echo   Starting Backend API Server (FastAPI + Uvicorn)...
echo   Running on: http://localhost:8765
echo   API Docs:   http://localhost:8765/docs
echo ======================================================================
echo.
cd /d "%~dp0"
python -m uvicorn api:app --host 127.0.0.1 --port 8765 --reload
pause
