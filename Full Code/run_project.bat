@echo off
title NeuroDetect - AI-Powered EEG Diagnostic System
color 0B

echo ======================================================================
echo          NeuroDetect: EEG Analysis for Alzheimer's ^& Depression
echo                      Capstone Project Launcher
echo ======================================================================
echo.

:: Move to script directory
cd /d "%~dp0"

echo [1/3] Checking project directory...
echo       Path: %CD%
echo.

echo [2/3] Launching web browser in 2 seconds...
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:5000"
echo.

echo [3/3] Starting NeuroDetect Flask Server...
echo ======================================================================
echo  Server is starting on http://127.0.0.1:5000
echo  Press CTRL+C in this terminal window to stop the server.
echo ======================================================================
echo.

:: Try 'py' launcher first, then fallback to 'python'
where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    py app.py
) else (
    python app.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Failed to launch application.
    echo Please make sure Python and all dependencies are installed.
    pause
)
