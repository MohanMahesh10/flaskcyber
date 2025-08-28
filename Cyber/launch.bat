@echo off
title ML-Enhanced Cybersecurity System

echo ============================================================
echo   🔐 ML-Enhanced Cybersecurity System
echo   Starting Application...
echo ============================================================
echo.
echo 📱 The application will open in your default web browser
echo 🔗 URL: http://localhost:8501  
echo.
echo 💡 To stop the application, press Ctrl+C in this window
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python not found!
    echo Please install Python or run install.bat first
    pause
    exit /b 1
)

REM Launch the Streamlit application
python -m streamlit run app.py

echo.
echo 👋 Application stopped.
pause
