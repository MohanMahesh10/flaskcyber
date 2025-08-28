@echo off
echo ============================================================
echo  🔐 ML-Enhanced Cybersecurity System - Flask Version
echo ============================================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.7+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python is installed
echo.

echo Installing/checking dependencies...
pip install -r requirements_flask.txt >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    echo Please run: pip install -r requirements_flask.txt
    pause
    exit /b 1
)

echo ✅ Dependencies installed
echo.

echo Testing application components...
python test_app.py
if %errorlevel% neq 0 (
    echo ❌ Application test failed
    pause
    exit /b 1
)

echo.
echo 🚀 Starting Flask application...
echo Open your browser and navigate to: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python app_flask.py

pause
