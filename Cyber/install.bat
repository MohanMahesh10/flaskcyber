@echo off
echo ============================================================
echo   ML-Enhanced Cybersecurity System - Windows Installation
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add to PATH" during installation
    pause
    exit /b 1
)

echo Installing required packages...
echo.

REM Install packages one by one for better error handling
python -m pip install --upgrade pip
python -m pip install streamlit
python -m pip install pandas
python -m pip install numpy
python -m pip install scikit-learn
python -m pip install matplotlib
python -m pip install seaborn
python -m pip install plotly
python -m pip install tensorflow
python -m pip install xgboost
python -m pip install lightgbm
python -m pip install cryptography
python -m pip install pillow
python -m pip install opencv-python
python -m pip install imbalanced-learn
python -m pip install joblib
python -m pip install psutil
python -m pip install pycryptodome

echo.
echo ============================================================
echo   Installation completed!
echo ============================================================
echo.
echo To run the application:
echo   1. Open Command Prompt or PowerShell
echo   2. Navigate to this folder
echo   3. Run: streamlit run app.py
echo   4. Open http://localhost:8501 in your browser
echo.
pause
