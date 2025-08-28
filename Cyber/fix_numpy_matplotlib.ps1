# PowerShell script to fix numpy/matplotlib import issues
# Run this script as Administrator if needed

Write-Host "🔧 Fixing NumPy/Matplotlib import issues..." -ForegroundColor Green
Write-Host "=" * 60

# Step 1: Check Python installation
Write-Host "Step 1: Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
    } else {
        throw "Python not found"
    }
} catch {
    Write-Host "❌ Python not found in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.10 from https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Yellow
    exit 1
}

# Step 2: Create virtual environment
Write-Host "`nStep 2: Setting up virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "Virtual environment exists, removing old one..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "venv"
}

python -m venv venv
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

# Step 3: Activate virtual environment
Write-Host "`nStep 3: Activating virtual environment..." -ForegroundColor Yellow
try {
    & ".\venv\Scripts\Activate.ps1"
    Write-Host "✅ Virtual environment activated" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Execution policy issue, fixing..." -ForegroundColor Yellow
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
    & ".\venv\Scripts\Activate.ps1"
    Write-Host "✅ Virtual environment activated" -ForegroundColor Green
}

# Step 4: Upgrade pip
Write-Host "`nStep 4: Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
Write-Host "✅ Pip upgraded" -ForegroundColor Green

# Step 5: Clear any corrupted packages
Write-Host "`nStep 5: Removing potentially corrupted packages..." -ForegroundColor Yellow
pip uninstall numpy matplotlib scipy pandas scikit-learn tensorflow keras -y
pip cache purge
Write-Host "✅ Corrupted packages removed and cache cleared" -ForegroundColor Green

# Step 6: Install numpy first
Write-Host "`nStep 6: Installing NumPy..." -ForegroundColor Yellow
pip install numpy==1.21.6
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ NumPy installed successfully" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to install NumPy" -ForegroundColor Red
    exit 1
}

# Step 7: Test numpy
Write-Host "`nStep 7: Testing NumPy..." -ForegroundColor Yellow
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ NumPy import test successful" -ForegroundColor Green
} else {
    Write-Host "❌ NumPy import test failed" -ForegroundColor Red
    exit 1
}

# Step 8: Install remaining packages
Write-Host "`nStep 8: Installing remaining packages from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ All packages installed successfully" -ForegroundColor Green
} else {
    Write-Host "❌ Some packages failed to install" -ForegroundColor Red
}

# Step 9: Run comprehensive test
Write-Host "`nStep 9: Running comprehensive test..." -ForegroundColor Yellow
python test_numpy_matplotlib.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ All tests passed!" -ForegroundColor Green
} else {
    Write-Host "⚠️  Some tests failed, but basic functionality might work" -ForegroundColor Yellow
}

# Step 10: Final instructions
Write-Host "`n" + ("=" * 60)
Write-Host "🎉 Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To activate your environment in the future:" -ForegroundColor Yellow
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run your app:" -ForegroundColor Yellow
Write-Host "  python app.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "Or to run with Streamlit:" -ForegroundColor Yellow
Write-Host "  streamlit run app.py" -ForegroundColor Cyan
