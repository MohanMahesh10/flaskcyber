# ML-Enhanced Cybersecurity System - PowerShell Installation Script

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  ML-Enhanced Cybersecurity System - PowerShell Installation" -ForegroundColor Cyan  
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://python.org" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add to PATH' during installation" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "📦 Installing required packages..." -ForegroundColor Yellow
Write-Host ""

$packages = @(
    "streamlit",
    "pandas", 
    "numpy",
    "scikit-learn",
    "matplotlib",
    "seaborn", 
    "plotly",
    "tensorflow",
    "xgboost",
    "lightgbm", 
    "cryptography",
    "pillow",
    "opencv-python",
    "imbalanced-learn",
    "joblib",
    "psutil",
    "pycryptodome"
)

# Upgrade pip first
Write-Host "⬆️ Upgrading pip..." -ForegroundColor Blue
python -m pip install --upgrade pip

$failedPackages = @()

foreach ($package in $packages) {
    Write-Host "📦 Installing $package..." -ForegroundColor Blue
    
    try {
        $result = python -m pip install $package 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ $package installed successfully" -ForegroundColor Green
        } else {
            Write-Host "❌ Failed to install $package" -ForegroundColor Red
            $failedPackages += $package
        }
    } catch {
        Write-Host "❌ Error installing $package : $_" -ForegroundColor Red
        $failedPackages += $package
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Installation Summary" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$successCount = $packages.Count - $failedPackages.Count
Write-Host "✅ Successfully installed: $successCount/$($packages.Count) packages" -ForegroundColor Green

if ($failedPackages.Count -gt 0) {
    Write-Host "❌ Failed packages: $($failedPackages -join ', ')" -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 You can try installing failed packages manually with:" -ForegroundColor Yellow
    Write-Host "python -m pip install <package_name>" -ForegroundColor Yellow
} else {
    Write-Host "🎉 All packages installed successfully!" -ForegroundColor Green
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  How to Run the Application" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "1. Open Command Prompt or PowerShell" -ForegroundColor White
Write-Host "2. Navigate to this folder: $PWD" -ForegroundColor White
Write-Host "3. Run: streamlit run app.py" -ForegroundColor Yellow
Write-Host "4. Open http://localhost:8501 in your browser" -ForegroundColor White
Write-Host ""
Write-Host "🚀 Quick start: Run 'streamlit run app.py' in this directory" -ForegroundColor Green
Write-Host ""

Read-Host "Press Enter to continue"
