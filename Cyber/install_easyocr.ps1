Write-Host "Installing EasyOCR and dependencies for image-based intrusion detection..." -ForegroundColor Green
Write-Host ""

Write-Host "Installing EasyOCR..." -ForegroundColor Yellow
pip install easyocr

Write-Host ""
Write-Host "Installing additional dependencies..." -ForegroundColor Yellow
pip install Pillow
pip install opencv-python
pip install torch torchvision

Write-Host ""
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To test the installation, run: python test_easyocr.py" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to continue"
