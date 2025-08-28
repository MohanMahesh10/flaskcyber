@echo off
echo Installing EasyOCR and dependencies for image-based intrusion detection...
echo.

echo Installing EasyOCR...
pip install easyocr

echo.
echo Installing additional dependencies...
pip install Pillow
pip install opencv-python
pip install torch torchvision

echo.
echo Installation complete!
echo.
echo To test the installation, run: python test_easyocr.py
echo.
pause
