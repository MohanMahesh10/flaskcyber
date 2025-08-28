#!/usr/bin/env python3
"""
Installation script for OCR dependencies
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and return success status."""
    print(f"\n🔄 {description}...")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout.strip():
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr.strip() if e.stderr else str(e)}")
        return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def install_ocr_dependencies():
    """Install OCR dependencies step by step."""
    print("=== Installing OCR Dependencies ===")
    print("This will install EasyOCR and its dependencies...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"\nPython version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 7):
        print("❌ Python 3.7+ is required for EasyOCR")
        return False
    
    # Install dependencies in order
    dependencies = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu", "Installing PyTorch (CPU version)"),
        ("pip install opencv-python-headless", "Installing OpenCV"),
        ("pip install easyocr", "Installing EasyOCR"),
        ("pip install Pillow", "Installing/Upgrading Pillow"),
        ("pip install numpy", "Installing/Upgrading NumPy"),
    ]
    
    success_count = 0
    for command, description in dependencies:
        if run_command(command, description):
            success_count += 1
        else:
            print(f"⚠️ Failed to install {description}")
            # Continue with other installations
    
    print(f"\n=== Installation Summary ===")
    print(f"Successfully installed: {success_count}/{len(dependencies)} packages")
    
    if success_count == len(dependencies):
        print("🎉 All OCR dependencies installed successfully!")
        return True
    else:
        print("⚠️ Some installations failed. OCR may not work properly.")
        return False

def test_ocr_installation():
    """Test if OCR installation works."""
    print("\n=== Testing OCR Installation ===")
    
    try:
        print("Testing EasyOCR import...")
        import easyocr
        print("✅ EasyOCR imported successfully")
        
        print("Testing EasyOCR initialization...")
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("✅ EasyOCR reader initialized successfully")
        
        print("🎉 OCR installation test passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ OCR test failed: {e}")
        return False

def main():
    """Main installation function."""
    print("OCR Dependencies Installer")
    print("="*50)
    
    # Ask user for confirmation
    response = input("Do you want to install OCR dependencies? (y/N): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Installation cancelled.")
        return
    
    # Install dependencies
    install_success = install_ocr_dependencies()
    
    # Test installation
    if install_success:
        test_success = test_ocr_installation()
        if test_success:
            print("\n🎉 OCR setup completed successfully!")
            print("You can now use image-based intrusion detection.")
        else:
            print("\n⚠️ Installation completed but testing failed.")
            print("There may be compatibility issues.")
    else:
        print("\n❌ Installation failed.")
        print("Please check the error messages above.")
        
    print("\nNext steps:")
    print("1. Run the Flask application: python app_flask.py")
    print("2. Test phishing detection: python test_phishing_detection.py")
    print("3. Upload an image with text to test OCR functionality")

if __name__ == "__main__":
    main()
