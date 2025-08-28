#!/usr/bin/env python3
"""
Test script to verify EasyOCR installation and functionality.
Run this script to check if EasyOCR is working properly.
"""

import sys
import traceback

def test_easyocr_import():
    """Test if EasyOCR can be imported."""
    print("Testing EasyOCR import...")
    try:
        import easyocr
        print("✅ EasyOCR imported successfully")
        print(f"   Version: {easyocr.__version__}")
        return True
    except ImportError as e:
        print(f"❌ EasyOCR import failed: {e}")
        print("   To install EasyOCR, run: pip install easyocr")
        return False
    except Exception as e:
        print(f"❌ Unexpected error importing EasyOCR: {e}")
        return False

def test_easyocr_reader():
    """Test if EasyOCR reader can be initialized."""
    print("\nTesting EasyOCR reader initialization...")
    try:
        import easyocr
        reader = easyocr.Reader(['en'])
        print("✅ EasyOCR reader initialized successfully")
        return True
    except Exception as e:
        print(f"❌ EasyOCR reader initialization failed: {e}")
        traceback.print_exc()
        return False

def test_pil_import():
    """Test if PIL/Pillow can be imported."""
    print("\nTesting PIL/Pillow import...")
    try:
        from PIL import Image
        print("✅ PIL/Pillow imported successfully")
        return True
    except ImportError as e:
        print(f"❌ PIL/Pillow import failed: {e}")
        print("   To install Pillow, run: pip install Pillow")
        return False

def test_simple_intrusion_detector():
    """Test the simple intrusion detector module."""
    print("\nTesting SimpleIntrusionDetector module...")
    try:
        from modules.simple_intrusion_detector import SimpleIntrusionDetector, OCR_AVAILABLE
        
        print(f"   OCR_AVAILABLE: {OCR_AVAILABLE}")
        
        if OCR_AVAILABLE:
            detector = SimpleIntrusionDetector()
            print("✅ SimpleIntrusionDetector created successfully")
            
            # Test text analysis
            test_text = "This is a test message with no threats"
            result = detector.analyze_text(test_text)
            print(f"   Text analysis test: {result['is_intrusion']}")
            
            return True
        else:
            print("❌ OCR not available in SimpleIntrusionDetector")
            return False
            
    except Exception as e:
        print(f"❌ SimpleIntrusionDetector test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("EasyOCR and Intrusion Detection System Test")
    print("=" * 60)
    
    tests = [
        ("EasyOCR Import", test_easyocr_import),
        ("EasyOCR Reader", test_easyocr_reader),
        ("PIL/Pillow Import", test_pil_import),
        ("SimpleIntrusionDetector", test_simple_intrusion_detector)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your system is ready for image-based intrusion detection.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("\nCommon solutions:")
        print("1. Install EasyOCR: pip install easyocr")
        print("2. Install Pillow: pip install Pillow")
        print("3. Check Python environment and dependencies")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
