#!/usr/bin/env python3
"""
Test script to verify phishing detection functionality
"""

import sys
import os

# Add the modules directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from simple_intrusion_detector import SimpleIntrusionDetector

def test_phishing_detection():
    """Test various phishing attack patterns."""
    detector = SimpleIntrusionDetector()
    
    # Test cases with different phishing patterns
    test_cases = [
        {
            'text': 'Dear customer, your account will be closed unless you verify your identity immediately. Click here to update your password.',
            'expected': True,
            'description': 'Banking phishing email'
        },
        {
            'text': 'Urgent action required! Your PayPal account has been suspended. Click here to resolve this issue now.',
            'expected': True,
            'description': 'PayPal impersonation phishing'
        },
        {
            'text': 'Congratulations! You have won the lottery! Claim your prize now before it expires today.',
            'expected': True,
            'description': 'Prize scam phishing'
        },
        {
            'text': 'Security alert: Unusual activity detected on your account. Verify payment method to prevent account closure.',
            'expected': True,
            'description': 'Security alert phishing'
        },
        {
            'text': 'Your computer is infected with a virus! Call Microsoft support immediately to fix your computer.',
            'expected': True,
            'description': 'Tech support scam'
        },
        {
            'text': 'Bitcoin investment opportunity - double your bitcoin with guaranteed returns. Act now!',
            'expected': True,
            'description': 'Cryptocurrency scam'
        },
        {
            'text': 'Hello, this is just a normal email about meeting tomorrow at 3 PM.',
            'expected': False,
            'description': 'Normal email (should not trigger)'
        },
        {
            'text': 'Thank you for your purchase. Your order will be delivered within 3-5 business days.',
            'expected': False,
            'description': 'Legitimate business email (should not trigger)'
        }
    ]
    
    print("=== Phishing Detection Test Results ===\n")
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        result = detector.analyze_text(test_case['text'])
        
        # Check if phishing was detected
        phishing_detected = any(threat['category'] == 'phishing' for threat in result['threats_detected'])
        
        # Test result
        test_passed = phishing_detected == test_case['expected']
        
        print(f"Test {i}: {test_case['description']}")
        print(f"Text: \"{test_case['text'][:80]}{'...' if len(test_case['text']) > 80 else ''}\"")
        print(f"Expected phishing: {test_case['expected']}")
        print(f"Detected phishing: {phishing_detected}")
        print(f"Result: {'✅ PASS' if test_passed else '❌ FAIL'}")
        
        if phishing_detected:
            phishing_threats = [threat for threat in result['threats_detected'] if threat['category'] == 'phishing']
            if phishing_threats:
                print(f"Phishing patterns found: {phishing_threats[0]['matches']}")
                print(f"Confidence: {result['confidence_score']}")
                print(f"Severity: {result['severity']}")
        
        print("-" * 80)
        
        if test_passed:
            passed += 1
        else:
            failed += 1
    
    print(f"\n=== Test Summary ===")
    print(f"Total tests: {len(test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(passed/len(test_cases))*100:.1f}%")
    
    # Show available phishing patterns
    stats = detector.get_threat_statistics()
    phishing_stats = stats['categories']['phishing']
    print(f"\nPhishing patterns available: {phishing_stats['pattern_count']}")
    print(f"OCR available: {stats['ocr_available']}")
    
    return failed == 0

def test_ocr_installation():
    """Test if OCR functionality is working."""
    print("\n=== OCR Installation Test ===")
    
    try:
        import easyocr
        print("✅ EasyOCR module imported successfully")
        
        # Try to create a reader
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("✅ EasyOCR reader initialized successfully")
        print("✅ OCR functionality is working")
        return True
        
    except ImportError as e:
        print(f"❌ EasyOCR import failed: {e}")
        print("To install EasyOCR, run:")
        print("pip install easyocr torch torchvision")
        return False
        
    except Exception as e:
        print(f"❌ EasyOCR initialization failed: {e}")
        print("There may be dependency issues with EasyOCR")
        return False

if __name__ == "__main__":
    print("Testing Intrusion Detection System...\n")
    
    # Test phishing detection
    phishing_ok = test_phishing_detection()
    
    # Test OCR functionality
    ocr_ok = test_ocr_installation()
    
    print(f"\n=== Overall Results ===")
    print(f"Phishing detection: {'✅ Working' if phishing_ok else '❌ Issues found'}")
    print(f"OCR functionality: {'✅ Working' if ocr_ok else '❌ Not available'}")
    
    if phishing_ok and ocr_ok:
        print("🎉 All systems operational!")
    elif phishing_ok:
        print("⚠️ Phishing detection working, but OCR needs setup")
    else:
        print("❌ Issues detected - check the logs above")
