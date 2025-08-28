#!/usr/bin/env python3
"""
Test script to check image upload functionality locally
"""

import requests
import io
from PIL import Image, ImageDraw, ImageFont
import time

def create_test_image_with_text():
    """Create a simple test image with some text."""
    # Create a white image
    width, height = 400, 200
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Add some text that should trigger phishing detection
    test_text = "URGENT: Click here to verify your account immediately!"
    
    # Try to use a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Draw the text
    draw.text((10, 50), test_text, fill='red', font=font)
    draw.text((10, 100), "Your account will be suspended", fill='black', font=font)
    draw.text((10, 150), "Act now to prevent closure", fill='blue', font=font)
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    
    return buffer.getvalue()

def test_local_image_analysis():
    """Test image analysis locally without server."""
    print("🧪 Testing image analysis locally...")
    
    try:
        from modules.simple_intrusion_detector import SimpleIntrusionDetector
        detector = SimpleIntrusionDetector()
        
        # Create test image
        image_data = create_test_image_with_text()
        print(f"✅ Created test image: {len(image_data)} bytes")
        
        # Test local analysis
        print("🔍 Analyzing image locally...")
        result = detector.analyze_image_file(image_data)
        
        print(f"Analysis complete:")
        print(f"  - Success: {result.get('severity', 'ERROR') != 'ERROR'}")
        print(f"  - Severity: {result.get('severity', 'UNKNOWN')}")
        print(f"  - Threats detected: {len(result.get('threats_detected', []))}")
        
        if result.get('threats_detected'):
            for threat in result['threats_detected']:
                print(f"    • {threat['category']}: {threat['matches']}")
        
        if 'ocr_info' in result:
            print(f"  - OCR extracted {result['ocr_info']['extracted_text_length']} characters")
            print(f"  - Preview: {result['ocr_info']['extracted_text_preview']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Local test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_server_endpoint():
    """Test the Flask server endpoint."""
    print("\n🌐 Testing Flask server endpoint...")
    
    server_url = "http://localhost:5000"
    
    try:
        # Check if server is running
        response = requests.get(f"{server_url}/api/system-status", timeout=5)
        print("✅ Server is running")
        
        # Test diagnostics endpoint
        response = requests.get(f"{server_url}/api/system-diagnostics", timeout=10)
        if response.status_code == 200:
            diag = response.json()
            if diag.get('success'):
                print("✅ Diagnostics endpoint working")
                print(f"  - Simple IDS: {diag['diagnostics']['modules']['simple_ids']}")
                print(f"  - OCR Available: {diag['diagnostics']['ocr']['available']}")
            else:
                print(f"⚠️ Diagnostics issues: {diag.get('error')}")
        
        # Create test image
        image_data = create_test_image_with_text()
        
        # Test image upload
        print("📤 Testing image upload...")
        files = {
            'file': ('test_image.png', io.BytesIO(image_data), 'image/png')
        }
        
        response = requests.post(f"{server_url}/api/analyze-image", files=files, timeout=30)
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ Image analysis successful!")
                analysis = result['result']
                print(f"  - Threats: {len(analysis.get('threats_detected', []))}")
                print(f"  - Severity: {analysis.get('severity', 'NONE')}")
                if analysis.get('threats_detected'):
                    for threat in analysis['threats_detected']:
                        print(f"    • {threat['category']}: {threat['matches']}")
            else:
                print(f"❌ Server analysis failed: {result.get('message')}")
        else:
            print(f"❌ Server request failed: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Flask server. Make sure it's running on localhost:5000")
        print("   Run: python app_flask.py")
        return False
    except Exception as e:
        print(f"❌ Server test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🧪 Image Upload and Analysis Test Suite")
    print("=" * 50)
    
    # Test 1: Local analysis
    local_success = test_local_image_analysis()
    
    # Test 2: Server endpoint
    server_success = test_server_endpoint()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"  Local Analysis: {'✅ PASS' if local_success else '❌ FAIL'}")
    print(f"  Server Endpoint: {'✅ PASS' if server_success else '❌ FAIL'}")
    
    if local_success and server_success:
        print("🎉 All tests passed! Image analysis is working correctly.")
    elif local_success:
        print("⚠️ Local analysis works, but server has issues. Check Flask app.")
    else:
        print("❌ Issues detected. Check the error messages above.")

if __name__ == "__main__":
    main()
