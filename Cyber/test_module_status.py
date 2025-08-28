#!/usr/bin/env python3
"""
Simple test to check module status
"""

try:
    from modules.simple_intrusion_detector import SimpleIntrusionDetector
    print("✅ SimpleIntrusionDetector imported successfully")
    
    detector = SimpleIntrusionDetector()
    print("✅ SimpleIntrusionDetector initialized successfully")
    
    stats = detector.get_threat_statistics()
    print(f"OCR Available: {stats['ocr_available']}")
    print(f"Total patterns: {stats['total_patterns']}")
    print(f"Phishing patterns: {stats['categories']['phishing']['pattern_count']}")
    
    # Test text analysis
    result = detector.analyze_text("Click here urgently to verify your account before it expires")
    print(f"Test text analysis - Threats detected: {len(result['threats_detected'])}")
    if result['threats_detected']:
        for threat in result['threats_detected']:
            print(f"  - {threat['category']}: {threat['matches']}")
    
    print("✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
