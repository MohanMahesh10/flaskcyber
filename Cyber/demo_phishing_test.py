#!/usr/bin/env python3
"""
Simple demonstration of enhanced phishing detection
"""

import sys
import os

# Add the modules directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from simple_intrusion_detector import SimpleIntrusionDetector

def demo_phishing_detection():
    """Demonstrate phishing detection with various examples."""
    detector = SimpleIntrusionDetector()
    
    print("🔍 Enhanced Phishing Detection Demonstration")
    print("=" * 60)
    
    # Test examples that should trigger phishing detection
    examples = [
        "Your account has been suspended due to unusual activity. Click here to verify your identity and restore access immediately.",
        "URGENT: Security alert! Your PayPal account will be closed unless you update your payment method within 24 hours.",
        "Congratulations! You are the lucky winner of our $1,000,000 lottery! Claim your prize before it expires today.",
        "Your computer is infected with a dangerous virus! Call Microsoft support at this number immediately to fix the problem.",
        "Amazing crypto opportunity! Double your bitcoin investment with our guaranteed returns program. Act now!",
        "This is just a normal email about tomorrow's team meeting at 2 PM in the conference room."  # Should NOT trigger
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n📧 Example {i}:")
        print(f"Text: \"{example}\"")
        print("-" * 50)
        
        result = detector.analyze_text(example)
        
        if result['is_intrusion']:
            print(f"🚨 PHISHING DETECTED!")
            print(f"   Confidence: {result['confidence_score']}")
            print(f"   Severity: {result['severity']}")
            print(f"   Total threats: {len(result['threats_detected'])}")
            
            # Show specific phishing patterns found
            for threat in result['threats_detected']:
                if threat['category'] == 'phishing':
                    print(f"   🎯 Phishing patterns: {', '.join(threat['matches'])}")
                else:
                    print(f"   🔍 {threat['category']}: {', '.join(threat['matches'])}")
        else:
            print("✅ No threats detected - appears legitimate")
        
        print()
    
    # Show statistics
    stats = detector.get_threat_statistics()
    print("📊 Detection System Statistics:")
    print(f"   Total threat categories: {stats['total_categories']}")
    print(f"   Total patterns monitored: {stats['total_patterns']}")
    print(f"   Phishing patterns: {stats['categories']['phishing']['pattern_count']}")
    print(f"   OCR capability: {'✅ Available' if stats['ocr_available'] else '❌ Not available'}")

if __name__ == "__main__":
    demo_phishing_detection()
