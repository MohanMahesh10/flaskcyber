#!/usr/bin/env python3
"""
Create test images with text for testing OCR functionality.
This script generates images with various types of text to test intrusion detection.
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_test_image(text, filename, size=(800, 400), bg_color=(255, 255, 255), text_color=(0, 0, 0)):
    """Create a test image with the given text."""
    # Create image
    image = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(image)
    
    # Try to use a default font, fallback to basic if not available
    try:
        # Try to use a system font
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 24)
        except:
            # Use default font
            font = ImageFont.load_default()
    
    # Calculate text position (center)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    # Draw text
    draw.text((x, y), text, fill=text_color, font=font)
    
    # Save image
    image.save(filename)
    print(f"Created test image: {filename}")
    return filename

def main():
    """Create various test images."""
    # Create test directory
    test_dir = "test_images"
    os.makedirs(test_dir, exist_ok=True)
    
    # Test cases
    test_cases = [
        # Normal text
        ("Hello World! This is a normal message.", "normal_text.png"),
        
        # SQL Injection attempt
        ("SELECT * FROM users WHERE id = 1 OR 1=1; DROP TABLE users;", "sql_injection.png"),
        
        # XSS attempt
        ("<script>alert('XSS Attack');</script>", "xss_attack.png"),
        
        # Command injection
        ("; rm -rf /; cat /etc/passwd", "command_injection.png"),
        
        # Path traversal
        ("../../../etc/passwd", "path_traversal.png"),
        
        # Malware keywords
        ("ransomware attack detected, trojan horse found", "malware_keywords.png"),
        
        # Phishing attempt
        ("URGENT: Your account has been suspended. Click here immediately!", "phishing.png"),
        
        # Network attack tools
        ("nmap scan results, metasploit exploit, burp suite", "network_attacks.png"),
        
        # Mixed threats
        ("Login attempt: admin OR 1=1; <script>eval('alert')</script>", "mixed_threats.png"),
        
        # Clean technical text
        ("System status: CPU 45%, Memory 67%, Network: OK", "clean_technical.png")
    ]
    
    print("Creating test images for OCR testing...")
    print("=" * 50)
    
    created_files = []
    
    for text, filename in test_cases:
        filepath = os.path.join(test_dir, filename)
        try:
            create_test_image(text, filepath)
            created_files.append(filepath)
        except Exception as e:
            print(f"Error creating {filename}: {e}")
    
    print("\n" + "=" * 50)
    print(f"Created {len(created_files)} test images in '{test_dir}' directory")
    print("\nYou can now use these images to test the OCR functionality:")
    print("1. Upload them through the Flask web interface")
    print("2. Use them to test the intrusion detection system")
    print("3. Verify that different threat types are detected correctly")
    
    # Create a summary file
    summary_file = os.path.join(test_dir, "test_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Test Images Summary\n")
        f.write("=" * 30 + "\n\n")
        for text, filename in test_cases:
            f.write(f"File: {filename}\n")
            f.write(f"Text: {text}\n")
            f.write(f"Expected: {'Threat' if any(keyword in text.lower() for keyword in ['script', 'drop', 'rm', 'etc', 'ransomware', 'trojan', 'phishing', 'nmap', 'metasploit']) else 'Clean'}\n")
            f.write("-" * 30 + "\n\n")
    
    print(f"\nTest summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
