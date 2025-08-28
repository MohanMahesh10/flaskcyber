import re
import csv
import io
import hashlib
import time
from PIL import Image
import numpy as np
try:
    import easyocr
    OCR_AVAILABLE = True
    print("EasyOCR successfully imported")
except ImportError as e:
    OCR_AVAILABLE = False
    print(f"EasyOCR import failed: {e}")
    print("To install EasyOCR, run: pip install easyocr")
except Exception as e:
    OCR_AVAILABLE = False
    print(f"EasyOCR error: {e}")

class SimpleIntrusionDetector:
    """
    Simple rule-based intrusion detection system that analyzes text for malicious patterns.
    """
    
    def __init__(self):
        # Define malicious keywords and patterns
        self.malicious_keywords = {
            'sql_injection': [
                'union select', 'drop table', 'delete from', 'insert into', 
                'update set', "' or 1=1", '" or 1=1', 'exec master',
                'xp_cmdshell', 'sp_executesql', '--', ';--', '/*', '*/',
                'waitfor delay', 'benchmark(', 'sleep(', 'pg_sleep'
            ],
            'xss_attack': [
                '<script', '</script>', 'javascript:', 'onload=', 'onerror=',
                'onclick=', 'onmouseover=', 'alert(', 'document.cookie',
                'window.location', 'eval(', 'innerHTML', 'document.write'
            ],
            'command_injection': [
                '; ls', '; cat', '; rm', '; wget', '; curl', '| nc',
                '&& rm', '|| rm', '$(', '`', 'bash -c', 'sh -c',
                '/bin/sh', '/bin/bash', 'cmd.exe', 'powershell',
                'system(', 'exec(', 'passthru(', 'shell_exec('
            ],
            'path_traversal': [
                '../', '..\\', '..../', '....\\', '/etc/passwd',
                '/etc/shadow', 'c:\\windows', '..\\..\\windows',
                '%2e%2e%2f', '%2e%2e\\', 'file://', 'ftp://'
            ],
            'malware_signatures': [
                'wannacry', 'ransomware', 'cryptolocker', 'trojan',
                'backdoor', 'keylogger', 'botnet', 'rootkit',
                'virus', 'malware', 'exploit', 'payload'
            ],
            'phishing': [
                # Common phishing phrases
                'urgent action required', 'verify your account', 'suspended account',
                'click here immediately', 'limited time offer', 'congratulations you won',
                'nigerian prince', 'inheritance', 'lottery winner', 'tax refund',
                
                # Banking/Financial phishing
                'account will be closed', 'unauthorized access detected', 'security alert',
                'confirm your identity', 'update payment method', 'billing information',
                'payment declined', 'your card has been suspended', 'account locked',
                'fraud detected', 'unusual activity', 'verify payment', 'confirm transaction',
                
                # Social engineering
                'act now', 'expires today', 'final notice', 'immediate response required',
                'don\'t delay', 'reply within 24 hours', 'time sensitive', 'deadline approaching',
                'call us immediately', 'contact us now', 'resolve this issue',
                
                # Tech support scams
                'your computer is infected', 'virus detected', 'system compromised',
                'microsoft support', 'apple support', 'tech support', 'computer problem',
                'call tech support', 'remote access', 'fix your computer',
                
                # Prize/Money scams
                'you have won', 'claim your prize', 'congratulations winner',
                'million dollars', 'inheritance fund', 'beneficiary', 'claim money',
                'lottery notification', 'sweepstakes winner', 'cash prize',
                
                # Credential harvesting
                'update your password', 'confirm your password', 'reset password',
                'security verification', 'account verification', 'confirm login',
                'verify identity', 'two-factor authentication', 'security code',
                
                # URL/Link indicators
                'click here', 'click below', 'click this link', 'download attachment',
                'open attachment', 'view document', 'download now', 'install now',
                
                # Impersonation
                'from: paypal', 'from: amazon', 'from: microsoft', 'from: apple',
                'from: bank', 'from: irs', 'from: government', 'official notice',
                
                # Cryptocurrency scams
                'bitcoin investment', 'crypto opportunity', 'investment opportunity',
                'double your bitcoin', 'guaranteed returns', 'crypto giveaway'
                ,
                # Single-word and short variants to catch minimal inputs
                'phishing', 'phish', 'phishing attempt', 'phishing email', 'phishing link'
            ],
            'network_attacks': [
                'syn flood', 'ddos', 'dos attack', 'port scan', 'nmap',
                'metasploit', 'burp suite', 'wireshark', 'tcpdump',
                'nessus', 'nikto', 'sqlmap', 'hydra', 'john the ripper'
            ],
            'suspicious_activity': [
                'password', 'admin', 'root', 'administrator', 'login',
                'credential', 'authentication', 'bypass', 'privilege escalation',
                'buffer overflow', 'format string', 'race condition'
            ]
        }
        
        # Compiled regex patterns for better performance
        self.compiled_patterns = {}
        for category, keywords in self.malicious_keywords.items():
            pattern = '|'.join(re.escape(keyword.lower()) for keyword in keywords)
            self.compiled_patterns[category] = re.compile(pattern, re.IGNORECASE)
        
        # Severity levels
        self.severity_levels = {
            'sql_injection': 'HIGH',
            'xss_attack': 'HIGH', 
            'command_injection': 'CRITICAL',
            'path_traversal': 'HIGH',
            'malware_signatures': 'CRITICAL',
            'phishing': 'MEDIUM',
            'network_attacks': 'HIGH',
            'suspicious_activity': 'LOW'
        }
    
    def analyze_text(self, text):
        """
        Analyze text for intrusion patterns and return detailed results.
        """
        if not text or not isinstance(text, str):
            return {
                'is_intrusion': False,
                'confidence_score': 0,
                'threats_detected': [],
                'severity': 'NONE',
                'details': 'No text provided for analysis'
            }
        
        text_lower = text.lower()
        detected_threats = []
        total_matches = 0
        highest_severity = 'NONE'
        
        # Analyze each category
        for category, pattern in self.compiled_patterns.items():
            matches = pattern.findall(text_lower)
            if matches:
                severity = self.severity_levels[category]
                unique_matches = list(set(matches))  # Remove duplicates
                
                threat_info = {
                    'category': category,
                    'severity': severity,
                    'matches': unique_matches,
                    'match_count': len(matches),
                    'description': self._get_threat_description(category)
                }
                detected_threats.append(threat_info)
                total_matches += len(matches)
                
                # Update highest severity
                if self._is_higher_severity(severity, highest_severity):
                    highest_severity = severity
        
        # Additional heuristic phishing detection to reduce false negatives
        try:
            phishing_signals = ['verify', 'account', 'password', 'click', 'link', 'login', 'confirm', 'update', 'urgent', 'suspended', 'locked']
            url_indicators = ['http://', 'https://', 'www.', 'bit.ly', 'tinyurl', 't.co', 'goo.gl']
            found_signals = [w for w in phishing_signals if w in text_lower]
            has_url = any(ind in text_lower for ind in url_indicators)
            co_occurrence = ('click' in text_lower and 'here' in text_lower) or ('verify' in text_lower and 'account' in text_lower)
            if (len(found_signals) >= 2 and has_url) or co_occurrence:
                # Only add if not already present
                already_reported = any(t.get('category') == 'phishing' for t in detected_threats)
                if not already_reported:
                    threat_info = {
                        'category': 'phishing',
                        'severity': 'MEDIUM',
                        'matches': list(set(found_signals + (['url'] if has_url else []))),
                        'match_count': max(1, len(found_signals)),
                        'description': self._get_threat_description('phishing') + ' (heuristic)'
                    }
                    detected_threats.append(threat_info)
                    total_matches += threat_info['match_count']
                    if self._is_higher_severity('MEDIUM', highest_severity):
                        highest_severity = 'MEDIUM'
        except Exception:
            pass

        # Calculate confidence score
        confidence_score = min(100, total_matches * 10 + len(detected_threats) * 15)
        
        # Determine if it's an intrusion
        is_intrusion = len(detected_threats) > 0
        
        return {
            'is_intrusion': is_intrusion,
            'confidence_score': confidence_score,
            'threats_detected': detected_threats,
            'severity': highest_severity,
            'total_matches': total_matches,
            'analysis_time': time.time(),
            'text_length': len(text),
            'details': f"Analyzed {len(text)} characters, found {len(detected_threats)} threat categories with {total_matches} total matches"
        }
    
    def analyze_csv_file(self, file_content):
        """
        Analyze CSV file content for intrusions in all text fields.
        """
        try:
            # Parse CSV content
            csv_reader = csv.reader(io.StringIO(file_content))
            rows = list(csv_reader)
            
            if not rows:
                return {
                    'is_intrusion': False,
                    'confidence_score': 0,
                    'threats_detected': [],
                    'severity': 'NONE',
                    'details': 'Empty CSV file'
                }
            
            # Combine all text from CSV
            all_text = []
            for row_idx, row in enumerate(rows):
                for col_idx, cell in enumerate(row):
                    if cell and isinstance(cell, str):
                        all_text.append(f"Row {row_idx}, Col {col_idx}: {cell}")
            
            combined_text = " ".join(all_text)
            
            # Analyze combined text
            result = self.analyze_text(combined_text)
            result['csv_info'] = {
                'total_rows': len(rows),
                'total_cells': sum(len(row) for row in rows),
                'text_cells': len(all_text)
            }
            result['details'] = f"CSV Analysis: {len(rows)} rows, {len(all_text)} text cells analyzed"
            
            return result
            
        except Exception as e:
            return {
                'is_intrusion': False,
                'confidence_score': 0,
                'threats_detected': [],
                'severity': 'ERROR',
                'details': f"Error parsing CSV: {str(e)}"
            }
    
    def analyze_image_file(self, image_path_or_content):
        """
        Extract text from image using OCR and analyze for intrusions.
        """
        if not OCR_AVAILABLE:
            return {
                'is_intrusion': False,
                'confidence_score': 0,
                'threats_detected': [],
                'severity': 'ERROR',
                'details': 'OCR functionality not available. Install with: pip install easyocr torch torchvision'
            }
        
        try:
            # Open and validate image
            if isinstance(image_path_or_content, str):
                image = Image.open(image_path_or_content)
            else:
                image = Image.open(io.BytesIO(image_path_or_content))
            
            print(f"Image opened: size={image.size}, mode={image.mode}")
            
            # Validate image size
            if image.size[0] * image.size[1] > 10000000:  # 10MP limit
                image = image.resize((1920, 1080), Image.Resampling.LANCZOS)
                print(f"Image resized to: {image.size}")
            
            # Convert image to RGB if necessary (EasyOCR works better with RGB)
            if image.mode != 'RGB':
                image = image.convert('RGB')
                print(f"Image converted to RGB mode")
            
            # Initialize OCR with error handling
            try:
                print("Initializing EasyOCR reader...")
                reader = easyocr.Reader(['en'], gpu=False, verbose=False)  # Force CPU, disable verbose
                print("OCR Reader initialized successfully")
            except Exception as ocr_init_error:
                print(f"OCR initialization failed: {ocr_init_error}")
                return {
                    'is_intrusion': False,
                    'confidence_score': 0,
                    'threats_detected': [],
                    'severity': 'ERROR',
                    'details': f'OCR initialization failed: {str(ocr_init_error)}'
                }
            
            # Extract text using EasyOCR
            try:
                print("Processing image with OCR...")
                # Convert PIL image to numpy array for EasyOCR
                np_image = np.array(image)
                print(f"Image converted to numpy array: shape={np_image.shape}")
                
                # Process with timeout protection (Windows-compatible)
                import threading
                import time as time_module
                
                # OCR processing with timeout
                ocr_result = [None]
                ocr_error = [None]
                
                def ocr_worker():
                    try:
                        ocr_result[0] = reader.readtext(np_image, paragraph=False)
                    except Exception as e:
                        ocr_error[0] = e
                
                # Start OCR processing in a separate thread
                thread = threading.Thread(target=ocr_worker)
                thread.daemon = True
                thread.start()
                
                # Wait for completion with timeout
                thread.join(timeout=30.0)  # 30 second timeout
                
                if thread.is_alive():
                    # Thread is still running, OCR timed out
                    raise TimeoutError("OCR processing timed out after 30 seconds")
                
                if ocr_error[0]:
                    raise ocr_error[0]
                
                ocr_results = ocr_result[0]
                if ocr_results is None:
                    raise RuntimeError("OCR processing failed to return results")
                
                print(f"OCR completed, found {len(ocr_results)} text regions")
                
            except TimeoutError:
                return {
                    'is_intrusion': False,
                    'confidence_score': 0,
                    'threats_detected': [],
                    'severity': 'ERROR',
                    'details': 'OCR processing timed out (30s limit exceeded)'
                }
            except Exception as ocr_process_error:
                print(f"OCR processing error: {ocr_process_error}")
                return {
                    'is_intrusion': False,
                    'confidence_score': 0,
                    'threats_detected': [],
                    'severity': 'ERROR',
                    'details': f'OCR processing failed: {str(ocr_process_error)}'
                }
            
            # Extract and combine detected text
            extracted_texts = []
            for result in ocr_results:
                if len(result) >= 2 and result[1]:  # Ensure result has text
                    text = str(result[1]).strip()
                    if text:  # Only add non-empty text
                        extracted_texts.append(text)
            
            extracted_text = ' '.join(extracted_texts)
            print(f"Extracted text: {len(extracted_text)} characters")
            
            if not extracted_text.strip():
                return {
                    'is_intrusion': False,
                    'confidence_score': 0,
                    'threats_detected': [],
                    'severity': 'NONE',
                    'details': f'No text found in image (processed {len(ocr_results)} regions)'
                }
            
            # Analyze extracted text
            result = self.analyze_text(extracted_text)
            result['ocr_info'] = {
                'image_size': image.size,
                'image_mode': image.mode,
                'ocr_regions_found': len(ocr_results),
                'extracted_text_length': len(extracted_text),
                'extracted_text_preview': extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text,
                'processing_method': 'EasyOCR'
            }
            result['details'] = f"OCR Analysis: Extracted {len(extracted_text)} characters from {image.size} image ({len(ocr_results)} text regions)"
            
            return result
            
        except Exception as e:
            import traceback
            error_details = f"Error processing image: {str(e)}"
            traceback_info = traceback.format_exc()
            print(f"{error_details}\nTraceback: {traceback_info}")
            
            return {
                'is_intrusion': False,
                'confidence_score': 0,
                'threats_detected': [],
                'severity': 'ERROR',
                'details': f'Image analysis failed: {error_details}'
            }
    
    def _get_threat_description(self, category):
        """Get description for threat category."""
        descriptions = {
            'sql_injection': 'SQL Injection attack patterns detected',
            'xss_attack': 'Cross-Site Scripting (XSS) patterns detected',
            'command_injection': 'Command injection patterns detected',
            'path_traversal': 'Path traversal attack patterns detected',
            'malware_signatures': 'Malware-related keywords detected',
            'phishing': 'Phishing attempt patterns detected',
            'network_attacks': 'Network attack tool signatures detected',
            'suspicious_activity': 'Suspicious security-related activity detected'
        }
        return descriptions.get(category, 'Unknown threat category')
    
    def _is_higher_severity(self, severity1, severity2):
        """Compare severity levels."""
        severity_order = ['NONE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        try:
            return severity_order.index(severity1) > severity_order.index(severity2)
        except ValueError:
            return False
    
    def get_threat_statistics(self):
        """Get statistics about available threat patterns."""
        stats = {}
        total_patterns = 0
        
        for category, keywords in self.malicious_keywords.items():
            pattern_count = len(keywords)
            stats[category] = {
                'pattern_count': pattern_count,
                'severity': self.severity_levels[category]
            }
            total_patterns += pattern_count
        
        return {
            'total_categories': len(self.malicious_keywords),
            'total_patterns': total_patterns,
            'categories': stats,
            'ocr_available': OCR_AVAILABLE
        }
    
    def add_custom_patterns(self, category, patterns, severity='MEDIUM'):
        """Add custom threat patterns."""
        if category not in self.malicious_keywords:
            self.malicious_keywords[category] = []
            self.severity_levels[category] = severity
        
        # Add new patterns
        self.malicious_keywords[category].extend(patterns)
        
        # Recompile pattern
        pattern = '|'.join(re.escape(keyword.lower()) for keyword in self.malicious_keywords[category])
        self.compiled_patterns[category] = re.compile(pattern, re.IGNORECASE)
        
        return f"Added {len(patterns)} patterns to {category} category"
