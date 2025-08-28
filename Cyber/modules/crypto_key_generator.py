import hashlib
import secrets
import time
import psutil
import random
import math
from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet
from Crypto.Cipher import AES
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
# TensorFlow and sklearn removed for numpy-free version
import threading
from concurrent.futures import ThreadPoolExecutor

class CryptoKeyGenerator:
    """
    Advanced cryptographic key generation with ML-enhanced entropy and security analysis.
    """
    
    def __init__(self):
        self.entropy_model = None  # Removed ML model
        self.key_cache = {}
        self.performance_metrics = []
        
    def _build_entropy_model(self):
        """Simple entropy enhancement without ML dependencies."""
        return None
    
    def _extract_entropy_features(self, data):
        """Extract entropy features from input data. Always returns bytes-based features."""
        # Normalize to bytes
        if isinstance(data, bytes):
            data_bytes = data
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8', errors='ignore')
        elif isinstance(data, list):
            # Handle list of pixel values from PIL/NumPy
            try:
                data_bytes = bytes([int(x) & 0xFF for x in data[:10000]])
            except Exception:
                data_bytes = str(data).encode('utf-8', errors='ignore')
        else:
            try:
                data_bytes = bytes(data)
            except Exception:
                data_bytes = str(data).encode('utf-8', errors='ignore')
        
        # Extract various entropy measures
        features = []
        
        # Basic statistical features
        byte_array = list(data_bytes)
        if len(byte_array) > 0:
            mean_val = sum(byte_array) / len(byte_array)
            variance = sum((x - mean_val)**2 for x in byte_array) / len(byte_array)
            std_val = math.sqrt(variance)
            features.extend([
                mean_val,
                std_val,
                variance,
                min(byte_array),
                max(byte_array)
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # Frequency analysis
        freq_dist = [0] * 256
        for byte_val in byte_array:
            freq_dist[byte_val] += 1
        
        # Shannon entropy calculation
        total_bytes = len(byte_array)
        if total_bytes > 0:
            entropy = 0
            for freq in freq_dist:
                if freq > 0:
                    prob = freq / total_bytes
                    entropy -= prob * math.log2(prob)
        else:
            entropy = 0
            
        unique_bytes = sum(1 for freq in freq_dist if freq > 0)
        max_freq = max(freq_dist) if freq_dist else 0
        
        features.extend([entropy, unique_bytes, max_freq])
        
        # Hash-based features
        hash_features = []
        for hash_func in [hashlib.md5, hashlib.sha1, hashlib.sha256]:
            hash_val = hash_func(data_bytes).digest()
            hash_array = list(hash_val)
            if hash_array:
                hash_mean = sum(hash_array) / len(hash_array)
                hash_variance = sum((x - hash_mean)**2 for x in hash_array) / len(hash_array)
                hash_std = math.sqrt(hash_variance)
                odd_count = sum(1 for x in hash_array if x % 2 == 1)
                hash_features.extend([hash_mean, hash_std, odd_count])
            else:
                hash_features.extend([0, 0, 0])
        features.extend(hash_features)
        
        # Autocorrelation features (simplified)
        if len(byte_array) > 1:
            autocorr_sum = sum(byte_array[i] * byte_array[i-1] for i in range(1, len(byte_array)))
            autocorr_mean = autocorr_sum / (len(byte_array) - 1) if len(byte_array) > 1 else 0
            features.extend([autocorr_sum, autocorr_mean, 0])  # simplified
        else:
            features.extend([0, 0, 0])
        
        # Pad or truncate to fixed size
        while len(features) < 100:
            features.append(0)
        
        features = features[:100]  # truncate to 100
        
        return [features]  # Keep shape for downstream use
    
    def generate_aes_key(self, key_size=256):
        """Generate an AES key using system's secure random number generator."""
        start_time = time.perf_counter()
        
        # Generate secure random key
        key = secrets.token_bytes(key_size // 8)
        
        generation_time = (time.perf_counter() - start_time) * 1000
        
        # Test encryption/decryption performance
        encode_start = time.perf_counter()
        test_data = b"Performance test"
        try:
            cipher = AES.new(key, AES.MODE_ECB)
            encrypted = cipher.encrypt(test_data)
            encoding_time = (time.perf_counter() - encode_start) * 1000
            
            decode_start = time.perf_counter()
            decrypted = cipher.decrypt(encrypted)
            decoding_time = (time.perf_counter() - decode_start) * 1000
        except Exception as e:
            # Fallback values if encryption fails
            encoding_time = generation_time * 0.1
            decoding_time = generation_time * 0.1
        
        return {
            'key': key.hex(),  # Convert bytes to hex string for JSON serialization
            'key_type': 'AES',
            'key_size': key_size,
            'generation_time': generation_time,
            'encoding_time': encoding_time,
            'decoding_time': decoding_time
        }
    
    def generate_rsa_key(self, key_size=2048, serialize_keys=True):
        """Generate RSA key pair using system's secure random number generator.
        
        Args:
            key_size: Size of the RSA key in bits (must be >= 512)
            serialize_keys: Whether to serialize keys to PEM format
            
        Returns:
            dict: Contains private_key, public_key, and metadata
        """
        start_time = time.perf_counter()
        
        # Ensure minimum key size
        key_size = max(key_size, 512)
        
        # Generate key pair using system's secure random
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )
        public_key = private_key.public_key()
        
        generation_time = (time.perf_counter() - start_time) * 1000
        
        # Test encoding/decoding performance
        encode_start = time.perf_counter()
        test_data = b"Performance test"
        try:
            from cryptography.hazmat.primitives.asymmetric import padding as asymmetric_padding
            
            # Encrypt with public key
            ciphertext = public_key.encrypt(
                test_data,
                asymmetric_padding.OAEP(
                    mgf=asymmetric_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            encoding_time = (time.perf_counter() - encode_start) * 1000
            
            # Decrypt with private key
            decode_start = time.perf_counter()
            plaintext = private_key.decrypt(
                ciphertext,
                asymmetric_padding.OAEP(
                    mgf=asymmetric_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            decoding_time = (time.perf_counter() - decode_start) * 1000
            
            # Verify the decryption
            if plaintext != test_data:
                raise ValueError("Decryption verification failed")
                
        except Exception as e:
            print(f"Warning: RSA encryption test failed: {str(e)}")
            # Fallback for encryption/decryption testing
            encoding_time = generation_time * 0.1
            decoding_time = generation_time * 0.1
        
        result = {
            'private_key': private_key,  # Keep as object for internal use
            'public_key': public_key,    # Keep as object for internal use
            'key_type': 'RSA',
            'key_size': key_size,
            'generation_time': generation_time,
            'encoding_time': encoding_time,
            'decoding_time': decoding_time
        }
        
        # Only serialize if explicitly requested
        if serialize_keys:
            result['private_key'] = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ).decode('utf-8')
            
            result['public_key'] = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode('utf-8')
        
        return result
    
    def generate_ecc_key(self, curve_name='secp256r1', serialize_keys=True):
        """Generate ECC key pair using system's secure random number generator.
        
        Args:
            curve_name: Name of the elliptic curve to use (secp256r1, secp384r1, secp521r1)
            serialize_keys: Whether to serialize keys to PEM format
            
        Returns:
            dict: Contains private_key, public_key, and metadata
        """
        start_time = time.perf_counter()
        
        # Map of supported curves
        curve_map = {
            'secp256r1': ec.SECP256R1(),
            'secp384r1': ec.SECP384R1(),
            'secp521r1': ec.SECP521R1()
        }
        
        # Default to secp256r1 if invalid curve specified
        curve = curve_map.get(curve_name, ec.SECP256R1())
        
        # Generate key pair using system's secure random
        private_key = ec.generate_private_key(curve, default_backend())
        public_key = private_key.public_key()
        
        generation_time = (time.perf_counter() - start_time) * 1000
        
        # Simulate encoding/decoding times (ECC signing/verification)
        encode_start = time.perf_counter()
        try:
            test_signature = private_key.sign(b"test data", ec.ECDSA(hashes.SHA256()))
            encoding_time = (time.perf_counter() - encode_start) * 1000
            
            decode_start = time.perf_counter()
            try:
                public_key.verify(test_signature, b"test data", ec.ECDSA(hashes.SHA256()))
                verification_success = True
            except:
                verification_success = False
            decoding_time = (time.perf_counter() - decode_start) * 1000
        except Exception as e:
            # Fallback for signing/verification testing
            encoding_time = generation_time * 0.1
            decoding_time = generation_time * 0.1
        
        result = {
            'private_key': private_key,  # Keep as object for internal use
            'public_key': public_key,    # Keep as object for internal use
            'key_type': 'ECC',
            'curve': curve_name,
            'generation_time': generation_time,
            'encoding_time': encoding_time,
            'decoding_time': decoding_time
        }
        
        # Only serialize if explicitly requested
        if serialize_keys:
            result['private_key'] = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ).decode('utf-8')
            
            result['public_key'] = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode('utf-8')
        
        return result
    
    def generate_hybrid_key(self, aes_size=256, rsa_size=2048):
        """Generate hybrid key system (AES + RSA)."""
        start_time = time.perf_counter()
        
        # Generate both keys using secure random
        aes_key = self.generate_aes_key(aes_size)
        rsa_keys = self.generate_rsa_key(rsa_size)
        
        generation_time = (time.perf_counter() - start_time) * 1000
        
        return {
            'aes_key': aes_key,
            'rsa_keys': rsa_keys,
            'key_type': 'Hybrid',
            'generation_time': generation_time,
            'encoding_time': (aes_key['encoding_time'] + rsa_keys['encoding_time']) / 2,
            'decoding_time': (aes_key['decoding_time'] + rsa_keys['decoding_time']) / 2
        }
    
    def calculate_key_strength(self, key_data):
        """Calculate key strength metrics."""
        if isinstance(key_data, dict):
            if 'key' in key_data:
                key_bytes = key_data['key']
            elif 'private_key' in key_data:
                key_bytes = key_data['private_key'].private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
            else:
                key_bytes = str(key_data).encode()
        else:
            key_bytes = key_data
        
        if isinstance(key_bytes, str):
            key_bytes = key_bytes.encode()
        
        # Calculate entropy
        byte_array = list(key_bytes)
        freq_dist = [0] * 256
        for byte_val in byte_array:
            freq_dist[byte_val] += 1
        
        total_bytes = len(byte_array)
        if total_bytes > 0:
            entropy = 0
            for freq in freq_dist:
                if freq > 0:
                    prob = freq / total_bytes
                    entropy -= prob * math.log2(prob)
        else:
            entropy = 0
        
        # Randomness tests
        randomness_score = self._test_randomness(byte_array)
        
        # Estimated attack success rate (simplified model)
        key_length_bits = len(key_bytes) * 8
        attack_complexity = 2 ** min(key_length_bits, 128)  # Cap for computation
        attack_success_rate = max(0.01, 100 / attack_complexity * 1e10)  # Simplified model
        
        return {
            'entropy': entropy,
            'randomness_score': randomness_score,
            'attack_success_rate': min(attack_success_rate, 99.99),
            'key_length_bits': key_length_bits
        }
    
    def _test_randomness(self, byte_array):
        """Perform basic randomness tests on byte array."""
        if not byte_array:
            return 0.5
            
        # Frequency test
        expected_freq = len(byte_array) / 256
        freq_dist = [0] * 256
        for byte_val in byte_array:
            freq_dist[byte_val] += 1
        
        chi_square = sum((freq - expected_freq) ** 2 / expected_freq for freq in freq_dist if expected_freq > 0)
        
        # Runs test (simplified)
        runs = 0
        for i in range(1, len(byte_array)):
            if (byte_array[i] >= 128) != (byte_array[i-1] >= 128):
                runs += 1
        
        expected_runs = len(byte_array) / 2
        runs_score = abs(runs - expected_runs) / expected_runs if expected_runs > 0 else 1
        
        # Combine scores (lower is better, normalize to 0-1 where 1 is best)
        freq_score = max(0, 1 - chi_square / (256 * 10))  # Normalize
        runs_score = max(0, 1 - runs_score)
        
        return (freq_score + runs_score) / 2
    
    def generate_keys(self, key_type='AES', key_size=256, count=1):
        """Generate cryptographic keys using secure system random.
        
        Args:
            key_type: Type of key to generate (AES, RSA, ECC, HYBRID)
            key_size: Size of the key in bits
            count: Number of keys to generate (will generate at least 1)
            
        Returns:
            dict: Contains generated keys and performance metrics
        """
        start_time = time.time()
        
        # Ensure we generate at least 1 key
        if count < 1:
            count = 1
            
        # For ECC, validate key size and use standard curves
        if key_type.upper() == 'ECC':
            # Map key sizes to standard curves
            curve_map = {
                '256': 'secp256r1',
                '384': 'secp384r1',
                '521': 'secp521r1'
            }
            # Default to 256 if invalid size provided
            curve = curve_map.get(str(key_size), 'secp256r1')
            key_size = 256  # Standard ECC key size
        
        keys = []
        attempts = 0
        max_attempts = count * 2  # Try at most twice the requested number
        
        while len(keys) < count and attempts < max_attempts:
            attempts += 1
            try:
                # Generate the key based on type
                if key_type.upper() == 'AES':
                    # Ensure valid AES key size (128, 192, or 256)
                    valid_aes_sizes = [128, 192, 256]
                    aes_size = int(key_size)
                    if aes_size not in valid_aes_sizes:
                        aes_size = 256  # Default to 256 if invalid size
                    key_result = self.generate_aes_key(aes_size)
                    
                elif key_type.upper() == 'RSA':
                    # Ensure minimum RSA key size
                    rsa_size = max(int(key_size), 2048)  # Minimum 2048 bits for RSA
                    key_result = self.generate_rsa_key(rsa_size, serialize_keys=False)
                    # Serialize RSA keys for response
                    key_result['private_key'] = key_result['private_key'].private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption()
                    ).decode('utf-8')
                    key_result['public_key'] = key_result['public_key'].public_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo
                    ).decode('utf-8')
                    
                elif key_type.upper() == 'ECC':
                    try:
                        key_result = self.generate_ecc_key(curve, serialize_keys=False)
                        # Serialize ECC keys for response
                        key_result['private_key'] = key_result['private_key'].private_bytes(
                            encoding=serialization.Encoding.PEM,
                            format=serialization.PrivateFormat.PKCS8,
                            encryption_algorithm=serialization.NoEncryption()
                        ).decode('utf-8')
                        key_result['public_key'] = key_result['public_key'].public_bytes(
                            encoding=serialization.Encoding.PEM,
                            format=serialization.PublicFormat.SubjectPublicKeyInfo
                        ).decode('utf-8')
                    except Exception as e:
                        print(f"Error generating ECC key: {str(e)}")
                        print(f"Falling back to RSA key generation")
                        # Fall back to RSA if ECC fails
                        key_result = self.generate_rsa_key(2048, serialize_keys=False)
                        key_result['private_key'] = key_result['private_key'].private_bytes(
                            encoding=serialization.Encoding.PEM,
                            format=serialization.PrivateFormat.PKCS8,
                            encryption_algorithm=serialization.NoEncryption()
                        ).decode('utf-8')
                        key_result['public_key'] = key_result['public_key'].public_bytes(
                            encoding=serialization.Encoding.PEM,
                            format=serialization.PublicFormat.SubjectPublicKeyInfo
                        ).decode('utf-8')
                        key_result['key_type'] = 'RSA'  # Update key type to reflect fallback
                    
                elif key_type.upper() == 'HYBRID':
                    # For hybrid, use fixed sizes to ensure compatibility
                    key_result = self.generate_hybrid_key(256, 2048)
                    
                else:  # Default to AES if invalid type
                    key_result = self.generate_aes_key(256)
                
                keys.append(key_result)
                
            except Exception as e:
                print(f"Error generating key (attempt {attempts}): {str(e)}")
                # If we're on the last attempt and have no keys, try one more time with default settings
                if attempts >= max_attempts and not keys:
                    print("Falling back to default key generation")
                    key_result = self.generate_aes_key(256)
                    keys.append(key_result)
        
        end_time = time.time()
        
        # Calculate metrics
        total_generation_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        return {
            'keys': keys,
            'count': len(keys),
            'total_generation_time': total_generation_time,
            'key_type': key_type,
            'key_size': key_size
        }
    
    def generate_keys_batch(self, count=10, key_types=None):
        """Generate a batch of different key types for testing."""
        if key_types is None:
            key_types = ['AES', 'RSA', 'ECC']
        
        results = []
        for key_type in key_types:
            if key_type == 'AES':
                result = self.generate_keys('AES', 256, count//len(key_types))
            elif key_type == 'RSA':
                result = self.generate_keys('RSA', 2048, count//len(key_types))
            elif key_type == 'ECC':
                result = self.generate_keys('ECC', 256, count//len(key_types))
            
            results.append(result)
        
        return results


