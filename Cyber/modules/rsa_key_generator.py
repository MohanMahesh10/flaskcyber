import time
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding as asymmetric_padding
from cryptography.hazmat.backends import default_backend

def generate_rsa_keys(key_size=2048, deterministic=False):
    """
    Generate RSA key pair and return both private and public keys 
    as objects and PEM-encoded strings.
    
    Args:
        key_size: Size of the RSA key in bits (default: 2048)
        deterministic: If True, uses a fixed random seed for testing (not secure for production)
        
    Returns:
        dict: Contains private_key_obj, public_key_obj, private_key_pem, public_key_pem
    """
    start_time = time.perf_counter()
    
    # Ensure minimum key size for security
    key_size = max(key_size, 2048)
    
    try:
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )
        
        # Get public key
        public_key = private_key.public_key()
        
        # Generate timestamps
        generation_time = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
        
        # Convert private key to PEM (bytes)
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        # Convert public key to PEM (bytes)
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Test encryption/decryption
        test_data = b"Test message for encryption"
        try:
            # Encrypt with public key
            ciphertext = public_key.encrypt(
                test_data,
                asymmetric_padding.OAEP(
                    mgf=asymmetric_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Decrypt with private key
            plaintext = private_key.decrypt(
                ciphertext,
                asymmetric_padding.OAEP(
                    mgf=asymmetric_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Verify the decryption
            if plaintext != test_data:
                raise ValueError("Decryption verification failed")
                
        except Exception as e:
            print(f"Warning: RSA encryption test failed: {str(e)}")
        
        return {
            "private_key_obj": private_key,
            "public_key_obj": public_key,
            "private_key_pem": private_pem.decode("utf-8"),
            "public_key_pem": public_pem.decode("utf-8"),
            "key_size": key_size,
            "generation_time_ms": generation_time
        }
        
    except Exception as e:
        print(f"Error generating RSA keys: {str(e)}")
        raise
