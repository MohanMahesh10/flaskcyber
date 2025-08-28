"""
Simple, reliable RSA key generation with minimal dependencies.
"""
import time
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

def generate_rsa_keypair(key_size=2048):
    """
    Generate an RSA key pair with the specified key size.
    
    Args:
        key_size: Size of the RSA key in bits (default: 2048)
        
    Returns:
        dict: Contains private and public keys in PEM format
    """
    # Ensure key size is at least 1024 bits for security
    key_size = max(1024, key_size)
    
    try:
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )
        
        # Get public key
        public_key = private_key.public_key()
        
        # Serialize private key to PEM format
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode('utf-8')
        
        # Serialize public key to PEM format
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
        
        return {
            'success': True,
            'private_key': private_pem,
            'public_key': public_pem,
            'key_size': key_size,
            'key_type': 'RSA',
            'timestamp': int(time.time())
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'key_size': key_size,
            'key_type': 'RSA'
        }
