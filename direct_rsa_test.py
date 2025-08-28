import rsa
import time

def generate_rsa_key(bits=2048):
    print(f"Generating {bits}-bit RSA key pair...")
    start_time = time.time()
    
    try:
        # Generate RSA keys
        (pubkey, privkey) = rsa.newkeys(bits)
        
        # Get keys in PEM format
        pubkey_pem = pubkey.save_pkcs1().decode('utf-8')
        privkey_pem = privkey.save_pkcs1().decode('utf-8')
        
        print(f"Successfully generated {bits}-bit RSA keys in {time.time() - start_time:.2f} seconds")
        print(f"Public key (first 50 chars): {pubkey_pem[:50]}...")
        print(f"Private key (first 50 chars): {privkey_pem[:50]}...")
        
        # Test encryption/decryption
        message = b"Test message for RSA encryption"
        encrypted = rsa.encrypt(message, pubkey)
        decrypted = rsa.decrypt(encrypted, privkey)
        
        if decrypted == message:
            print("✓ Encryption/decryption test passed")
        else:
            print("✗ Encryption/decryption test failed")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test with different key sizes
    for bits in [2048, 3072, 4096]:
        print("\n" + "="*50)
        generate_rsa_key(bits)
        print("="*50 + "\n")
