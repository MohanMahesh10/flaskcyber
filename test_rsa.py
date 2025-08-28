import rsa
import time

def test_rsa_generation():
    print("Testing RSA key generation...")
    start_time = time.time()
    
    try:
        # Try different key sizes
        for bits in [2048, 3072, 4096]:
            print(f"\nGenerating {bits}-bit RSA key pair...")
            start = time.time()
            
            # Generate keys
            (pubkey, privkey) = rsa.newkeys(bits)
            
            # Get keys in PEM format
            pubkey_pem = pubkey.save_pkcs1().decode('utf-8')
            privkey_pem = privkey.save_pkcs1().decode('utf-8')
            
            # Print some info
            print(f"Generated {bits}-bit keys in {time.time() - start:.2f} seconds")
            print(f"Public key length: {len(pubkey_pem)} characters")
            print(f"Private key length: {len(privkey_pem)} characters")
            
            # Test encryption/decryption
            message = "Test message".encode('utf-8')
            encrypted = rsa.encrypt(message, pubkey)
            decrypted = rsa.decrypt(encrypted, privkey)
            
            if decrypted == message:
                print("✓ Encryption/decryption test passed")
            else:
                print("✗ Encryption/decryption test failed")
                
    except Exception as e:
        print(f"Error during RSA key generation: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\nTotal test time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    test_rsa_generation()
