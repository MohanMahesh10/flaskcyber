import rsa
import time
import sys
import os

def test_rsa():
    print("Testing RSA key generation...")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"RSA module path: {os.path.dirname(rsa.__file__)}")
    
    try:
        print("\n1. Testing small key generation (512 bits for test)...")
        start_time = time.time()
        (pubkey, privkey) = rsa.newkeys(512)  # Using smaller key for test
        print(f"✓ Small key generation successful! Took {time.time() - start_time:.2f} seconds")
        
        # Test encryption/decryption with small key
        message = b"Test message"
        print("\n2. Testing encryption with small key...")
        encrypted = rsa.encrypt(message, pubkey)
        print(f"✓ Encryption successful. Ciphertext length: {len(encrypted)} bytes")
        
        print("3. Testing decryption with small key...")
        decrypted = rsa.decrypt(encrypted, privkey)
        
        if message == decrypted:
            print("✓ Decryption successful! Original message recovered.")
        else:
            print("✗ Decryption failed: Message mismatch")
        
        # Now test with 2048 bits
        print("\n4. Testing 2048-bit key generation...")
        start_time = time.time()
        (pubkey, privkey) = rsa.newkeys(2048)
        print(f"✓ 2048-bit key generation successful! Took {time.time() - start_time:.2f} seconds")
        
        print("5. Testing encryption with 2048-bit key...")
        encrypted = rsa.encrypt(message, pubkey)
        print(f"✓ Encryption successful. Ciphertext length: {len(encrypted)} bytes")
        
        print("6. Testing decryption with 2048-bit key...")
        decrypted = rsa.decrypt(encrypted, privkey)
        
        if message == decrypted:
            print("✓ Decryption successful! Original message recovered.")
        else:
            print("✗ Decryption failed: Message mismatch")
        
    except Exception as e:
        print(f"\n✗ Error during test: {str(e)}")
        import traceback
        print("\nStack trace:")
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting RSA test...")
    test_rsa()
    print("\nTest completed.")
