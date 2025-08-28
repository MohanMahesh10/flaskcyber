print("Starting minimal RSA test...")
try:
    import rsa
    print("1. RSA module imported successfully")
    
    print("2. Generating 512-bit RSA key pair...")
    (pub, priv) = rsa.newkeys(512)
    print("3. Key generation successful!")
    
    message = b"Hello, RSA!"
    print(f"4. Testing with message: {message.decode()}")
    
    encrypted = rsa.encrypt(message, pub)
    print(f"5. Encryption successful. Ciphertext length: {len(encrypted)} bytes")
    
    decrypted = rsa.decrypt(encrypted, priv)
    print(f"6. Decryption successful. Decrypted: {decrypted.decode()}")
    
    if message == decrypted:
        print("\n✓ Test passed! RSA is working correctly.")
    else:
        print("\n✗ Test failed: Decrypted message doesn't match original")
        
except Exception as e:
    print(f"\n✗ Error: {str(e)}")
    import traceback
    print("\nStack trace:")
    traceback.print_exc()

print("\nTest completed.")
