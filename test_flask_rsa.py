import requests
import json

def test_rsa_endpoint():
    url = "http://localhost:5000/crypto"
    
    # Test RSA key generation
    print("Testing RSA key generation...")
    for bits in [2048, 3072, 4096]:
        print(f"\nGenerating {bits}-bit RSA key pair...")
        try:
            response = requests.post(
                url,
                data={
                    'key_type': 'RSA',
                    'key_size': bits,
                    'num_keys': 1
                },
                headers={'X-Requested-With': 'XMLHttpRequest'}
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print(f"✓ Successfully generated {bits}-bit RSA keys")
                    print(f"Public key length: {len(result.get('public_key', ''))} chars")
                    print(f"Private key length: {len(result.get('private_key', ''))} chars")
                else:
                    print(f"✗ Failed to generate keys: {result.get('message', 'Unknown error')}")
            else:
                print(f"✗ Request failed with status {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"✗ Error during request: {str(e)}")

if __name__ == "__main__":
    test_rsa_endpoint()
