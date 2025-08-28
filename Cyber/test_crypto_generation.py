#!/usr/bin/env python3
"""
Test script to verify the crypto generation works independently
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from modules.crypto_key_generator import CryptoKeyGenerator
import json

def test_crypto_generation():
    print("Testing crypto key generation...")
    
    # Initialize the generator
    crypto_generator = CryptoKeyGenerator()
    
    # Test parameters
    test_cases = [
        {'key_type': 'AES', 'key_size': 256, 'count': 2},
        {'key_type': 'RSA', 'key_size': 2048, 'count': 1},
        {'key_type': 'ECC', 'key_size': 256, 'count': 1},
        {'key_type': 'Hybrid', 'key_size': 256, 'count': 1},
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {test_case}")
        
        try:
            results = crypto_generator.generate_keys_with_ml(
                key_type=test_case['key_type'],
                key_size=test_case['key_size'],
                count=test_case['count'],
                input_data="Sample test data for entropy",
                use_ml=True
            )
            
            print(f"Results keys: {results.get('keys', [])}")
            print(f"Number of keys generated: {len(results.get('keys', []))}")
            
            # Test what the Flask API would send
            flask_response = {
                'success': True,
                'message': f'Generated {test_case["count"]} keys successfully!',
                'results': {
                    'count': test_case['count'],
                    'generation_time': results.get('total_generation_time', 0),
                    'encoding_time': results.get('encoding_time', 0),
                    'decoding_time': results.get('decoding_time', 0),
                    'cpu_cycles': results.get('cpu_cycles', 0),
                    'entropy_score': results.get('entropy_score', 0),
                    'attack_success_rate': results.get('attack_success_rate', 0),
                    'keys': results.get('keys', [])  # This should contain the keys
                }
            }
            
            print(f"Flask response would contain {len(flask_response['results']['keys'])} keys")
            
            # Show the first key structure if available
            if flask_response['results']['keys']:
                first_key = flask_response['results']['keys'][0]
                print(f"First key structure:")
                print(f"  - key_type: {first_key.get('key_type', 'MISSING')}")
                print(f"  - key_size: {first_key.get('key_size', 'MISSING')}")
                if first_key.get('key_type') == 'AES':
                    print(f"  - key: {first_key.get('key', 'MISSING')[:50]}...")
                elif first_key.get('key_type') in ['RSA', 'ECC']:
                    print(f"  - private_key: {str(first_key.get('private_key', 'MISSING'))[:50]}...")
                    print(f"  - public_key: {str(first_key.get('public_key', 'MISSING'))[:50]}...")
            
        except Exception as e:
            print(f"ERROR in test case: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_crypto_generation()
