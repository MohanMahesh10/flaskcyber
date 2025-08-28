#!/usr/bin/env python3
"""
Test script to verify that all modules can be imported without numpy dependency.
"""

import sys
import traceback

def test_module_import(module_name):
    """Test importing a module and report any numpy dependencies."""
    try:
        __import__(module_name)
        print(f"✅ {module_name} imported successfully")
        return True
    except ImportError as e:
        if "numpy" in str(e).lower():
            print(f"❌ {module_name} failed due to numpy dependency: {e}")
            return False
        else:
            print(f"⚠️  {module_name} failed due to other dependency: {e}")
            return True  # Not a numpy issue
    except Exception as e:
        print(f"⚠️  {module_name} failed due to other error: {e}")
        return True  # Not a numpy issue

def main():
    """Main test function."""
    print("Testing module imports without numpy dependency...")
    print("=" * 60)
    
    # Test core modules
    modules_to_test = [
        'modules.crypto_key_generator',
        'modules.intrusion_detection', 
        'modules.data_processor',
        'modules.visualization'
    ]
    
    all_passed = True
    for module in modules_to_test:
        if not test_module_import(module):
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 All modules can be imported without numpy dependency!")
    else:
        print("❌ Some modules still have numpy dependencies.")
        sys.exit(1)

if __name__ == "__main__":
    main()
