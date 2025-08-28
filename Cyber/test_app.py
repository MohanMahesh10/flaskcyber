#!/usr/bin/env python3
"""
Test script to verify Flask app functionality
Run this to check if all components are working correctly
"""

def test_imports():
    """Test if all required modules can be imported."""
    try:
        print("Testing imports...")
        
        # Test Flask and basic libraries
        from flask import Flask, render_template, request, jsonify
        print("✅ Flask imports successful")
        
        # Test our custom modules
        from modules.crypto_key_generator import CryptoKeyGenerator
        print("✅ CryptoKeyGenerator import successful")
        
        from modules.intrusion_detection import IntrusionDetectionSystem
        print("✅ IntrusionDetectionSystem import successful")
        
        from modules.data_processor import DataProcessor
        print("✅ DataProcessor import successful")
        
        from modules.visualization_flask import VisualizerFlask
        print("✅ VisualizerFlask import successful")
        
        from utils.metrics_flask import MetricsCalculatorFlask
        print("✅ MetricsCalculatorFlask import successful")
        
        print("\n🎉 All imports successful! The Flask app should work correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nPlease install missing dependencies using:")
        print("pip install -r requirements_flask.txt")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_functionality():
    """Test basic functionality of each module."""
    try:
        print("\nTesting basic functionality...")
        
        # Test crypto key generator
        crypto_gen = CryptoKeyGenerator()
        test_key = crypto_gen.generate_aes_key(256, "test data", False)
        print(f"✅ Crypto key generation: {test_key['key_type']} key generated")
        
        # Test IDS system
        ids_system = IntrusionDetectionSystem()
        test_data = [{"test": "data"}]
        results = ids_system.predict_batch(test_data)
        print(f"✅ IDS system: {results['total_samples']} samples processed")
        
        # Test data processor
        data_proc = DataProcessor()
        sample_data, labels = data_proc.generate_ids_training_data(100)
        print(f"✅ Data processor: Generated {len(sample_data)} training samples")
        
        # Test visualizer
        visualizer = VisualizerFlask()
        chart_data = visualizer.get_model_comparison_data()
        print(f"✅ Visualizer: Generated chart data with {len(chart_data['datasets'])} datasets")
        
        # Test metrics calculator
        metrics_calc = MetricsCalculatorFlask()
        y_true = [0, 1, 0, 1, 0]
        y_pred = [0, 1, 1, 1, 0]
        metrics = metrics_calc.calculate_classification_metrics(y_true, y_pred)
        print(f"✅ Metrics calculator: Accuracy = {metrics['accuracy']:.3f}")
        
        print("\n🚀 All functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("🔐 ML-Enhanced Cybersecurity System - Flask Version")
    print("Testing Application Components")
    print("=" * 60)
    
    # Test imports
    if not test_imports():
        return
    
    # Test functionality
    if not test_functionality():
        return
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("Your Flask application is ready to run!")
    print("\nTo start the application:")
    print("1. Install dependencies: pip install -r requirements_flask.txt")
    print("2. Run the app: python app_flask.py")
    print("3. Open browser: http://localhost:5000")
    print("=" * 60)

if __name__ == "__main__":
    main()
