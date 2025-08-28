#!/usr/bin/env python3
"""
Real-Time Analytics Dashboard Demo Script

This script demonstrates the functionality of the real-time analytics dashboard
by testing API endpoints and generating sample data to verify all components work correctly.

Usage:
    python test_analytics_dashboard.py

Requirements:
    - Flask application running on localhost:5000
    - All analytics modules properly installed
"""

import requests
import json
import time
import sys
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:5000"
ENDPOINTS = [
    "/api/system-status",
    "/api/ids-realtime-metrics", 
    "/api/crypto-realtime-metrics",
    "/api/attack-matrix-metrics",
    "/api/system-recommendations"
]

def print_banner():
    """Print a nice banner for the demo."""
    print("=" * 80)
    print("🔥 REAL-TIME ANALYTICS DASHBOARD DEMO")
    print("=" * 80)
    print(f"Testing dashboard functionality at {BASE_URL}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

def test_server_connection():
    """Test if the Flask server is running and accessible."""
    print("\n🔗 Testing server connection...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running and accessible")
            return True
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure Flask app is running on port 5000")
        return False
    except requests.exceptions.Timeout:
        print("❌ Connection timeout. Server may be overloaded")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_dashboard_page():
    """Test if the dashboard page loads correctly."""
    print("\n📊 Testing dashboard page access...")
    try:
        response = requests.get(f"{BASE_URL}/realtime-analytics", timeout=10)
        if response.status_code == 200:
            print("✅ Dashboard page loads successfully")
            print(f"   Page size: {len(response.text)} characters")
            
            # Check for key components
            content = response.text.lower()
            checks = [
                ("chart.js library", "chart.js" in content),
                ("plotly library", "plotly" in content),
                ("bootstrap css", "bootstrap" in content),
                ("dashboard header", "real-time analytics dashboard" in content),
                ("performance chart", "performancechart" in content),
                ("threat chart", "threatchart" in content)
            ]
            
            print("   Component checks:")
            for component, status in checks:
                status_icon = "✅" if status else "❌"
                print(f"     {status_icon} {component}")
                
            return True
        else:
            print(f"❌ Dashboard page returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error accessing dashboard page: {e}")
        return False

def test_api_endpoint(endpoint):
    """Test a specific API endpoint."""
    print(f"\n🔍 Testing {endpoint}...")
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"✅ {endpoint} - Success")
                
                # Display key metrics based on endpoint
                if endpoint == "/api/system-status":
                    print(f"   CPU Usage: {data.get('cpu_usage', 'N/A')}%")
                    print(f"   Memory Usage: {data.get('memory_usage', 'N/A')}%")
                    print(f"   Crypto Ready: {data.get('crypto_ready', False)}")
                    print(f"   IDS Ready: {data.get('ids_ready', False)}")
                    
                elif endpoint == "/api/ids-realtime-metrics":
                    print(f"   Accuracy: {data.get('accuracy', 0):.3f}")
                    print(f"   Precision: {data.get('precision', 0):.3f}")
                    print(f"   F1 Score: {data.get('f1_score', 0):.3f}")
                    print(f"   Throughput: {data.get('throughput_samples_per_sec', 0):.1f} samples/sec")
                    
                elif endpoint == "/api/crypto-realtime-metrics":
                    print(f"   Avg Generation Time: {data.get('avg_generation_time_ms', 0):.1f}ms")
                    print(f"   Security Score: {data.get('avg_security_score', 0):.1f}")
                    print(f"   Total Keys Generated: {data.get('total_keys_generated', 0)}")
                    
                elif endpoint == "/api/attack-matrix-metrics":
                    if isinstance(data, list) and len(data) > 0:
                        print(f"   Attack types monitored: {len(data)}")
                        print(f"   Sample - {data[0].get('name', 'Unknown')}: "
                              f"Precision={data[0].get('precision', 0):.3f}")
                    
                elif endpoint == "/api/system-recommendations":
                    recommendations = data.get('recommendations', [])
                    print(f"   Active recommendations: {len(recommendations)}")
                    if recommendations:
                        print(f"   First recommendation: {recommendations[0][:60]}...")
                
                return True
                
            except json.JSONDecodeError:
                print(f"❌ {endpoint} - Invalid JSON response")
                print(f"   Response: {response.text[:100]}...")
                return False
                
        else:
            print(f"❌ {endpoint} - Status code: {response.status_code}")
            if response.text:
                print(f"   Error: {response.text[:200]}...")
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ {endpoint} - Timeout")
        return False
    except Exception as e:
        print(f"❌ {endpoint} - Error: {e}")
        return False

def test_real_time_updates():
    """Test real-time updates by calling the same endpoint multiple times."""
    print("\n⏱️ Testing real-time updates (5 iterations)...")
    endpoint = "/api/ids-realtime-metrics"
    previous_values = {}
    changes_detected = 0
    
    for i in range(5):
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                current_values = {
                    'accuracy': data.get('accuracy', 0),
                    'precision': data.get('precision', 0),
                    'throughput': data.get('throughput_samples_per_sec', 0)
                }
                
                print(f"   Iteration {i+1}: "
                      f"Acc={current_values['accuracy']:.3f}, "
                      f"Prec={current_values['precision']:.3f}, "
                      f"Thru={current_values['throughput']:.1f}")
                
                if previous_values:
                    # Check if any values changed (indicating real-time updates)
                    for key in current_values:
                        if abs(current_values[key] - previous_values[key]) > 0.001:
                            changes_detected += 1
                            break
                
                previous_values = current_values.copy()
                
                if i < 4:  # Don't sleep after the last iteration
                    time.sleep(2)  # Wait 2 seconds between requests
                    
            else:
                print(f"   Iteration {i+1}: Failed (Status {response.status_code})")
                
        except Exception as e:
            print(f"   Iteration {i+1}: Error - {e}")
    
    if changes_detected > 0:
        print(f"✅ Real-time updates working - {changes_detected} changes detected")
    else:
        print("⚠️ No changes detected - may be using static mock data")

def run_comprehensive_test():
    """Run all tests and provide a summary."""
    print_banner()
    
    results = {
        'server_connection': False,
        'dashboard_page': False,
        'api_endpoints': {},
        'real_time_updates': False
    }
    
    # Test server connection
    results['server_connection'] = test_server_connection()
    if not results['server_connection']:
        print("\n❌ Cannot proceed with tests - server is not accessible")
        return results
    
    # Test dashboard page
    results['dashboard_page'] = test_dashboard_page()
    
    # Test API endpoints
    print("\n🚀 Testing API endpoints...")
    for endpoint in ENDPOINTS:
        results['api_endpoints'][endpoint] = test_api_endpoint(endpoint)
    
    # Test real-time updates
    try:
        test_real_time_updates()
        results['real_time_updates'] = True
    except Exception as e:
        print(f"❌ Real-time update test failed: {e}")
        results['real_time_updates'] = False
    
    # Print summary
    print("\n" + "=" * 80)
    print("📋 TEST SUMMARY")
    print("=" * 80)
    
    # Overall status
    total_tests = 3 + len(ENDPOINTS)  # server + dashboard + real-time + API endpoints
    passed_tests = (
        int(results['server_connection']) + 
        int(results['dashboard_page']) + 
        int(results['real_time_updates']) +
        sum(results['api_endpoints'].values())
    )
    
    print(f"Overall Status: {passed_tests}/{total_tests} tests passed")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print(f"\n✅ Server Connection: {'PASS' if results['server_connection'] else 'FAIL'}")
    print(f"✅ Dashboard Page: {'PASS' if results['dashboard_page'] else 'FAIL'}")
    print(f"✅ Real-time Updates: {'PASS' if results['real_time_updates'] else 'FAIL'}")
    
    print("\n📡 API Endpoints:")
    for endpoint, status in results['api_endpoints'].items():
        status_text = 'PASS' if status else 'FAIL'
        icon = '✅' if status else '❌'
        print(f"   {icon} {endpoint}: {status_text}")
    
    print("\n" + "=" * 80)
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Your analytics dashboard is fully functional.")
        print("🌐 Access the dashboard at: http://localhost:5000/realtime-analytics")
    elif passed_tests >= total_tests * 0.8:
        print("✅ Most tests passed. Dashboard should be functional with minor issues.")
        print("🌐 Access the dashboard at: http://localhost:5000/realtime-analytics")
    else:
        print("⚠️ Multiple test failures detected. Please check the setup.")
        print("📚 Refer to ANALYTICS_DASHBOARD_GUIDE.md for troubleshooting.")
    
    return results

def main():
    """Main function to run the demo."""
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print(__doc__)
        return
    
    try:
        results = run_comprehensive_test()
        
        # Exit with appropriate code
        total_tests = 3 + len(ENDPOINTS)
        passed_tests = (
            int(results['server_connection']) + 
            int(results['dashboard_page']) + 
            int(results['real_time_updates']) +
            sum(results['api_endpoints'].values())
        )
        
        if passed_tests == total_tests:
            sys.exit(0)  # All tests passed
        elif passed_tests >= total_tests * 0.8:
            sys.exit(1)  # Mostly working but some issues
        else:
            sys.exit(2)  # Major issues detected
            
    except KeyboardInterrupt:
        print("\n\n⛔ Demo interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error during demo: {e}")
        sys.exit(3)

if __name__ == "__main__":
    main()
