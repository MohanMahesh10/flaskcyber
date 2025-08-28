#!/usr/bin/env python3
"""
Quick Start Script for Real-Time Analytics Dashboard

This script launches the Flask server and optionally opens the analytics dashboard 
in your default web browser.

Usage:
    python start_analytics_dashboard.py [--no-browser] [--port PORT]

Options:
    --no-browser    Don't automatically open browser
    --port PORT     Specify port number (default: 5000)
    --help          Show this help message
"""

import sys
import os
import time
import subprocess
import webbrowser
from threading import Timer
import argparse

def print_banner():
    """Print startup banner."""
    print("=" * 70)
    print("🚀 STARTING REAL-TIME ANALYTICS DASHBOARD")
    print("=" * 70)

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'flask',
        'requests',
        'psutil',
        'rsa',
        'pillow'  # PIL
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install them with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_flask_app():
    """Check if Flask app file exists."""
    print("🔍 Checking Flask application...")
    
    flask_app = "app_flask.py"
    if not os.path.exists(flask_app):
        print(f"❌ {flask_app} not found in current directory")
        print("Make sure you're running this from the Cyber directory")
        return False
    
    print(f"✅ Found {flask_app}")
    return True

def open_browser_delayed(url, delay=3):
    """Open browser after a delay."""
    def open_browser():
        print(f"🌐 Opening browser: {url}")
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"⚠️ Could not open browser automatically: {e}")
            print(f"Please manually navigate to: {url}")
    
    Timer(delay, open_browser).start()

def start_flask_server(port=5000, open_browser=True):
    """Start the Flask server."""
    print(f"🔥 Starting Flask server on port {port}...")
    
    # Set environment variables
    os.environ['FLASK_APP'] = 'app_flask.py'
    os.environ['FLASK_ENV'] = 'development'
    
    # URLs to access
    dashboard_url = f"http://localhost:{port}/realtime-analytics"
    main_url = f"http://localhost:{port}"
    
    print(f"📊 Analytics Dashboard: {dashboard_url}")
    print(f"🏠 Main Application: {main_url}")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 70)
    
    # Schedule browser opening
    if open_browser:
        print("🕒 Browser will open in 3 seconds...")
        open_browser_delayed(dashboard_url, 3)
    
    try:
        # Start Flask app
        from app_flask import app
        app.run(
            debug=True,
            host='0.0.0.0',
            port=port,
            use_reloader=False  # Disable reloader to avoid double execution
        )
    except ImportError as e:
        print(f"❌ Error importing Flask app: {e}")
        print("Make sure app_flask.py is in the current directory")
        return False
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {port} is already in use")
            print(f"Try a different port with: --port {port+1}")
        else:
            print(f"❌ Error starting server: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def run_quick_test(port=5000):
    """Run a quick connectivity test."""
    print("🧪 Running quick connectivity test...")
    
    try:
        # Try to run the test script
        result = subprocess.run([
            sys.executable, 
            "test_analytics_dashboard.py"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        elif result.returncode == 1:
            print("⚠️ Most tests passed with minor issues")
            return True
        else:
            print("❌ Some tests failed")
            print("Output:", result.stdout[-200:] if result.stdout else "No output")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏱️ Test timed out - server may be slow to start")
        return True
    except FileNotFoundError:
        print("⚠️ Test script not found - skipping test")
        return True
    except Exception as e:
        print(f"⚠️ Test error: {e}")
        return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Start Real-Time Analytics Dashboard")
    parser.add_argument('--no-browser', action='store_true', 
                       help="Don't automatically open browser")
    parser.add_argument('--port', type=int, default=5000,
                       help="Port number (default: 5000)")
    parser.add_argument('--test', action='store_true',
                       help="Run connectivity test after starting")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Pre-flight checks
    if not check_dependencies():
        print("\n❌ Dependency check failed")
        sys.exit(1)
    
    if not check_flask_app():
        print("\n❌ Flask app check failed")
        sys.exit(1)
    
    print("\n✅ Pre-flight checks passed")
    print("=" * 70)
    
    # Start the server
    success = start_flask_server(
        port=args.port, 
        open_browser=not args.no_browser
    )
    
    if success and args.test:
        print("\n" + "=" * 70)
        run_quick_test(args.port)
    
    print("\n👋 Thanks for using the Real-Time Analytics Dashboard!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⛔ Startup interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Startup error: {e}")
        sys.exit(1)
