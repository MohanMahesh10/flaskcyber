#!/usr/bin/env python3
"""
Deployment script for ML-Enhanced Cybersecurity System
Automates the setup and deployment process
"""

import subprocess
import sys
import os
import platform
import importlib.util

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible.")
    return True

def check_pip():
    """Check if pip is available."""
    try:
        import pip
        print("✅ pip is available.")
        return True
    except ImportError:
        print("❌ pip is not available. Please install pip first.")
        return False

def install_requirements():
    """Install required packages."""
    print("\n📦 Installing required packages...")
    
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("✅ All packages installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def check_required_packages():
    """Check if all required packages are installed."""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'scikit-learn', 
        'matplotlib', 'seaborn', 'plotly', 'tensorflow',
        'xgboost', 'lightgbm', 'cryptography', 'pillow', 'cv2'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        package_name = 'opencv-python' if package == 'cv2' else package
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are installed.")
    return True

def create_sample_data():
    """Ensure sample data exists."""
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"✅ Created {data_dir} directory.")
    
    sample_file = os.path.join(data_dir, "sample_network_data.csv")
    if os.path.exists(sample_file):
        print("✅ Sample data file exists.")
    else:
        print("⚠️ Sample data file not found. Using built-in data generation.")
    
    return True

def check_system_resources():
    """Check system resources."""
    try:
        import psutil
        
        # Check RAM
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        
        if total_gb < 4:
            print(f"⚠️ System has {total_gb:.1f}GB RAM. 8GB+ recommended for optimal performance.")
        else:
            print(f"✅ System has {total_gb:.1f}GB RAM.")
        
        # Check CPU cores
        cpu_cores = psutil.cpu_count(logical=False)
        print(f"✅ System has {cpu_cores} CPU cores.")
        
    except ImportError:
        print("⚠️ Could not check system resources (psutil not available).")
    
    return True

def test_streamlit():
    """Test if Streamlit can be launched."""
    print("\n🧪 Testing Streamlit installation...")
    
    try:
        # Just check if streamlit can be imported and version obtained
        result = subprocess.run([sys.executable, "-c", "import streamlit; print(streamlit.__version__)"], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Streamlit {version} is working correctly.")
            return True
        else:
            print(f"❌ Streamlit test failed: {result.stderr}")
            return False
    
    except subprocess.TimeoutExpired:
        print("❌ Streamlit test timed out.")
        return False
    except Exception as e:
        print(f"❌ Streamlit test failed: {e}")
        return False

def launch_application():
    """Launch the Streamlit application."""
    print("\n🚀 Launching ML-Enhanced Cybersecurity System...")
    print("📱 The application will open in your default web browser.")
    print("🔗 URL: http://localhost:8501")
    print("\n💡 To stop the application, press Ctrl+C in this terminal.")
    print("=" * 60)
    
    try:
        # Launch Streamlit app
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to launch application: {e}")
        print("\n🔍 Troubleshooting:")
        print("1. Check if all dependencies are installed")
        print("2. Verify app.py exists in the current directory")
        print("3. Check for any error messages above")

def print_banner():
    """Print application banner."""
    banner = """
    ╔══════════════════════════════════════════════════╗
    ║         🔐 ML-Enhanced Cybersecurity System      ║
    ║     Cryptographic Key Generation & IDS Platform   ║
    ╚══════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Main deployment function."""
    print_banner()
    
    print("🏁 Starting deployment process...\n")
    
    # Check system compatibility
    if not check_python_version():
        sys.exit(1)
    
    if not check_pip():
        sys.exit(1)
    
    # Install requirements if not already installed
    if not check_required_packages():
        print("\n📦 Some packages are missing. Installing...")
        if not install_requirements():
            print("\n❌ Deployment failed during package installation.")
            sys.exit(1)
    
    # Verify installation
    if not check_required_packages():
        print("\n❌ Package installation verification failed.")
        sys.exit(1)
    
    # Check system resources
    check_system_resources()
    
    # Create necessary directories and files
    create_sample_data()
    
    # Test Streamlit
    if not test_streamlit():
        print("\n❌ Streamlit test failed. Please check your installation.")
        sys.exit(1)
    
    print("\n✅ All checks passed! System is ready to launch.")
    
    # Ask user if they want to launch immediately
    try:
        launch_now = input("\n🚀 Launch the application now? (y/n): ").lower().strip()
        if launch_now in ['y', 'yes', '']:
            launch_application()
        else:
            print("\n📝 To launch later, run: streamlit run app.py")
            print("🔗 The application will be available at: http://localhost:8501")
    
    except KeyboardInterrupt:
        print("\n👋 Deployment completed. Run 'streamlit run app.py' to launch.")

if __name__ == "__main__":
    main()
