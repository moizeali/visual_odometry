#!/usr/bin/env python3
"""
Installation and setup script for Visual Odometry Enhanced System
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print installation banner"""
    print("=" * 60)
    print("🚀 Visual Odometry Enhanced System - Installation Script")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False

    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_requirements():
    """Install Python requirements"""
    print("\n📦 Installing Python dependencies...")

    try:
        # Install main requirements
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Main dependencies installed")

        # Install development dependencies (optional)
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-e", ".[dev]"
            ])
            print("✅ Development dependencies installed")
        except subprocess.CalledProcessError:
            print("⚠️  Development dependencies skipped (optional)")

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

    return True

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")

    directories = [
        "datasets",
        "logs",
        "outputs",
        "temp"
    ]

    for dir_name in directories:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        print(f"✅ Created: {dir_name}/")

    return True

def check_opencv():
    """Check OpenCV installation and features"""
    print("\n🔍 Checking OpenCV installation...")

    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")

        # Check for non-free algorithms (SIFT, SURF)
        try:
            sift = cv2.SIFT_create()
            print("✅ SIFT algorithm available")
        except AttributeError:
            print("⚠️  SIFT not available (install opencv-contrib-python for full features)")

        return True
    except ImportError:
        print("❌ OpenCV not found")
        return False

def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")

    try:
        # Test imports
        sys.path.insert(0, str(Path.cwd() / "backend"))

        from core.visual_odometry import CameraParams
        from data.dataset_loader import DatasetManager
        from utils.metrics import MetricsCollector

        print("✅ Core modules imported successfully")

        # Test basic functionality
        camera_params = CameraParams(718.856, 718.856, 607.1928, 185.2157, 1241, 376)
        dataset_manager = DatasetManager()
        metrics = MetricsCollector()

        print("✅ Basic functionality test passed")
        return True

    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

def print_usage_instructions():
    """Print usage instructions"""
    print("\n" + "=" * 60)
    print("🎉 Installation completed successfully!")
    print("=" * 60)
    print()
    print("📋 Quick Start Commands:")
    print()
    print("1. 🖥️  Start the web server:")
    print("   python start_server.py")
    print()
    print("2. 🧪 Run example processing:")
    print("   python run_example.py")
    print()
    print("3. 📊 Open Jupyter notebook:")
    print("   jupyter notebook notebooks/visual_odometry_demo.ipynb")
    print()
    print("4. 🌐 Access web interface:")
    print("   http://localhost:8000")
    print()
    print("📚 Documentation:")
    print("   - README.md: Complete project documentation")
    print("   - QUICKSTART.md: Quick start guide")
    print("   - notebooks/: Interactive examples")
    print()
    print("💡 Tip: Check logs/ directory for debugging information")
    print()

def main():
    """Main installation function"""
    print_banner()

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Check if we're in the right directory
    if not Path("README.md").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)

    # Install requirements
    if not install_requirements():
        print("❌ Installation failed during dependency installation")
        sys.exit(1)

    # Create directories
    if not create_directories():
        print("❌ Installation failed during directory creation")
        sys.exit(1)

    # Check OpenCV
    check_opencv()

    # Test installation
    if not test_installation():
        print("❌ Installation test failed")
        print("💡 Try running: pip install -r requirements.txt")
        sys.exit(1)

    # Print usage instructions
    print_usage_instructions()

if __name__ == "__main__":
    main()