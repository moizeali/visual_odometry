#!/usr/bin/env python3
"""
Local development server launcher for Visual Odometry System
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_requirements():
    """Check if required dependencies are installed"""
    try:
        import cv2
        import numpy as np
        import fastapi
        print("✅ Core dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Installing requirements...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "backend/requirements.txt"])
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False

def setup_directories():
    """Create necessary directories"""
    directories = ["datasets", "logs", "frontend/templates", "frontend/static"]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("✅ Directories created")

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting Visual Odometry Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("🔧 Press Ctrl+C to stop the server")
    print("-" * 50)

    # Change to backend directory
    os.chdir("backend")

    # Start the server
    try:
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server error: {e}")

def main():
    print("🎯 Enhanced Visual Odometry System")
    print("=" * 40)

    # Check if we're in the right directory
    if not Path("backend/app.py").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)

    # Setup
    setup_directories()

    if not check_requirements():
        print("❌ Failed to setup requirements. Please install manually:")
        print("   pip install -r backend/requirements.txt")
        sys.exit(1)

    # Open browser after a short delay
    def open_browser():
        time.sleep(3)  # Wait for server to start
        webbrowser.open("http://localhost:8000")

    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    # Start server
    start_server()

if __name__ == "__main__":
    main()