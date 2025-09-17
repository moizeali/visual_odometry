#!/usr/bin/env python3
"""
Startup script for Visual Odometry System
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add backend to Python path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

def start_server():
    """Start the FastAPI server"""
    print("Starting Visual Odometry System...")
    print("Dashboard will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)

    try:
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped successfully")
    except Exception as e:
        print(f"ERROR: Error starting server: {e}")

if __name__ == "__main__":
    start_server()