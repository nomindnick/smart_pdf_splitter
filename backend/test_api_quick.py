#!/usr/bin/env python3
"""Quick API test to verify the integration is working."""

import requests
import json

def test_api_health():
    """Test basic API functionality."""
    base_url = "http://localhost:8000"
    
    print("Testing API Health Check...")
    try:
        # First, let's try to start a simple server
        import subprocess
        import time
        
        # Kill any existing process
        subprocess.run(["pkill", "-f", "uvicorn"], capture_output=True)
        time.sleep(2)
        
        # Start server
        print("Starting API server...")
        proc = subprocess.Popen(
            ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8001"],
            cwd="/home/nick/Projects/smart_pdf_splitter/backend",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        time.sleep(5)
        
        # Test health endpoint
        print("Testing health endpoint...")
        response = requests.get("http://localhost:8001/health", timeout=5)
        print(f"Health check status: {response.status_code}")
        if response.status_code == 200:
            print("Health check response:", response.json())
        
        # Test detection presets
        print("\nTesting detection presets endpoint...")
        response = requests.get("http://localhost:8001/api/v2/detection-presets", timeout=5)
        print(f"Detection presets status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("Available presets:")
            for preset in data.get('presets', []):
                print(f"  - {preset['name']}: {preset['description']}")
        
        # Test root endpoint
        print("\nTesting root endpoint...")
        response = requests.get("http://localhost:8001/", timeout=5)
        print(f"Root endpoint status: {response.status_code}")
        if response.status_code == 200:
            print("Root response:", response.json())
        
        # Kill server
        proc.terminate()
        proc.wait()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_api_health()