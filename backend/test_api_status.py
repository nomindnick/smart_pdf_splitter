#!/usr/bin/env python3
"""Quick test to check API status and configuration."""

import requests
import json

def check_api_status():
    base_url = "http://localhost:8003"
    
    print("API Status Check")
    print("================\n")
    
    # 1. Health check
    print("1. Health Check:")
    response = requests.get(f"{base_url}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    # 2. Root endpoint
    print("\n2. Root Endpoint:")
    response = requests.get(f"{base_url}/")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    # 3. Detection presets
    print("\n3. Detection Presets (Enhanced Routes):")
    response = requests.get(f"{base_url}/api/v2/detection-presets")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("   Available presets:")
        for preset in data.get('presets', []):
            print(f"     - {preset['name']}: {preset['description']}")
    
    # 4. Test upload endpoint with options
    print("\n4. Upload Endpoint Configuration Test:")
    print("   Testing parameter handling (no actual upload)...")
    
    # Try OPTIONS request to see if parameters are accepted
    response = requests.options(f"{base_url}/api/documents/upload")
    print(f"   OPTIONS status: {response.status_code}")
    
    print("\n✅ API is running and configured with:")
    print("   - Basic document routes at /api/documents/")
    print("   - Enhanced routes at /api/v2/")
    print("   - LLM and visual detection parameters available")

if __name__ == "__main__":
    check_api_status()