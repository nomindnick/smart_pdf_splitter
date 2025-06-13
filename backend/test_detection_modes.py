#!/usr/bin/env python3
"""Test the different detection modes."""

import requests
import json
import time
from pathlib import Path

def test_detection_modes():
    """Test different detection modes."""
    
    base_url = "http://localhost:8003"
    
    # Check health
    print("Checking API health...")
    response = requests.get(f"{base_url}/health")
    print(f"Health status: {response.json()}")
    
    # Check detection presets
    print("\nChecking detection presets...")
    response = requests.get(f"{base_url}/api/v2/detection-presets")
    if response.status_code == 200:
        presets = response.json()
        print("Available presets:")
        for preset in presets.get('presets', []):
            print(f"  - {preset['name']}: {preset['description']}")
    
    # Find test file
    test_file = Path("/home/nick/Projects/smart_pdf_splitter/tests/test_files/Test_PDF_Set_1.pdf")
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return
    
    print(f"\nUsing test file: {test_file}")
    
    # Test basic upload (text only)
    print("\n" + "="*60)
    print("Testing Basic Detection (text only)")
    print("="*60)
    
    with open(test_file, 'rb') as f:
        files = {'file': ('test.pdf', f, 'application/pdf')}
        params = {
            'enable_visual_detection': False,
            'enable_llm_detection': False,
            'enable_intelligent_ocr': False
        }
        
        print("Uploading file...")
        response = requests.post(
            f"{base_url}/api/documents/upload",
            files=files,
            params=params
        )
    
    if response.status_code == 200:
        upload_data = response.json()
        document_id = upload_data['document_id']
        print(f"Upload successful. Document ID: {document_id}")
        
        # Poll for completion
        print("Processing...")
        for i in range(30):  # Max 2.5 minutes
            time.sleep(5)
            status_response = requests.get(f"{base_url}/api/documents/{document_id}/status")
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                status = status_data['status']
                
                if status == 'completed':
                    print(f"Processing completed!")
                    boundaries = status_data.get('detected_boundaries', [])
                    print(f"Detected {len(boundaries)} boundaries")
                    
                    # Show first few boundaries
                    for idx, boundary in enumerate(boundaries[:3]):
                        print(f"\nBoundary {idx+1}:")
                        print(f"  Pages: {boundary['start_page']}-{boundary['end_page']}")
                        print(f"  Confidence: {boundary['confidence']:.2%}")
                        print(f"  Type: {boundary.get('document_type', 'unknown')}")
                    
                    if len(boundaries) > 3:
                        print(f"\n... and {len(boundaries) - 3} more boundaries")
                    
                    break
                elif status == 'failed':
                    print(f"Processing failed: {status_data.get('message', 'Unknown error')}")
                    break
                else:
                    print(f"Status: {status}...")
        
        # Clean up
        print("\nCleaning up...")
        requests.delete(f"{base_url}/api/documents/{document_id}")
        
    else:
        print(f"Upload failed: {response.status_code}")
        print(response.text)
    
    # Test with visual detection
    print("\n" + "="*60)
    print("Testing Visual Detection")
    print("="*60)
    
    with open(test_file, 'rb') as f:
        files = {'file': ('test.pdf', f, 'application/pdf')}
        params = {
            'enable_visual_detection': True,
            'enable_llm_detection': False,
            'enable_intelligent_ocr': False
        }
        
        print("Uploading file with visual detection enabled...")
        response = requests.post(
            f"{base_url}/api/documents/upload",
            files=files,
            params=params
        )
    
    if response.status_code == 200:
        upload_data = response.json()
        document_id = upload_data['document_id']
        print(f"Upload successful. Document ID: {document_id}")
        
        # Wait a bit longer for visual processing
        print("Processing with visual detection (this may take longer)...")
        time.sleep(10)
        
        # Check status
        status_response = requests.get(f"{base_url}/api/documents/{document_id}/status")
        if status_response.status_code == 200:
            status_data = status_response.json()
            print(f"Status: {status_data['status']}")
            
            if status_data['status'] == 'completed':
                boundaries = status_data.get('detected_boundaries', [])
                print(f"Detected {len(boundaries)} boundaries with visual detection")
        
        # Clean up
        requests.delete(f"{base_url}/api/documents/{document_id}")


if __name__ == "__main__":
    test_detection_modes()